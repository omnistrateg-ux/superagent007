"""
Microstructure Data Storage Layer

Append-only SQLite storage for orderflow research.
All timestamps normalized to MSK (UTC+3).

Tables:
- raw_trades: tick-level trade tape with aggressor side
- raw_quotes: best bid/ask snapshots
- raw_depth: L2 orderbook depth (top N levels)
- raw_oi: open interest updates (futures)
- session_labels: derived session labels per interval
- collection_stats: data quality metrics per day
- collection_gaps: detected gaps / reconnects
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

MSK = ZoneInfo("Europe/Moscow")


# =============================================================================
# SESSION DEFINITIONS
# =============================================================================

@dataclass
class SessionWindow:
    """Trading session time window."""
    name: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    trading: bool = True


MOEX_SESSIONS = [
    SessionWindow("morning_auction", 6, 50, 7, 0, trading=False),
    SessionWindow("morning_forts", 7, 0, 10, 0, trading=True),
    SessionWindow("stock_auction", 9, 50, 10, 0, trading=False),
    SessionWindow("opening_drive", 10, 0, 10, 15, trading=True),
    SessionWindow("morning_active", 10, 15, 11, 30, trading=True),
    SessionWindow("midday", 11, 30, 13, 0, trading=True),
    SessionWindow("lunch", 13, 0, 14, 0, trading=True),
    SessionWindow("clearing_1", 14, 0, 14, 5, trading=False),
    SessionWindow("afternoon", 14, 5, 16, 0, trading=True),
    SessionWindow("preclose", 16, 0, 18, 40, trading=True),
    SessionWindow("close_auction", 18, 40, 18, 50, trading=False),
    SessionWindow("clearing_2", 18, 45, 19, 5, trading=False),
    SessionWindow("evening", 19, 5, 23, 50, trading=True),
]


def get_session_label(ts: datetime) -> str:
    """Get session label for timestamp (MSK)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=MSK)
    else:
        ts = ts.astimezone(MSK)

    t = ts.time()
    for session in MOEX_SESSIONS:
        start = datetime.strptime(f"{session.start_hour}:{session.start_minute}", "%H:%M").time()
        end = datetime.strptime(f"{session.end_hour}:{session.end_minute}", "%H:%M").time()
        if start <= t < end:
            return session.name

    return "outside_hours"


def to_msk(ts: datetime) -> datetime:
    """Convert timestamp to MSK timezone."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=MSK)
    return ts.astimezone(MSK)


def msk_now() -> datetime:
    """Get current time in MSK."""
    return datetime.now(MSK)


# =============================================================================
# STORAGE CLASS
# =============================================================================

class MicrostructureStorage:
    """
    SQLite storage for microstructure research data.

    Features:
    - Append-only tables for raw data
    - All timestamps in MSK (Europe/Moscow)
    - Gap detection and quality stats
    - Batch inserts for performance
    """

    SCHEMA = """
    -- Raw trade tape
    CREATE TABLE IF NOT EXISTS raw_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,                     -- ISO timestamp (MSK)
        ticker TEXT NOT NULL,
        price REAL NOT NULL,
        qty INTEGER NOT NULL,
        side TEXT NOT NULL,                   -- 'BUY', 'SELL', 'UNKNOWN'
        trade_id TEXT,                        -- Exchange trade ID if available
        session TEXT,                         -- Session label
        collected_ts TEXT NOT NULL            -- When we received this
    );
    CREATE INDEX IF NOT EXISTS idx_raw_trades_ticker_ts ON raw_trades(ticker, ts);
    CREATE INDEX IF NOT EXISTS idx_raw_trades_session ON raw_trades(session, ts);

    -- Best bid/ask quotes
    CREATE TABLE IF NOT EXISTS raw_quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        ticker TEXT NOT NULL,
        bid_price REAL,
        bid_size INTEGER,
        ask_price REAL,
        ask_size INTEGER,
        spread_bps REAL,                      -- Spread in basis points
        session TEXT,
        collected_ts TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_raw_quotes_ticker_ts ON raw_quotes(ticker, ts);

    -- L2 orderbook depth
    CREATE TABLE IF NOT EXISTS raw_depth (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        ticker TEXT NOT NULL,
        bid_prices TEXT NOT NULL,             -- JSON array
        bid_sizes TEXT NOT NULL,              -- JSON array
        ask_prices TEXT NOT NULL,             -- JSON array
        ask_sizes TEXT NOT NULL,              -- JSON array
        levels INTEGER NOT NULL,              -- Number of levels
        imbalance_5 REAL,                     -- Top-5 imbalance
        session TEXT,
        collected_ts TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_raw_depth_ticker_ts ON raw_depth(ticker, ts);

    -- Open Interest (futures only)
    CREATE TABLE IF NOT EXISTS raw_oi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        ticker TEXT NOT NULL,
        open_interest INTEGER NOT NULL,
        oi_change INTEGER,                    -- Change from previous
        session TEXT,
        collected_ts TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_raw_oi_ticker_ts ON raw_oi(ticker, ts);

    -- Collection statistics per day/ticker
    CREATE TABLE IF NOT EXISTS collection_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,                   -- YYYY-MM-DD
        ticker TEXT NOT NULL,
        trades_count INTEGER DEFAULT 0,
        quotes_count INTEGER DEFAULT 0,
        depth_count INTEGER DEFAULT 0,
        oi_count INTEGER DEFAULT 0,
        first_ts TEXT,
        last_ts TEXT,
        gaps_count INTEGER DEFAULT 0,
        coverage_pct REAL,                    -- % of expected intervals covered
        unknown_side_pct REAL,                -- % trades with UNKNOWN side
        UNIQUE(date, ticker)
    );
    CREATE INDEX IF NOT EXISTS idx_collection_stats_date ON collection_stats(date);

    -- Detected gaps and reconnects
    CREATE TABLE IF NOT EXISTS collection_gaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        ticker TEXT NOT NULL,
        gap_type TEXT NOT NULL,               -- 'gap', 'reconnect', 'slow'
        gap_seconds REAL,                     -- Duration of gap
        prev_ts TEXT,
        next_ts TEXT,
        notes TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_collection_gaps_date ON collection_gaps(ts);

    -- State for incremental collection
    CREATE TABLE IF NOT EXISTS collection_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_ts TEXT NOT NULL
    );
    """

    def __init__(self, db_path: Path, batch_size: int = 100):
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self._conn: Optional[sqlite3.Connection] = None

        # Batch buffers
        self._trades_buffer: List[tuple] = []
        self._quotes_buffer: List[tuple] = []
        self._depth_buffer: List[tuple] = []
        self._oi_buffer: List[tuple] = []

        # Last timestamps for gap detection
        self._last_ts: Dict[str, datetime] = {}
        self._last_oi: Dict[str, int] = {}

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.executescript(self.SCHEMA)
        conn.commit()
        logger.info(f"Microstructure storage initialized: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Get or create connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None  # Autocommit for WAL
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-65536")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Flush buffers and close connection."""
        self.flush_all()
        if self._conn:
            self._conn.close()
            self._conn = None

    # === TRADE TAPE ===

    def add_trade(
        self,
        ts: datetime,
        ticker: str,
        price: float,
        qty: int,
        side: str,
        trade_id: Optional[str] = None,
    ) -> None:
        """Add trade to buffer."""
        ts_msk = to_msk(ts)
        session = get_session_label(ts_msk)
        now = msk_now()

        # Gap detection
        self._check_gap(ticker, ts_msk, "trade")

        self._trades_buffer.append((
            ts_msk.isoformat(),
            ticker,
            price,
            qty,
            side.upper() if side else "UNKNOWN",
            trade_id,
            session,
            now.isoformat(),
        ))

        if len(self._trades_buffer) >= self.batch_size:
            self._flush_trades()

    def _flush_trades(self) -> None:
        """Flush trades buffer to database."""
        if not self._trades_buffer:
            return

        conn = self._connect()
        conn.executemany(
            """INSERT INTO raw_trades (ts, ticker, price, qty, side, trade_id, session, collected_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            self._trades_buffer
        )
        logger.debug(f"Flushed {len(self._trades_buffer)} trades")
        self._trades_buffer = []

    # === QUOTES (BBO) ===

    def add_quote(
        self,
        ts: datetime,
        ticker: str,
        bid_price: Optional[float],
        bid_size: Optional[int],
        ask_price: Optional[float],
        ask_size: Optional[int],
    ) -> None:
        """Add best bid/ask quote."""
        ts_msk = to_msk(ts)
        session = get_session_label(ts_msk)
        now = msk_now()

        # Calculate spread
        spread_bps = None
        if bid_price and ask_price and bid_price > 0:
            mid = (bid_price + ask_price) / 2
            spread_bps = ((ask_price - bid_price) / mid) * 10000

        self._check_gap(ticker, ts_msk, "quote")

        self._quotes_buffer.append((
            ts_msk.isoformat(),
            ticker,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            spread_bps,
            session,
            now.isoformat(),
        ))

        if len(self._quotes_buffer) >= self.batch_size:
            self._flush_quotes()

    def _flush_quotes(self) -> None:
        """Flush quotes buffer."""
        if not self._quotes_buffer:
            return

        conn = self._connect()
        conn.executemany(
            """INSERT INTO raw_quotes (ts, ticker, bid_price, bid_size, ask_price, ask_size, spread_bps, session, collected_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            self._quotes_buffer
        )
        logger.debug(f"Flushed {len(self._quotes_buffer)} quotes")
        self._quotes_buffer = []

    # === DEPTH (L2) ===

    def add_depth(
        self,
        ts: datetime,
        ticker: str,
        bid_prices: List[float],
        bid_sizes: List[int],
        ask_prices: List[float],
        ask_sizes: List[int],
    ) -> None:
        """Add L2 orderbook depth snapshot."""
        ts_msk = to_msk(ts)
        session = get_session_label(ts_msk)
        now = msk_now()

        levels = min(len(bid_prices), len(ask_prices))

        # Calculate top-5 imbalance
        imbalance_5 = None
        if levels >= 5:
            bid_vol = sum(bid_sizes[:5])
            ask_vol = sum(ask_sizes[:5])
            total = bid_vol + ask_vol
            if total > 0:
                imbalance_5 = (bid_vol - ask_vol) / total

        self._depth_buffer.append((
            ts_msk.isoformat(),
            ticker,
            json.dumps(bid_prices),
            json.dumps(bid_sizes),
            json.dumps(ask_prices),
            json.dumps(ask_sizes),
            levels,
            imbalance_5,
            session,
            now.isoformat(),
        ))

        if len(self._depth_buffer) >= self.batch_size:
            self._flush_depth()

    def _flush_depth(self) -> None:
        """Flush depth buffer."""
        if not self._depth_buffer:
            return

        conn = self._connect()
        conn.executemany(
            """INSERT INTO raw_depth (ts, ticker, bid_prices, bid_sizes, ask_prices, ask_sizes, levels, imbalance_5, session, collected_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            self._depth_buffer
        )
        logger.debug(f"Flushed {len(self._depth_buffer)} depth snapshots")
        self._depth_buffer = []

    # === OPEN INTEREST ===

    def add_oi(
        self,
        ts: datetime,
        ticker: str,
        open_interest: int,
    ) -> None:
        """Add Open Interest update."""
        ts_msk = to_msk(ts)
        session = get_session_label(ts_msk)
        now = msk_now()

        # Calculate OI change
        oi_change = None
        if ticker in self._last_oi:
            oi_change = open_interest - self._last_oi[ticker]
        self._last_oi[ticker] = open_interest

        self._oi_buffer.append((
            ts_msk.isoformat(),
            ticker,
            open_interest,
            oi_change,
            session,
            now.isoformat(),
        ))

        if len(self._oi_buffer) >= self.batch_size:
            self._flush_oi()

    def _flush_oi(self) -> None:
        """Flush OI buffer."""
        if not self._oi_buffer:
            return

        conn = self._connect()
        conn.executemany(
            """INSERT INTO raw_oi (ts, ticker, open_interest, oi_change, session, collected_ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            self._oi_buffer
        )
        logger.debug(f"Flushed {len(self._oi_buffer)} OI updates")
        self._oi_buffer = []

    # === GAP DETECTION ===

    def _check_gap(self, ticker: str, ts: datetime, data_type: str) -> None:
        """Check for data gaps."""
        key = f"{ticker}:{data_type}"

        if key in self._last_ts:
            gap = (ts - self._last_ts[key]).total_seconds()

            # Thresholds by data type
            thresholds = {
                "trade": 30,   # 30 sec without trades = gap
                "quote": 10,   # 10 sec without quotes
                "depth": 5,    # 5 sec without depth
            }

            threshold = thresholds.get(data_type, 30)

            if gap > threshold:
                self._record_gap(ticker, ts, gap, data_type)

        self._last_ts[key] = ts

    def _record_gap(
        self,
        ticker: str,
        ts: datetime,
        gap_seconds: float,
        data_type: str,
    ) -> None:
        """Record detected gap."""
        conn = self._connect()

        gap_type = "gap" if gap_seconds > 60 else "slow"
        prev_ts = self._last_ts.get(f"{ticker}:{data_type}")

        conn.execute(
            """INSERT INTO collection_gaps (ts, ticker, gap_type, gap_seconds, prev_ts, next_ts, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ts.isoformat(),
                ticker,
                gap_type,
                gap_seconds,
                prev_ts.isoformat() if prev_ts else None,
                ts.isoformat(),
                f"{data_type} gap",
            )
        )
        logger.warning(f"Gap detected: {ticker} {data_type} {gap_seconds:.1f}s")

    def record_reconnect(self, ticker: str, ts: Optional[datetime] = None) -> None:
        """Record a reconnection event."""
        conn = self._connect()
        ts = ts or msk_now()

        conn.execute(
            """INSERT INTO collection_gaps (ts, ticker, gap_type, gap_seconds, notes)
               VALUES (?, ?, 'reconnect', 0, 'reconnected')""",
            (ts.isoformat(), ticker)
        )

    # === FLUSH ALL ===

    def flush_all(self) -> None:
        """Flush all buffers to database."""
        self._flush_trades()
        self._flush_quotes()
        self._flush_depth()
        self._flush_oi()

    # === STATE MANAGEMENT ===

    def get_state(self, key: str) -> Optional[str]:
        """Get state value."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT value FROM collection_state WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        """Set state value."""
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO collection_state (key, value, updated_ts)
               VALUES (?, ?, ?)""",
            (key, value, msk_now().isoformat())
        )

    # === DAILY STATS ===

    def update_daily_stats(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Calculate and store daily collection statistics."""
        conn = self._connect()
        target_date = target_date or date.today()
        date_str = target_date.isoformat()

        # Get unique tickers with data today
        cur = conn.execute(
            """SELECT DISTINCT ticker FROM raw_trades WHERE ts LIKE ? || '%'
               UNION SELECT DISTINCT ticker FROM raw_quotes WHERE ts LIKE ? || '%'""",
            (date_str, date_str)
        )
        tickers = [row["ticker"] for row in cur.fetchall()]

        stats = {}
        for ticker in tickers:
            # Count by type
            trades_count = conn.execute(
                "SELECT COUNT(*) as c FROM raw_trades WHERE ticker=? AND ts LIKE ? || '%'",
                (ticker, date_str)
            ).fetchone()["c"]

            quotes_count = conn.execute(
                "SELECT COUNT(*) as c FROM raw_quotes WHERE ticker=? AND ts LIKE ? || '%'",
                (ticker, date_str)
            ).fetchone()["c"]

            depth_count = conn.execute(
                "SELECT COUNT(*) as c FROM raw_depth WHERE ticker=? AND ts LIKE ? || '%'",
                (ticker, date_str)
            ).fetchone()["c"]

            oi_count = conn.execute(
                "SELECT COUNT(*) as c FROM raw_oi WHERE ticker=? AND ts LIKE ? || '%'",
                (ticker, date_str)
            ).fetchone()["c"]

            # First/last timestamps
            first_ts = conn.execute(
                """SELECT MIN(ts) as ts FROM (
                    SELECT ts FROM raw_trades WHERE ticker=? AND ts LIKE ? || '%'
                    UNION ALL SELECT ts FROM raw_quotes WHERE ticker=? AND ts LIKE ? || '%'
                )""",
                (ticker, date_str, ticker, date_str)
            ).fetchone()["ts"]

            last_ts = conn.execute(
                """SELECT MAX(ts) as ts FROM (
                    SELECT ts FROM raw_trades WHERE ticker=? AND ts LIKE ? || '%'
                    UNION ALL SELECT ts FROM raw_quotes WHERE ticker=? AND ts LIKE ? || '%'
                )""",
                (ticker, date_str, ticker, date_str)
            ).fetchone()["ts"]

            # Gaps count
            gaps_count = conn.execute(
                "SELECT COUNT(*) as c FROM collection_gaps WHERE ticker=? AND ts LIKE ? || '%'",
                (ticker, date_str)
            ).fetchone()["c"]

            # Unknown side percentage
            unknown_count = conn.execute(
                "SELECT COUNT(*) as c FROM raw_trades WHERE ticker=? AND ts LIKE ? || '%' AND side='UNKNOWN'",
                (ticker, date_str)
            ).fetchone()["c"]

            unknown_side_pct = (unknown_count / trades_count * 100) if trades_count > 0 else 0

            # Calculate coverage (% of expected 1-second intervals)
            coverage_pct = None
            if first_ts and last_ts:
                first_dt = datetime.fromisoformat(first_ts)
                last_dt = datetime.fromisoformat(last_ts)
                expected_seconds = (last_dt - first_dt).total_seconds()
                if expected_seconds > 0:
                    # Rough estimate: should have quote every ~0.5s
                    expected_quotes = expected_seconds * 2
                    coverage_pct = min(100.0, (quotes_count / expected_quotes) * 100)

            # Upsert stats
            conn.execute(
                """INSERT OR REPLACE INTO collection_stats
                   (date, ticker, trades_count, quotes_count, depth_count, oi_count,
                    first_ts, last_ts, gaps_count, coverage_pct, unknown_side_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date_str, ticker, trades_count, quotes_count, depth_count, oi_count,
                    first_ts, last_ts, gaps_count, coverage_pct, unknown_side_pct
                )
            )

            stats[ticker] = {
                "trades": trades_count,
                "quotes": quotes_count,
                "depth": depth_count,
                "oi": oi_count,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "gaps": gaps_count,
                "coverage_pct": coverage_pct,
                "unknown_side_pct": unknown_side_pct,
            }

        return stats

    # === QUERIES ===

    def get_trades(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        session: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load trades as DataFrame."""
        conn = self._connect()

        query = "SELECT * FROM raw_trades WHERE ticker = ?"
        params: List[Any] = [ticker]

        if start:
            query += " AND ts >= ?"
            params.append(to_msk(start).isoformat())
        if end:
            query += " AND ts <= ?"
            params.append(to_msk(end).isoformat())
        if session:
            query += " AND session = ?"
            params.append(session)

        query += " ORDER BY ts"

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    def get_depth(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load depth snapshots as DataFrame."""
        conn = self._connect()

        query = "SELECT * FROM raw_depth WHERE ticker = ?"
        params: List[Any] = [ticker]

        if start:
            query += " AND ts >= ?"
            params.append(to_msk(start).isoformat())
        if end:
            query += " AND ts <= ?"
            params.append(to_msk(end).isoformat())

        query += " ORDER BY ts"

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            df["bid_prices"] = df["bid_prices"].apply(json.loads)
            df["bid_sizes"] = df["bid_sizes"].apply(json.loads)
            df["ask_prices"] = df["ask_prices"].apply(json.loads)
            df["ask_sizes"] = df["ask_sizes"].apply(json.loads)
        return df

    def get_collection_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Get collection statistics."""
        conn = self._connect()

        query = "SELECT * FROM collection_stats WHERE 1=1"
        params: List[Any] = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date DESC, ticker"

        return pd.read_sql_query(query, conn, params=params)

    def get_summary(self) -> Dict[str, Any]:
        """Get overall storage summary."""
        conn = self._connect()

        summary = {
            "db_path": str(self.db_path),
            "db_size_mb": self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0,
        }

        for table in ["raw_trades", "raw_quotes", "raw_depth", "raw_oi"]:
            count = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
            summary[f"{table}_count"] = count

        # Date range
        first_ts = conn.execute(
            "SELECT MIN(ts) as ts FROM raw_trades"
        ).fetchone()["ts"]
        last_ts = conn.execute(
            "SELECT MAX(ts) as ts FROM raw_trades"
        ).fetchone()["ts"]

        summary["first_ts"] = first_ts
        summary["last_ts"] = last_ts

        # Gaps count
        gaps = conn.execute(
            "SELECT COUNT(*) as c FROM collection_gaps"
        ).fetchone()["c"]
        summary["total_gaps"] = gaps

        return summary
