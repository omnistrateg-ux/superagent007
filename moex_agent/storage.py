"""
MOEX Agent v2 Storage Layer

SQLite database operations with WAL mode for performance.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("moex_agent.storage")


def connect(db_path: Path, optimize: bool = True) -> sqlite3.Connection:
    """
    Connect to SQLite with performance optimizations.

    Args:
        db_path: Path to SQLite database file
        optimize: Whether to apply performance optimizations

    Returns:
        SQLite connection object
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    if optimize:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")

    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def database(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections."""
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
        logger.debug("Database connection closed")


def init_db(conn: sqlite3.Connection, schema_path: Path) -> None:
    """Initialize database schema from SQL file."""
    sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()
    logger.info(f"Database initialized from {schema_path}")


def upsert_many(
    conn: sqlite3.Connection,
    table: str,
    columns: Tuple[str, ...],
    rows: Iterable[Tuple[Any, ...]],
) -> int:
    """Insert or replace multiple rows."""
    rows_list = list(rows)
    if not rows_list:
        return 0

    cols = ",".join(columns)
    qmarks = ",".join(["?"] * len(columns))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({qmarks})"
    cur = conn.executemany(sql, rows_list)
    conn.commit()
    return cur.rowcount


def get_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Get value from state table."""
    cur = conn.execute("SELECT value FROM state WHERE key=?", (key,))
    row = cur.fetchone()
    return None if row is None else str(row["value"])


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set value in state table."""
    conn.execute("INSERT OR REPLACE INTO state(key,value) VALUES (?,?)", (key, value))
    conn.commit()


_ALLOWED_TABLES = frozenset({"candles", "quotes", "alerts", "trades", "state"})


def get_max_ts(conn: sqlite3.Connection, table: str = "candles") -> Optional[str]:
    """Get maximum timestamp from a table."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    cur = conn.execute(f"SELECT MAX(ts) as max_ts FROM {table}")
    row = cur.fetchone()
    return row["max_ts"] if row and row["max_ts"] else None


def get_window(
    conn: sqlite3.Connection,
    minutes: int,
    anchor_ts: Optional[str] = None,
    interval: int = 1,
) -> pd.DataFrame:
    """
    Get candles window ending at anchor timestamp.

    Args:
        conn: SQLite connection
        minutes: Window size in minutes
        anchor_ts: End timestamp or None for latest
        interval: Candle interval

    Returns:
        DataFrame with candle data
    """
    if anchor_ts is None:
        anchor_ts = get_max_ts(conn)

    if anchor_ts is None:
        return pd.DataFrame(columns=["secid", "ts", "open", "high", "low", "close", "value", "volume"])

    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = ?
      AND ts >= datetime(?, ?)
      AND ts <= ?
    ORDER BY secid, ts
    """
    modifier = f"-{minutes} minutes"
    df = pd.read_sql_query(q, conn, params=(interval, anchor_ts, modifier, anchor_ts))
    return df


def get_recent_candles(
    conn: sqlite3.Connection,
    days: int = 3,
    interval: int = 1,
) -> pd.DataFrame:
    """Get candles for the last N days."""
    return get_window(conn, minutes=days * 24 * 60, interval=interval)


def save_alert(
    conn: sqlite3.Connection,
    secid: str,
    direction: str,
    horizon: str,
    p: float,
    signal_type: str,
    entry: Optional[float] = None,
    take: Optional[float] = None,
    stop: Optional[float] = None,
    ttl_minutes: Optional[int] = None,
    anomaly_score: Optional[float] = None,
    payload_json: Optional[str] = None,
) -> int:
    """Save alert to database."""
    created_ts = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO alerts (created_ts, secid, direction, horizon, p, signal_type, entry, take, stop, ttl_minutes, anomaly_score, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (created_ts, secid, direction, horizon, p, signal_type, entry, take, stop, ttl_minutes, anomaly_score, payload_json),
    )
    conn.commit()
    return cur.lastrowid or 0


def mark_alert_sent(conn: sqlite3.Connection, alert_id: int) -> None:
    """Mark an alert as sent."""
    conn.execute("UPDATE alerts SET sent = 1 WHERE id = ?", (alert_id,))
    conn.commit()


def get_alerts(
    conn: sqlite3.Connection,
    limit: int = 100,
    sent_only: bool = False,
) -> List[sqlite3.Row]:
    """Get recent alerts from database."""
    where = "WHERE sent = 1" if sent_only else ""
    q = f"""
    SELECT * FROM alerts
    {where}
    ORDER BY created_ts DESC
    LIMIT ?
    """
    cur = conn.execute(q, (limit,))
    return cur.fetchall()


def save_trade(
    conn: sqlite3.Connection,
    secid: str,
    direction: str,
    entry_price: float,
    exit_price: Optional[float],
    pnl: Optional[float],
    status: str,
    horizon: str,
) -> int:
    """Save trade to database."""
    created_ts = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO trades (created_ts, secid, direction, entry_price, exit_price, pnl, status, horizon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (created_ts, secid, direction, entry_price, exit_price, pnl, status, horizon),
    )
    conn.commit()
    return cur.lastrowid or 0


def get_daily_trades(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Get trades from today."""
    q = """
    SELECT * FROM trades
    WHERE date(created_ts) = date('now')
    ORDER BY created_ts DESC
    """
    cur = conn.execute(q)
    return cur.fetchall()
