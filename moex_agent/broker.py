"""
MOEX Agent v2.5 BCS QUIK API Client

Phase 2: Data collection for microstructure features.

Collects:
- Order book snapshots (every 500ms)
- Trade tape (all trades)
- Open Interest (futures)

Storage:
- SQLite tables: orderbook_snapshots, trades, oi_history
- ~50-100 MB/day for 4 futures + 10 stocks

BCS QUIK API provides:
- Aggregated order book (sum volume per level, not individual orders)
- Depth: up to 20-50 levels
- Update frequency: ~200-500ms
- Trade tape: price, volume, time, direction (buy/sell)
- Open Interest: updates after each trade (futures)

NOT available through QUIK:
- Full Orders Log (individual order add/cancel events)
- Event-based OFI/MLOFI
- Cancel/add ratio, order lifetime
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .microstructure import MicrostructureFeatures, OrderBookSnapshot, Trade

logger = logging.getLogger(__name__)


@dataclass
class BCSConfig:
    """BCS QUIK connection configuration."""
    host: str = "127.0.0.1"
    port: int = 34130
    timeout_sec: float = 5.0
    reconnect_delay_sec: float = 10.0


class BCSClient:
    """
    BCS QUIK API client for microstructure data collection.

    This is a placeholder/mock implementation.
    Real implementation requires QUIK Lua scripts or QUIK2Python bridge.

    Typical QUIK integration options:
    1. QUIK Lua scripts → write to shared memory/files
    2. QUIK2Python / QuikPy library
    3. QUIK DDE → Excel → Python
    4. Trans2QUIK.dll via ctypes

    For now, this provides the interface that real implementation should follow.
    """

    def __init__(self, config: Optional[BCSConfig] = None):
        self.config = config or BCSConfig()
        self.connected = False
        self.subscribed_tickers: List[str] = []

        # Microstructure feature calculators per ticker
        self.micro_features: Dict[str, MicrostructureFeatures] = {}

        # Callbacks for real-time updates
        self.on_book_update: Optional[Callable[[str, OrderBookSnapshot], None]] = None
        self.on_trade: Optional[Callable[[str, Trade], None]] = None
        self.on_oi_update: Optional[Callable[[str, float], None]] = None

    def connect(self) -> bool:
        """
        Connect to QUIK terminal.

        Returns:
            True if connected successfully.
        """
        # TODO: Implement real QUIK connection
        logger.warning("BCSClient.connect() - MOCK implementation")
        logger.info(f"Would connect to QUIK at {self.config.host}:{self.config.port}")
        self.connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from QUIK."""
        logger.info("Disconnecting from QUIK")
        self.connected = False
        self.subscribed_tickers = []

    def subscribe(self, tickers: List[str]) -> bool:
        """
        Subscribe to order book and trade updates for tickers.

        Args:
            tickers: List of ticker symbols (e.g., ["SBER", "GAZP", "BR"])

        Returns:
            True if subscribed successfully.
        """
        if not self.connected:
            logger.error("Not connected to QUIK")
            return False

        for ticker in tickers:
            if ticker not in self.subscribed_tickers:
                self.subscribed_tickers.append(ticker)
                self.micro_features[ticker] = MicrostructureFeatures(ticker)
                logger.info(f"Subscribed to {ticker}")

        return True

    def unsubscribe(self, tickers: List[str]) -> None:
        """Unsubscribe from tickers."""
        for ticker in tickers:
            if ticker in self.subscribed_tickers:
                self.subscribed_tickers.remove(ticker)
                self.micro_features.pop(ticker, None)
                logger.info(f"Unsubscribed from {ticker}")

    def get_orderbook(self, ticker: str) -> Optional[OrderBookSnapshot]:
        """
        Get current order book snapshot.

        Args:
            ticker: Ticker symbol

        Returns:
            OrderBookSnapshot or None if not available
        """
        # TODO: Implement real QUIK data fetch
        logger.debug(f"get_orderbook({ticker}) - MOCK")
        return None

    def get_micro_features(self, ticker: str, price_change: float = 0.0) -> Dict[str, float]:
        """
        Get current microstructure features for ticker.

        Args:
            ticker: Ticker symbol
            price_change: Recent price change for OI divergence

        Returns:
            Dict of feature_name -> value
        """
        if ticker not in self.micro_features:
            return {}

        return self.micro_features[ticker].get_features(price_change)

    def get_spread_bps(self, ticker: str) -> float:
        """Get current spread in basis points."""
        if ticker not in self.micro_features:
            return 0.0
        return self.micro_features[ticker].get_spread_bps()

    # === DATA COLLECTION ===

    def _handle_book_update(self, ticker: str, snapshot: OrderBookSnapshot) -> None:
        """Internal handler for order book updates."""
        if ticker in self.micro_features:
            self.micro_features[ticker].add_book_snapshot(snapshot)

        if self.on_book_update:
            self.on_book_update(ticker, snapshot)

    def _handle_trade(self, ticker: str, trade: Trade) -> None:
        """Internal handler for trade updates."""
        if ticker in self.micro_features:
            self.micro_features[ticker].add_trade(trade)

        if self.on_trade:
            self.on_trade(ticker, trade)

    def _handle_oi_update(self, ticker: str, oi: float) -> None:
        """Internal handler for OI updates (futures)."""
        if ticker in self.micro_features:
            self.micro_features[ticker].add_oi(pd.Timestamp.now(tz='UTC'), oi)

        if self.on_oi_update:
            self.on_oi_update(ticker, oi)


class MicrostructureStorage:
    """
    SQLite storage for microstructure data.

    Tables:
    - orderbook_snapshots: ticker, timestamp, bid_prices, bid_vols, ask_prices, ask_vols
    - trades: ticker, timestamp, price, volume, side
    - oi_history: ticker, timestamp, open_interest
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        bid_prices TEXT,  -- JSON array
        bid_volumes TEXT, -- JSON array
        ask_prices TEXT,  -- JSON array
        ask_volumes TEXT  -- JSON array
    );

    CREATE INDEX IF NOT EXISTS idx_ob_ticker_ts ON orderbook_snapshots(ticker, timestamp);

    CREATE TABLE IF NOT EXISTS micro_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        price REAL NOT NULL,
        volume REAL NOT NULL,
        side TEXT NOT NULL  -- 'buy' or 'sell'
    );

    CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON micro_trades(ticker, timestamp);

    CREATE TABLE IF NOT EXISTS oi_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        open_interest REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_oi_ticker_ts ON oi_history(ticker, timestamp);
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()
        logger.info(f"Microstructure DB initialized: {self.db_path}")

    def save_orderbook(self, snapshot: OrderBookSnapshot) -> None:
        """Save order book snapshot."""
        import json

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO orderbook_snapshots
            (ticker, timestamp, bid_prices, bid_volumes, ask_prices, ask_volumes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.ticker,
                snapshot.timestamp.isoformat(),
                json.dumps(snapshot.bid_prices),
                json.dumps(snapshot.bid_volumes),
                json.dumps(snapshot.ask_prices),
                json.dumps(snapshot.ask_volumes),
            )
        )
        conn.commit()
        conn.close()

    def save_trade(self, trade: Trade) -> None:
        """Save trade."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO micro_trades (ticker, timestamp, price, volume, side)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                trade.ticker,
                trade.timestamp.isoformat(),
                trade.price,
                trade.volume,
                trade.side,
            )
        )
        conn.commit()
        conn.close()

    def save_oi(self, ticker: str, timestamp: pd.Timestamp, oi: float) -> None:
        """Save Open Interest."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO oi_history (ticker, timestamp, open_interest)
            VALUES (?, ?, ?)
            """,
            (ticker, timestamp.isoformat(), oi)
        )
        conn.commit()
        conn.close()

    def load_orderbooks(
        self,
        ticker: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> List[OrderBookSnapshot]:
        """Load order book snapshots from database."""
        import json

        conn = sqlite3.connect(self.db_path)
        query = "SELECT timestamp, bid_prices, bid_volumes, ask_prices, ask_volumes FROM orderbook_snapshots WHERE ticker = ?"
        params = [ticker]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        snapshots = []
        for row in rows:
            snapshots.append(OrderBookSnapshot(
                timestamp=pd.Timestamp(row[0]),
                ticker=ticker,
                bid_prices=json.loads(row[1]) if row[1] else [],
                bid_volumes=json.loads(row[2]) if row[2] else [],
                ask_prices=json.loads(row[3]) if row[3] else [],
                ask_volumes=json.loads(row[4]) if row[4] else [],
            ))

        return snapshots

    def load_trades(
        self,
        ticker: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> List[Trade]:
        """Load trades from database."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT timestamp, price, volume, side FROM micro_trades WHERE ticker = ?"
        params = [ticker]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(Trade(
                timestamp=pd.Timestamp(row[0]),
                ticker=ticker,
                price=row[1],
                volume=row[2],
                side=row[3],
            ))

        return trades

    def get_data_stats(self) -> Dict[str, int]:
        """Get counts of stored data."""
        conn = sqlite3.connect(self.db_path)

        stats = {}
        for table in ["orderbook_snapshots", "micro_trades", "oi_history"]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()
        return stats
