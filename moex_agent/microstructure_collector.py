"""
MOEX Microstructure Data Collector

Collects tick-level orderflow data for research:
- Trade tape with aggressor side
- Best bid/ask quotes
- L2 orderbook depth
- Open Interest (futures)

Data Sources:
1. QUIK via LuaSocket/QuikPy (production)
2. ISS API polling (fallback, no aggressor side)
3. Simulated (development/testing)

Usage:
    # Start collector
    python -m moex_agent.microstructure_collector --tickers BR,RI,SBER --mode quik

    # Dry run with simulated data
    python -m moex_agent.microstructure_collector --tickers SBER --mode sim --duration 60

    # ISS fallback (no aggressor side!)
    python -m moex_agent.microstructure_collector --tickers SBER --mode iss
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import random

from .microstructure_storage import MicrostructureStorage, msk_now, MSK

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class RawTrade:
    """Raw trade from tape."""
    ts: datetime
    ticker: str
    price: float
    qty: int
    side: str  # 'BUY', 'SELL', 'UNKNOWN'
    trade_id: Optional[str] = None


@dataclass
class RawQuote:
    """Best bid/ask quote."""
    ts: datetime
    ticker: str
    bid_price: Optional[float]
    bid_size: Optional[int]
    ask_price: Optional[float]
    ask_size: Optional[int]


@dataclass
class RawDepth:
    """L2 orderbook snapshot."""
    ts: datetime
    ticker: str
    bid_prices: List[float]
    bid_sizes: List[int]
    ask_prices: List[float]
    ask_sizes: List[int]


@dataclass
class RawOI:
    """Open Interest update."""
    ts: datetime
    ticker: str
    open_interest: int


class CollectorMode(Enum):
    """Data source mode."""
    QUIK = "quik"      # Real QUIK connection
    ISS = "iss"        # ISS API fallback
    SIM = "sim"        # Simulated data


# =============================================================================
# DATA SOURCE INTERFACE
# =============================================================================

class DataSource(ABC):
    """Abstract data source interface."""

    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.running = False

        # Callbacks
        self.on_trade: Optional[Callable[[RawTrade], None]] = None
        self.on_quote: Optional[Callable[[RawQuote], None]] = None
        self.on_depth: Optional[Callable[[RawDepth], None]] = None
        self.on_oi: Optional[Callable[[RawOI], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None

    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source. Returns True if successful."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from data source."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start receiving data."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop receiving data."""
        pass

    @property
    @abstractmethod
    def provides_aggressor_side(self) -> bool:
        """Returns True if source provides aggressor side."""
        pass


# =============================================================================
# QUIK DATA SOURCE (PRODUCTION) - Wrapper for quik_source module
# =============================================================================

@dataclass
class QUIKConfig:
    """QUIK connection configuration."""
    host: str = "127.0.0.1"
    port: int = 34130
    timeout_sec: float = 5.0
    reconnect_delay_sec: float = 10.0
    max_reconnect_attempts: int = 10
    depth_levels: int = 20
    snapshot_interval_ms: int = 500
    heartbeat_interval_sec: float = 30.0


class QUIKDataSourceWrapper(DataSource):
    """
    QUIK data source wrapper using quik_source.QUIKDataSource.

    Provides aggressor side via OnAllTrade callback.
    Supports QuikPy and Lua socket bridge backends.

    Requirements:
    - QUIK terminal running
    - BCS broker account
    - QuikPy installed: pip install quikpy
    """

    def __init__(self, tickers: List[str], config: Optional[QUIKConfig] = None):
        super().__init__(tickers)
        self.config = config or QUIKConfig()
        self._quik_source = None

    def connect(self) -> bool:
        """Connect to QUIK via quik_source module."""
        try:
            from .quik_source import QUIKDataSource as RealQUIKSource, QUIKConfig as RealConfig

            real_config = RealConfig(
                host=self.config.host,
                port=self.config.port,
                timeout_sec=self.config.timeout_sec,
                reconnect_delay_sec=self.config.reconnect_delay_sec,
                max_reconnect_attempts=self.config.max_reconnect_attempts,
                depth_levels=self.config.depth_levels,
                snapshot_interval_ms=self.config.snapshot_interval_ms,
                heartbeat_interval_sec=self.config.heartbeat_interval_sec,
            )

            self._quik_source = RealQUIKSource(self.tickers, real_config)
            return self._quik_source.connect()

        except ImportError as e:
            logger.error(f"Failed to import quik_source: {e}")
            return False

        except Exception as e:
            logger.error(f"QUIK connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from QUIK."""
        if self._quik_source:
            self._quik_source.disconnect()

    def start(self) -> None:
        """Start receiving data."""
        if not self._quik_source:
            logger.error("QUIK source not initialized")
            return

        self.running = True

        # Wire up callbacks
        self._quik_source.on_trade = self._forward_trade
        self._quik_source.on_quote = self._forward_quote
        self._quik_source.on_depth = self._forward_depth
        self._quik_source.on_oi = self._forward_oi
        self._quik_source.on_error = self._forward_error

        self._quik_source.start()

    def _forward_trade(self, trade) -> None:
        """Forward trade from quik_source."""
        if self.on_trade:
            self.on_trade(RawTrade(
                ts=trade.ts,
                ticker=trade.ticker,
                price=trade.price,
                qty=trade.qty,
                side=trade.side,
                trade_id=trade.trade_id,
            ))

    def _forward_quote(self, quote) -> None:
        """Forward quote from quik_source."""
        if self.on_quote:
            self.on_quote(RawQuote(
                ts=quote.ts,
                ticker=quote.ticker,
                bid_price=quote.bid_price,
                bid_size=quote.bid_size,
                ask_price=quote.ask_price,
                ask_size=quote.ask_size,
            ))

    def _forward_depth(self, depth) -> None:
        """Forward depth from quik_source."""
        if self.on_depth:
            self.on_depth(RawDepth(
                ts=depth.ts,
                ticker=depth.ticker,
                bid_prices=depth.bid_prices,
                bid_sizes=depth.bid_sizes,
                ask_prices=depth.ask_prices,
                ask_sizes=depth.ask_sizes,
            ))

    def _forward_oi(self, oi) -> None:
        """Forward OI from quik_source."""
        if self.on_oi:
            self.on_oi(RawOI(
                ts=oi.ts,
                ticker=oi.ticker,
                open_interest=oi.open_interest,
            ))

    def _forward_error(self, context: str, error: Exception) -> None:
        """Forward error from quik_source."""
        if self.on_error:
            self.on_error(context, error)

    def stop(self) -> None:
        """Stop receiving data."""
        self.running = False
        if self._quik_source:
            self._quik_source.stop()

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get QUIK-specific quality metrics."""
        if self._quik_source:
            return self._quik_source.get_quality_metrics()
        return {}

    @property
    def provides_aggressor_side(self) -> bool:
        """QUIK provides aggressor side via OnAllTrade."""
        return True


# =============================================================================
# ISS DATA SOURCE (FALLBACK)
# =============================================================================

class ISSDataSource(DataSource):
    """
    ISS API data source (fallback mode).

    WARNING: Does NOT provide aggressor side for trades!
    Only use if QUIK is unavailable.

    Uses MOEX ISS API endpoints:
    - /securities/{secid}.json for quotes
    - /engines/stock/markets/shares/securities/{secid}/trades.json for trades
    """

    # Ticker to ISS parameters mapping
    TICKER_MAP = {
        # Stocks (TQBR board)
        "SBER": ("stock", "shares", "TQBR", "SBER"),
        "GAZP": ("stock", "shares", "TQBR", "GAZP"),
        "LKOH": ("stock", "shares", "TQBR", "LKOH"),
        "ROSN": ("stock", "shares", "TQBR", "ROSN"),
        "GMKN": ("stock", "shares", "TQBR", "GMKN"),
        "NVTK": ("stock", "shares", "TQBR", "NVTK"),
        # Futures (RFUD board)
        "BR": ("futures", "forts", "RFUD", "BR"),
        "RI": ("futures", "forts", "RFUD", "RI"),
        "MX": ("futures", "forts", "RFUD", "MX"),
        "Si": ("futures", "forts", "RFUD", "Si"),
    }

    ISS_BASE = "https://iss.moex.com/iss"

    def __init__(self, tickers: List[str], poll_interval_sec: float = 1.0):
        super().__init__(tickers)
        self.poll_interval_sec = poll_interval_sec
        self._thread: Optional[threading.Thread] = None
        self._last_trade_id: Dict[str, int] = {}
        self._session = None

    def connect(self) -> bool:
        """ISS API doesn't require connection."""
        import requests
        self._session = requests.Session()
        logger.warning("ISS mode: NO AGGRESSOR SIDE available!")
        logger.warning("Trade side will be 'UNKNOWN' - research value limited!")
        return True

    def disconnect(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._session = None

    def start(self) -> None:
        """Start polling ISS API."""
        self.running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _get_ticker_params(self, ticker: str) -> tuple:
        """Get ISS parameters for ticker."""
        if ticker in self.TICKER_MAP:
            return self.TICKER_MAP[ticker]
        # Default to stock
        return ("stock", "shares", "TQBR", ticker)

    def _fetch_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch current quote from ISS."""
        if not self._session:
            return None

        engine, market, board, secid = self._get_ticker_params(ticker)

        try:
            url = f"{self.ISS_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}.json"
            params = {"iss.meta": "off", "iss.only": "marketdata"}

            r = self._session.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()

            marketdata = data.get("marketdata", {}).get("data", [])
            if not marketdata:
                return None

            # Find columns
            columns = data.get("marketdata", {}).get("columns", [])
            row = marketdata[0]

            def get_col(name: str) -> Any:
                try:
                    idx = columns.index(name)
                    return row[idx] if idx < len(row) else None
                except (ValueError, IndexError):
                    return None

            return {
                "bid": get_col("BID"),
                "bid_size": get_col("BIDDEPTH") or get_col("BIDDEPTHT"),
                "ask": get_col("OFFER"),
                "ask_size": get_col("OFFERDEPTH") or get_col("OFFERDEPTHT"),
                "last": get_col("LAST"),
                "volume": get_col("VOLTODAY"),
            }

        except Exception as e:
            logger.debug(f"Failed to fetch quote for {ticker}: {e}")
            return None

    def _fetch_trades(self, ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent trades from ISS."""
        if not self._session:
            return []

        engine, market, board, secid = self._get_ticker_params(ticker)

        try:
            url = f"{self.ISS_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/trades.json"
            params = {"iss.meta": "off", "limit": limit}

            r = self._session.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()

            trades_data = data.get("trades", {}).get("data", [])
            columns = data.get("trades", {}).get("columns", [])

            if not trades_data:
                return []

            result = []
            for row in trades_data:
                trade = {}
                for i, col in enumerate(columns):
                    if i < len(row):
                        trade[col] = row[i]
                result.append(trade)

            return result

        except Exception as e:
            logger.debug(f"Failed to fetch trades for {ticker}: {e}")
            return []

    def _poll_loop(self) -> None:
        """Poll ISS API for data."""
        while self.running:
            ts = msk_now()

            for ticker in self.tickers:
                try:
                    # Fetch quote
                    quote_data = self._fetch_quote(ticker)
                    if quote_data and self.on_quote:
                        quote = RawQuote(
                            ts=ts,
                            ticker=ticker,
                            bid_price=quote_data.get("bid"),
                            bid_size=quote_data.get("bid_size"),
                            ask_price=quote_data.get("ask"),
                            ask_size=quote_data.get("ask_size"),
                        )
                        self.on_quote(quote)

                    # Fetch recent trades
                    trades = self._fetch_trades(ticker, limit=50)
                    if trades and self.on_trade:
                        for trade_data in trades:
                            trade_id = trade_data.get("TRADENO") or trade_data.get("tradeno")
                            if trade_id and trade_id > self._last_trade_id.get(ticker, 0):
                                # ISS doesn't provide aggressor side!
                                trade_ts = trade_data.get("SYSTIME") or trade_data.get("systime") or ts.isoformat()
                                if isinstance(trade_ts, str):
                                    try:
                                        trade_ts = datetime.fromisoformat(trade_ts.replace("Z", "+00:00"))
                                    except Exception:
                                        trade_ts = ts

                                trade = RawTrade(
                                    ts=trade_ts if isinstance(trade_ts, datetime) else ts,
                                    ticker=ticker,
                                    price=float(trade_data.get("PRICE") or trade_data.get("price") or 0),
                                    qty=int(trade_data.get("QUANTITY") or trade_data.get("quantity") or 0),
                                    side="UNKNOWN",  # ISS limitation!
                                    trade_id=str(trade_id),
                                )
                                self.on_trade(trade)
                                self._last_trade_id[ticker] = trade_id

                except Exception as e:
                    logger.error(f"ISS poll error for {ticker}: {e}")
                    if self.on_error:
                        self.on_error(ticker, e)

            time.sleep(self.poll_interval_sec)

    def stop(self) -> None:
        """Stop polling."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def provides_aggressor_side(self) -> bool:
        """ISS does NOT provide aggressor side."""
        return False


# =============================================================================
# SIMULATED DATA SOURCE (TESTING)
# =============================================================================

class SimulatedDataSource(DataSource):
    """
    Simulated data for development and testing.

    Generates realistic orderflow patterns:
    - Random walk price
    - Correlated bid/ask sizes
    - Trade clustering (bursts)
    - Session-aware activity levels
    """

    def __init__(
        self,
        tickers: List[str],
        interval_ms: int = 100,
        trade_prob: float = 0.3,
    ):
        super().__init__(tickers)
        self.interval_ms = interval_ms
        self.trade_prob = trade_prob
        self._thread: Optional[threading.Thread] = None
        self._prices: Dict[str, float] = {}
        self._trade_count = 0

    def connect(self) -> bool:
        """Initialize simulated market."""
        logger.info("Simulated data source initialized")

        # Initialize prices
        base_prices = {
            "BR": 85.0,
            "RI": 110000.0,
            "MX": 3200.0,
            "Si": 92.0,
            "SBER": 280.0,
            "GAZP": 170.0,
            "LKOH": 7200.0,
            "ROSN": 550.0,
        }
        for ticker in self.tickers:
            self._prices[ticker] = base_prices.get(ticker, 1000.0)

        return True

    def disconnect(self) -> None:
        """Nothing to disconnect."""
        pass

    def start(self) -> None:
        """Start generating simulated data."""
        self.running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()

    def _generate_loop(self) -> None:
        """Generate simulated market data."""
        while self.running:
            ts = msk_now()

            for ticker in self.tickers:
                self._generate_tick(ticker, ts)

            time.sleep(self.interval_ms / 1000.0)

    def _generate_tick(self, ticker: str, ts: datetime) -> None:
        """Generate one tick of data for ticker."""
        # Random walk with mean reversion
        price = self._prices[ticker]
        drift = (random.random() - 0.5) * 0.0005
        price *= (1 + drift)
        self._prices[ticker] = price

        # Spread based on ticker liquidity
        spreads = {"BR": 0.01, "RI": 10.0, "SBER": 0.05, "GAZP": 0.1}
        spread = spreads.get(ticker, price * 0.0002)

        bid = price - spread / 2
        ask = price + spread / 2

        # Generate quote
        if self.on_quote:
            quote = RawQuote(
                ts=ts,
                ticker=ticker,
                bid_price=bid,
                bid_size=random.randint(10, 200),
                ask_price=ask,
                ask_size=random.randint(10, 200),
            )
            self.on_quote(quote)

        # Generate trade with probability
        if self.on_trade and random.random() < self.trade_prob:
            # Determine aggressor side
            side = "BUY" if random.random() > 0.5 else "SELL"
            trade_price = ask if side == "BUY" else bid

            self._trade_count += 1
            trade = RawTrade(
                ts=ts,
                ticker=ticker,
                price=trade_price,
                qty=random.randint(1, 20),
                side=side,
                trade_id=str(self._trade_count),
            )
            self.on_trade(trade)

        # Generate depth occasionally
        if self.on_depth and random.random() < 0.2:
            levels = 10
            bid_prices = [bid - spread * i for i in range(levels)]
            ask_prices = [ask + spread * i for i in range(levels)]

            # Sizes with some correlation
            base_size = random.randint(20, 100)
            bid_sizes = [max(1, base_size + random.randint(-20, 20)) for _ in range(levels)]
            ask_sizes = [max(1, base_size + random.randint(-20, 20)) for _ in range(levels)]

            depth = RawDepth(
                ts=ts,
                ticker=ticker,
                bid_prices=bid_prices,
                bid_sizes=bid_sizes,
                ask_prices=ask_prices,
                ask_sizes=ask_sizes,
            )
            self.on_depth(depth)

        # Generate OI for futures
        if self.on_oi and ticker in ("BR", "RI", "MX", "Si") and random.random() < 0.05:
            oi = RawOI(
                ts=ts,
                ticker=ticker,
                open_interest=random.randint(100000, 500000),
            )
            self.on_oi(oi)

    def stop(self) -> None:
        """Stop generating data."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def provides_aggressor_side(self) -> bool:
        """Simulated data provides aggressor side."""
        return True


# =============================================================================
# COLLECTOR
# =============================================================================

@dataclass
class CollectorStats:
    """Collection statistics."""
    start_time: datetime = field(default_factory=msk_now)
    trades_count: int = 0
    quotes_count: int = 0
    depth_count: int = 0
    oi_count: int = 0
    errors_count: int = 0
    unknown_side_count: int = 0
    reconnect_count: int = 0
    last_flush_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    @property
    def unknown_side_pct(self) -> float:
        """Percentage of trades with unknown side."""
        if self.trades_count == 0:
            return 0.0
        return (self.unknown_side_count / self.trades_count) * 100


class MicrostructureCollector:
    """
    Main collector orchestrating data source and storage.

    Responsibilities:
    - Connect to data source
    - Store incoming data
    - Periodic flush and stats
    - Graceful shutdown
    """

    def __init__(
        self,
        tickers: List[str],
        storage: MicrostructureStorage,
        mode: CollectorMode = CollectorMode.SIM,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.tickers = tickers
        self.storage = storage
        self.mode = mode
        self.config = config or {}
        self.running = False

        self.stats = CollectorStats()
        self._source: Optional[DataSource] = None
        self._stats_thread: Optional[threading.Thread] = None

    def _create_source(self) -> DataSource:
        """Create data source based on mode."""
        if self.mode == CollectorMode.QUIK:
            quik_config = QUIKConfig(**{
                k: v for k, v in self.config.items()
                if k in QUIKConfig.__dataclass_fields__
            })
            return QUIKDataSource(self.tickers, quik_config)

        elif self.mode == CollectorMode.ISS:
            return ISSDataSource(
                self.tickers,
                poll_interval_sec=self.config.get("poll_interval_sec", 1.0)
            )

        else:  # SIM
            return SimulatedDataSource(
                self.tickers,
                interval_ms=self.config.get("interval_ms", 100),
                trade_prob=self.config.get("trade_prob", 0.3),
            )

    def start(self) -> bool:
        """Start collection."""
        logger.info(f"Starting collector: mode={self.mode.value}, tickers={self.tickers}")

        # Create and connect source
        self._source = self._create_source()

        if not self._source.connect():
            logger.error("Failed to connect to data source")
            return False

        # Warn if no aggressor side
        if not self._source.provides_aggressor_side:
            logger.warning("=" * 60)
            logger.warning("WARNING: Data source does NOT provide aggressor side!")
            logger.warning("Trade side will be 'UNKNOWN'. Research value limited.")
            logger.warning("=" * 60)

        # Set up callbacks
        self._source.on_trade = self._handle_trade
        self._source.on_quote = self._handle_quote
        self._source.on_depth = self._handle_depth
        self._source.on_oi = self._handle_oi
        self._source.on_error = self._handle_error

        # Set up reconnect callback for QUIK mode
        if hasattr(self._source, 'on_reconnect'):
            self._source.on_reconnect = self._handle_reconnect

        # Start receiving
        self.running = True
        self._source.start()

        # Start stats reporter
        self._stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        self._stats_thread.start()

        logger.info("Collector started")
        return True

    def stop(self) -> None:
        """Stop collection gracefully."""
        logger.info("Stopping collector...")
        self.running = False

        if self._source:
            self._source.stop()
            self._source.disconnect()

        # Flush remaining data
        self.storage.flush_all()

        # Update daily stats
        try:
            self.storage.update_daily_stats()
        except Exception as e:
            logger.error(f"Failed to update daily stats: {e}")

        logger.info(f"Collector stopped. Stats: {self._format_stats()}")

    def _handle_trade(self, trade: RawTrade) -> None:
        """Handle incoming trade."""
        self.storage.add_trade(
            ts=trade.ts,
            ticker=trade.ticker,
            price=trade.price,
            qty=trade.qty,
            side=trade.side,
            trade_id=trade.trade_id,
        )
        self.stats.trades_count += 1
        self.stats.last_trade_time = msk_now()
        if trade.side == "UNKNOWN":
            self.stats.unknown_side_count += 1

    def _handle_quote(self, quote: RawQuote) -> None:
        """Handle incoming quote."""
        self.storage.add_quote(
            ts=quote.ts,
            ticker=quote.ticker,
            bid_price=quote.bid_price,
            bid_size=quote.bid_size,
            ask_price=quote.ask_price,
            ask_size=quote.ask_size,
        )
        self.stats.quotes_count += 1

    def _handle_depth(self, depth: RawDepth) -> None:
        """Handle incoming depth snapshot."""
        self.storage.add_depth(
            ts=depth.ts,
            ticker=depth.ticker,
            bid_prices=depth.bid_prices,
            bid_sizes=depth.bid_sizes,
            ask_prices=depth.ask_prices,
            ask_sizes=depth.ask_sizes,
        )
        self.stats.depth_count += 1

    def _handle_oi(self, oi: RawOI) -> None:
        """Handle incoming OI update."""
        self.storage.add_oi(
            ts=oi.ts,
            ticker=oi.ticker,
            open_interest=oi.open_interest,
        )
        self.stats.oi_count += 1

    def _handle_error(self, ticker: str, error: Exception) -> None:
        """Handle data source error."""
        self.stats.errors_count += 1
        logger.error(f"Data source error for {ticker}: {error}")

    def _stats_loop(self) -> None:
        """Periodically report stats and flush."""
        flush_interval = 60  # seconds
        last_flush = time.time()

        while self.running:
            time.sleep(5)

            # Periodic flush
            if time.time() - last_flush > flush_interval:
                self.storage.flush_all()
                self.stats.last_flush_time = msk_now()
                last_flush = time.time()

            # Log stats every minute
            if int(time.time()) % 60 < 5:
                logger.info(f"[STATS] {self._format_stats()}")

    def _handle_reconnect(self) -> None:
        """Handle reconnection event."""
        self.stats.reconnect_count += 1
        logger.warning(f"Reconnected (total: {self.stats.reconnect_count})")
        # Record in storage for all tickers
        for ticker in self.tickers:
            self.storage.record_reconnect(ticker)

    def _format_stats(self) -> str:
        """Format stats for logging."""
        runtime = (msk_now() - self.stats.start_time).total_seconds()
        return (
            f"trades={self.stats.trades_count} "
            f"quotes={self.stats.quotes_count} "
            f"depth={self.stats.depth_count} "
            f"oi={self.stats.oi_count} "
            f"errors={self.stats.errors_count} "
            f"unknown_side={self.stats.unknown_side_pct:.1f}% "
            f"reconnects={self.stats.reconnect_count} "
            f"runtime={runtime:.0f}s"
        )

    def run(self, duration_sec: Optional[int] = None) -> None:
        """Run collector (blocking)."""
        if not self.start():
            return

        try:
            if duration_sec:
                logger.info(f"Running for {duration_sec} seconds...")
                time.sleep(duration_sec)
            else:
                logger.info("Running indefinitely. Press Ctrl+C to stop.")
                while self.running:
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.stop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MOEX Microstructure Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simulated data (for testing)
    python -m moex_agent.microstructure_collector --tickers SBER,GAZP --mode sim --duration 60

    # ISS fallback (no aggressor side!)
    python -m moex_agent.microstructure_collector --tickers SBER --mode iss

    # QUIK (production - requires running QUIK terminal)
    python -m moex_agent.microstructure_collector --tickers BR,RI --mode quik
        """
    )

    parser.add_argument(
        "--tickers", type=str, default="BR,RI,SBER",
        help="Comma-separated list of tickers"
    )
    parser.add_argument(
        "--mode", type=str, default="sim",
        choices=["quik", "iss", "sim"],
        help="Data source mode"
    )
    parser.add_argument(
        "--db-path", type=str, default="data/microstructure.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Duration in seconds (None = run until Ctrl+C)"
    )
    parser.add_argument(
        "--interval-ms", type=int, default=500,
        help="Snapshot interval in milliseconds"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    )

    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(",")]

    # Create storage
    storage = MicrostructureStorage(Path(args.db_path))

    # Create collector
    mode = CollectorMode(args.mode)
    collector = MicrostructureCollector(
        tickers=tickers,
        storage=storage,
        mode=mode,
        config={
            "snapshot_interval_ms": args.interval_ms,
            "interval_ms": args.interval_ms,
        },
    )

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        collector.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    collector.run(duration_sec=args.duration)

    # Print final summary
    summary = storage.get_summary()
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
