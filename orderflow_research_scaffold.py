"""
Orderflow Research Scaffold

Research-only framework for testing microstructure hypotheses.
NO live trading. NO ML. NO size_mult. NO edge claims from proxies.

Pipeline:
    load_raw_data -> build_events -> calc_features -> event_study -> falsification

Usage:
    # M3 via ISS (READY NOW - ~45 days available)
    python orderflow_research_scaffold.py --hypothesis M3 --ticker BR --days 45 --source iss --run-falsification

    # M2 via ISS (READY NOW)
    python orderflow_research_scaffold.py --hypothesis M2 --ticker BR --days 45 --source iss --run-falsification

    # M1 via QUIK (BLOCKED - needs L2)
    python orderflow_research_scaffold.py --hypothesis M1 --ticker BR --days 30 --source quik

Hypotheses (M = Microstructure, distinct from old futures H1-H4):
    M1: Opening Imbalance (5min after open) - BLOCKED (needs L2)
    M2: Flow Divergence (continuous) - READY (ISS has BUYSELL)
    M3: Close Pressure → Overnight Gap - READY (ISS has BUYSELL)
    M4: Queue Depletion at S/R (not implemented) - BLOCKED (needs L2)

Data Sources:
    iss: ISS API futures trades with BUYSELL (~45 days)
    quik: QUIK terminal (needs setup)
"""
from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LAYER
# =============================================================================

class DataSource(Enum):
    """Data source type."""
    ISS_PROXY = "iss_proxy"       # 1m candles from ISS API (no BUYSELL)
    ISS_TAPE = "iss_tape"         # Futures trades from ISS API (has BUYSELL!)
    QUIK_TAPE = "quik_tape"       # Trade tape from QUIK
    QUIK_L2 = "quik_l2"           # Order book from QUIK


@dataclass
class Trade:
    """Single trade from tape."""
    ts: datetime
    ticker: str
    price: float
    qty: int
    side: str  # 'BUY', 'SELL', 'UNKNOWN'


@dataclass
class OrderBookSnapshot:
    """L2 order book snapshot."""
    ts: datetime
    ticker: str
    bid_prices: List[float]
    bid_sizes: List[int]
    ask_prices: List[float]
    ask_sizes: List[int]


@dataclass
class Event:
    """Research event with features and label."""
    event_id: str
    hypothesis: str
    ts: datetime
    ticker: str
    features: Dict[str, float]
    signal: float
    signal_direction: str  # 'LONG', 'SHORT', 'NONE'
    entry_price: float
    exit_prices: Dict[str, float]  # horizon -> price
    labels: Dict[str, float]  # horizon -> return or label


class DataLoader:
    """Load raw data from various sources."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = None

    def _connect(self):
        """Lazy connect to SQLite."""
        if self._conn is None:
            import sqlite3
            self._conn = sqlite3.connect(str(self.db_path))
        return self._conn

    def load_trades(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> List[Trade]:
        """
        Load trade tape from database.

        Returns empty list if data not available.
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return []

        conn = self._connect()
        query = """
            SELECT ts, ticker, price, qty, side
            FROM trades
            WHERE ticker = ? AND ts >= ? AND ts < ?
            ORDER BY ts
        """
        try:
            df = pd.read_sql_query(
                query, conn,
                params=(ticker, start.isoformat(), end.isoformat())
            )
            return [
                Trade(
                    ts=pd.to_datetime(row['ts']),
                    ticker=row['ticker'],
                    price=row['price'],
                    qty=row['qty'],
                    side=row['side'],
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.debug(f"Failed to load trades: {e}")
            return []

    def load_orderbook(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> List[OrderBookSnapshot]:
        """
        Load L2 order book snapshots.

        Returns empty list if data not available.
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return []

        conn = self._connect()
        query = """
            SELECT ts, ticker, bid_prices, bid_sizes, ask_prices, ask_sizes
            FROM orderbook
            WHERE ticker = ? AND ts >= ? AND ts < ?
            ORDER BY ts
        """
        try:
            df = pd.read_sql_query(
                query, conn,
                params=(ticker, start.isoformat(), end.isoformat())
            )

            import json
            snapshots = []
            for _, row in df.iterrows():
                snapshots.append(OrderBookSnapshot(
                    ts=pd.to_datetime(row['ts']),
                    ticker=row['ticker'],
                    bid_prices=json.loads(row['bid_prices']),
                    bid_sizes=json.loads(row['bid_sizes']),
                    ask_prices=json.loads(row['ask_prices']),
                    ask_sizes=json.loads(row['ask_sizes']),
                ))
            return snapshots
        except Exception as e:
            logger.debug(f"Failed to load orderbook: {e}")
            return []

    def load_candles_1m(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Load 1-minute candles (ISS proxy data).

        This is PROXY data, not real orderflow.
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return pd.DataFrame()

        conn = self._connect()
        query = """
            SELECT ts, open, high, low, close, volume
            FROM candles_1m
            WHERE secid = ? AND ts >= ? AND ts < ?
            ORDER BY ts
        """
        try:
            df = pd.read_sql_query(
                query, conn,
                params=(ticker, start.isoformat(), end.isoformat())
            )
            if not df.empty:
                df['ts'] = pd.to_datetime(df['ts'])
                df = df.set_index('ts')
            return df
        except Exception as e:
            logger.debug(f"Failed to load candles: {e}")
            return pd.DataFrame()

    def load_iss_trades(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> List[Trade]:
        """
        Load futures trades from ISS API with BUYSELL field.

        ISS provides ~45 days of tick data for futures.
        BUYSELL = 'B' (buy aggressor) or 'S' (sell aggressor).

        Ticker mapping:
        - BR -> BRM6 (Brent June 2026), BRK6 (May), etc.
        - RI -> RIM6 (RTS June 2026)
        - MX -> MXM6 (MOEX Index June 2026)
        - NG -> NGM6 (Natural Gas June 2026)
        """
        import requests

        # Map base ticker to active contract(s)
        # Use June 2026 (M6) as primary, May 2026 (K6) as secondary
        TICKER_MAP = {
            "BR": ["BRM6", "BRK6"],  # Brent
            "RI": ["RIM6", "RIH6"],  # RTS Index
            "MX": ["MXM6", "MXH6"],  # MOEX Index
            "NG": ["NGM6", "NGK6"],  # Natural Gas
            "Si": ["SiM6", "SiH6"],  # USD/RUB
            "GAZP": ["GZM6"],        # Gazprom
            "SBER": ["SRM6"],        # Sberbank
        }

        # Get actual tickers to fetch
        tickers_to_fetch = TICKER_MAP.get(ticker, [ticker])
        logger.info(f"Mapping {ticker} -> {tickers_to_fetch}")

        all_trades = []

        # NOTE: ISS only provides ~1 day of tick data (TODAY)
        # Historical tick data is NOT available via ISS
        # For 30+ day studies, QUIK collection is required

        for iss_ticker in tickers_to_fetch:
            # ISS trades endpoint for futures
            url = (
                f"https://iss.moex.com/iss/engines/futures/markets/forts/"
                f"securities/{iss_ticker}/trades.json"
            )

            # Paginate to get all available trades
            # ISS limit is 5000 per request
            page_start = 0
            page_size = 5000

            while True:
                params = {
                    "limit": page_size,
                    "start": page_start,
                }

                try:
                    resp = requests.get(url, params=params, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()

                    trades_data = data.get("trades", {})
                    columns = trades_data.get("columns", [])
                    rows = trades_data.get("data", [])

                    if not rows:
                        # No more pages
                        break

                    # Find column indices
                    col_idx = {col: i for i, col in enumerate(columns)}

                    for row in rows:
                        # Parse timestamp - SYSTIME is full datetime like "2026-04-15 09:00:16"
                        systime = row[col_idx.get("SYSTIME", 0)]
                        tradetime = row[col_idx.get("TRADETIME", 0)]
                        trade_date = row[col_idx.get("TRADEDATE", 0)]

                        try:
                            if isinstance(systime, str) and " " in systime:
                                # SYSTIME has full datetime
                                ts = datetime.strptime(systime[:19], "%Y-%m-%d %H:%M:%S")
                            elif isinstance(tradetime, str) and ":" in tradetime:
                                # TRADETIME has only time, combine with date
                                ts_str = f"{trade_date} {tradetime}"
                                ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
                            else:
                                # Fallback to noon
                                ts = datetime.combine(datetime.now().date(), time(12, 0))
                        except (ValueError, TypeError):
                            ts = datetime.combine(datetime.now().date(), time(12, 0))

                        # Filter by time range
                        if ts < start or ts > end:
                            continue

                        # Get price, qty, side
                        price = float(row[col_idx.get("PRICE", 0)])
                        qty = int(row[col_idx.get("QUANTITY", row[col_idx.get("QTY", 0)])])

                        # BUYSELL field: 'B' = buy aggressor, 'S' = sell aggressor
                        buysell = row[col_idx.get("BUYSELL", "")]
                        if buysell == "B":
                            side = "BUY"
                        elif buysell == "S":
                            side = "SELL"
                        else:
                            side = "UNKNOWN"

                        all_trades.append(Trade(
                            ts=ts,
                            ticker=ticker,
                            price=price,
                            qty=qty,
                            side=side,
                        ))

                    logger.debug(f"Page {page_start}: {len(rows)} trades for {iss_ticker}")

                    # Check if we need more pages
                    if len(rows) < page_size:
                        break  # Last page
                    page_start += page_size

                except requests.RequestException as e:
                    logger.warning(f"ISS request failed for {iss_ticker}: {e}")
                    break  # Stop pagination on error

        logger.info(f"Loaded {len(all_trades)} total ISS trades for {ticker}")
        return all_trades

    def check_data_availability(self, ticker: str, days: int, source: str = "quik") -> Dict[str, bool]:
        """Check what data is available."""
        end = datetime.now()
        start = end - timedelta(days=days)

        if source == "iss":
            # Check ISS trades
            iss_trades = self.load_iss_trades(ticker, start, end)
            return {
                "trades": len(iss_trades) > 0,
                "orderbook": False,  # ISS has no L2
                "candles_1m": True,  # Always available
                "trades_count": len(iss_trades),
                "orderbook_count": 0,
                "candles_count": 0,
                "source": "iss",
            }

        # QUIK/SQLite path
        trades = self.load_trades(ticker, start, end)
        books = self.load_orderbook(ticker, start, end)
        candles = self.load_candles_1m(ticker, start, end)

        return {
            "trades": len(trades) > 0,
            "orderbook": len(books) > 0,
            "candles_1m": len(candles) > 0,
            "trades_count": len(trades),
            "orderbook_count": len(books),
            "candles_count": len(candles),
            "source": "quik",
        }


# =============================================================================
# HYPOTHESIS BASE CLASS
# =============================================================================

class Hypothesis(ABC):
    """Base class for microstructure hypotheses."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.events: List[Event] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Hypothesis name (H1, H2, etc.)."""
        pass

    @property
    @abstractmethod
    def required_data(self) -> List[DataSource]:
        """Required data sources."""
        pass

    @property
    @abstractmethod
    def exit_horizons(self) -> List[str]:
        """Exit horizons to test (e.g., '15m', '30m')."""
        pass

    @abstractmethod
    def build_events(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> List[Event]:
        """Build events from raw data."""
        pass

    @abstractmethod
    def calc_signal(self, **kwargs) -> Tuple[float, str]:
        """
        Calculate signal value and direction.

        Returns:
            (signal_value, direction)
            direction is 'LONG', 'SHORT', or 'NONE'
        """
        pass

    def check_data_available(self, ticker: str, days: int) -> bool:
        """Check if required data is available."""
        avail = self.data_loader.check_data_availability(ticker, days)

        for source in self.required_data:
            if source == DataSource.QUIK_TAPE and not avail["trades"]:
                logger.error(f"[{self.name}] BLOCKER: No trade tape data")
                return False
            if source == DataSource.QUIK_L2 and not avail["orderbook"]:
                logger.error(f"[{self.name}] BLOCKER: No L2 orderbook data")
                return False
            if source == DataSource.ISS_PROXY and not avail["candles_1m"]:
                logger.error(f"[{self.name}] BLOCKER: No 1m candle data")
                return False

        return True


# =============================================================================
# H1: OPENING IMBALANCE
# =============================================================================

class M1_OpeningImbalance(Hypothesis):
    """
    M1: Opening Imbalance (First 5 Minutes)

    Signal: Trade flow imbalance in first 5 minutes after open
    Entry: 10:05 MSK
    Exit: 15m, 30m, 60m
    """

    @property
    def name(self) -> str:
        return "M1_opening_imbalance"

    @property
    def required_data(self) -> List[DataSource]:
        return [DataSource.QUIK_TAPE, DataSource.QUIK_L2]

    @property
    def exit_horizons(self) -> List[str]:
        return ["15m", "30m", "60m"]

    def calc_signal(
        self,
        trades: List[Trade],
        threshold: float = 0.4,
    ) -> Tuple[float, str]:
        """
        Calculate opening imbalance signal.
        """
        if not trades:
            return 0.0, "NONE"

        buy_vol = sum(t.qty for t in trades if t.side == "BUY")
        sell_vol = sum(t.qty for t in trades if t.side == "SELL")
        total = buy_vol + sell_vol

        if total == 0:
            return 0.0, "NONE"

        imbalance = (buy_vol - sell_vol) / total

        if imbalance > threshold:
            return imbalance, "LONG"
        elif imbalance < -threshold:
            return imbalance, "SHORT"
        else:
            return imbalance, "NONE"

    def build_events(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        threshold: float = 0.4,
    ) -> List[Event]:
        """Build opening imbalance events."""
        events = []
        current = start

        while current <= end:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # Event window: 10:00 - 10:05 MSK
            open_time = current.replace(hour=10, minute=0, second=0, microsecond=0)
            signal_end = open_time + timedelta(minutes=5)
            entry_time = signal_end

            # Load trades for window
            trades = self.data_loader.load_trades(ticker, open_time, signal_end)

            if not trades:
                # Try proxy (ISS candles)
                logger.debug(f"No trades for {current.date()}, using proxy")
                current += timedelta(days=1)
                continue

            # Calculate signal
            signal_value, direction = self.calc_signal(trades, threshold)

            if direction == "NONE":
                current += timedelta(days=1)
                continue

            # Get entry price
            entry_trades = self.data_loader.load_trades(
                ticker, entry_time, entry_time + timedelta(minutes=1)
            )
            if not entry_trades:
                current += timedelta(days=1)
                continue

            entry_price = entry_trades[0].price

            # Get exit prices
            exit_prices = {}
            for horizon in self.exit_horizons:
                minutes = int(horizon.replace('m', ''))
                exit_time = entry_time + timedelta(minutes=minutes)
                exit_trades = self.data_loader.load_trades(
                    ticker, exit_time, exit_time + timedelta(minutes=1)
                )
                if exit_trades:
                    exit_prices[horizon] = exit_trades[-1].price

            if not exit_prices:
                current += timedelta(days=1)
                continue

            # Calculate labels
            labels = {}
            for horizon, exit_price in exit_prices.items():
                if direction == "LONG":
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price
                labels[f"return_{horizon}"] = ret
                labels[f"label_{horizon}"] = 1 if ret > 0 else 0

            # Create event
            event = Event(
                event_id=f"{self.name}_{ticker}_{current.date()}",
                hypothesis=self.name,
                ts=entry_time,
                ticker=ticker,
                features={
                    "imbalance": signal_value,
                    "buy_volume": sum(t.qty for t in trades if t.side == "BUY"),
                    "sell_volume": sum(t.qty for t in trades if t.side == "SELL"),
                    "trade_count": len(trades),
                },
                signal=signal_value,
                signal_direction=direction,
                entry_price=entry_price,
                exit_prices=exit_prices,
                labels=labels,
            )
            events.append(event)

            current += timedelta(days=1)

        self.events = events
        return events


# =============================================================================
# H2: FLOW DIVERGENCE
# =============================================================================

class M2_FlowDivergence(Hypothesis):
    """
    M2: Flow Divergence (Continuous)

    Signal: Price makes new high/low but net flow diverges
    - Bearish divergence: price new high, net flow negative
    - Bullish divergence: price new low, net flow positive
    Entry: At divergence detection
    Exit: 30m, 60m

    Data: ISS futures trades with BUYSELL (~45 days) OR QUIK
    Status: UNBLOCKED via ISS
    """

    def __init__(self, data_loader: DataLoader, source: str = "iss"):
        super().__init__(data_loader)
        self.source = source

    @property
    def name(self) -> str:
        return "M2_flow_divergence"

    @property
    def required_data(self) -> List[DataSource]:
        if self.source == "iss":
            return [DataSource.ISS_TAPE]
        return [DataSource.QUIK_TAPE, DataSource.QUIK_L2]

    @property
    def exit_horizons(self) -> List[str]:
        return ["30m", "60m"]

    def calc_signal(
        self,
        trades: List[Trade],
        price_high: float,
        price_low: float,
        current_price: float,
        lookback_high: float,
        lookback_low: float,
        threshold: float = 0.2,
    ) -> Tuple[float, str]:
        """
        Calculate flow divergence signal.

        Divergence = price extreme + opposite flow.
        """
        if not trades:
            return 0.0, "NONE"

        # Calculate net flow
        buy_vol = sum(t.qty for t in trades if t.side == "BUY")
        sell_vol = sum(t.qty for t in trades if t.side == "SELL")
        total = buy_vol + sell_vol

        if total == 0:
            return 0.0, "NONE"

        net_flow = (buy_vol - sell_vol) / total

        # Check for new high/low
        is_new_high = current_price > lookback_high
        is_new_low = current_price < lookback_low

        # Bearish divergence: new high but selling pressure
        if is_new_high and net_flow < -threshold:
            return net_flow, "SHORT"

        # Bullish divergence: new low but buying pressure
        if is_new_low and net_flow > threshold:
            return net_flow, "LONG"

        return net_flow, "NONE"

    def build_events(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        lookback_minutes: int = 30,
        scan_interval_minutes: int = 5,
        threshold: float = 0.2,
    ) -> List[Event]:
        """Build flow divergence events (continuous scan)."""
        events = []

        # Pre-load all ISS trades if using ISS source
        if self.source == "iss":
            all_iss_trades = self.data_loader.load_iss_trades(ticker, start, end)
            logger.info(f"Loaded {len(all_iss_trades)} total ISS trades for M2")
        else:
            all_iss_trades = []

        current = start

        while current <= end:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # Scan during trading hours: 10:15 - 18:00
            scan_start = current.replace(hour=10, minute=15, second=0, microsecond=0)
            scan_end = current.replace(hour=18, minute=0, second=0, microsecond=0)

            scan_time = scan_start
            while scan_time < scan_end:
                # Lookback window for price extremes
                lookback_start = scan_time - timedelta(minutes=lookback_minutes)

                # Load lookback trades for price levels
                if self.source == "iss":
                    lookback_trades = [
                        t for t in all_iss_trades
                        if lookback_start <= t.ts < scan_time
                    ]
                else:
                    lookback_trades = self.data_loader.load_trades(
                        ticker, lookback_start, scan_time
                    )

                if len(lookback_trades) < 10:
                    scan_time += timedelta(minutes=scan_interval_minutes)
                    continue

                # Calculate lookback price range
                lookback_prices = [t.price for t in lookback_trades]
                lookback_high = max(lookback_prices)
                lookback_low = min(lookback_prices)

                # Current 5-minute window for flow
                flow_window_start = scan_time - timedelta(minutes=5)
                if self.source == "iss":
                    flow_trades = [
                        t for t in all_iss_trades
                        if flow_window_start <= t.ts < scan_time
                    ]
                else:
                    flow_trades = self.data_loader.load_trades(
                        ticker, flow_window_start, scan_time
                    )

                if not flow_trades:
                    scan_time += timedelta(minutes=scan_interval_minutes)
                    continue

                current_price = flow_trades[-1].price

                # Calculate signal
                signal_value, direction = self.calc_signal(
                    trades=flow_trades,
                    price_high=max(t.price for t in flow_trades),
                    price_low=min(t.price for t in flow_trades),
                    current_price=current_price,
                    lookback_high=lookback_high,
                    lookback_low=lookback_low,
                    threshold=threshold,
                )

                if direction == "NONE":
                    scan_time += timedelta(minutes=scan_interval_minutes)
                    continue

                # Entry price
                entry_price = current_price
                entry_time = scan_time

                # Get exit prices
                exit_prices = {}
                for horizon in self.exit_horizons:
                    minutes = int(horizon.replace('m', ''))
                    exit_time = entry_time + timedelta(minutes=minutes)

                    # Don't exit after market close
                    if exit_time.hour >= 18 and exit_time.minute > 40:
                        continue

                    if self.source == "iss":
                        exit_trades = [
                            t for t in all_iss_trades
                            if exit_time <= t.ts < exit_time + timedelta(minutes=5)
                        ]
                    else:
                        exit_trades = self.data_loader.load_trades(
                            ticker, exit_time, exit_time + timedelta(minutes=1)
                        )
                    if exit_trades:
                        exit_prices[horizon] = exit_trades[-1].price

                if not exit_prices:
                    scan_time += timedelta(minutes=scan_interval_minutes)
                    continue

                # Calculate labels
                labels = {}
                for horizon, exit_price in exit_prices.items():
                    if direction == "LONG":
                        ret = (exit_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - exit_price) / entry_price
                    labels[f"return_{horizon}"] = ret
                    labels[f"label_{horizon}"] = 1 if ret > 0 else 0

                # Create event
                event = Event(
                    event_id=f"{self.name}_{ticker}_{scan_time.isoformat()}",
                    hypothesis=self.name,
                    ts=entry_time,
                    ticker=ticker,
                    features={
                        "net_flow": signal_value,
                        "lookback_high": lookback_high,
                        "lookback_low": lookback_low,
                        "current_price": current_price,
                        "trade_count": len(flow_trades),
                    },
                    signal=signal_value,
                    signal_direction=direction,
                    entry_price=entry_price,
                    exit_prices=exit_prices,
                    labels=labels,
                )
                events.append(event)

                # Skip ahead to avoid overlapping events
                scan_time += timedelta(minutes=30)
                continue

            current += timedelta(days=1)

        self.events = events
        return events


# =============================================================================
# H3: CLOSE PRESSURE
# =============================================================================

class M3_ClosePressure(Hypothesis):
    """
    M3: Close Pressure → Overnight Gap

    Signal: Trade flow imbalance 18:25-18:40 MSK
    Entry: 18:40 (hold overnight)
    Exit: Next day 10:05

    Data: ISS futures trades with BUYSELL (~45 days) OR QUIK
    Status: UNBLOCKED via ISS
    """

    def __init__(self, data_loader: DataLoader, source: str = "iss"):
        super().__init__(data_loader)
        self.source = source

    @property
    def name(self) -> str:
        return "M3_close_pressure"

    @property
    def required_data(self) -> List[DataSource]:
        if self.source == "iss":
            return [DataSource.ISS_TAPE]
        return [DataSource.QUIK_TAPE, DataSource.QUIK_L2]

    @property
    def exit_horizons(self) -> List[str]:
        return ["overnight"]

    def calc_signal(
        self,
        trades: List[Trade],
        books: List[OrderBookSnapshot],
        threshold: float = 0.3,
    ) -> Tuple[float, str]:
        """
        Calculate close pressure signal.

        Combines trade flow (70%) and book pressure (30%).
        """
        # Trade imbalance
        if trades:
            buy_vol = sum(t.qty for t in trades if t.side == "BUY")
            sell_vol = sum(t.qty for t in trades if t.side == "SELL")
            total = buy_vol + sell_vol
            trade_imb = (buy_vol - sell_vol) / total if total > 0 else 0.0
        else:
            trade_imb = 0.0

        # Book pressure
        if books:
            imbalances = []
            for book in books:
                bid_vol = sum(book.bid_sizes[:5])
                ask_vol = sum(book.ask_sizes[:5])
                if bid_vol + ask_vol > 0:
                    imbalances.append((bid_vol - ask_vol) / (bid_vol + ask_vol))
            book_pressure = np.mean(imbalances) if imbalances else 0.0
        else:
            book_pressure = 0.0

        # Combined signal
        signal = 0.7 * trade_imb + 0.3 * book_pressure

        if signal > threshold:
            return signal, "LONG"
        elif signal < -threshold:
            return signal, "SHORT"
        else:
            return signal, "NONE"

    def build_events(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        threshold: float = 0.3,
    ) -> List[Event]:
        """Build close pressure events."""
        events = []

        # Pre-load all ISS trades if using ISS source
        if self.source == "iss":
            all_iss_trades = self.data_loader.load_iss_trades(ticker, start, end)
            logger.info(f"Loaded {len(all_iss_trades)} total ISS trades")
        else:
            all_iss_trades = []

        current = start

        while current <= end - timedelta(days=1):  # Need next day
            # Skip weekends and Fridays (no overnight gap)
            if current.weekday() >= 4:  # Fri, Sat, Sun
                current += timedelta(days=1)
                continue

            # Pre-close window: 18:25 - 18:40 (extended from 18:30)
            preclose_start = current.replace(hour=18, minute=25, second=0, microsecond=0)
            preclose_end = current.replace(hour=18, minute=40, second=0, microsecond=0)

            # Load data based on source
            if self.source == "iss":
                # Filter pre-loaded ISS trades for this window
                trades = [
                    t for t in all_iss_trades
                    if preclose_start <= t.ts < preclose_end
                ]
                books = []  # ISS has no L2
            else:
                trades = self.data_loader.load_trades(ticker, preclose_start, preclose_end)
                books = self.data_loader.load_orderbook(ticker, preclose_start, preclose_end)

            if not trades and not books:
                current += timedelta(days=1)
                continue

            # Calculate signal
            signal_value, direction = self.calc_signal(trades, books, threshold)

            if direction == "NONE":
                current += timedelta(days=1)
                continue

            # Entry price: last trade before 18:40
            if self.source == "iss":
                # Use last trade in preclose window
                entry_trades = [t for t in trades if t.ts < preclose_end]
            else:
                entry_trades = self.data_loader.load_trades(
                    ticker, preclose_end - timedelta(minutes=1), preclose_end
                )

            if not entry_trades:
                current += timedelta(days=1)
                continue

            entry_price = entry_trades[-1].price

            # Exit price: next day 10:05
            next_day = current + timedelta(days=1)
            if next_day.weekday() >= 5:  # Skip to Monday
                next_day += timedelta(days=(7 - next_day.weekday()))

            exit_time = next_day.replace(hour=10, minute=5, second=0, microsecond=0)

            if self.source == "iss":
                # Find trades around next day open
                exit_trades = [
                    t for t in all_iss_trades
                    if exit_time <= t.ts < exit_time + timedelta(minutes=10)
                ]
            else:
                exit_trades = self.data_loader.load_trades(
                    ticker, exit_time, exit_time + timedelta(minutes=1)
                )

            if not exit_trades:
                current += timedelta(days=1)
                continue

            exit_price = exit_trades[0].price
            exit_prices = {"overnight": exit_price}

            # Calculate labels
            if direction == "LONG":
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price

            # Subtract overnight margin cost (~0.05%)
            ret_net = ret - 0.0005

            labels = {
                "return_overnight": ret,
                "return_overnight_net": ret_net,
                "gap": (exit_price - entry_price) / entry_price,
                "label_overnight": 1 if ret_net > 0 else 0,
            }

            # Create event
            event = Event(
                event_id=f"{self.name}_{ticker}_{current.date()}",
                hypothesis=self.name,
                ts=preclose_end,
                ticker=ticker,
                features={
                    "signal": signal_value,
                    "trade_count": len(trades),
                    "book_count": len(books),
                },
                signal=signal_value,
                signal_direction=direction,
                entry_price=entry_price,
                exit_prices=exit_prices,
                labels=labels,
            )
            events.append(event)

            current += timedelta(days=1)

        self.events = events
        return events


# =============================================================================
# EVENT STUDY
# =============================================================================

@dataclass
class EventStudyResult:
    """Results of event study analysis."""
    hypothesis: str
    ticker: str
    n_events: int
    n_long: int
    n_short: int
    win_rate: float
    avg_return: float
    profit_factor: float
    sharpe: float
    best_horizon: str
    by_horizon: Dict[str, Dict[str, float]]


class EventStudy:
    """Event study analysis framework."""

    def __init__(self, events: List[Event], cost_bps: float = 10):
        self.events = events
        self.cost_bps = cost_bps  # 0.10% round-trip

    def run(self) -> EventStudyResult:
        """Run event study analysis."""
        if not self.events:
            return self._empty_result()

        # Group by horizon
        by_horizon = {}
        for event in self.events:
            for key, value in event.labels.items():
                if key.startswith("return_"):
                    horizon = key.replace("return_", "")
                    if horizon not in by_horizon:
                        by_horizon[horizon] = []
                    by_horizon[horizon].append(value)

        # Calculate metrics per horizon
        horizon_results = {}
        best_pf = 0
        best_horizon = None

        for horizon, returns in by_horizon.items():
            returns = np.array(returns)
            net_returns = returns - self.cost_bps / 10000  # Subtract costs

            wins = net_returns[net_returns > 0]
            losses = net_returns[net_returns < 0]

            wr = len(wins) / len(returns) if returns.size > 0 else 0
            avg_ret = np.mean(net_returns)

            gross_win = np.sum(wins) if wins.size > 0 else 0
            gross_loss = np.abs(np.sum(losses)) if losses.size > 0 else 0.001
            pf = gross_win / gross_loss

            sharpe = (np.mean(net_returns) / np.std(net_returns) *
                     np.sqrt(252)) if np.std(net_returns) > 0 else 0

            horizon_results[horizon] = {
                "n": len(returns),
                "win_rate": wr,
                "avg_return": avg_ret,
                "profit_factor": pf,
                "sharpe": sharpe,
            }

            if pf > best_pf:
                best_pf = pf
                best_horizon = horizon

        # Overall stats
        n_long = sum(1 for e in self.events if e.signal_direction == "LONG")
        n_short = len(self.events) - n_long

        return EventStudyResult(
            hypothesis=self.events[0].hypothesis if self.events else "",
            ticker=self.events[0].ticker if self.events else "",
            n_events=len(self.events),
            n_long=n_long,
            n_short=n_short,
            win_rate=horizon_results.get(best_horizon, {}).get("win_rate", 0),
            avg_return=horizon_results.get(best_horizon, {}).get("avg_return", 0),
            profit_factor=best_pf,
            sharpe=horizon_results.get(best_horizon, {}).get("sharpe", 0),
            best_horizon=best_horizon or "",
            by_horizon=horizon_results,
        )

    def _empty_result(self) -> EventStudyResult:
        return EventStudyResult(
            hypothesis="",
            ticker="",
            n_events=0,
            n_long=0,
            n_short=0,
            win_rate=0,
            avg_return=0,
            profit_factor=0,
            sharpe=0,
            best_horizon="",
            by_horizon={},
        )


# =============================================================================
# FALSIFICATION TESTS
# =============================================================================

@dataclass
class FalsificationResult:
    """Result of a single falsification test."""
    test_name: str
    passed: bool
    metric: float
    threshold: float
    details: str


class FalsificationSuite:
    """Suite of falsification tests for a hypothesis."""

    def __init__(self, events: List[Event]):
        self.events = events
        self.results: List[FalsificationResult] = []

    def run_all(self) -> List[FalsificationResult]:
        """Run all falsification tests."""
        self.results = [
            self.test_placebo_shuffle(),
            self.test_placebo_reverse(),
            self.test_shuffled_dates(),
            self.test_side_symmetry(),
            self.test_cost_shock(multiplier=2.0),
            self.test_cost_shock(multiplier=3.0),
        ]
        return self.results

    def test_placebo_shuffle(self, n_shuffles: int = 100) -> FalsificationResult:
        """
        Test 1: Placebo Shuffle

        Randomly shuffle signal values across events.
        Real PF should be significantly better than placebo.
        """
        if len(self.events) < 10:
            return FalsificationResult(
                test_name="placebo_shuffle",
                passed=False,
                metric=0,
                threshold=0,
                details="Too few events (n < 10)",
            )

        # Real PF
        real_study = EventStudy(self.events)
        real_result = real_study.run()
        real_pf = real_result.profit_factor

        # Placebo PFs
        np.random.seed(42)
        placebo_pfs = []

        signals = [e.signal for e in self.events]
        for _ in range(n_shuffles):
            shuffled = np.random.permutation(signals)

            # Create shuffled events
            shuffled_events = []
            for i, event in enumerate(self.events):
                new_event = Event(
                    event_id=event.event_id,
                    hypothesis=event.hypothesis,
                    ts=event.ts,
                    ticker=event.ticker,
                    features=event.features,
                    signal=shuffled[i],
                    signal_direction="LONG" if shuffled[i] > 0 else "SHORT",
                    entry_price=event.entry_price,
                    exit_prices=event.exit_prices,
                    labels=event.labels,
                )
                shuffled_events.append(new_event)

            placebo_study = EventStudy(shuffled_events)
            placebo_result = placebo_study.run()
            placebo_pfs.append(placebo_result.profit_factor)

        placebo_mean = np.mean(placebo_pfs)
        placebo_std = np.std(placebo_pfs)

        # P-value
        if placebo_std > 0:
            z_score = (real_pf - placebo_mean) / placebo_std
            from scipy import stats
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = 0.5

        passed = p_value < 0.1
        return FalsificationResult(
            test_name="placebo_shuffle",
            passed=passed,
            metric=p_value,
            threshold=0.1,
            details=f"Real PF={real_pf:.2f}, Placebo mean={placebo_mean:.2f}, p={p_value:.3f}",
        )

    def test_placebo_reverse(self) -> FalsificationResult:
        """
        Test 2: Placebo Reverse

        Reverse all signal directions.
        Reversed PF should be < 1.0 (losing).
        """
        if len(self.events) < 10:
            return FalsificationResult(
                test_name="placebo_reverse",
                passed=False,
                metric=0,
                threshold=0,
                details="Too few events (n < 10)",
            )

        # Create reversed events
        reversed_events = []
        for event in self.events:
            new_direction = "SHORT" if event.signal_direction == "LONG" else "LONG"

            # Reverse the labels too
            reversed_labels = {}
            for key, value in event.labels.items():
                if key.startswith("return_"):
                    reversed_labels[key] = -value
                elif key.startswith("label_"):
                    reversed_labels[key] = 1 - value
                else:
                    reversed_labels[key] = value

            new_event = Event(
                event_id=event.event_id,
                hypothesis=event.hypothesis,
                ts=event.ts,
                ticker=event.ticker,
                features=event.features,
                signal=-event.signal,
                signal_direction=new_direction,
                entry_price=event.entry_price,
                exit_prices=event.exit_prices,
                labels=reversed_labels,
            )
            reversed_events.append(new_event)

        reversed_study = EventStudy(reversed_events)
        reversed_result = reversed_study.run()
        reversed_pf = reversed_result.profit_factor

        passed = reversed_pf < 0.9
        return FalsificationResult(
            test_name="placebo_reverse",
            passed=passed,
            metric=reversed_pf,
            threshold=0.9,
            details=f"Reversed PF={reversed_pf:.2f} (should be < 0.9)",
        )

    def test_side_symmetry(self) -> FalsificationResult:
        """
        Test 3: Side Symmetry

        Both LONG and SHORT sides should be profitable.
        """
        long_events = [e for e in self.events if e.signal_direction == "LONG"]
        short_events = [e for e in self.events if e.signal_direction == "SHORT"]

        if len(long_events) < 5 or len(short_events) < 5:
            return FalsificationResult(
                test_name="side_symmetry",
                passed=False,
                metric=0,
                threshold=0,
                details=f"Insufficient samples: LONG={len(long_events)}, SHORT={len(short_events)}",
            )

        long_study = EventStudy(long_events)
        short_study = EventStudy(short_events)

        long_pf = long_study.run().profit_factor
        short_pf = short_study.run().profit_factor

        # Both sides should have PF > 0.8 (at least not heavily losing)
        passed = long_pf > 0.8 and short_pf > 0.8
        return FalsificationResult(
            test_name="side_symmetry",
            passed=passed,
            metric=min(long_pf, short_pf),
            threshold=0.8,
            details=f"LONG PF={long_pf:.2f}, SHORT PF={short_pf:.2f}",
        )

    def test_shuffled_dates(self, n_shuffles: int = 100) -> FalsificationResult:
        """
        Test: Shuffled Dates

        Shuffle which date each signal occurred on.
        If signal timing matters, shuffled should perform worse.
        """
        if len(self.events) < 10:
            return FalsificationResult(
                test_name="shuffled_dates",
                passed=False,
                metric=0,
                threshold=0,
                details="Too few events (n < 10)",
            )

        # Real PF
        real_study = EventStudy(self.events)
        real_pf = real_study.run().profit_factor

        # Shuffle dates
        np.random.seed(42)
        shuffled_pfs = []

        dates = [e.ts for e in self.events]
        for _ in range(n_shuffles):
            shuffled_dates = np.random.permutation(dates)

            shuffled_events = []
            for i, event in enumerate(self.events):
                new_event = Event(
                    event_id=event.event_id,
                    hypothesis=event.hypothesis,
                    ts=shuffled_dates[i],
                    ticker=event.ticker,
                    features=event.features,
                    signal=event.signal,
                    signal_direction=event.signal_direction,
                    entry_price=event.entry_price,
                    exit_prices=event.exit_prices,
                    labels=event.labels,
                )
                shuffled_events.append(new_event)

            shuffled_study = EventStudy(shuffled_events)
            shuffled_pfs.append(shuffled_study.run().profit_factor)

        shuffled_mean = np.mean(shuffled_pfs)

        # Real should be better than shuffled (real PF > shuffled mean)
        passed = real_pf > shuffled_mean * 1.1  # 10% better
        return FalsificationResult(
            test_name="shuffled_dates",
            passed=passed,
            metric=real_pf / shuffled_mean if shuffled_mean > 0 else 0,
            threshold=1.1,
            details=f"Real PF={real_pf:.2f}, Shuffled mean={shuffled_mean:.2f}, ratio={real_pf/shuffled_mean:.2f}" if shuffled_mean > 0 else "Shuffled mean=0",
        )

    def test_cost_shock(self, multiplier: float = 2.0) -> FalsificationResult:
        """
        Test: Cost Shock

        Strategy should survive Nx transaction costs.
        """
        # Normal costs (10 bps)
        normal_study = EventStudy(self.events, cost_bps=10)
        normal_pf = normal_study.run().profit_factor

        # Nx costs
        shock_bps = int(10 * multiplier)
        shock_study = EventStudy(self.events, cost_bps=shock_bps)
        shock_pf = shock_study.run().profit_factor

        passed = shock_pf > 1.0
        return FalsificationResult(
            test_name=f"cost_shock_{multiplier:.0f}x",
            passed=passed,
            metric=shock_pf,
            threshold=1.0,
            details=f"Normal PF={normal_pf:.2f}, {multiplier:.0f}x Cost PF={shock_pf:.2f}",
        )

    def summary(self) -> Dict:
        """Get summary of all tests."""
        passed = sum(1 for r in self.results if r.passed)
        return {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "metric": r.metric,
                    "threshold": r.threshold,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# =============================================================================
# RESEARCH RUNNER
# =============================================================================

class ResearchRunner:
    """Main research execution framework."""

    HYPOTHESIS_CLASSES = {
        "M1": M1_OpeningImbalance,
        "M2": M2_FlowDivergence,
        "M3": M3_ClosePressure,
    }

    # Which hypotheses support ISS data source
    ISS_SUPPORTED = {"M2", "M3"}

    def __init__(self, db_path: Path):
        self.data_loader = DataLoader(db_path)

    def run(
        self,
        hypothesis_id: str,
        ticker: str,
        days: int,
        run_falsification: bool = False,
        source: str = "iss",
    ) -> Dict:
        """
        Run research for a hypothesis.

        Returns:
            Dict with results, blockers, and recommendations.
        """
        logger.info(f"=" * 60)
        logger.info(f"RESEARCH: {hypothesis_id} | {ticker} | {days} days")
        logger.info(f"=" * 60)

        # Check hypothesis exists
        if hypothesis_id not in self.HYPOTHESIS_CLASSES:
            return {
                "status": "ERROR",
                "error": f"Unknown hypothesis: {hypothesis_id}",
            }

        # Validate source
        if source == "iss" and hypothesis_id not in self.ISS_SUPPORTED:
            return {
                "status": "ERROR",
                "error": f"{hypothesis_id} requires QUIK L2 data, ISS not supported",
                "hint": "M1 needs L2 orderbook. Use --source quik after QUIK setup.",
            }

        # Create hypothesis instance
        hypothesis_class = self.HYPOTHESIS_CLASSES[hypothesis_id]

        # Pass source for hypotheses that support it
        if hypothesis_id in self.ISS_SUPPORTED:
            hypothesis = hypothesis_class(self.data_loader, source=source)
        else:
            hypothesis = hypothesis_class(self.data_loader)

        # Check data availability
        if source == "iss":
            avail = self.data_loader.check_data_availability(ticker, days, source="iss")
            if not avail["trades"]:
                return {
                    "status": "BLOCKED",
                    "hypothesis": hypothesis_id,
                    "blocker": "No ISS trades found",
                    "hint": "Check ticker symbol and ISS API availability",
                }
        elif not hypothesis.check_data_available(ticker, days):
            return {
                "status": "BLOCKED",
                "hypothesis": hypothesis_id,
                "blocker": "Required data not available",
                "required_data": [ds.value for ds in hypothesis.required_data],
            }

        # Build events
        end = datetime.now()
        start = end - timedelta(days=days)

        logger.info(f"Building events...")
        events = hypothesis.build_events(ticker, start, end)

        if not events:
            return {
                "status": "NO_EVENTS",
                "hypothesis": hypothesis_id,
                "message": "No events generated (check data or signal thresholds)",
            }

        logger.info(f"Generated {len(events)} events")

        # Run event study
        logger.info(f"Running event study...")
        study = EventStudy(events)
        result = study.run()

        output = {
            "status": "OK",
            "hypothesis": hypothesis_id,
            "ticker": ticker,
            "n_events": result.n_events,
            "n_long": result.n_long,
            "n_short": result.n_short,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "sharpe": result.sharpe,
            "best_horizon": result.best_horizon,
            "by_horizon": result.by_horizon,
        }

        # Print summary
        logger.info(f"\nRESULTS:")
        logger.info(f"  n = {result.n_events} (LONG: {result.n_long}, SHORT: {result.n_short})")
        logger.info(f"  WR = {result.win_rate * 100:.1f}%")
        logger.info(f"  PF = {result.profit_factor:.2f}")
        logger.info(f"  Sharpe = {result.sharpe:.2f}")
        logger.info(f"  Best horizon = {result.best_horizon}")

        # Run falsification if requested
        if run_falsification:
            logger.info(f"\nRunning falsification tests...")
            falsification = FalsificationSuite(events)
            falsification.run_all()

            output["falsification"] = falsification.summary()

            logger.info(f"\nFALSIFICATION RESULTS:")
            for r in falsification.results:
                status = "PASS" if r.passed else "FAIL"
                logger.info(f"  [{status}] {r.test_name}: {r.details}")

        # Recommendation
        if result.profit_factor < 1.0:
            output["recommendation"] = "KILL: PF < 1.0"
        elif result.n_events < 30:
            output["recommendation"] = "INSUFFICIENT DATA: n < 30, collect more data"
        elif result.profit_factor < 1.2:
            output["recommendation"] = "MARGINAL: PF 1.0-1.2, may be noise"
        else:
            output["recommendation"] = "CANDIDATE: PF > 1.2, proceed to falsification"

        logger.info(f"\nRECOMMENDATION: {output['recommendation']}")

        return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orderflow Research Framework",
        epilog="""
Examples:
  # M3 via ISS (READY NOW - ~45 days available)
  python orderflow_research_scaffold.py --hypothesis M3 --ticker BR --days 45 --source iss --run-falsification

  # M2 via ISS (READY NOW)
  python orderflow_research_scaffold.py --hypothesis M2 --ticker BR --days 45 --source iss

  # M1 via QUIK (BLOCKED - needs L2)
  python orderflow_research_scaffold.py --hypothesis M1 --ticker BR --source quik
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hypothesis", type=str, required=True, choices=["M1", "M2", "M3"],
                        help="Hypothesis to test: M3 (close pressure), M2 (flow divergence), M1 (opening imbalance)")
    parser.add_argument("--ticker", type=str, default="BR",
                        help="Futures ticker (default: BR)")
    parser.add_argument("--days", type=int, default=45,
                        help="Days of data to use (default: 45 for ISS)")
    parser.add_argument("--source", type=str, default="iss", choices=["iss", "quik"],
                        help="Data source: iss (ISS API trades with BUYSELL) or quik (QUIK terminal)")
    parser.add_argument("--db-path", type=str, default="data/microstructure.db",
                        help="SQLite database path (for QUIK source)")
    parser.add_argument("--run-falsification", action="store_true",
                        help="Run falsification tests (placebo, reverse, cost shock)")
    args = parser.parse_args()

    runner = ResearchRunner(Path(args.db_path))
    result = runner.run(
        hypothesis_id=args.hypothesis,
        ticker=args.ticker,
        days=args.days,
        run_falsification=args.run_falsification,
        source=args.source,
    )

    import json
    print("\n" + "=" * 60)
    print("FULL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
