"""
MOEX Agent v2.5 Microstructure Features

Phase 2: Order book and trade tape features for entry timing.

Two-level architecture:
1. Alpha Model (existing): OHLCV → direction prediction (10m/30m/1h)
2. Entry Timing Model (new): microstructure → entry quality (1-5 min)

Features available through BCS QUIK API:
- Aggregated order book (not individual orders)
- Trade tape with direction (buy/sell)
- Open Interest updates (futures)

NOT available (requires Full Orders Log):
- Event-based OFI/MLOFI
- Cancel/add ratio
- Order lifetime/persistence
- Queue depletion in pure form
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Single order book snapshot."""
    timestamp: pd.Timestamp
    ticker: str
    bid_prices: List[float]  # Best to worst
    bid_volumes: List[float]
    ask_prices: List[float]  # Best to worst
    ask_volumes: List[float]

    @property
    def mid_price(self) -> float:
        """Mid price between best bid and ask."""
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        return (self.bid_prices[0] + self.ask_prices[0]) / 2

    @property
    def spread(self) -> float:
        """Absolute spread."""
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        return self.ask_prices[0] - self.bid_prices[0]

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = self.mid_price
        if mid <= 0:
            return 0.0
        return (self.spread / mid) * 10000


@dataclass
class Trade:
    """Single trade from tape."""
    timestamp: pd.Timestamp
    ticker: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'

    @property
    def signed_volume(self) -> float:
        """Positive for buys, negative for sells."""
        return self.volume if self.side == 'buy' else -self.volume


class MicrostructureFeatures:
    """
    Calculate microstructure features from order book and trade tape.

    All features aggregated in windows: 5s, 10s, 30s, 60s.
    Alpha model uses 30s and 60s.
    Entry timing uses 5s and 10s.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.book_history: List[OrderBookSnapshot] = []
        self.trade_history: List[Trade] = []
        self.oi_history: List[Tuple[pd.Timestamp, float]] = []
        self.session_spreads: List[float] = []
        self.cvd: float = 0.0  # Cumulative Volume Delta

    def add_book_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add order book snapshot to history."""
        self.book_history.append(snapshot)
        self.session_spreads.append(snapshot.spread_bps)

        # Keep last 5 minutes of data
        cutoff = snapshot.timestamp - pd.Timedelta(minutes=5)
        self.book_history = [b for b in self.book_history if b.timestamp >= cutoff]

    def add_trade(self, trade: Trade) -> None:
        """Add trade to history and update CVD."""
        self.trade_history.append(trade)
        self.cvd += trade.signed_volume

        # Keep last 5 minutes of data
        cutoff = trade.timestamp - pd.Timedelta(minutes=5)
        self.trade_history = [t for t in self.trade_history if t.timestamp >= cutoff]

    def add_oi(self, timestamp: pd.Timestamp, oi: float) -> None:
        """Add Open Interest update (futures only)."""
        self.oi_history.append((timestamp, oi))

        # Keep last 5 minutes
        cutoff = timestamp - pd.Timedelta(minutes=5)
        self.oi_history = [(ts, o) for ts, o in self.oi_history if ts >= cutoff]

    # === ORDER BOOK FEATURES ===

    def imbalance_top_n(self, n: int = 5) -> float:
        """
        Order book imbalance at top N levels.

        (sum_bid_1..n - sum_ask_1..n) / (sum_bid_1..n + sum_ask_1..n)

        Returns:
            Imbalance in [-1, 1]. Positive = more bids (buying pressure).
        """
        if not self.book_history:
            return 0.0

        book = self.book_history[-1]
        bid_vol = sum(book.bid_volumes[:n]) if book.bid_volumes else 0
        ask_vol = sum(book.ask_volumes[:n]) if book.ask_volumes else 0

        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def microprice_gap(self) -> float:
        """
        Microprice deviation from mid price.

        microprice = (ask1 × bid_vol1 + bid1 × ask_vol1) / (bid_vol1 + ask_vol1)
        gap = (microprice - mid) / spread

        Returns:
            Gap in [-0.5, 0.5]. Positive = price pressure up.
        """
        if not self.book_history:
            return 0.0

        book = self.book_history[-1]
        if not book.bid_prices or not book.ask_prices:
            return 0.0
        if not book.bid_volumes or not book.ask_volumes:
            return 0.0

        bid1, ask1 = book.bid_prices[0], book.ask_prices[0]
        bid_vol1, ask_vol1 = book.bid_volumes[0], book.ask_volumes[0]

        total_vol = bid_vol1 + ask_vol1
        if total_vol <= 0:
            return 0.0

        microprice = (ask1 * bid_vol1 + bid1 * ask_vol1) / total_vol
        mid = (bid1 + ask1) / 2
        spread = ask1 - bid1

        if spread <= 0:
            return 0.0

        return (microprice - mid) / spread

    def spread_vs_median(self) -> float:
        """
        Current spread vs session median.

        Returns:
            Ratio. >1.5 = abnormally wide spread.
        """
        if not self.book_history or not self.session_spreads:
            return 1.0

        current = self.book_history[-1].spread_bps
        median = np.median(self.session_spreads)

        if median <= 0:
            return 1.0

        return current / median

    def depth_ratio(self, n: int = 5) -> float:
        """
        Total bid volume / total ask volume at top N levels.

        Returns:
            Ratio. >1 = more buyers.
        """
        if not self.book_history:
            return 1.0

        book = self.book_history[-1]
        bid_vol = sum(book.bid_volumes[:n]) if book.bid_volumes else 0
        ask_vol = sum(book.ask_volumes[:n]) if book.ask_volumes else 0

        if ask_vol <= 0:
            return 1.0

        return bid_vol / ask_vol

    def depth_change(self, window_sec: int = 10) -> float:
        """
        Change in total depth over window.

        Proxy for queue depletion.
        Rapid drop in bid depth = liquidity leaving = bearish.

        Returns:
            Percent change. Negative = depth decreasing.
        """
        if len(self.book_history) < 2:
            return 0.0

        current = self.book_history[-1]
        cutoff = current.timestamp - pd.Timedelta(seconds=window_sec)

        # Find snapshot closest to cutoff
        prev_snapshots = [b for b in self.book_history if b.timestamp <= cutoff]
        if not prev_snapshots:
            return 0.0

        prev = prev_snapshots[-1]

        current_depth = sum(current.bid_volumes[:5]) + sum(current.ask_volumes[:5])
        prev_depth = sum(prev.bid_volumes[:5]) + sum(prev.ask_volumes[:5])

        if prev_depth <= 0:
            return 0.0

        return (current_depth - prev_depth) / prev_depth

    def imbalance_trend(self, window_sec: int = 30) -> float:
        """
        Slope of imbalance over window.

        Proxy for OFI (Order Flow Imbalance).
        Sustained positive trend = growing buy pressure.

        Returns:
            Slope of imbalance. Positive = increasing buy pressure.
        """
        if len(self.book_history) < 3:
            return 0.0

        current = self.book_history[-1]
        cutoff = current.timestamp - pd.Timedelta(seconds=window_sec)

        relevant = [b for b in self.book_history if b.timestamp >= cutoff]
        if len(relevant) < 3:
            return 0.0

        # Calculate imbalance for each snapshot
        imbalances = []
        for book in relevant:
            bid_vol = sum(book.bid_volumes[:5]) if book.bid_volumes else 0
            ask_vol = sum(book.ask_volumes[:5]) if book.ask_volumes else 0
            total = bid_vol + ask_vol
            if total > 0:
                imbalances.append((bid_vol - ask_vol) / total)
            else:
                imbalances.append(0.0)

        if len(imbalances) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(imbalances))
        slope = np.polyfit(x, imbalances, 1)[0]

        return slope

    def bid_wall_ratio(self, n: int = 5) -> float:
        """
        Max bid volume vs mean bid volume.

        Detects large orders ("walls").

        Returns:
            Ratio. >3 = significant wall.
        """
        if not self.book_history:
            return 1.0

        book = self.book_history[-1]
        if not book.bid_volumes or len(book.bid_volumes) < n:
            return 1.0

        vols = book.bid_volumes[:n]
        mean_vol = np.mean(vols)

        if mean_vol <= 0:
            return 1.0

        return max(vols) / mean_vol

    # === TRADE TAPE FEATURES ===

    def signed_trade_imbalance(self, window_sec: int = 30) -> float:
        """
        (buy_vol - sell_vol) / total_vol over window.

        Most reliable micro feature - actual executions.

        Returns:
            Imbalance in [-1, 1]. Positive = net buying.
        """
        if not self.trade_history:
            return 0.0

        current_ts = self.trade_history[-1].timestamp
        cutoff = current_ts - pd.Timedelta(seconds=window_sec)

        relevant = [t for t in self.trade_history if t.timestamp >= cutoff]
        if not relevant:
            return 0.0

        buy_vol = sum(t.volume for t in relevant if t.side == 'buy')
        sell_vol = sum(t.volume for t in relevant if t.side == 'sell')
        total = buy_vol + sell_vol

        if total <= 0:
            return 0.0

        return (buy_vol - sell_vol) / total

    def get_cvd(self) -> float:
        """
        Cumulative Volume Delta for session.

        CVD divergence vs price = early reversal signal.
        """
        return self.cvd

    def large_trade_ratio(self, threshold_lots: float = 50) -> float:
        """
        Fraction of volume from large trades (>threshold).

        High ratio = institutional activity.
        """
        if not self.trade_history:
            return 0.0

        total_vol = sum(t.volume for t in self.trade_history)
        large_vol = sum(t.volume for t in self.trade_history if t.volume >= threshold_lots)

        if total_vol <= 0:
            return 0.0

        return large_vol / total_vol

    def trade_intensity(self, window_sec: int = 30) -> float:
        """
        Trades per second vs session average.

        Acceleration = something happening.

        Returns:
            Ratio. >1.5 = elevated activity.
        """
        if not self.trade_history:
            return 1.0

        current_ts = self.trade_history[-1].timestamp
        cutoff = current_ts - pd.Timedelta(seconds=window_sec)

        recent = [t for t in self.trade_history if t.timestamp >= cutoff]
        recent_rate = len(recent) / window_sec if window_sec > 0 else 0

        # Session average (use all history)
        if len(self.trade_history) < 10:
            return 1.0

        first_ts = self.trade_history[0].timestamp
        session_seconds = (current_ts - first_ts).total_seconds()

        if session_seconds <= 0:
            return 1.0

        session_rate = len(self.trade_history) / session_seconds

        if session_rate <= 0:
            return 1.0

        return recent_rate / session_rate

    def avg_aggressor_size(self, side: str, window_sec: int = 60) -> float:
        """
        Average size of aggressive orders by side.

        Increasing size = larger players entering.
        """
        if not self.trade_history:
            return 0.0

        current_ts = self.trade_history[-1].timestamp
        cutoff = current_ts - pd.Timedelta(seconds=window_sec)

        relevant = [t for t in self.trade_history
                   if t.timestamp >= cutoff and t.side == side]

        if not relevant:
            return 0.0

        return np.mean([t.volume for t in relevant])

    # === FUTURES FEATURES ===

    def oi_change_pct(self, window_sec: int = 60) -> float:
        """
        Open Interest change over window.
        """
        if len(self.oi_history) < 2:
            return 0.0

        current_ts, current_oi = self.oi_history[-1]
        cutoff = current_ts - pd.Timedelta(seconds=window_sec)

        prev_oi_data = [(ts, oi) for ts, oi in self.oi_history if ts <= cutoff]
        if not prev_oi_data:
            return 0.0

        _, prev_oi = prev_oi_data[-1]

        if prev_oi <= 0:
            return 0.0

        return (current_oi - prev_oi) / prev_oi

    def oi_price_divergence(self, price_change: float) -> int:
        """
        OI vs price divergence signal.

        +1: price↑ OI↑ = new longs (trend confirmed)
        -1: price↑ OI↓ = short covering (trend weak)
        0: inconclusive
        """
        oi_change = self.oi_change_pct(60)

        if abs(price_change) < 0.001 or abs(oi_change) < 0.01:
            return 0

        if price_change > 0 and oi_change > 0:
            return 1  # New longs
        elif price_change > 0 and oi_change < 0:
            return -1  # Short covering
        elif price_change < 0 and oi_change > 0:
            return -1  # New shorts
        elif price_change < 0 and oi_change < 0:
            return 1  # Long liquidation

        return 0

    # === AGGREGATE FEATURE VECTOR ===

    def get_features(self, price_change: float = 0.0) -> Dict[str, float]:
        """
        Get all microstructure features as dict.

        Args:
            price_change: Recent price change for OI divergence calc.

        Returns:
            Dict of feature_name -> value
        """
        return {
            # Order book
            "micro_imbalance_5": self.imbalance_top_n(5),
            "micro_imbalance_10": self.imbalance_top_n(10),
            "micro_microprice_gap": self.microprice_gap(),
            "micro_spread_vs_median": self.spread_vs_median(),
            "micro_depth_ratio": self.depth_ratio(5),
            "micro_depth_change_10s": self.depth_change(10),
            "micro_depth_change_30s": self.depth_change(30),
            "micro_imbalance_trend_30s": self.imbalance_trend(30),
            "micro_bid_wall": self.bid_wall_ratio(5),

            # Trade tape
            "micro_trade_imbalance_10s": self.signed_trade_imbalance(10),
            "micro_trade_imbalance_30s": self.signed_trade_imbalance(30),
            "micro_trade_imbalance_60s": self.signed_trade_imbalance(60),
            "micro_cvd": self.get_cvd(),
            "micro_large_trade_ratio": self.large_trade_ratio(50),
            "micro_trade_intensity": self.trade_intensity(30),
            "micro_avg_buy_size": self.avg_aggressor_size("buy", 60),
            "micro_avg_sell_size": self.avg_aggressor_size("sell", 60),

            # Futures
            "micro_oi_change": self.oi_change_pct(60),
            "micro_oi_divergence": float(self.oi_price_divergence(price_change)),
        }

    def get_spread_bps(self) -> float:
        """Get current spread in basis points."""
        if not self.book_history:
            return 0.0
        return self.book_history[-1].spread_bps


# Feature column names for ML
MICRO_FEATURE_COLS = [
    "micro_imbalance_5",
    "micro_imbalance_10",
    "micro_microprice_gap",
    "micro_spread_vs_median",
    "micro_depth_ratio",
    "micro_depth_change_10s",
    "micro_depth_change_30s",
    "micro_imbalance_trend_30s",
    "micro_bid_wall",
    "micro_trade_imbalance_10s",
    "micro_trade_imbalance_30s",
    "micro_trade_imbalance_60s",
    "micro_cvd",
    "micro_large_trade_ratio",
    "micro_trade_intensity",
    "micro_avg_buy_size",
    "micro_avg_sell_size",
    "micro_oi_change",
    "micro_oi_divergence",
]

assert len(MICRO_FEATURE_COLS) == 19, f"Expected 19 micro features, got {len(MICRO_FEATURE_COLS)}"
