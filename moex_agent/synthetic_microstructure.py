"""
MOEX Agent v2.5 Synthetic Microstructure Generator

Phase 2: Generate realistic microstructure data from OHLCV for testing.

This allows testing the Entry Timing Model architecture before collecting
real order book and trade data from QUIK.

Approach:
- Use OHLCV candles to infer market state
- Generate order book snapshots based on volume, volatility, and price movement
- Generate trade tape based on price direction and volume profile
- Create entry timing labels based on intra-candle price paths

Limitations:
- Synthetic data cannot capture real queue dynamics
- OI data is estimated, not real
- Better than nothing for architecture validation
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .microstructure import (
    MicrostructureFeatures,
    OrderBookSnapshot,
    Trade,
    MICRO_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


class SyntheticMicrostructureGenerator:
    """
    Generate synthetic microstructure data from OHLCV candles.

    For each 1-min candle, generates:
    - 2 order book snapshots (at 0s and 30s)
    - 10-50 trades based on volume

    Features are calibrated to be statistically similar to real data.
    """

    def __init__(
        self,
        depth_levels: int = 10,
        base_spread_bps: float = 5.0,
        tick_size: float = 0.01,
        avg_trades_per_min: int = 20,
        seed: Optional[int] = None,
    ):
        """
        Args:
            depth_levels: Order book depth (levels per side)
            base_spread_bps: Average spread in basis points
            tick_size: Minimum price increment
            avg_trades_per_min: Average number of trades per minute
            seed: Random seed for reproducibility
        """
        self.depth_levels = depth_levels
        self.base_spread_bps = base_spread_bps
        self.tick_size = tick_size
        self.avg_trades_per_min = avg_trades_per_min
        self.rng = np.random.default_rng(seed)

    def generate_from_candles(
        self,
        df: pd.DataFrame,
        ticker: str = "SYNTH",
    ) -> Tuple[List[OrderBookSnapshot], List[Trade]]:
        """
        Generate microstructure data from OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be DatetimeIndex
            ticker: Ticker symbol for generated data

        Returns:
            (list of OrderBookSnapshot, list of Trade)
        """
        if df.empty:
            return [], []

        snapshots = []
        trades = []

        for idx, row in df.iterrows():
            ts = pd.Timestamp(idx)
            o, h, l, c, vol = row['open'], row['high'], row['low'], row['close'], row['volume']

            # Skip invalid rows
            if pd.isna(o) or pd.isna(vol) or vol <= 0:
                continue

            # Generate data for this candle
            candle_snapshots, candle_trades = self._generate_candle_data(
                ts, ticker, o, h, l, c, vol
            )
            snapshots.extend(candle_snapshots)
            trades.extend(candle_trades)

        logger.info(f"Generated {len(snapshots)} snapshots and {len(trades)} trades from {len(df)} candles")
        return snapshots, trades

    def _generate_candle_data(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        o: float,
        h: float,
        l: float,
        c: float,
        vol: float,
    ) -> Tuple[List[OrderBookSnapshot], List[Trade]]:
        """Generate snapshots and trades for a single candle."""
        snapshots = []
        trades = []

        # Volatility estimate from candle range
        candle_range = (h - l) / o if o > 0 else 0.01
        volatility_mult = max(0.5, min(3.0, candle_range / 0.005))  # Normalized to ~0.5%

        # Direction: positive if close > open
        is_up = c >= o
        price_change = (c - o) / o if o > 0 else 0

        # Spread widens with volatility
        spread_bps = self.base_spread_bps * volatility_mult
        spread = o * spread_bps / 10000

        # === Generate 2 snapshots per minute ===
        for offset_sec in [0, 30]:
            snap_ts = timestamp + pd.Timedelta(seconds=offset_sec)

            # Price at this point (linear interpolation)
            progress = offset_sec / 60.0
            price = o + (c - o) * progress

            # Add noise to imbalance based on direction
            imbalance_bias = 0.1 if is_up else -0.1
            imbalance = np.clip(imbalance_bias + self.rng.normal(0, 0.15), -0.5, 0.5)

            snapshot = self._generate_snapshot(
                snap_ts, ticker, price, spread, imbalance, volatility_mult
            )
            snapshots.append(snapshot)

        # === Generate trades ===
        n_trades = int(self.rng.poisson(self.avg_trades_per_min * vol / 1000))
        n_trades = max(5, min(100, n_trades))

        # Distribute trades through the minute
        trade_times = sorted(self.rng.uniform(0, 60, n_trades))

        # Trade direction bias based on candle
        buy_prob = 0.6 if is_up else 0.4

        # Total volume to distribute
        remaining_vol = vol
        vol_per_trade = remaining_vol / n_trades

        for t_offset in trade_times:
            trade_ts = timestamp + pd.Timedelta(seconds=t_offset)

            # Price at this time
            progress = t_offset / 60.0

            # Simulate intra-candle path (not just linear)
            # Use a parabola that touches high/low
            if is_up:
                # Up candle: dip first, then rise
                mid_point = (l - o) / (h - l + 0.001)  # Where is low relative to range
                if progress < 0.3:
                    price = o + (l - o) * (progress / 0.3)
                else:
                    price = l + (c - l) * ((progress - 0.3) / 0.7)
            else:
                # Down candle: rise first, then fall
                if progress < 0.3:
                    price = o + (h - o) * (progress / 0.3)
                else:
                    price = h + (c - h) * ((progress - 0.3) / 0.7)

            # Trade size (log-normal distribution)
            trade_vol = vol_per_trade * self.rng.lognormal(0, 0.5)
            trade_vol = max(1, min(vol / 3, trade_vol))

            # Trade side
            side = 'buy' if self.rng.random() < buy_prob else 'sell'

            # Price slightly inside spread based on side
            if side == 'buy':
                trade_price = price + spread / 4  # Buy at ask
            else:
                trade_price = price - spread / 4  # Sell at bid

            trades.append(Trade(
                timestamp=trade_ts,
                ticker=ticker,
                price=trade_price,
                volume=trade_vol,
                side=side,
            ))

            remaining_vol -= trade_vol
            if remaining_vol <= 0:
                break

        return snapshots, trades

    def _generate_snapshot(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        mid_price: float,
        spread: float,
        imbalance: float,
        volatility_mult: float,
    ) -> OrderBookSnapshot:
        """Generate a single order book snapshot."""
        bid1 = mid_price - spread / 2
        ask1 = mid_price + spread / 2

        # Round to tick size
        bid1 = round(bid1 / self.tick_size) * self.tick_size
        ask1 = round(ask1 / self.tick_size) * self.tick_size

        # Generate price levels
        bid_prices = [bid1 - i * self.tick_size for i in range(self.depth_levels)]
        ask_prices = [ask1 + i * self.tick_size for i in range(self.depth_levels)]

        # Base volume profile (exponential decay from best price)
        base_vols = [100 * np.exp(-0.3 * i) for i in range(self.depth_levels)]

        # Apply imbalance (more volume on one side)
        if imbalance > 0:
            bid_mult = 1 + imbalance
            ask_mult = 1 - imbalance * 0.5
        else:
            bid_mult = 1 + imbalance * 0.5
            ask_mult = 1 - imbalance

        bid_volumes = [v * bid_mult * self.rng.uniform(0.5, 1.5) for v in base_vols]
        ask_volumes = [v * ask_mult * self.rng.uniform(0.5, 1.5) for v in base_vols]

        # Add occasional "wall" (large order at one level)
        if self.rng.random() < 0.1:
            wall_level = self.rng.integers(1, min(5, self.depth_levels))
            if imbalance > 0:
                bid_volumes[wall_level] *= 5
            else:
                ask_volumes[wall_level] *= 5

        return OrderBookSnapshot(
            timestamp=timestamp,
            ticker=ticker,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
        )


def generate_micro_features_from_candles(
    df: pd.DataFrame,
    ticker: str = "SYNTH",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate microstructure features DataFrame from OHLCV candles.

    Convenience function for training Entry Timing Model.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        ticker: Ticker symbol
        seed: Random seed

    Returns:
        DataFrame with MICRO_FEATURE_COLS as columns, same index as input
    """
    generator = SyntheticMicrostructureGenerator(seed=seed)
    snapshots, trades = generator.generate_from_candles(df, ticker)

    if not snapshots:
        logger.warning("No snapshots generated")
        return pd.DataFrame(columns=MICRO_FEATURE_COLS, index=df.index)

    # Build MicrostructureFeatures tracker
    micro = MicrostructureFeatures(ticker)

    # Map snapshots and trades to candle timestamps
    # For each candle, we calculate features from all data up to that point
    results = []
    snap_idx = 0
    trade_idx = 0

    for candle_ts in df.index:
        candle_end = pd.Timestamp(candle_ts) + pd.Timedelta(minutes=1)

        # Add all snapshots up to this candle
        while snap_idx < len(snapshots) and snapshots[snap_idx].timestamp < candle_end:
            micro.add_book_snapshot(snapshots[snap_idx])
            snap_idx += 1

        # Add all trades up to this candle
        while trade_idx < len(trades) and trades[trade_idx].timestamp < candle_end:
            micro.add_trade(trades[trade_idx])
            trade_idx += 1

        # Calculate price change for OI divergence
        if candle_ts in df.index:
            row = df.loc[candle_ts]
            price_change = (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0
        else:
            price_change = 0

        # Get features
        features = micro.get_features(price_change)
        features['timestamp'] = candle_ts
        results.append(features)

    # Build DataFrame
    result_df = pd.DataFrame(results)
    result_df.set_index('timestamp', inplace=True)

    # Ensure all columns exist
    for col in MICRO_FEATURE_COLS:
        if col not in result_df.columns:
            result_df[col] = 0.0

    return result_df[MICRO_FEATURE_COLS]


def create_entry_timing_labels(
    df: pd.DataFrame,
    horizon_bars: int = 5,
    favorable_threshold: float = 0.3,
) -> pd.Series:
    """
    Create entry timing labels based on future price path.

    Label = 1 if entry within next `horizon_bars` has favorable outcome.

    "Favorable" is defined as:
    - For LONG: price doesn't drop more than 0.5*ATR before rising 1*ATR
    - For SHORT: price doesn't rise more than 0.5*ATR before dropping 1*ATR

    This is a simplification; real labels would use tick-by-tick data.

    Args:
        df: DataFrame with OHLCV + 'direction' column (LONG/SHORT from alpha model)
        horizon_bars: How many bars forward to check
        favorable_threshold: Threshold for favorable move (fraction of ATR)

    Returns:
        Series of 0/1 labels
    """
    if 'atr_14' not in df.columns:
        # Calculate ATR if not present
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

    labels = pd.Series(0, index=df.index)

    for i in range(len(df) - horizon_bars):
        idx = df.index[i]
        direction = df.loc[idx, 'direction'] if 'direction' in df.columns else 'LONG'
        atr = df.loc[idx, 'atr_14']
        entry_price = df.loc[idx, 'close']

        if pd.isna(atr) or atr <= 0:
            continue

        # Look at future bars
        future_slice = df.iloc[i+1:i+1+horizon_bars]

        if direction == 'LONG':
            # Good entry if max drawdown is small and we hit target
            max_low = future_slice['low'].min()
            max_high = future_slice['high'].max()

            drawdown = (entry_price - max_low) / atr if atr > 0 else 999
            gain = (max_high - entry_price) / atr if atr > 0 else 0

            # Favorable if gain > 1 ATR and drawdown < 0.5 ATR
            if gain >= 1.0 and drawdown < 0.5:
                labels[idx] = 1

        else:  # SHORT
            max_high = future_slice['high'].max()
            min_low = future_slice['low'].min()

            drawup = (max_high - entry_price) / atr if atr > 0 else 999
            gain = (entry_price - min_low) / atr if atr > 0 else 0

            if gain >= 1.0 and drawup < 0.5:
                labels[idx] = 1

    return labels


if __name__ == "__main__":
    # Test synthetic generation
    import sqlite3

    logging.basicConfig(level=logging.INFO)

    # Load sample data
    db_path = Path("data/moex_agent.sqlite")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT * FROM candles WHERE ticker='SBER' ORDER BY timestamp LIMIT 1000",
            conn
        )
        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Generate features
            micro_df = generate_micro_features_from_candles(df, "SBER")
            print(f"\nGenerated {len(micro_df)} rows of microstructure features")
            print(f"Feature columns: {list(micro_df.columns)}")
            print(f"\nSample statistics:")
            print(micro_df.describe())
        else:
            print("No data found")
    else:
        print(f"Database not found: {db_path}")
