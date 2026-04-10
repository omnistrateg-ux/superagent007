"""
MOEX Agent v2 Order Flow Features

Advanced features for real predictive power:
- Volume Imbalance (buy vs sell pressure)
- Trade Flow (institutional activity detection)
- Spread Dynamics (liquidity changes)
- Price-Volume Divergence (trend weakness detection)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("moex_agent.orderflow")


def estimate_buy_volume(row: pd.Series) -> float:
    """
    Estimate buy volume using candle structure.

    Uses the position of close relative to high-low range.
    Close near high = more buying, close near low = more selling.
    """
    high, low, close, volume = row["high"], row["low"], row["close"], row["volume"]

    if high == low or volume == 0:
        return volume * 0.5

    # Buy ratio: where close is in the range [low, high]
    buy_ratio = (close - low) / (high - low)
    return volume * buy_ratio


def compute_volume_imbalance(df: pd.DataFrame, windows: list = [5, 15, 30]) -> pd.DataFrame:
    """
    Compute Volume Imbalance features.

    buy_volume / total_volume over different windows.
    Values > 0.5 indicate buying pressure, < 0.5 selling pressure.
    """
    result = pd.DataFrame(index=df.index)

    # Estimate buy volume for each candle
    buy_vol = df.apply(estimate_buy_volume, axis=1)
    total_vol = df["volume"]

    for w in windows:
        buy_sum = buy_vol.rolling(window=w, min_periods=1).sum()
        total_sum = total_vol.rolling(window=w, min_periods=1).sum()

        # Volume imbalance ratio
        result[f"vol_imbalance_{w}m"] = (buy_sum / total_sum.replace(0, np.nan)).fillna(0.5)

        # Imbalance change (momentum of buying/selling)
        result[f"vol_imbalance_chg_{w}m"] = result[f"vol_imbalance_{w}m"].diff(w).fillna(0)

    # Cross-window imbalance divergence (short vs long term)
    if 5 in windows and 30 in windows:
        result["vol_imbalance_divergence"] = (
            result["vol_imbalance_5m"] - result["vol_imbalance_30m"]
        )

    return result


def compute_trade_flow(df: pd.DataFrame, windows: list = [5, 15, 30]) -> pd.DataFrame:
    """
    Compute Trade Flow features.

    - Average trade size (value / volume)
    - Trade count proxy (volume / avg_trade_size)
    - Large trade ratio
    """
    result = pd.DataFrame(index=df.index)

    # Average trade size (RUB per share)
    # Using value (turnover in RUB) / volume (shares)
    avg_trade_size = (df["value"] / df["volume"].replace(0, np.nan)).fillna(0)
    result["avg_trade_size"] = avg_trade_size

    for w in windows:
        # Rolling average trade size
        rolling_avg = avg_trade_size.rolling(window=w, min_periods=1).mean()
        result[f"avg_trade_size_{w}m"] = rolling_avg

        # Normalized: current vs rolling average
        result[f"trade_size_ratio_{w}m"] = (
            avg_trade_size / rolling_avg.replace(0, np.nan)
        ).fillna(1.0)

        # Large trade detection: trades > 2x rolling average
        is_large = (avg_trade_size > 2 * rolling_avg).astype(float)
        result[f"large_trade_ratio_{w}m"] = is_large.rolling(window=w, min_periods=1).mean()

        # Volume intensity (volume relative to rolling average)
        vol_avg = df["volume"].rolling(window=w, min_periods=1).mean()
        result[f"volume_intensity_{w}m"] = (
            df["volume"] / vol_avg.replace(0, np.nan)
        ).fillna(1.0)

    # Institutional activity proxy: large trades + high value
    result["institutional_score"] = (
        result.get("large_trade_ratio_15m", 0) * 0.5 +
        (result.get("trade_size_ratio_15m", 1) - 1).clip(0, 2) * 0.5
    )

    return result


def compute_spread_dynamics(
    df: pd.DataFrame,
    spread_col: str = "spread_pct",
    windows: list = [15, 60],
) -> pd.DataFrame:
    """
    Compute Spread Dynamics features.

    - Current spread vs average
    - Spread narrowing/widening momentum
    """
    result = pd.DataFrame(index=df.index)

    # If no spread column, estimate from high-low
    if spread_col not in df.columns:
        # Estimate spread as (high - low) / close
        spread = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0) * 100
    else:
        spread = df[spread_col]

    result["spread_current"] = spread

    for w in windows:
        # Rolling average spread
        spread_avg = spread.rolling(window=w, min_periods=1).mean()
        result[f"spread_avg_{w}m"] = spread_avg

        # Spread ratio: current / average
        # < 1 = tighter spread = more liquidity
        result[f"spread_ratio_{w}m"] = (
            spread / spread_avg.replace(0, np.nan)
        ).fillna(1.0)

        # Spread momentum (is spread narrowing?)
        result[f"spread_momentum_{w}m"] = -spread.diff(w).fillna(0)

    # Liquidity improvement score
    # Positive = spread narrowing, negative = spread widening
    result["liquidity_score"] = (
        (1 - result.get("spread_ratio_15m", 1)).clip(-1, 1) * 0.5 +
        result.get("spread_momentum_15m", 0).clip(-0.1, 0.1) * 5
    )

    return result


def compute_price_volume_divergence(
    df: pd.DataFrame,
    windows: list = [5, 15, 30],
) -> pd.DataFrame:
    """
    Compute Price-Volume Divergence features.

    Detects weak trends:
    - Price up + volume down = weak bullish
    - Price down + volume down = weak bearish
    - Price up + volume up = strong bullish
    - Price down + volume up = strong bearish (capitulation)
    """
    result = pd.DataFrame(index=df.index)

    for w in windows:
        # Price change over window
        price_chg = df["close"].pct_change(w).fillna(0)

        # Volume change over window
        vol_avg_now = df["volume"].rolling(window=w, min_periods=1).mean()
        vol_avg_prev = df["volume"].shift(w).rolling(window=w, min_periods=1).mean()
        vol_chg = ((vol_avg_now / vol_avg_prev.replace(0, np.nan)) - 1).fillna(0)

        result[f"price_chg_{w}m"] = price_chg
        result[f"volume_chg_{w}m"] = vol_chg

        # Divergence: positive when price and volume move together
        # Negative when they diverge (weak trend signal)
        result[f"pv_divergence_{w}m"] = np.sign(price_chg) * vol_chg

        # Trend strength: |price_chg| * volume_chg
        # High when big move with increasing volume
        result[f"trend_strength_{w}m"] = abs(price_chg) * (1 + vol_chg.clip(-0.5, 2))

        # Weak trend flag: price moving but volume declining
        weak_up = (price_chg > 0.001) & (vol_chg < -0.1)
        weak_down = (price_chg < -0.001) & (vol_chg < -0.1)
        result[f"weak_trend_{w}m"] = (weak_up | weak_down).astype(float)

    # Aggregate divergence score
    # Positive = confirmed trend, negative = weak/divergent trend
    result["divergence_score"] = (
        result.get("pv_divergence_5m", 0) * 0.5 +
        result.get("pv_divergence_15m", 0) * 0.3 +
        result.get("pv_divergence_30m", 0) * 0.2
    )

    return result


def compute_momentum_quality(df: pd.DataFrame, windows: list = [5, 15]) -> pd.DataFrame:
    """
    Compute Momentum Quality features.

    Not just direction, but quality of the move.
    """
    result = pd.DataFrame(index=df.index)

    for w in windows:
        # Price momentum
        ret = df["close"].pct_change(w).fillna(0)

        # Volatility over the window
        vol = df["close"].pct_change().rolling(window=w, min_periods=1).std().fillna(0)

        # Momentum quality: return / volatility (like mini Sharpe)
        result[f"momentum_quality_{w}m"] = (ret / vol.replace(0, np.nan)).fillna(0).clip(-5, 5)

        # Consistency: how many candles moved in same direction
        direction = np.sign(df["close"].diff())
        same_dir = (direction == direction.shift(1)).astype(float)
        result[f"momentum_consistency_{w}m"] = same_dir.rolling(window=w, min_periods=1).mean()

        # Acceleration: is momentum increasing?
        ret_prev = df["close"].pct_change(w).shift(w).fillna(0)
        result[f"momentum_accel_{w}m"] = ret - ret_prev

    return result


def compute_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all order flow features for a single ticker.

    Args:
        df: DataFrame with columns [ts, open, high, low, close, value, volume]
            Optionally: spread_pct

    Returns:
        DataFrame with order flow features
    """
    # Compute all feature groups
    vi = compute_volume_imbalance(df)      # Volume Imbalance
    tf = compute_trade_flow(df)            # Trade Flow
    sd = compute_spread_dynamics(df)       # Spread Dynamics
    pvd = compute_price_volume_divergence(df)  # Price-Volume Divergence
    mq = compute_momentum_quality(df)      # Momentum Quality

    # Single concat is O(n) instead of 5x O(n) for sequential concats
    result = pd.concat([vi, tf, sd, pvd, mq], axis=1)

    return result


def build_orderflow_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Build order flow features for all tickers.

    Args:
        candles: DataFrame with [secid, ts, open, high, low, close, value, volume]

    Returns:
        DataFrame with [secid, ts] + order flow features
    """
    logger.info("Building order flow features...")

    results = []
    tickers = candles["secid"].unique()

    for secid in tickers:
        ticker_data = candles[candles["secid"] == secid].copy()
        ticker_data = ticker_data.sort_values("ts").reset_index(drop=True)

        if len(ticker_data) < 30:
            continue

        features = compute_orderflow_features(ticker_data)
        features["secid"] = secid
        features["ts"] = ticker_data["ts"].values

        results.append(features)

    if not results:
        return pd.DataFrame()

    result = pd.concat(results, ignore_index=True)
    logger.info(f"Order flow features: {len(result):,} rows, {len(result.columns) - 2} features")

    return result


# List of order flow feature columns
ORDERFLOW_FEATURE_COLS = [
    # Volume Imbalance
    "vol_imbalance_5m",
    "vol_imbalance_15m",
    "vol_imbalance_30m",
    "vol_imbalance_chg_5m",
    "vol_imbalance_chg_15m",
    "vol_imbalance_chg_30m",
    "vol_imbalance_divergence",

    # Trade Flow
    "avg_trade_size",
    "avg_trade_size_5m",
    "avg_trade_size_15m",
    "avg_trade_size_30m",
    "trade_size_ratio_5m",
    "trade_size_ratio_15m",
    "trade_size_ratio_30m",
    "large_trade_ratio_5m",
    "large_trade_ratio_15m",
    "large_trade_ratio_30m",
    "volume_intensity_5m",
    "volume_intensity_15m",
    "volume_intensity_30m",
    "institutional_score",

    # Spread Dynamics
    "spread_current",
    "spread_avg_15m",
    "spread_avg_60m",
    "spread_ratio_15m",
    "spread_ratio_60m",
    "spread_momentum_15m",
    "spread_momentum_60m",
    "liquidity_score",

    # Price-Volume Divergence
    "price_chg_5m",
    "price_chg_15m",
    "price_chg_30m",
    "volume_chg_5m",
    "volume_chg_15m",
    "volume_chg_30m",
    "pv_divergence_5m",
    "pv_divergence_15m",
    "pv_divergence_30m",
    "trend_strength_5m",
    "trend_strength_15m",
    "trend_strength_30m",
    "weak_trend_5m",
    "weak_trend_15m",
    "weak_trend_30m",
    "divergence_score",

    # Momentum Quality
    "momentum_quality_5m",
    "momentum_quality_15m",
    "momentum_consistency_5m",
    "momentum_consistency_15m",
    "momentum_accel_5m",
    "momentum_accel_15m",
]
