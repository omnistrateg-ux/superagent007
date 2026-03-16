"""
MOEX Agent v2 Technical Features

30 technical indicators for ML model.
FEATURE_COLS is the canonical list used everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical feature columns - USE THIS EVERYWHERE
FEATURE_COLS = [
    # Returns (5)
    "r_1m", "r_5m", "r_10m", "r_30m", "r_60m",
    # Turnover (3)
    "turn_1m", "turn_5m", "turn_10m",
    # ATR & VWAP (2)
    "atr_14", "dist_vwap_atr",
    # RSI (2)
    "rsi_14", "rsi_7",
    # MACD (3)
    "macd", "macd_signal", "macd_hist",
    # Bollinger Bands (2)
    "bb_position", "bb_width",
    # Stochastic (2)
    "stoch_k", "stoch_d",
    # ADX (1)
    "adx",
    # OBV (1)
    "obv_change",
    # Momentum (2)
    "momentum_10", "momentum_30",
    # Volatility (2)
    "volatility_10", "volatility_30",
    # Moving averages (3)
    "price_sma20_ratio", "price_sma50_ratio", "sma20_sma50_ratio",
    # Volume (1)
    "volume_sma_ratio",
    # Extra (1)
    "hl_range",
]

assert len(FEATURE_COLS) == 30, f"Expected 30 features, got {len(FEATURE_COLS)}"


def compute_atr(g: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD indicator."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    position = (close - lower) / (upper - lower + 1e-9)
    return upper, lower, position


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
    """Stochastic Oscillator."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=period, adjust=False).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def build_feature_frame(candles_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature DataFrame from 1-minute candles.

    Args:
        candles_1m: DataFrame with columns [secid, ts, open, high, low, close, value, volume]

    Returns:
        DataFrame with 30 features per row
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    feats = []
    for secid, g in df.groupby("secid", sort=False):
        g = g.set_index("ts")
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float)

        g_feat = pd.DataFrame(index=g.index)
        g_feat["secid"] = secid

        # Returns
        g_feat["r_1m"] = close.pct_change(1)
        g_feat["r_5m"] = close.pct_change(5)
        g_feat["r_10m"] = close.pct_change(10)
        g_feat["r_30m"] = close.pct_change(30)
        g_feat["r_60m"] = close.pct_change(60)

        # Turnover
        g_feat["turn_1m"] = g["value"].astype(float)
        g_feat["turn_5m"] = g["value"].astype(float).rolling(5).sum()
        g_feat["turn_10m"] = g["value"].astype(float).rolling(10).sum()

        # ATR & VWAP
        g_feat["atr_14"] = compute_atr(g, 14)
        vwap_30 = (g["value"].astype(float).rolling(30).sum() / (volume.rolling(30).sum() + 1e-9))
        g_feat["dist_vwap_atr"] = (close - vwap_30) / (g_feat["atr_14"] + 1e-9)

        # RSI
        g_feat["rsi_14"] = compute_rsi(close, 14)
        g_feat["rsi_7"] = compute_rsi(close, 7)

        # MACD
        macd_line, signal_line, histogram = compute_macd(close)
        g_feat["macd"] = macd_line
        g_feat["macd_signal"] = signal_line
        g_feat["macd_hist"] = histogram

        # Bollinger Bands
        bb_upper, bb_lower, bb_position = compute_bollinger(close, 20, 2.0)
        g_feat["bb_position"] = bb_position
        g_feat["bb_width"] = (bb_upper - bb_lower) / (close + 1e-9)

        # Stochastic
        stoch_k, stoch_d = compute_stochastic(high, low, close)
        g_feat["stoch_k"] = stoch_k
        g_feat["stoch_d"] = stoch_d

        # ADX
        g_feat["adx"] = compute_adx(high, low, close)

        # OBV
        obv = compute_obv(close, volume)
        obv_mean = obv.rolling(100, min_periods=10).mean().abs() + 1e-9
        g_feat["obv_change"] = obv.diff(10) / obv_mean

        # Momentum
        g_feat["momentum_10"] = close / close.shift(10) - 1
        g_feat["momentum_30"] = close / close.shift(30) - 1

        # Volatility
        g_feat["volatility_10"] = close.pct_change().rolling(10).std()
        g_feat["volatility_30"] = close.pct_change().rolling(30).std()

        # Moving averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        g_feat["price_sma20_ratio"] = close / (sma_20 + 1e-9)
        g_feat["price_sma50_ratio"] = close / (sma_50 + 1e-9)
        g_feat["sma20_sma50_ratio"] = sma_20 / (sma_50 + 1e-9)

        # Volume
        g_feat["volume_sma_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

        # High-Low range
        g_feat["hl_range"] = (high - low) / (close + 1e-9)

        # Store close for backtesting
        g_feat["close"] = close

        feats.append(g_feat.reset_index().rename(columns={"index": "ts"}))

    out = pd.concat(feats, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
