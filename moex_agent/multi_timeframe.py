"""
MOEX Agent v2 Multi-Timeframe Analysis

Trend analysis using higher timeframes:
- SMA crossover (SMA20 vs SMA50)
- ADX for trend strength
- Trend alignment checks
- Position sizing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("moex_agent.multi_timeframe")


class TrendDirection(Enum):
    """Trend direction classification."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class TrendState:
    """Trend analysis result."""
    direction: TrendDirection
    strength: float  # 0-100 (ADX)
    sma20: float
    sma50: float
    adx: float
    is_strong: bool  # ADX > 25

    def __repr__(self) -> str:
        return f"TrendState({self.direction.value}, ADX={self.adx:.1f}, strong={self.is_strong})"


def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Average Directional Index (ADX).

    Args:
        df: DataFrame with high, low, close columns
        period: ADX period (default 14)

    Returns:
        Current ADX value
    """
    if len(df) < period * 2:
        return 0.0

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages (Wilder's smoothing)
    def wilder_smooth(arr, n):
        result = np.zeros_like(arr, dtype=float)
        result[n-1] = np.sum(arr[:n])
        for i in range(n, len(arr)):
            result[i] = result[i-1] - result[i-1]/n + arr[i]
        return result

    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * wilder_smooth(minus_dm, period) / (atr + 1e-10)

    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, period)

    # Return last valid ADX
    valid_adx = adx[~np.isnan(adx)]
    return float(valid_adx[-1]) if len(valid_adx) > 0 else 0.0


def analyze_trend(ticker: str, candles_1h: pd.DataFrame) -> TrendState:
    """
    Analyze trend using 1-hour candles.

    Uses:
    - SMA20 vs SMA50 crossover for direction
    - ADX for trend strength

    Args:
        ticker: Security ticker
        candles_1h: DataFrame with OHLCV data (1-hour timeframe)

    Returns:
        TrendState with direction and strength
    """
    if len(candles_1h) < 50:
        return TrendState(
            direction=TrendDirection.SIDEWAYS,
            strength=0.0,
            sma20=0.0,
            sma50=0.0,
            adx=0.0,
            is_strong=False,
        )

    close = candles_1h["close"]

    # Compute SMAs
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])

    # Compute ADX
    adx = _compute_adx(candles_1h, period=14)

    # Determine trend direction
    if sma20 > sma50 * 1.005:  # 0.5% buffer
        direction = TrendDirection.UP
    elif sma20 < sma50 * 0.995:
        direction = TrendDirection.DOWN
    else:
        direction = TrendDirection.SIDEWAYS

    # Strong trend if ADX > 25
    is_strong = adx > 25

    return TrendState(
        direction=direction,
        strength=adx,
        sma20=sma20,
        sma50=sma50,
        adx=adx,
        is_strong=is_strong,
    )


def check_trend_alignment(
    trend_state: TrendState,
    direction: str,
) -> Tuple[bool, str]:
    """
    Check if signal direction aligns with higher timeframe trend.

    Rules:
    - LONG only allowed when trend is UP
    - SHORT only allowed when trend is DOWN
    - SIDEWAYS allows both but with warning

    Args:
        trend_state: Current trend analysis
        direction: Signal direction ("LONG" or "SHORT")

    Returns:
        (is_aligned, reason) tuple
    """
    direction_upper = direction.upper() if isinstance(direction, str) else direction.value.upper()

    if trend_state.direction == TrendDirection.UP:
        if direction_upper == "LONG":
            return True, "Aligned: LONG with UP trend"
        else:
            return False, f"Misaligned: SHORT against UP trend (ADX={trend_state.adx:.1f})"

    elif trend_state.direction == TrendDirection.DOWN:
        if direction_upper == "SHORT":
            return True, "Aligned: SHORT with DOWN trend"
        else:
            return False, f"Misaligned: LONG against DOWN trend (ADX={trend_state.adx:.1f})"

    else:  # SIDEWAYS
        # Allow but warn - weaker signals in sideways markets
        if trend_state.is_strong:
            return True, f"Sideways but strong ADX={trend_state.adx:.1f}"
        else:
            return True, f"Sideways trend, weak ADX={trend_state.adx:.1f}"


def compute_entry_target_stop(
    price: float,
    direction: str,
    atr: float,
    risk_reward: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Compute entry, target, and stop prices.

    Args:
        price: Current price (entry)
        direction: "LONG" or "SHORT"
        atr: Average True Range
        risk_reward: Risk/reward ratio (default 2.0)

    Returns:
        (entry, target, stop) tuple
    """
    direction_upper = direction.upper() if isinstance(direction, str) else direction.value.upper()

    stop_distance = atr * 1.5  # 1.5 ATR stop
    target_distance = stop_distance * risk_reward

    entry = price

    if direction_upper == "LONG":
        stop = price - stop_distance
        target = price + target_distance
    else:
        stop = price + stop_distance
        target = price - target_distance

    return entry, target, stop


def compute_position_size(
    equity: float,
    entry: float,
    stop: float,
    max_risk_pct: float = 0.5,
) -> int:
    """
    Compute position size based on risk.

    Args:
        equity: Account equity
        entry: Entry price
        stop: Stop loss price
        max_risk_pct: Maximum risk per trade (default 0.5%)

    Returns:
        Number of shares/lots
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0

    max_risk_amount = equity * (max_risk_pct / 100)
    position_size = int(max_risk_amount / risk_per_share)

    return max(0, position_size)
