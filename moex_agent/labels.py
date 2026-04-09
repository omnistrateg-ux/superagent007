"""
MOEX Agent v2.4 Label Generation

Binary labels for ML training, net of fees.

v2.4 Changes (Phase 0 - Fix Leakage):
- CRITICAL: ATR now lagged by 1 bar to avoid look-ahead bias
- ATR[i] uses only data from bars [0..i-1], not including bar i
- This prevents barrier levels from "knowing" current bar volatility
- CRITICAL: Entry price is now NEXT bar's open + slippage (0.03%)
- Signal at bar i → execution at bar i+1 open
- This reflects realistic execution, not instant fill at current close

v2.1 Changes:
- Added make_atr_trend_labels() for volatility-adjusted targets
- ATR-based take/stop instead of fixed percentages
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def make_time_exit_labels(
    candles_1m: pd.DataFrame,
    horizons: List[Tuple[str, int]],
    fee_bps: float = 8.0,
) -> pd.DataFrame:
    """
    Create binary labels for time-based exits.

    Label = 1 if return after H minutes > 0 after round-trip fee.

    Args:
        candles_1m: DataFrame with columns [secid, ts, close, ...]
        horizons: List of (name, minutes) tuples
        fee_bps: Round-trip cost in basis points (default 8)

    Returns:
        DataFrame with columns [secid, ts, y_time_5m, y_time_10m, ...]
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    labels = []
    cost = fee_bps / 10000.0

    for secid, g in df.groupby("secid", sort=False):
        g = g.set_index("ts")
        close = g["close"].astype(float)
        y = pd.DataFrame(index=g.index)
        y["secid"] = secid

        for name, minutes in horizons:
            fut = close.shift(-minutes)
            ret = (fut / close) - 1.0
            ret_net = ret - cost
            y[f"y_time_{name}"] = (ret_net > 0).astype(int)

        labels.append(y.reset_index().rename(columns={"index": "ts"}))

    return pd.concat(labels, ignore_index=True)


def make_price_exit_labels(
    candles_1m: pd.DataFrame,
    take_atr: float = 0.8,
    stop_atr: float = 0.6,
    max_bars: int = 60,
) -> pd.DataFrame:
    """
    Create binary labels for price-based exits (take profit / stop loss).

    Label = 1 if take profit hit before stop loss within max_bars.

    Args:
        candles_1m: DataFrame with candle data
        take_atr: Take profit distance in ATR units
        stop_atr: Stop loss distance in ATR units
        max_bars: Maximum bars to wait for exit

    Returns:
        DataFrame with y_price column
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    from .features import compute_atr

    labels = []

    slippage = 0.0003  # 0.03% slippage

    for secid, g in df.groupby("secid", sort=False):
        g = g.set_index("ts").reset_index()
        open_prices = g["open"].astype(float).values
        close = g["close"].astype(float).values
        high = g["high"].astype(float).values
        low = g["low"].astype(float).values

        # ATR with 1-bar lag to avoid look-ahead bias
        atr = compute_atr(g.set_index("ts"), 14).shift(1).values
        n = len(close)
        y_price = []

        for i in range(n):
            # Need i >= 1 because of ATR lag, and i+1 for next bar entry
            if i < 1 or i >= n - max_bars - 1 or pd.isna(atr[i]) or atr[i] <= 0:
                y_price.append(0)
                continue

            # CRITICAL: Entry at NEXT bar's open + slippage
            entry = open_prices[i + 1] * (1 + slippage)
            take = entry + take_atr * atr[i]
            stop = entry - stop_atr * atr[i]

            hit_take = False
            for j in range(i + 1, min(i + 1 + max_bars + 1, n)):
                if high[j] >= take:
                    hit_take = True
                    break
                if low[j] <= stop:
                    break

            y_price.append(1 if hit_take else 0)

        g["y_price"] = y_price
        g["secid"] = secid
        labels.append(g[["secid", "ts", "y_price"]])

    return pd.concat(labels, ignore_index=True)


def make_trend_following_labels(
    candles_1m: pd.DataFrame,
    horizons: List[Tuple[str, int]],
    take_pct: float = 1.5,
    stop_pct: float = 0.75,
    fee_bps: float = 8.0,
) -> pd.DataFrame:
    """
    Create trend-following labels with asymmetric R:R.

    LONG label (1): price reaches +take_pct% BEFORE falling -stop_pct%
    SHORT label (-1): price falls -take_pct% BEFORE rising +stop_pct%
    No signal (0): neither condition met within horizon

    This creates a 2:1 R:R ratio by default (1.5% take / 0.75% stop).

    Args:
        candles_1m: DataFrame with columns [secid, ts, open, high, low, close]
        horizons: List of (name, minutes) tuples
        take_pct: Take profit percentage (default 1.5%)
        stop_pct: Stop loss percentage (default 0.75%)
        fee_bps: Round-trip fee in basis points

    Returns:
        DataFrame with columns [secid, ts, y_trend_5m, y_trend_10m, ...]
        Values: 1 (LONG), -1 (SHORT), 0 (no signal)
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    fee = fee_bps / 10000.0
    slippage = 0.0003  # 0.03% slippage

    # Adjust multipliers to include slippage for LONG entry (buy higher)
    take_mult = 1 + take_pct / 100.0 + fee
    stop_mult_long = 1 - stop_pct / 100.0 - fee
    take_mult_short = 1 - take_pct / 100.0 - fee
    stop_mult_short = 1 + stop_pct / 100.0 + fee

    all_labels = []

    for secid, g in df.groupby("secid", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        open_prices = g["open"].astype(float).values
        close = g["close"].astype(float).values
        high = g["high"].astype(float).values
        low = g["low"].astype(float).values
        n = len(close)

        result = pd.DataFrame({"secid": secid, "ts": g["ts"]})

        for name, minutes in horizons:
            labels = np.zeros(n, dtype=int)

            for i in range(n - minutes - 1):  # -1 for next bar entry
                # CRITICAL: Entry at NEXT bar's open + slippage
                entry = open_prices[i + 1] * (1 + slippage)
                take_long = entry * take_mult
                stop_long = entry * stop_mult_long
                take_short = entry * take_mult_short
                stop_short = entry * stop_mult_short

                long_win = False
                long_loss = False
                short_win = False
                short_loss = False

                # Check future bars within horizon
                for j in range(i + 1, min(i + minutes + 1, n)):
                    # LONG: check if take hit before stop
                    if not long_win and not long_loss:
                        if high[j] >= take_long:
                            long_win = True
                        elif low[j] <= stop_long:
                            long_loss = True

                    # SHORT: check if take hit before stop
                    if not short_win and not short_loss:
                        if low[j] <= take_short:
                            short_win = True
                        elif high[j] >= stop_short:
                            short_loss = True

                    # Early exit if both decided
                    if (long_win or long_loss) and (short_win or short_loss):
                        break

                # Assign label
                if long_win and not short_win:
                    labels[i] = 1  # LONG signal
                elif short_win and not long_win:
                    labels[i] = -1  # SHORT signal
                elif long_win and short_win:
                    labels[i] = 0  # Ambiguous - no signal
                else:
                    labels[i] = 0  # Neither hit - no signal

            result[f"y_trend_{name}"] = labels

        all_labels.append(result)

    return pd.concat(all_labels, ignore_index=True)


def make_atr_trend_labels(
    candles_1m: pd.DataFrame,
    horizons: List[Tuple[str, int]],
    take_atr_mult: float = 2.0,
    stop_atr_mult: float = 1.0,
    atr_period: int = 14,
    fee_bps: float = 8.0,
) -> pd.DataFrame:
    """
    Create trend-following labels with ATR-based take/stop levels.

    v2.1: Volatility-adjusted targets instead of fixed percentages.
    This adapts to market conditions - wider targets in volatile markets,
    tighter targets in calm markets.

    LONG label (1): price reaches +take_atr_mult*ATR BEFORE falling -stop_atr_mult*ATR
    SHORT label (-1): price falls -take_atr_mult*ATR BEFORE rising +stop_atr_mult*ATR
    No signal (0): neither condition met within horizon

    Default R:R is 2:1 (take=2*ATR, stop=1*ATR).

    Args:
        candles_1m: DataFrame with columns [secid, ts, open, high, low, close]
        horizons: List of (name, minutes) tuples
        take_atr_mult: Take profit in ATR units (default 2.0)
        stop_atr_mult: Stop loss in ATR units (default 1.0)
        atr_period: Period for ATR calculation (default 14)
        fee_bps: Round-trip fee in basis points

    Returns:
        DataFrame with columns [secid, ts, y_atr_5m, y_atr_10m, ...]
        Values: 1 (LONG), -1 (SHORT), 0 (no signal)
    """
    from .features import compute_atr

    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    fee = fee_bps / 10000.0

    all_labels = []

    # Default slippage for label creation (conservative estimate)
    # 0.05% for liquid stocks, but we use 0.03% as average
    slippage = 0.0003

    for secid, g in df.groupby("secid", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        g_indexed = g.set_index("ts")

        open_prices = g["open"].astype(float).values
        close = g["close"].astype(float).values
        high = g["high"].astype(float).values
        low = g["low"].astype(float).values
        n = len(close)

        # Calculate ATR with 1-bar lag to avoid look-ahead bias
        # ATR[i] should only use data from bars [0..i-1], not including bar i
        atr_series = compute_atr(g_indexed, period=atr_period).shift(1)
        atr = atr_series.values

        result = pd.DataFrame({"secid": secid, "ts": g["ts"]})

        for name, minutes in horizons:
            labels = np.zeros(n, dtype=int)

            for i in range(n - minutes - 1):  # -1 for next bar entry
                # Skip if ATR is not available or too small
                # With shift(1), first valid ATR is at index atr_period + 1
                if i <= atr_period or pd.isna(atr[i]) or atr[i] <= 0:
                    labels[i] = 0
                    continue

                # CRITICAL: Entry at NEXT bar's open + slippage
                # Signal at bar i, execution at bar i+1
                entry = open_prices[i + 1] * (1 + slippage)
                current_atr = atr[i]

                # ATR-based levels with fee adjustment
                take_long = entry + take_atr_mult * current_atr
                stop_long = entry - stop_atr_mult * current_atr
                take_short = entry - take_atr_mult * current_atr
                stop_short = entry + stop_atr_mult * current_atr

                # Adjust for fees (makes targets slightly harder to hit)
                fee_adj = entry * fee
                take_long += fee_adj
                stop_long -= fee_adj
                take_short -= fee_adj
                stop_short += fee_adj

                long_win = False
                long_loss = False
                short_win = False
                short_loss = False

                # Check future bars starting from entry bar (i+1) within horizon
                for j in range(i + 1, min(i + 1 + minutes + 1, n)):
                    # LONG: check if take hit before stop
                    if not long_win and not long_loss:
                        if high[j] >= take_long:
                            long_win = True
                        elif low[j] <= stop_long:
                            long_loss = True

                    # SHORT: check if take hit before stop
                    if not short_win and not short_loss:
                        if low[j] <= take_short:
                            short_win = True
                        elif high[j] >= stop_short:
                            short_loss = True

                    # Early exit if both decided
                    if (long_win or long_loss) and (short_win or short_loss):
                        break

                # Assign label
                if long_win and not short_win:
                    labels[i] = 1  # LONG signal
                elif short_win and not long_win:
                    labels[i] = -1  # SHORT signal
                elif long_win and short_win:
                    labels[i] = 0  # Ambiguous - no signal
                else:
                    labels[i] = 0  # Neither hit - no signal

            result[f"y_atr_{name}"] = labels

        all_labels.append(result)

    return pd.concat(all_labels, ignore_index=True)
