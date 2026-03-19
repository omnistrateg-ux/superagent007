"""
MOEX Agent v2 Label Generation

Binary labels for ML training, net of 8bps fee.
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

    for secid, g in df.groupby("secid", sort=False):
        g = g.set_index("ts").reset_index()
        close = g["close"].astype(float).values
        high = g["high"].astype(float).values
        low = g["low"].astype(float).values

        atr = compute_atr(g.set_index("ts"), 14).values
        n = len(close)
        y_price = []

        for i in range(n):
            if i >= n - max_bars or pd.isna(atr[i]) or atr[i] <= 0:
                y_price.append(0)
                continue

            entry = close[i]
            take = entry + take_atr * atr[i]
            stop = entry - stop_atr * atr[i]

            hit_take = False
            for j in range(i + 1, min(i + max_bars + 1, n)):
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
    take_mult = 1 + take_pct / 100.0 + fee
    stop_mult_long = 1 - stop_pct / 100.0 - fee
    take_mult_short = 1 - take_pct / 100.0 - fee
    stop_mult_short = 1 + stop_pct / 100.0 + fee

    all_labels = []

    for secid, g in df.groupby("secid", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        close = g["close"].astype(float).values
        high = g["high"].astype(float).values
        low = g["low"].astype(float).values
        n = len(close)

        result = pd.DataFrame({"secid": secid, "ts": g["ts"]})

        for name, minutes in horizons:
            labels = np.zeros(n, dtype=int)

            for i in range(n - minutes):
                entry = close[i]
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
