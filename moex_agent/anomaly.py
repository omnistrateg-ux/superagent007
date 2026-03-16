"""
MOEX Agent v2 Anomaly Detection

MAD z-score based anomaly detection for price/volume spikes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Direction(str, Enum):
    """Signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class AnomalyResult:
    """Detected anomaly."""
    secid: str
    score: float
    direction: Direction
    z_ret_5m: float
    z_vol_5m: float
    ret_5m: float
    turnover_5m: float
    spread_bps: Optional[float]
    volume_spike: float


def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_z(value: float, hist: np.ndarray) -> float:
    """
    Robust z-score using MAD instead of std.

    More resistant to outliers than standard z-score.
    """
    hist = hist[~np.isnan(hist)]
    if hist.size < 30:
        return 0.0
    med = float(np.median(hist))
    mad = _mad(hist)
    if mad < 1e-12:
        return 0.0
    return float((value - med) / (1.4826 * mad))


def compute_anomalies(
    candles_1m: pd.DataFrame,
    quotes: Dict[str, Dict],
    min_turnover_rub_5m: float,
    max_spread_bps: float,
    top_n: int,
    min_abs_z_ret: float = 0.8,
) -> List[AnomalyResult]:
    """
    Detect price/volume anomalies using MAD z-scores.

    Args:
        candles_1m: DataFrame with [secid, ts, close, value, volume]
        quotes: Dict of {secid: {bid, ask, last, ...}}
        min_turnover_rub_5m: Minimum 5-min turnover
        max_spread_bps: Maximum spread penalty threshold
        top_n: Return top N anomalies
        min_abs_z_ret: Minimum |z_ret| to be considered

    Returns:
        List of AnomalyResult sorted by score descending
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    out: List[AnomalyResult] = []

    for secid, g in df.groupby("secid", sort=False):
        if len(g) < 200:
            continue

        g = g.set_index("ts")

        # 5-minute return
        r5 = g["close"].pct_change(5).iloc[-1]
        turn5 = g["value"].rolling(5).sum().iloc[-1]

        if pd.isna(r5) or pd.isna(turn5):
            continue

        # Historical series for robust z (last 2000 points)
        r5_hist = g["close"].pct_change(5).tail(2000).to_numpy(dtype=float)
        turn5_hist = g["value"].rolling(5).sum().tail(2000).to_numpy(dtype=float)

        z_ret = robust_z(float(r5), r5_hist)
        z_vol = robust_z(float(turn5), turn5_hist)

        # Volume spike
        vol_avg = g["volume"].tail(100).mean()
        vol_current = g["volume"].iloc[-1]
        volume_spike = float(vol_current / vol_avg) if vol_avg > 0 else 1.0

        # Spread calculation
        q = quotes.get(secid, {})
        spread_bps = None
        bid = q.get("bid")
        ask = q.get("ask")
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread_bps = float((ask - bid) / mid * 10000)

        # Direction
        direction = Direction.LONG if r5 > 0 else Direction.SHORT

        # Skip weak signals
        abs_z_ret = abs(z_ret)
        if abs_z_ret < min_abs_z_ret:
            continue

        # Scoring formula
        vol_bonus = 0.3 * np.clip(z_vol, 0, 4)
        spike_bonus = 0.2 * np.clip(volume_spike - 1, 0, 5)
        score = abs_z_ret + vol_bonus + spike_bonus

        # Penalties
        if spread_bps is not None and spread_bps > max_spread_bps:
            score -= 1.5

        if turn5 < min_turnover_rub_5m:
            score -= 0.5

        if z_ret > 1.0 and z_vol < -1.0:
            score -= 0.8

        out.append(
            AnomalyResult(
                secid=secid,
                score=float(score),
                direction=direction,
                z_ret_5m=float(z_ret),
                z_vol_5m=float(z_vol),
                ret_5m=float(r5),
                turnover_5m=float(turn5),
                spread_bps=spread_bps,
                volume_spike=volume_spike,
            )
        )

    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_n]
