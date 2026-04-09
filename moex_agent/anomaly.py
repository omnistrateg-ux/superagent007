"""
MOEX Agent v2 Anomaly Detection

MAD z-score based anomaly detection for price/volume spikes.

v2.1: anomaly_score теперь feature, а не gate. Функция compute_anomaly_features()
добавляет z-scores как признаки для ML модели.
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


def compute_anomaly_features(candles_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute anomaly features for ALL candles (not just top-N).

    These features become inputs to the ML model instead of being
    used as a hard gate in the pipeline.

    Features added:
    - anomaly_z_ret_5m: MAD z-score of 5-min return
    - anomaly_z_vol_5m: MAD z-score of 5-min turnover
    - anomaly_score: composite score (|z_ret| + vol_bonus + spike_bonus)
    - anomaly_volume_spike: current volume / avg volume ratio
    - anomaly_direction: 1 for positive return, -1 for negative

    Args:
        candles_1m: DataFrame with [secid, ts, close, value, volume]

    Returns:
        DataFrame with [secid, ts, anomaly_*] columns
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    results = []

    for secid, g in df.groupby("secid", sort=False):
        g = g.set_index("ts").copy()

        if len(g) < 200:
            # Not enough data - return NaN features
            feat = pd.DataFrame(index=g.index)
            feat["secid"] = secid
            feat["anomaly_z_ret_5m"] = np.nan
            feat["anomaly_z_vol_5m"] = np.nan
            feat["anomaly_score"] = np.nan
            feat["anomaly_volume_spike"] = np.nan
            feat["anomaly_direction"] = 0
            results.append(feat.reset_index().rename(columns={"index": "ts"}))
            continue

        # 5-minute return series
        r5 = g["close"].pct_change(5)

        # 5-minute turnover series
        turn5 = g["value"].rolling(5).sum()

        # Volume spike: current / rolling mean
        vol_avg = g["volume"].rolling(100, min_periods=10).mean()
        volume_spike = g["volume"] / (vol_avg + 1e-9)

        # Calculate rolling MAD z-scores
        # Use expanding window for robustness, then take last 2000 points
        z_ret = pd.Series(index=g.index, dtype=float)
        z_vol = pd.Series(index=g.index, dtype=float)

        # Vectorized rolling z-score calculation
        window = 2000
        for i in range(len(g)):
            if i < 200:
                z_ret.iloc[i] = 0.0
                z_vol.iloc[i] = 0.0
                continue

            start_idx = max(0, i - window)

            # Return z-score
            r5_hist = r5.iloc[start_idx:i].dropna().values
            if len(r5_hist) >= 30:
                r5_val = r5.iloc[i]
                if not pd.isna(r5_val):
                    z_ret.iloc[i] = robust_z(float(r5_val), r5_hist)
                else:
                    z_ret.iloc[i] = 0.0
            else:
                z_ret.iloc[i] = 0.0

            # Turnover z-score
            turn5_hist = turn5.iloc[start_idx:i].dropna().values
            if len(turn5_hist) >= 30:
                turn5_val = turn5.iloc[i]
                if not pd.isna(turn5_val):
                    z_vol.iloc[i] = robust_z(float(turn5_val), turn5_hist)
                else:
                    z_vol.iloc[i] = 0.0
            else:
                z_vol.iloc[i] = 0.0

        # Composite anomaly score
        abs_z_ret = z_ret.abs()
        vol_bonus = 0.3 * z_vol.clip(0, 4)
        spike_bonus = 0.2 * (volume_spike - 1).clip(0, 5)
        anomaly_score = abs_z_ret + vol_bonus + spike_bonus

        # Direction: 1 for LONG (positive return), -1 for SHORT
        direction = np.where(r5 > 0, 1, np.where(r5 < 0, -1, 0))

        # Build result DataFrame
        feat = pd.DataFrame(index=g.index)
        feat["secid"] = secid
        feat["anomaly_z_ret_5m"] = z_ret
        feat["anomaly_z_vol_5m"] = z_vol
        feat["anomaly_score"] = anomaly_score
        feat["anomaly_volume_spike"] = volume_spike
        feat["anomaly_direction"] = direction

        results.append(feat.reset_index().rename(columns={"index": "ts"}))

    if not results:
        return pd.DataFrame(columns=[
            "secid", "ts", "anomaly_z_ret_5m", "anomaly_z_vol_5m",
            "anomaly_score", "anomaly_volume_spike", "anomaly_direction"
        ])

    return pd.concat(results, ignore_index=True)
