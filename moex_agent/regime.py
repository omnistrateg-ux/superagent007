"""
MOEX Agent v2.5 Regime Detection

Phase 3: Per-ticker regime classification using OHLCV features.

Two-level regime detection:
1. Market-wide (existing): IMOEX, USD/RUB → macro regime
2. Per-ticker (new): OHLCV features → micro regime

Per-ticker regimes:
- TREND_UP: Strong uptrend, high ADX, positive momentum
- TREND_DOWN: Strong downtrend, high ADX, negative momentum
- RANGE_LOW_VOL: Sideways, low volatility (mean-reversion)
- RANGE_HIGH_VOL: Sideways, high volatility (avoid)

Trading rules:
- TREND_UP: Only LONG signals (trend-following)
- TREND_DOWN: Only SHORT signals (trend-following)
- RANGE_LOW_VOL: Both directions (mean-reversion)
- RANGE_HIGH_VOL: Reduce size or skip (whipsaw risk)

Integration with Alpha Model:
- Alpha Model generates LONG/SHORT signals
- Regime filter checks direction alignment
- Misaligned signals are filtered out

Usage:
    from moex_agent.regime import RegimeDetector, filter_signal_by_regime

    detector = RegimeDetector()
    regime = detector.detect(df)  # df has OHLCV features

    # Filter signal
    allow, reason = filter_signal_by_regime("LONG", regime)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TickerRegime(str, Enum):
    """Per-ticker regime classification."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE_LOW_VOL = "range_low_vol"
    RANGE_HIGH_VOL = "range_high_vol"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: TickerRegime
    confidence: float  # 0-1
    volatility_percentile: float  # 0-100
    trend_strength: float  # ADX value
    momentum: float  # Recent return

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "volatility_percentile": self.volatility_percentile,
            "trend_strength": self.trend_strength,
            "momentum": self.momentum,
        }


# Features used for regime detection
REGIME_FEATURES = [
    "adx",           # Trend strength
    "volatility_30", # Recent volatility
    "r_60m",         # Hourly momentum
    "bb_width",      # Bollinger Band width
    "sma20_sma50_ratio",  # Trend direction
]


class RegimeDetector:
    """
    Per-ticker regime detector using OHLCV features.

    Uses rule-based detection with ML-refined thresholds.
    """

    # Default thresholds (can be calibrated via fit())
    ADX_TREND_THRESHOLD = 25.0
    VOL_HIGH_PERCENTILE = 75.0
    MOMENTUM_THRESHOLD = 0.005  # 0.5%

    def __init__(self):
        self.fitted = False
        self.thresholds = {
            "adx_trend": self.ADX_TREND_THRESHOLD,
            "vol_high_pct": self.VOL_HIGH_PERCENTILE,
            "momentum_threshold": self.MOMENTUM_THRESHOLD,
        }

        # Historical volatility for percentile calculation
        self.vol_history: List[float] = []
        self.max_vol_history = 1000

        # Optional ML clustering
        self.use_ml = False
        self.kmeans = None
        self.scaler = None
        self.cluster_to_regime = None

    def detect(self, features: pd.Series) -> RegimeState:
        """
        Detect regime from feature row.

        Args:
            features: Series with REGIME_FEATURES columns

        Returns:
            RegimeState with detected regime
        """
        # Extract features
        adx = features.get("adx", 20.0)
        volatility = features.get("volatility_30", 0.01)
        momentum = features.get("r_60m", 0.0)
        bb_width = features.get("bb_width", 0.02)
        sma_ratio = features.get("sma20_sma50_ratio", 1.0)

        # Handle NaN
        if pd.isna(adx):
            adx = 20.0
        if pd.isna(volatility):
            volatility = 0.01
        if pd.isna(momentum):
            momentum = 0.0

        # Update volatility history
        self._update_vol_history(volatility)

        # Calculate volatility percentile
        vol_percentile = self._get_vol_percentile(volatility)

        # ML-based detection if fitted
        if self.use_ml and self.kmeans is not None:
            return self._detect_ml(features, vol_percentile)

        # Rule-based detection
        return self._detect_rules(
            adx=adx,
            volatility=volatility,
            momentum=momentum,
            sma_ratio=sma_ratio,
            vol_percentile=vol_percentile,
        )

    def _detect_rules(
        self,
        adx: float,
        volatility: float,
        momentum: float,
        sma_ratio: float,
        vol_percentile: float,
    ) -> RegimeState:
        """Rule-based regime detection."""
        is_trending = adx >= self.thresholds["adx_trend"]
        is_high_vol = vol_percentile >= self.thresholds["vol_high_pct"]
        is_up_trend = sma_ratio > 1.0 and momentum > 0
        is_down_trend = sma_ratio < 1.0 and momentum < 0

        # Classify
        if is_trending:
            if is_up_trend:
                regime = TickerRegime.TREND_UP
                confidence = min(1.0, adx / 50.0)  # Stronger ADX = more confident
            elif is_down_trend:
                regime = TickerRegime.TREND_DOWN
                confidence = min(1.0, adx / 50.0)
            else:
                # Trending but direction unclear
                regime = TickerRegime.RANGE_HIGH_VOL if is_high_vol else TickerRegime.RANGE_LOW_VOL
                confidence = 0.5
        else:
            if is_high_vol:
                regime = TickerRegime.RANGE_HIGH_VOL
                confidence = vol_percentile / 100.0
            else:
                regime = TickerRegime.RANGE_LOW_VOL
                confidence = 1.0 - (vol_percentile / 100.0)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility_percentile=vol_percentile,
            trend_strength=adx,
            momentum=momentum,
        )

    def _detect_ml(self, features: pd.Series, vol_percentile: float) -> RegimeState:
        """ML-based regime detection using clustering."""
        # Prepare feature vector
        X = np.array([[
            features.get(col, 0.0) for col in REGIME_FEATURES
        ]])

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        cluster = self.kmeans.predict(X_scaled)[0]

        # Map cluster to regime
        regime = self.cluster_to_regime.get(cluster, TickerRegime.UNKNOWN)

        # Calculate confidence from distance to centroid
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - X_scaled, axis=1)
        min_dist = distances[cluster]
        max_dist = np.max(distances)
        confidence = 1.0 - (min_dist / (max_dist + 1e-9))

        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility_percentile=vol_percentile,
            trend_strength=features.get("adx", 20.0),
            momentum=features.get("r_60m", 0.0),
        )

    def _update_vol_history(self, volatility: float) -> None:
        """Update rolling volatility history."""
        self.vol_history.append(volatility)
        if len(self.vol_history) > self.max_vol_history:
            self.vol_history.pop(0)

    def _get_vol_percentile(self, volatility: float) -> float:
        """Get volatility percentile vs history."""
        if len(self.vol_history) < 10:
            return 50.0

        return float(np.percentile(
            [v for v in self.vol_history if v < volatility],
            100, method='lower'
        ) if volatility > min(self.vol_history) else 0.0)

    def fit(
        self,
        df: pd.DataFrame,
        use_ml: bool = True,
    ) -> Dict:
        """
        Fit regime detector on historical data.

        Args:
            df: DataFrame with OHLCV features
            use_ml: Whether to use ML clustering

        Returns:
            Fitting metrics
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, using rule-based only")
            use_ml = False

        metrics = {}

        # Extract features
        feature_cols = [c for c in REGIME_FEATURES if c in df.columns]
        if len(feature_cols) < len(REGIME_FEATURES):
            logger.warning(f"Missing features: {set(REGIME_FEATURES) - set(feature_cols)}")

        X = df[feature_cols].dropna().values

        if len(X) < 100:
            logger.warning("Not enough data for fitting")
            return {"error": "insufficient data"}

        logger.info(f"Fitting regime detector on {len(X)} samples")

        # Calibrate volatility thresholds
        vol_col = "volatility_30" if "volatility_30" in feature_cols else None
        if vol_col:
            vol_idx = feature_cols.index(vol_col)
            vol_data = X[:, vol_idx]
            self.thresholds["vol_high_pct"] = float(np.percentile(vol_data, 75))
            metrics["vol_75th"] = self.thresholds["vol_high_pct"]

        # Calibrate ADX threshold
        adx_col = "adx" if "adx" in feature_cols else None
        if adx_col:
            adx_idx = feature_cols.index(adx_col)
            adx_data = X[:, adx_idx]
            # Threshold at median - below = ranging, above = trending
            self.thresholds["adx_trend"] = float(np.percentile(adx_data, 50))
            metrics["adx_median"] = self.thresholds["adx_trend"]

        # ML clustering
        if use_ml and HAS_SKLEARN:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            self.kmeans.fit(X_scaled)

            # Map clusters to regimes based on centroid characteristics
            self.cluster_to_regime = self._map_clusters_to_regimes(
                self.kmeans.cluster_centers_,
                feature_cols,
            )

            self.use_ml = True
            metrics["ml_fitted"] = True
            metrics["cluster_mapping"] = {
                k: v.value for k, v in self.cluster_to_regime.items()
            }

        self.fitted = True
        logger.info(f"Regime detector fitted: {metrics}")

        return metrics

    def _map_clusters_to_regimes(
        self,
        centroids: np.ndarray,
        feature_cols: List[str],
    ) -> Dict[int, TickerRegime]:
        """Map cluster centroids to regime labels based on feature values."""
        mapping = {}

        adx_idx = feature_cols.index("adx") if "adx" in feature_cols else None
        vol_idx = feature_cols.index("volatility_30") if "volatility_30" in feature_cols else None
        mom_idx = feature_cols.index("r_60m") if "r_60m" in feature_cols else None
        sma_idx = feature_cols.index("sma20_sma50_ratio") if "sma20_sma50_ratio" in feature_cols else None

        for i in range(4):
            c = centroids[i]

            adx_val = c[adx_idx] if adx_idx is not None else 20
            vol_val = c[vol_idx] if vol_idx is not None else 0
            mom_val = c[mom_idx] if mom_idx is not None else 0
            sma_val = c[sma_idx] if sma_idx is not None else 1

            # High ADX clusters are trending
            is_trending = adx_val > 0  # Scaled, so 0 is median

            if is_trending:
                if mom_val > 0 or sma_val > 0:
                    mapping[i] = TickerRegime.TREND_UP
                else:
                    mapping[i] = TickerRegime.TREND_DOWN
            else:
                if vol_val > 0:
                    mapping[i] = TickerRegime.RANGE_HIGH_VOL
                else:
                    mapping[i] = TickerRegime.RANGE_LOW_VOL

        return mapping

    def save(self, path: Path) -> None:
        """Save fitted detector."""
        joblib.dump({
            "thresholds": self.thresholds,
            "use_ml": self.use_ml,
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "cluster_to_regime": {
                k: v.value for k, v in (self.cluster_to_regime or {}).items()
            },
            "vol_history": self.vol_history[-100:],  # Keep last 100
        }, path)
        logger.info(f"Saved regime detector: {path}")

    def load(self, path: Path) -> None:
        """Load fitted detector."""
        data = joblib.load(path)
        self.thresholds = data["thresholds"]
        self.use_ml = data.get("use_ml", False)
        self.kmeans = data.get("kmeans")
        self.scaler = data.get("scaler")
        self.vol_history = data.get("vol_history", [])

        if data.get("cluster_to_regime"):
            self.cluster_to_regime = {
                int(k): TickerRegime(v) for k, v in data["cluster_to_regime"].items()
            }

        self.fitted = True
        logger.info(f"Loaded regime detector: {path}")


def filter_signal_by_regime(
    direction: str,
    regime: RegimeState,
    allow_counter_trend: bool = False,
    strict_mode: bool = False,
) -> Tuple[bool, str]:
    """
    Filter Alpha Model signal by regime.

    Args:
        direction: "LONG" or "SHORT" from Alpha Model
        regime: Current regime state
        allow_counter_trend: Allow signals against trend (for mean-reversion)
        strict_mode: If True, only allow trend-aligned signals (blocks range trades)

    Returns:
        (should_allow, reason)
    """
    direction_upper = direction.upper()
    r = regime.regime

    # TREND_UP: only LONG
    if r == TickerRegime.TREND_UP:
        if direction_upper == "LONG":
            return True, f"LONG aligned with TREND_UP (ADX={regime.trend_strength:.1f})"
        else:
            if allow_counter_trend and regime.confidence < 0.7:
                return True, "SHORT allowed: weak trend"
            return False, f"SHORT blocked in TREND_UP (ADX={regime.trend_strength:.1f})"

    # TREND_DOWN: only SHORT
    if r == TickerRegime.TREND_DOWN:
        if direction_upper == "SHORT":
            return True, f"SHORT aligned with TREND_DOWN (ADX={regime.trend_strength:.1f})"
        else:
            if allow_counter_trend and regime.confidence < 0.7:
                return True, "LONG allowed: weak trend"
            return False, f"LONG blocked in TREND_DOWN (ADX={regime.trend_strength:.1f})"

    # RANGE_LOW_VOL: both directions (mean-reversion territory)
    if r == TickerRegime.RANGE_LOW_VOL:
        if strict_mode:
            return False, "Range regime blocked in strict mode"
        return True, f"Range regime: {direction_upper} allowed"

    # RANGE_HIGH_VOL: cautious
    if r == TickerRegime.RANGE_HIGH_VOL:
        if regime.volatility_percentile > 90:
            return False, f"High volatility ({regime.volatility_percentile:.0f}th pct): skip"
        if strict_mode:
            return False, "Range high-vol blocked in strict mode"
        return True, f"Elevated volatility: {direction_upper} with caution"

    # UNKNOWN: allow with warning
    return True, f"Unknown regime: {direction_upper} allowed"


def filter_signal_by_regime_quality(
    direction: str,
    regime: RegimeState,
    regime_win_rates: Dict[str, float],
    min_wr_threshold: float = 0.40,
) -> Tuple[bool, str]:
    """
    Filter signal based on historical regime win rates.

    Args:
        direction: "LONG" or "SHORT" from Alpha Model
        regime: Current regime state
        regime_win_rates: Dict of regime_name -> historical win rate
        min_wr_threshold: Minimum acceptable win rate

    Returns:
        (should_allow, reason)
    """
    regime_name = regime.regime.value
    historical_wr = regime_win_rates.get(regime_name)

    if historical_wr is None:
        return True, f"No historical data for {regime_name}"

    if historical_wr < min_wr_threshold:
        return False, f"{regime_name} WR={historical_wr:.1%} below threshold {min_wr_threshold:.1%}"

    return True, f"{regime_name} WR={historical_wr:.1%} OK"


def get_regime_position_multiplier(regime: RegimeState) -> float:
    """
    Get position size multiplier based on regime.

    Returns:
        Multiplier (0.0 = no trading, 1.0 = full size)
    """
    multipliers = {
        TickerRegime.TREND_UP: 1.0,      # Full size with trend
        TickerRegime.TREND_DOWN: 1.0,    # Full size with trend
        TickerRegime.RANGE_LOW_VOL: 0.8, # Slightly reduced
        TickerRegime.RANGE_HIGH_VOL: 0.5,# Half size
        TickerRegime.UNKNOWN: 0.5,
    }

    base = multipliers.get(regime.regime, 0.5)

    # Adjust by confidence
    if regime.confidence < 0.5:
        base *= 0.7

    # Adjust by volatility
    if regime.volatility_percentile > 80:
        base *= 0.8

    return max(0.2, min(1.0, base))


# ============================================================================
# Training and Backtesting
# ============================================================================

def train_regime_detector(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Tuple[RegimeDetector, Dict]:
    """
    Train regime detector on historical data.

    Args:
        df: DataFrame with OHLCV features
        save_path: Where to save the detector

    Returns:
        (detector, metrics)
    """
    detector = RegimeDetector()
    metrics = detector.fit(df, use_ml=HAS_SKLEARN)

    if save_path:
        detector.save(save_path)

    return detector, metrics


def backtest_regime_filter(
    df: pd.DataFrame,
    detector: RegimeDetector,
    direction_col: str = "direction",
    label_col: str = "label",
) -> Dict:
    """
    Backtest regime filtering impact.

    Args:
        df: DataFrame with features, direction, and outcome labels
        detector: Fitted regime detector
        direction_col: Column with LONG/SHORT direction
        label_col: Column with 0/1 outcome (1=win)

    Returns:
        Comparison metrics
    """
    results_baseline = []
    results_filtered = []
    regime_counts = {}

    for idx, row in df.iterrows():
        if direction_col not in row or label_col not in row:
            continue

        direction = row[direction_col]
        label = row[label_col]

        if direction not in ["LONG", "SHORT"]:
            continue
        if pd.isna(label):
            continue

        # Detect regime
        regime = detector.detect(row)

        # Count regimes
        r_name = regime.regime.value
        regime_counts[r_name] = regime_counts.get(r_name, 0) + 1

        # Baseline: all trades
        results_baseline.append({"regime": r_name, "outcome": label})

        # Filtered: regime-aligned only
        allow, _ = filter_signal_by_regime(direction, regime)
        if allow:
            results_filtered.append({"regime": r_name, "outcome": label})

    # Calculate metrics
    baseline_wr = np.mean([r["outcome"] for r in results_baseline]) if results_baseline else 0
    filtered_wr = np.mean([r["outcome"] for r in results_filtered]) if results_filtered else 0

    # Per-regime win rates
    regime_wrs = {}
    for r_name in regime_counts:
        outcomes = [r["outcome"] for r in results_baseline if r["regime"] == r_name]
        if outcomes:
            regime_wrs[r_name] = np.mean(outcomes)

    return {
        "baseline_trades": len(results_baseline),
        "baseline_win_rate": float(baseline_wr),
        "filtered_trades": len(results_filtered),
        "filtered_win_rate": float(filtered_wr),
        "improvement": float(filtered_wr - baseline_wr),
        "filter_rate": float(len(results_filtered) / max(len(results_baseline), 1)),
        "regime_distribution": regime_counts,
        "regime_win_rates": regime_wrs,
    }
