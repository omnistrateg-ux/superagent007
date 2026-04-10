"""
ML Predictor for 1h Futures (77-feature CatBoost model)

Loads the trained model and provides prediction interface for paper_futures.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("ml_predictor")

# Try loading joblib
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    log.warning("joblib not available - ML predictor disabled")


# Top 19 features from 1h model (after noise removal)
SELECTED_FEATURES = [
    "cal_day_of_week",
    "atr_14",
    "cal_time_normalized",
    "volatility_30",
    "cal_session_phase",
    "cal_liquidity_expected",
    "cal_is_quarter_end",
    "macd",
    "cal_is_friday",
    "cal_is_month_end",
    "cal_tax_day",
    "cross_sector_momentum",
    "cross_rel_strength_1h",
    "cal_is_month_start",
    "price_sma50_ratio",
    "r_60m",
    "dist_vwap_atr",
    "macd_signal",
    "cross_beta",
]


class MLPredictor:
    """
    ML-based 1h price direction predictor.

    Uses CatBoost model trained on 77 features, filtered to top 19.
    """

    MODEL_PATH = Path("models/model_full_77feat_catboost_60m.joblib")

    def __init__(self):
        self.model = None
        self.features = None
        self.loaded = False
        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        if not HAS_JOBLIB:
            return

        if not self.MODEL_PATH.exists():
            log.warning(f"Model not found: {self.MODEL_PATH}")
            return

        try:
            data = joblib.load(self.MODEL_PATH)
            self.model = data["model"]
            # Validate features are present in model file
            if "features" not in data:
                log.warning(f"Model file missing 'features' key, using SELECTED_FEATURES ({len(SELECTED_FEATURES)})")
            self.features = data.get("features", SELECTED_FEATURES)
            # Sanity check: warn if feature count differs significantly
            if len(self.features) != len(SELECTED_FEATURES) and "features" not in data:
                log.warning(f"Feature count mismatch: model expects {len(self.features)}, fallback has {len(SELECTED_FEATURES)}")
            self.loaded = True
            log.info(f"ML model loaded: {len(self.features)} features, AUC={data.get('metrics', {}).get('auc', '?')}")
        except Exception as e:
            log.error(f"Failed to load model: {e}")

    def predict(
        self,
        ticker: str,
        candles: List[Dict],
        current_price: float,
        ema: float,
        timestamp: datetime,
    ) -> Tuple[float, str]:
        """
        Predict 1h price direction.

        Args:
            ticker: Futures ticker (BR, MX, RI, NG)
            candles: Recent 1h candles (at least 60 for feature calculation)
            current_price: Current price
            ema: Current EMA-20
            timestamp: Current timestamp

        Returns:
            (probability, direction)
            - probability: 0.0-1.0, where >0.5 = LONG predicted
            - direction: "LONG" or "SHORT" based on probability
        """
        if not self.loaded or self.model is None:
            return 0.5, "NEUTRAL"

        try:
            features = self._build_features(ticker, candles, current_price, ema, timestamp)
            if features is None:
                return 0.5, "NEUTRAL"

            # Get probability of price going UP
            proba = self.model.predict_proba([features])[0][1]
            direction = "LONG" if proba > 0.5 else "SHORT"

            return float(proba), direction

        except Exception as e:
            log.debug(f"Prediction error: {e}")
            return 0.5, "NEUTRAL"

    def _build_features(
        self,
        ticker: str,
        candles: List[Dict],
        current_price: float,
        ema: float,
        timestamp: datetime,
    ) -> Optional[List[float]]:
        """Build feature vector from candle data."""
        if len(candles) < 60:
            return None

        # Convert candles to DataFrame
        df = pd.DataFrame(candles)
        if "close" not in df.columns:
            # Handle list format [open, close, high, low, begin]
            df = pd.DataFrame(candles, columns=["open", "close", "high", "low", "begin"])

        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close

        features = {}

        # Calendar features
        ts = timestamp
        h, m = ts.hour, ts.minute
        time_mins = h * 60 + m
        dow = ts.weekday()
        day = ts.day
        month = ts.month

        features["cal_day_of_week"] = dow / 4.0
        features["cal_time_normalized"] = max(0, min(1, (time_mins - 600) / 520))
        features["cal_is_friday"] = float(dow == 4)
        features["cal_is_month_end"] = float(day >= 28)
        features["cal_is_month_start"] = float(day <= 3)
        features["cal_is_quarter_end"] = float(month in [3, 6, 9, 12] and day >= 25)
        features["cal_tax_day"] = (day - 19) / 6.0 if 20 <= day <= 25 else 0.0

        # Session phase
        def get_session_phase(t):
            if 420 <= t < 600: return 0.2
            if 600 <= t < 615: return 0.3
            if 615 <= t < 690: return 0.4
            if 690 <= t < 780: return 0.5
            if 780 <= t < 840: return 0.6
            if 845 <= t < 960: return 0.7
            if 960 <= t < 1120: return 0.8
            if 1145 <= t < 1430: return 1.0
            return 0.0

        features["cal_session_phase"] = get_session_phase(time_mins)
        is_evening = 1145 <= time_mins < 1430
        features["cal_liquidity_expected"] = 1.0 - 0.7 * float(is_evening) - 0.3 * features["cal_is_friday"]

        # Technical features
        # ATR
        if len(df) >= 14:
            tr_list = []
            for i in range(1, min(15, len(df))):
                h_i, l_i, pc = high.iloc[i], low.iloc[i], close.iloc[i-1]
                tr = max(h_i - l_i, abs(h_i - pc), abs(l_i - pc))
                tr_list.append(tr)
            features["atr_14"] = np.mean(tr_list) if tr_list else 0
        else:
            features["atr_14"] = 0

        # Volatility
        returns = close.pct_change().dropna()
        features["volatility_30"] = returns.tail(30).std() if len(returns) >= 30 else 0

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        features["macd"] = macd_line.iloc[-1] if len(macd_line) > 0 else 0
        features["macd_signal"] = signal_line.iloc[-1] if len(signal_line) > 0 else 0

        # Price ratios
        sma_50 = close.rolling(50).mean()
        features["price_sma50_ratio"] = current_price / (sma_50.iloc[-1] + 1e-9) if len(sma_50) >= 50 else 1.0

        # Returns
        features["r_60m"] = close.pct_change(60).iloc[-1] if len(close) >= 61 else 0

        # VWAP distance (simplified)
        features["dist_vwap_atr"] = (current_price - ema) / (features["atr_14"] + 1e-9) if features["atr_14"] > 0 else 0

        # Cross-asset features (simplified)
        features["cross_sector_momentum"] = returns.tail(60).sum() if len(returns) >= 60 else 0
        features["cross_rel_strength_1h"] = features["r_60m"]
        vol_short = returns.tail(20).std() if len(returns) >= 20 else 0
        vol_long = returns.tail(60).std() if len(returns) >= 60 else 0
        features["cross_beta"] = vol_short / (vol_long + 1e-9) if vol_long > 0 else 1.0

        # Build feature vector in correct order
        feature_vector = []
        for feat_name in self.features:
            val = features.get(feat_name, 0.0)
            # Replace NaN/Inf with 0.0 to prevent model errors
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = 0.0
            feature_vector.append(val)

        return feature_vector

    def should_trade(
        self,
        ticker: str,
        direction: str,
        candles: List[Dict],
        current_price: float,
        ema: float,
        timestamp: datetime,
        min_confidence: float = 0.55,
    ) -> Tuple[bool, float, str]:
        """
        Check if trade should be taken based on ML prediction.

        Args:
            ticker: Futures ticker
            direction: Intended direction ("LONG" or "SHORT")
            candles: Recent candles
            current_price: Current price
            ema: Current EMA
            timestamp: Current timestamp
            min_confidence: Minimum probability threshold (default 0.55)

        Returns:
            (should_trade, confidence, reason)
        """
        proba, ml_direction = self.predict(ticker, candles, current_price, ema, timestamp)

        # Calculate confidence (distance from 0.5)
        confidence = abs(proba - 0.5) * 2  # Scale to 0-1

        # Check if ML agrees with intended direction
        if direction == "LONG":
            agrees = proba > 0.5
            if agrees and proba >= min_confidence:
                return True, confidence, f"ML confirms LONG (p={proba:.2f})"
            elif not agrees and (1 - proba) >= min_confidence:
                return False, confidence, f"ML contra: predicts SHORT (p={proba:.2f})"
            else:
                return True, confidence * 0.5, f"ML neutral (p={proba:.2f})"
        else:  # SHORT
            agrees = proba < 0.5
            if agrees and (1 - proba) >= min_confidence:
                return True, confidence, f"ML confirms SHORT (p={proba:.2f})"
            elif not agrees and proba >= min_confidence:
                return False, confidence, f"ML contra: predicts LONG (p={proba:.2f})"
            else:
                return True, confidence * 0.5, f"ML neutral (p={proba:.2f})"


# Singleton
_predictor: Optional[MLPredictor] = None


def get_ml_predictor() -> MLPredictor:
    """Get or create global MLPredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
