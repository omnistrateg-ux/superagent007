"""
MOEX Agent v2.3 ML Predictor

Model registry with safe inference.
Supports single models (LightGBM, LogReg), Ensemble models, and MetaLabelingPipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from .features import FEATURE_COLS
from .horizon_resolver import (
    HorizonResolver,
    ResolutionStrategy,
    ResolverDecision,
    Direction,
    get_horizon_resolver,
)

logger = logging.getLogger("moex_agent.predictor")

# Re-export FEATURE_COLS for convenience
__all__ = ["FEATURE_COLS", "ModelRegistry", "safe_predict_proba", "get_registry"]


def safe_predict_proba(model: Any, X: np.ndarray) -> float:
    """
    Safely extract P(class=1) from sklearn model.

    Handles edge cases:
    - model.classes_ may be [0, 1], [1, 0], or single-class
    - predict_proba may return shape (n, 1) or (n, 2)

    Args:
        model: Fitted sklearn classifier
        X: Feature array of shape (1, n_features)

    Returns:
        Probability of positive class in [0, 1]
    """
    try:
        proba = model.predict_proba(X)
    except Exception as e:
        logger.warning(f"predict_proba failed: {e}")
        return 0.5

    if proba is None:
        return 0.5

    proba = np.asarray(proba)
    classes = getattr(model, "classes_", None)

    if proba.ndim == 1:
        return float(proba[0])

    if proba.ndim != 2:
        return float(proba.ravel()[0]) if proba.size > 0 else 0.5

    # Single-class model
    if proba.shape[1] == 1:
        if classes is not None and len(classes) == 1:
            if classes[0] == 1 or classes[0] is True:
                return float(proba[0, 0])
            else:
                return 1.0 - float(proba[0, 0])
        return 0.5

    # Standard 2-class model
    if classes is not None:
        classes_list = list(classes)
        if 1 in classes_list:
            idx = classes_list.index(1)
            return float(proba[0, idx])
        if True in classes_list:
            idx = classes_list.index(True)
            return float(proba[0, idx])

    return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])


class ModelRegistry:
    """
    Thread-safe registry for loaded ML models.

    Provides lazy loading, caching, and safe prediction interface.
    Isotonic calibration ensures max p ~ 0.60.
    """

    # Horizons to exclude (negative Sharpe or poor performance)
    # Also exclude special models that aren't trading horizon models
    EXCLUDED_HORIZONS = {"1w", "entry_timing", "regime_detector"}

    def __init__(self, models_dir: Path = Path("./models")):
        self.models_dir = Path(models_dir)
        self._models: Dict[str, Any] = {}
        self._meta: Optional[Dict[str, Dict]] = None
        self._loaded = False

    def load(self) -> None:
        """Load all models from models_dir."""
        meta_path = self.models_dir / "meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {meta_path}\n"
                "Run 'python -m moex_agent train' first."
            )

        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._models = {}

        for horizon, info in self._meta.items():
            # Skip excluded horizons
            if horizon in self.EXCLUDED_HORIZONS:
                logger.info(f"Skipping excluded horizon: {horizon}")
                continue

            model_path = Path(info["path"])
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                self._models[horizon] = joblib.load(model_path)
                logger.debug(f"Loaded model: {horizon}")
            except Exception as e:
                logger.error(f"Failed to load model {horizon}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._models)} models: {list(self._models.keys())}")

    def ensure_loaded(self) -> None:
        """Load models if not already loaded."""
        if not self._loaded:
            self.load()

    @property
    def horizons(self) -> List[str]:
        """Get list of available horizons."""
        self.ensure_loaded()
        return list(self._models.keys())

    def predict(self, horizon: str, X: np.ndarray) -> float:
        """
        Predict P(success) for a given horizon.

        Args:
            horizon: Model horizon name (e.g., '5m', '1h')
            X: Feature array of shape (1, n_features) or (n_features,)

        Returns:
            Probability in [0, 1]
        """
        self.ensure_loaded()

        if horizon not in self._models:
            raise KeyError(f"Model for '{horizon}' not found. Available: {list(self._models.keys())}")

        model_data = self._models[horizon]

        # Handle new model format (dict with model, scaler, feature_indices)
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            scaler = model_data.get("scaler")  # Can be None for LightGBM
            feature_indices = model_data.get("feature_indices")

            # Ensure X is 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Select features if indices provided
            if feature_indices is not None:
                X_selected = X[:, feature_indices]
            else:
                X_selected = X

            # Check model type
            is_ensemble = hasattr(model, "models") and hasattr(model, "weights")
            is_meta_labeling = hasattr(model, "primary_model") and hasattr(model, "meta_model")

            if is_meta_labeling:
                # MetaLabelingPipeline handles everything internally
                return safe_predict_proba(model, X_selected)
            elif is_ensemble:
                # EnsembleClassifier handles scaling internally
                return safe_predict_proba(model, X_selected)
            elif scaler is not None:
                # Single model with scaler (LogReg)
                X_pred = scaler.transform(X_selected)
                return safe_predict_proba(model, X_pred)
            else:
                # Single model without scaler (LightGBM, CatBoost)
                return safe_predict_proba(model, X_selected)
        else:
            # Old model format (direct sklearn model)
            return safe_predict_proba(model_data, X)

    def predict_all(self, X: np.ndarray) -> Dict[str, float]:
        """Predict P(success) for all horizons."""
        self.ensure_loaded()
        return {h: self.predict(h, X) for h in self._models.keys()}

    def best_horizon(self, X: np.ndarray) -> Tuple[Optional[str], float]:
        """Find horizon with highest P(success)."""
        preds = self.predict_all(X)
        if not preds:
            return None, 0.0

        best_h = max(preds, key=preds.get)
        return best_h, preds[best_h]

    def resolve_horizons(
        self,
        X: np.ndarray,
        strategy: ResolutionStrategy = ResolutionStrategy.WEIGHTED_VOTE,
    ) -> Tuple[ResolverDecision, Dict[str, float]]:
        """
        Resolve multi-horizon predictions using HorizonResolver.

        Args:
            X: Feature vector
            strategy: Resolution strategy to use

        Returns:
            (ResolverDecision, all_predictions)
        """
        preds = self.predict_all(X)
        if not preds:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="no predictions available",
                horizons_agree=True,
                contributing_horizons=[],
                conflicts=[],
            ), {}

        resolver = get_horizon_resolver(strategy)
        decision = resolver.resolve(preds)

        return decision, preds


_registry: Optional[ModelRegistry] = None


def get_registry(models_dir: Path = Path("./models")) -> ModelRegistry:
    """Get or create global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(models_dir)
    return _registry


def reset_registry() -> None:
    """Reset global registry (for testing)."""
    global _registry
    _registry = None
