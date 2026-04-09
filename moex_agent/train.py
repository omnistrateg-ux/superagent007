"""
MOEX Agent v2.5 Model Training

Single CatBoost classifier (Phase 1 simplification).
Purged Walk-forward validation with embargo to prevent information leakage.

v2.5 Changes (Phase 1 - Simplification):
- DEFAULT: Single CatBoost instead of Ensemble
- Reason: CatBoost showed +11% edge on 10m vs +6% for Ensemble
- Simpler model = less overfitting risk
- Ensemble still available via use_ensemble=True
- Meta-labeling disabled by default (use_meta_labeling=False)

v2.4 Changes (Phase 0 - Fix Leakage):
- CRITICAL: PurgedTimeSeriesSplit with embargo between train/test
- Embargo = horizon_minutes to prevent overlapping target windows
- This is essential for honest OOS estimation in time series

v2.3 Changes:
- Added MetaLabelingPipeline: model1(direction) + model2(take/skip)
- Meta-model filters out low-quality signals from primary model
- Significantly improves precision at cost of fewer signals

v2.2 Changes:
- Added CatBoost as second gradient boosting model
- Ensemble classifier combining LightGBM + CatBoost + LogReg
- Soft voting with configurable weights
- Each model contributes to final probability

v2.1 Changes:
- Replaced LogisticRegression L1 with LightGBM
- Using native feature_importances_ instead of permutation importance
- No feature selection (LightGBM handles this internally)
- All 35 features used (30 TA + 5 anomaly)
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import load_config
from .features import FEATURE_COLS, build_feature_frame
from .labels import make_trend_following_labels, make_atr_trend_labels
from .storage import connect

# Try to import ML libraries (catch OSError for missing native libs like libomp)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except (ImportError, OSError):
    CatBoostClassifier = None
    HAS_CATBOOST = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.train")


class PurgedTimeSeriesSplit:
    """
    Purged Walk-Forward Cross-Validation with embargo.

    Unlike standard TimeSeriesSplit, this adds an embargo period between
    train and test sets to prevent information leakage through overlapping
    target windows.

    For example, with embargo=60 (1 hour on 1m bars):
    - If test starts at index 1000, train ends at index 939 (1000 - 60 - 1)
    - This ensures no label in train set uses future data that overlaps with test

    Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
    """

    def __init__(self, n_splits: int = 5, embargo: int = 60):
        """
        Args:
            n_splits: Number of splits (folds)
            embargo: Number of bars to skip between train and test
                    Should be >= max prediction horizon
        """
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Yields:
            (train_indices, test_indices) for each fold
        """
        n_samples = len(X)
        # Minimum test size
        test_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Test set boundaries
            test_start = (fold + 1) * test_size
            test_end = (fold + 2) * test_size if fold < self.n_splits - 1 else n_samples

            # Train set ends embargo bars before test starts
            train_end = test_start - self.embargo

            if train_end <= 0:
                # Not enough data for this fold with embargo
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def get_lgbm_params() -> Dict:
    """
    LightGBM hyperparameters optimized for trading signals.

    Conservative settings to avoid overfitting:
    - max_depth=6: prevents overly complex trees
    - min_child_samples=50: ensures stable leaf predictions
    - subsample/colsample: adds regularization
    - learning_rate=0.05: slow learning for stability
    """
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_catboost_params() -> Dict:
    """
    CatBoost hyperparameters optimized for trading signals.

    CatBoost advantages:
    - Better handling of categorical features (if any)
    - Ordered boosting reduces overfitting
    - Built-in regularization via random_strength
    """
    return {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.8,
        "border_count": 128,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
    }


class EnsembleClassifier:
    """
    Ensemble classifier combining LightGBM + CatBoost + LogReg.

    Uses soft voting: final probability = weighted average of model probabilities.

    Default weights: LightGBM=0.4, CatBoost=0.4, LogReg=0.2
    LogReg acts as a regularizer - simpler model that prevents overfitting.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_lightgbm: bool = True,
        use_catboost: bool = True,
        use_logreg: bool = True,
    ):
        self.weights = weights or {"lgbm": 0.4, "catboost": 0.4, "logreg": 0.2}
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.use_catboost = use_catboost and HAS_CATBOOST
        self.use_logreg = use_logreg

        self.models: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None
        self.fitted = False

        # Normalize weights based on available models
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0 based on available models."""
        active_weights = {}
        if self.use_lightgbm:
            active_weights["lgbm"] = self.weights.get("lgbm", 0.4)
        if self.use_catboost:
            active_weights["catboost"] = self.weights.get("catboost", 0.4)
        if self.use_logreg:
            active_weights["logreg"] = self.weights.get("logreg", 0.2)

        total = sum(active_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in active_weights.items()}
        else:
            # Fallback: LogReg only
            self.weights = {"logreg": 1.0}
            self.use_logreg = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """
        Fit all ensemble models.

        Args:
            X: Feature matrix
            y: Binary labels (0/1)

        Returns:
            self
        """
        # Scaler for LogReg
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train LightGBM
        if self.use_lightgbm:
            logger.info("    Training LightGBM...")
            lgbm = LGBMClassifier(**get_lgbm_params())
            lgbm.fit(X, y)  # LightGBM doesn't need scaling
            self.models["lgbm"] = lgbm

        # Train CatBoost
        if self.use_catboost:
            logger.info("    Training CatBoost...")
            cb = CatBoostClassifier(**get_catboost_params())
            cb.fit(X, y)  # CatBoost doesn't need scaling
            self.models["catboost"] = cb

        # Train LogReg (always train as baseline/regularizer)
        if self.use_logreg:
            logger.info("    Training LogReg...")
            logreg = LogisticRegression(
                penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=42
            )
            logreg.fit(X_scaled, y)
            self.models["logreg"] = logreg

        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using soft voting.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]

        # Weighted average of probabilities
        proba = np.zeros((n_samples, 2))

        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            if weight == 0:
                continue

            if name == "logreg":
                model_proba = model.predict_proba(X_scaled)
            else:
                model_proba = model.predict_proba(X)

            proba += weight * model_proba

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using soft voting."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Combined feature importances from all models.

        Weighted average of importances from each model.
        """
        if not self.fitted:
            return np.array([])

        n_features = None
        importances = None

        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            if weight == 0:
                continue

            if name == "lgbm":
                imp = model.feature_importances_
            elif name == "catboost":
                imp = model.feature_importances_
            elif name == "logreg":
                imp = np.abs(model.coef_[0])
            else:
                continue

            # Normalize to [0, 1] range
            if imp.max() > 0:
                imp = imp / imp.max()

            if n_features is None:
                n_features = len(imp)
                importances = np.zeros(n_features)

            importances += weight * imp

        return importances if importances is not None else np.array([])

    def get_model_info(self) -> Dict:
        """Get information about ensemble composition."""
        return {
            "models": list(self.models.keys()),
            "weights": self.weights,
            "n_models": len(self.models),
        }


class MetaLabelingPipeline:
    """
    Meta-labeling pipeline: model1(direction) + model2(take/skip).

    How it works:
    1. Primary model predicts direction (LONG probability)
    2. Meta-model predicts whether primary model's prediction will be correct
    3. Final signal = primary_signal if meta_model_confidence > threshold

    This significantly improves precision by filtering out low-quality signals.

    Meta-label creation:
    - For each sample where primary model predicts LONG (p > 0.5):
      - meta_label = 1 if actual outcome was LONG win
      - meta_label = 0 if actual outcome was loss or neutral
    """

    def __init__(
        self,
        primary_threshold: float = 0.55,
        meta_threshold: float = 0.50,
        use_ensemble: bool = True,
    ):
        """
        Initialize meta-labeling pipeline.

        Args:
            primary_threshold: Threshold for primary model to generate signal
            meta_threshold: Threshold for meta-model to approve signal
            use_ensemble: Use EnsembleClassifier for both models
        """
        self.primary_threshold = primary_threshold
        self.meta_threshold = meta_threshold
        self.use_ensemble = use_ensemble

        self.primary_model: Optional[object] = None
        self.meta_model: Optional[object] = None
        self.primary_scaler: Optional[StandardScaler] = None
        self.meta_scaler: Optional[StandardScaler] = None
        self.fitted = False

    def _create_model(self):
        """Create a model instance based on configuration."""
        if self.use_ensemble:
            return EnsembleClassifier()
        elif HAS_LIGHTGBM:
            return LGBMClassifier(**get_lgbm_params())
        else:
            return LogisticRegression(
                penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=42
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_binary: Optional[np.ndarray] = None,
    ) -> "MetaLabelingPipeline":
        """
        Fit meta-labeling pipeline.

        Args:
            X: Feature matrix
            y: Original labels (-1, 0, 1)
            y_binary: Binary labels (0/1) for primary model. If None, computed from y.

        Returns:
            self
        """
        if y_binary is None:
            y_binary = (y == 1).astype(int)

        logger.info("  [Meta-Labeling] Step 1: Training primary model...")

        # Step 1: Train primary model on binary labels
        self.primary_model = self._create_model()
        if self.use_ensemble or HAS_LIGHTGBM or HAS_CATBOOST:
            self.primary_model.fit(X, y_binary)
            primary_proba = self.primary_model.predict_proba(X)[:, 1]
        else:
            self.primary_scaler = StandardScaler()
            X_scaled = self.primary_scaler.fit_transform(X)
            self.primary_model.fit(X_scaled, y_binary)
            primary_proba = self.primary_model.predict_proba(X_scaled)[:, 1]

        # Step 2: Create meta-labels
        logger.info("  [Meta-Labeling] Step 2: Creating meta-labels...")

        # Samples where primary model predicts LONG
        primary_signals = primary_proba >= self.primary_threshold

        # Meta-label: 1 if primary model was correct (predicted LONG and actual was LONG)
        # 0 if primary model was wrong (predicted LONG but actual was not LONG)
        meta_labels = np.zeros(len(y), dtype=int)
        meta_labels[primary_signals & (y == 1)] = 1  # Correct LONG prediction

        # Only train meta-model on samples where primary model generated signal
        signal_mask = primary_signals
        n_signals = signal_mask.sum()
        n_correct = meta_labels[signal_mask].sum()

        logger.info(f"    Primary signals: {n_signals:,} ({n_signals/len(y)*100:.1f}%)")
        logger.info(f"    Correct signals: {n_correct:,} ({n_correct/n_signals*100:.1f}% of signals)")

        if n_signals < 100:
            logger.warning("    Too few signals for meta-model training, using primary model only")
            self.meta_model = None
            self.fitted = True
            return self

        # Step 3: Train meta-model on signal samples
        logger.info("  [Meta-Labeling] Step 3: Training meta-model...")

        X_meta = X[signal_mask]
        y_meta = meta_labels[signal_mask]

        # Add primary model probability as feature for meta-model
        X_meta_extended = np.column_stack([X_meta, primary_proba[signal_mask]])

        self.meta_model = self._create_model()
        if self.use_ensemble or HAS_LIGHTGBM or HAS_CATBOOST:
            self.meta_model.fit(X_meta_extended, y_meta)
        else:
            self.meta_scaler = StandardScaler()
            X_meta_scaled = self.meta_scaler.fit_transform(X_meta_extended)
            self.meta_model.fit(X_meta_scaled, y_meta)

        self.fitted = True
        logger.info("  [Meta-Labeling] Pipeline fitted successfully")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability using meta-labeling pipeline.

        Returns combined probability: P(direction) * P(meta_approve).

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        n_samples = X.shape[0]

        # Step 1: Get primary model predictions
        if self.primary_scaler is not None:
            X_primary = self.primary_scaler.transform(X)
        else:
            X_primary = X

        primary_proba = self.primary_model.predict_proba(X_primary)[:, 1]

        # If no meta-model, return primary predictions
        if self.meta_model is None:
            proba = np.zeros((n_samples, 2))
            proba[:, 1] = primary_proba
            proba[:, 0] = 1 - primary_proba
            return proba

        # Step 2: For samples with primary signal, get meta-model approval
        X_meta_extended = np.column_stack([X, primary_proba])

        if self.meta_scaler is not None:
            X_meta_scaled = self.meta_scaler.transform(X_meta_extended)
        else:
            X_meta_scaled = X_meta_extended

        meta_proba = self.meta_model.predict_proba(X_meta_scaled)[:, 1]

        # Combined probability: primary * meta
        # This effectively filters out signals where meta-model is not confident
        combined_proba = primary_proba * meta_proba

        proba = np.zeros((n_samples, 2))
        proba[:, 1] = combined_proba
        proba[:, 0] = 1 - combined_proba

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using meta-labeling pipeline."""
        proba = self.predict_proba(X)
        # Use combined threshold (primary * meta)
        threshold = self.primary_threshold * self.meta_threshold
        return (proba[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Combined feature importances (primary model only, meta adds 1 extra feature)."""
        if not self.fitted or self.primary_model is None:
            return np.array([])

        if hasattr(self.primary_model, "feature_importances_"):
            return self.primary_model.feature_importances_
        elif hasattr(self.primary_model, "coef_"):
            return np.abs(self.primary_model.coef_[0])

        return np.array([])

    def get_model_info(self) -> Dict:
        """Get information about meta-labeling pipeline."""
        return {
            "type": "meta_labeling",
            "primary_threshold": self.primary_threshold,
            "meta_threshold": self.meta_threshold,
            "has_meta_model": self.meta_model is not None,
            "use_ensemble": self.use_ensemble,
        }


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    take_pct: float = 1.5,
    stop_pct: float = 0.75,
    p_threshold: float = 0.55,
    use_ensemble: bool = False,
    use_meta_labeling: bool = False,
    embargo: int = 60,
) -> Dict:
    """
    Purged Walk-forward validation for trend-following strategy.

    Uses PurgedTimeSeriesSplit with embargo to prevent information leakage
    through overlapping target windows.

    Labels: 1=LONG win, -1=SHORT win, 0=no signal

    Args:
        X: Feature matrix
        y: Labels (-1, 0, 1)
        n_splits: Number of folds
        take_pct: Take profit percentage
        stop_pct: Stop loss percentage
        p_threshold: Probability threshold for taking trade
        use_ensemble: If True, use EnsembleClassifier instead of single model
        use_meta_labeling: If True, use MetaLabelingPipeline
        embargo: Bars to skip between train and test (default 60 = 1 hour on 1m)

    Returns:
        Dict with validation metrics
    """
    # Use purged CV with embargo to prevent leakage
    tscv = PurgedTimeSeriesSplit(n_splits=n_splits, embargo=embargo)

    results = {
        "trades": [],
        "pnl": [],
        "win_rates": [],
        "profit_factors": [],
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Convert to binary: 1=LONG, 0=everything else
        y_train_binary = (y_train == 1).astype(int)

        # Train model
        if use_meta_labeling:
            model = MetaLabelingPipeline(
                primary_threshold=p_threshold,
                use_ensemble=use_ensemble,
            )
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
        elif use_ensemble:
            model = EnsembleClassifier()
            model.fit(X_train, y_train_binary)
            y_proba = model.predict_proba(X_test)[:, 1]
        elif HAS_CATBOOST:
            # Phase 1: CatBoost is preferred single model
            model = CatBoostClassifier(**get_catboost_params())
            model.fit(X_train, y_train_binary, verbose=False)
            y_proba = model.predict_proba(X_test)[:, 1]
        elif HAS_LIGHTGBM:
            model = LGBMClassifier(**get_lgbm_params())
            model.fit(X_train, y_train_binary)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = LogisticRegression(
                penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=42
            )
            model.fit(X_train_scaled, y_train_binary)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # For meta-labeling, use combined threshold
        if use_meta_labeling:
            effective_threshold = p_threshold * 0.5  # Combined threshold
        else:
            effective_threshold = p_threshold

        fold_trades = []
        for i in range(len(y_proba)):
            if y_proba[i] < effective_threshold:
                continue

            # Check actual outcome
            if y_test[i] == 1:  # LONG won
                pnl_pct = take_pct
            elif y_test[i] == -1:  # SHORT won (we predicted LONG, so loss)
                pnl_pct = -stop_pct
            else:  # No clear outcome
                pnl_pct = -stop_pct * 0.5  # Partial loss

            fold_trades.append(pnl_pct)

        if fold_trades:
            wins = [t for t in fold_trades if t > 0]
            losses = [t for t in fold_trades if t <= 0]

            win_rate = len(wins) / len(fold_trades) * 100
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

            results["trades"].append(len(fold_trades))
            results["pnl"].append(sum(fold_trades))
            results["win_rates"].append(win_rate)
            results["profit_factors"].append(profit_factor)

    if results["win_rates"]:
        return {
            "avg_trades": np.mean(results["trades"]),
            "avg_pnl": np.mean(results["pnl"]),
            "avg_win_rate": np.mean(results["win_rates"]),
            "avg_profit_factor": np.mean(results["profit_factors"]),
            "std_win_rate": np.std(results["win_rates"]),
            "pnl": results["pnl"],
        }

    return {"avg_win_rate": 0, "avg_profit_factor": 0, "pnl": []}


def train_horizon_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    horizon: str,
    horizon_minutes: int = 60,
    use_ensemble: bool = False,
    use_meta_labeling: bool = False,
) -> Tuple[object, Dict, List[str]]:
    """
    Train model for one horizon.

    Args:
        X: Feature matrix (all features)
        y: Labels (-1, 0, 1)
        feature_names: List of feature names
        horizon: Horizon name (e.g., '5m', '1h')
        horizon_minutes: Horizon in minutes (used for embargo calculation)
        use_ensemble: If True, train EnsembleClassifier (LightGBM + CatBoost + LogReg)
        use_meta_labeling: If True, use MetaLabelingPipeline (model1+model2)

    Returns:
        (model_package, metrics, top_features)
    """
    logger.info(f"Training model for {horizon} (embargo={horizon_minutes} bars)...")

    # Convert to binary: 1=LONG, 0=everything else
    y_binary = (y == 1).astype(int)
    n_long = y_binary.sum()
    n_short = (y == -1).sum()
    n_neutral = (y == 0).sum()

    logger.info(f"  Data: {len(X):,} rows")
    logger.info(f"  Labels: LONG={n_long:,} ({n_long/len(y)*100:.1f}%), "
                f"SHORT={n_short:,} ({n_short/len(y)*100:.1f}%), "
                f"NEUTRAL={n_neutral:,} ({n_neutral/len(y)*100:.1f}%)")

    # Walk-forward validation with purged embargo
    logger.info(f"  Purged walk-forward validation (embargo={horizon_minutes} bars)...")
    wf_results = walk_forward_validation(
        X, y, n_splits=5,
        use_ensemble=use_ensemble,
        use_meta_labeling=use_meta_labeling,
        embargo=horizon_minutes,  # Embargo = horizon to prevent label leakage
    )

    wf_wr = wf_results.get("avg_win_rate", 0)
    wf_pf = wf_results.get("avg_profit_factor", 0)
    logger.info(f"  WF Results: WR={wf_wr:.1f}%, PF={wf_pf:.2f}")

    # Final model training on all data
    if use_meta_labeling:
        logger.info("  Training MetaLabelingPipeline...")
        pipeline = MetaLabelingPipeline(use_ensemble=use_ensemble)
        pipeline.fit(X, y)

        # Feature importance from primary model
        importance = pipeline.feature_importances_
        importance_pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [name for name, _ in importance_pairs[:10]]

        logger.info("  Top 10 features by importance:")
        for name, imp in importance_pairs[:10]:
            logger.info(f"    {name}: {imp:.3f}")

        model_info = pipeline.get_model_info()
        logger.info(f"  Meta-labeling: {model_info}")

        model_package = {
            "model": pipeline,
            "scaler": pipeline.primary_scaler,
            "feature_indices": list(range(len(feature_names))),
            "feature_names": feature_names,
            "feature_importances": dict(importance_pairs),
            "meta_labeling_info": model_info,
        }

        # Predictions stats
        y_proba = pipeline.predict_proba(X)[:, 1]
        model_type = "meta_labeling"

    elif use_ensemble:
        logger.info("  Training Ensemble (LightGBM + CatBoost + LogReg)...")
        ensemble = EnsembleClassifier()
        ensemble.fit(X, y_binary)

        # Combined feature importance
        importance = ensemble.feature_importances_
        importance_pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [name for name, _ in importance_pairs[:10]]

        logger.info("  Top 10 features by ensemble importance:")
        for name, imp in importance_pairs[:10]:
            logger.info(f"    {name}: {imp:.3f}")

        model_info = ensemble.get_model_info()
        logger.info(f"  Ensemble: {model_info['models']} with weights {model_info['weights']}")

        model_package = {
            "model": ensemble,
            "scaler": ensemble.scaler,  # Already fitted inside ensemble
            "feature_indices": list(range(len(feature_names))),
            "feature_names": feature_names,
            "feature_importances": dict(importance_pairs),
            "ensemble_info": model_info,
        }

        # Predictions stats
        y_proba = ensemble.predict_proba(X)[:, 1]
        model_type = "ensemble"

    elif HAS_CATBOOST:
        # Phase 1: CatBoost is preferred single model
        logger.info("  Training CatBoost model (Phase 1 default)...")
        model = CatBoostClassifier(**get_catboost_params())
        model.fit(X, y_binary, verbose=False)

        # Feature importance
        importance = model.feature_importances_
        importance_pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [name for name, _ in importance_pairs[:10]]

        logger.info("  Top 10 features by importance:")
        for name, imp in importance_pairs[:10]:
            logger.info(f"    {name}: {imp:.1f}")

        model_package = {
            "model": model,
            "scaler": None,  # CatBoost doesn't need scaling
            "feature_indices": list(range(len(feature_names))),
            "feature_names": feature_names,
            "feature_importances": dict(importance_pairs),
        }

        y_proba = model.predict_proba(X)[:, 1]
        model_type = "catboost"

    elif HAS_LIGHTGBM:
        logger.info("  Training LightGBM model...")
        model = LGBMClassifier(**get_lgbm_params())
        model.fit(X, y_binary)

        # Feature importance
        importance = model.feature_importances_
        importance_pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [name for name, _ in importance_pairs[:10]]

        logger.info("  Top 10 features by importance:")
        for name, imp in importance_pairs[:10]:
            logger.info(f"    {name}: {imp:.1f}")

        model_package = {
            "model": model,
            "scaler": None,  # LightGBM doesn't need scaling
            "feature_indices": list(range(len(feature_names))),
            "feature_names": feature_names,
            "feature_importances": dict(importance_pairs),
        }

        y_proba = model.predict_proba(X)[:, 1]
        model_type = "lightgbm"

    else:
        logger.info("  Training LogisticRegression (CatBoost/LightGBM not available)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=42
        )
        model.fit(X_scaled, y_binary)

        # Coefficient-based importance
        importance = np.abs(model.coef_[0])
        importance_pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [name for name, _ in importance_pairs[:10]]

        logger.info("  Top 10 features by coefficient:")
        for name, imp in importance_pairs[:10]:
            logger.info(f"    {name}: {imp:.4f}")

        model_package = {
            "model": model,
            "scaler": scaler,
            "feature_indices": list(range(len(feature_names))),
            "feature_names": feature_names,
            "feature_importances": dict(importance_pairs),
        }

        y_proba = model.predict_proba(X_scaled)[:, 1]
        model_type = "logreg"

    max_p = y_proba.max()
    mean_p = y_proba.mean()
    p90 = np.percentile(y_proba, 90)

    logger.info(f"  Probability stats: max={max_p:.2f}, mean={mean_p:.2f}, p90={p90:.2f}")

    wf_pnl_list = wf_results.get("pnl", [])
    if wf_pnl_list and len(wf_pnl_list) > 1:
        sharpe = np.mean(wf_pnl_list) / (np.std(wf_pnl_list) + 1e-9)
    else:
        sharpe = 0

    metrics = {
        "win_rate": wf_wr,
        "profit_factor": wf_pf,
        "sharpe": sharpe,
        "max_probability": float(max_p),
        "mean_probability": float(mean_p),
        "p90_probability": float(p90),
        "total_trades": int(wf_results.get("avg_trades", 0) * 5),
        "model_type": model_type,
    }

    logger.info(f"  Final: WR={wf_wr:.1f}%, PF={wf_pf:.2f}, Sharpe={sharpe:.2f}")

    return model_package, metrics, top_features


def main(
    train_days: Optional[int] = None,
    use_atr_labels: bool = True,
    use_ensemble: bool = False,  # Phase 1: Single CatBoost by default
    use_meta_labeling: bool = False,
):
    """
    Main training function.

    Args:
        train_days: If specified, only use first N days of data for training.
                   This enables walk-forward testing (train on past, test on future).
        use_atr_labels: If True (default), use ATR-based labels instead of fixed %.
        use_ensemble: If False (default after Phase 1), use single CatBoost.
                     Single model showed better edge on 10m (+11% vs +6% for ensemble).
        use_meta_labeling: If True, use MetaLabelingPipeline (direction + take/skip).
                          Disabled by default - enable only after proven edge.
    """
    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    # Log model configuration
    if use_meta_labeling:
        logger.info("Using MetaLabelingPipeline (direction + take/skip filter)")
        if use_ensemble:
            logger.info("  Primary & Meta models: Ensemble")
        elif HAS_LIGHTGBM:
            logger.info("  Primary & Meta models: LightGBM")
        else:
            logger.info("  Primary & Meta models: LogReg")
    elif use_ensemble:
        models_available = []
        if HAS_LIGHTGBM:
            models_available.append("LightGBM")
        if HAS_CATBOOST:
            models_available.append("CatBoost")
        models_available.append("LogReg")
        logger.info(f"Using Ensemble: {' + '.join(models_available)}")
    elif HAS_CATBOOST:
        logger.info("Using single CatBoost (Phase 1 default)")
    elif HAS_LIGHTGBM:
        logger.info("Using LightGBM for training")
    else:
        logger.info("CatBoost/LightGBM not available, using LogisticRegression")

    logger.info("Loading candles...")
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = 1
    ORDER BY secid, ts
    """
    candles = pd.read_sql_query(q, conn)
    logger.info(f"Loaded {len(candles):,} candles")

    # Walk-forward: only use first N days for training
    if train_days:
        candles["ts"] = pd.to_datetime(candles["ts"])
        min_date = candles["ts"].min()

        cutoff_date = min_date + pd.Timedelta(days=train_days)
        candles = candles[candles["ts"] < cutoff_date]
        logger.info(f"Walk-forward: using first {train_days} days (cutoff: {cutoff_date.date()})")
        logger.info(f"Training candles: {len(candles):,}")

    if len(candles) < 100_000:
        logger.warning("Low data volume! Recommend at least 100K candles.")

    # Build features (now includes anomaly features)
    logger.info(f"Building features ({len(FEATURE_COLS)} indicators)...")
    feats = build_feature_frame(candles, include_anomaly=True)

    # Create labels
    horizons = [(h.name, h.minutes) for h in cfg.horizons]

    if use_atr_labels:
        logger.info("Creating ATR-based labels (R:R = 2:1, take=2*ATR, stop=1*ATR)...")
        labels = make_atr_trend_labels(
            candles,
            horizons=horizons,
            take_atr_mult=2.0,
            stop_atr_mult=1.0,
            atr_period=14,
            fee_bps=cfg.fee_bps,
        )
        label_prefix = "y_atr_"
    else:
        logger.info("Creating fixed %-based labels (R:R = 2:1, take=1.5%, stop=0.75%)...")
        labels = make_trend_following_labels(
            candles,
            horizons=horizons,
            take_pct=1.5,
            stop_pct=0.75,
            fee_bps=cfg.fee_bps,
        )
        label_prefix = "y_trend_"

    # Merge
    df = feats.merge(labels, on=["secid", "ts"], how="inner")

    # Check features
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing features: {missing_cols}")
        return

    df = df.dropna(subset=FEATURE_COLS)
    logger.info(f"Training data: {len(df):,} rows")

    # Prepare arrays
    X = df[FEATURE_COLS].to_numpy(dtype=float)

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    meta = {}

    # Train for each horizon
    for name, minutes in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizon: {name} ({minutes} min)")
        logger.info(f"{'='*60}")

        ycol = f"{label_prefix}{name}"
        if ycol not in df.columns:
            logger.warning(f"Label {ycol} not found")
            continue

        y = df[ycol].to_numpy(dtype=int)

        model_package, metrics, top_features = train_horizon_model(
            X, y, FEATURE_COLS, horizon=name, horizon_minutes=minutes,
            use_ensemble=use_ensemble, use_meta_labeling=use_meta_labeling
        )

        # Model filename based on type
        if use_meta_labeling:
            model_path = models_dir / f"model_meta_{name}.joblib"
            model_type_str = "meta_labeling-v2.3"
        elif use_ensemble:
            model_path = models_dir / f"model_ensemble_{name}.joblib"
            model_type_str = "ensemble-v2.2"
        elif HAS_CATBOOST:
            model_path = models_dir / f"model_catboost_{name}.joblib"
            model_type_str = "catboost-v2.5"
        elif HAS_LIGHTGBM:
            model_path = models_dir / f"model_lgbm_{name}.joblib"
            model_type_str = "lgbm-v2.1"
        else:
            model_path = models_dir / f"model_logreg_{name}.joblib"
            model_type_str = "logreg-v2.1"

        joblib.dump(model_package, model_path)

        meta[name] = {
            "type": model_type_str,
            "path": str(model_path),
            "features": FEATURE_COLS,
            "top_features": top_features,
            "metrics": metrics,
            "trained_at": datetime.now().isoformat(),
            "data_size": len(df),
            "label_type": "atr" if use_atr_labels else "fixed_pct",
        }

        # Add ensemble info if applicable
        if use_ensemble and "ensemble_info" in model_package:
            meta[name]["ensemble_info"] = model_package["ensemble_info"]

        # Add meta-labeling info if applicable
        if use_meta_labeling and "meta_labeling_info" in model_package:
            meta[name]["meta_labeling_info"] = model_package["meta_labeling_info"]

        logger.info(f"Model saved: {model_path}")

    # Save metadata
    meta_path = models_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)

    for horizon, info in meta.items():
        m = info["metrics"]
        logger.info(f"{horizon}: WR={m['win_rate']:.1f}%, PF={m['profit_factor']:.2f}, "
                   f"Sharpe={m['sharpe']:.2f}, MaxP={m['max_probability']:.2f}, "
                   f"Model={m['model_type']}")
        logger.info(f"  Top features: {info['top_features'][:5]}")

    logger.info("\nTraining complete!")
    conn.close()


if __name__ == "__main__":
    main()
