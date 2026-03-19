"""
MOEX Agent v2 Model Training

LogisticRegression with L1 regularization + Trend-Following labels.
Feature selection via permutation importance.
Walk-forward validation for honest OOS estimation.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .features import FEATURE_COLS, build_feature_frame
from .labels import make_trend_following_labels
from .storage import connect

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.train")

# Top features to keep after feature selection
TOP_N_FEATURES = 10


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_top: int = 10,
) -> Tuple[List[str], List[int]]:
    """
    Select top N features using permutation importance.

    Args:
        X: Feature matrix
        y: Binary labels
        feature_names: List of feature names
        n_top: Number of top features to keep

    Returns:
        (selected_feature_names, selected_indices)
    """
    logger.info(f"  Selecting top {n_top} features via permutation importance...")

    # Train a simple model for feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.1,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Permutation importance
    result = permutation_importance(
        model, X_scaled, y,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    # Sort by importance
    importance = result.importances_mean
    indices = np.argsort(importance)[::-1][:n_top]

    selected_names = [feature_names[i] for i in indices]
    selected_importance = [importance[i] for i in indices]

    logger.info(f"  Top {n_top} features:")
    for name, imp in zip(selected_names, selected_importance):
        logger.info(f"    {name}: {imp:.4f}")

    return selected_names, list(indices)


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    take_pct: float = 1.5,
    stop_pct: float = 0.75,
) -> Dict:
    """
    Walk-forward validation for trend-following strategy.

    Labels: 1=LONG win, -1=SHORT win, 0=no signal
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        "trades": [],
        "pnl": [],
        "win_rates": [],
        "profit_factors": [],
    }

    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Only train on LONG signals (y=1) for simplicity
        # Convert: 1 -> 1 (LONG), -1 -> 0 (treat SHORT as negative), 0 -> skip
        y_train_binary = (y_train == 1).astype(int)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.1,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_binary)

        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        fold_trades = []
        for i in range(len(y_proba)):
            if y_proba[i] < 0.55:  # Threshold for taking trade
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
) -> Tuple[object, Dict, List[str]]:
    """
    Train LogisticRegression model for one horizon.

    Returns:
        (model, metrics, selected_features)
    """
    logger.info(f"Training model for {horizon}...")

    # Convert to binary: 1=LONG, 0=everything else
    y_binary = (y == 1).astype(int)
    n_long = y_binary.sum()
    n_short = (y == -1).sum()
    n_neutral = (y == 0).sum()

    logger.info(f"  Data: {len(X):,} rows")
    logger.info(f"  Labels: LONG={n_long:,} ({n_long/len(y)*100:.1f}%), "
                f"SHORT={n_short:,} ({n_short/len(y)*100:.1f}%), "
                f"NEUTRAL={n_neutral:,} ({n_neutral/len(y)*100:.1f}%)")

    # Feature selection
    selected_names, selected_indices = select_top_features(
        X, y_binary, feature_names, n_top=TOP_N_FEATURES
    )
    X_selected = X[:, selected_indices]

    # Walk-forward validation
    logger.info("  Walk-forward validation...")
    wf_results = walk_forward_validation(X_selected, y, n_splits=5)

    wf_wr = wf_results.get("avg_win_rate", 0)
    wf_pf = wf_results.get("avg_profit_factor", 0)
    logger.info(f"  WF Results: WR={wf_wr:.1f}%, PF={wf_pf:.2f}")

    # Final model training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.1,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y_binary)

    # Check coefficients
    n_nonzero = np.sum(model.coef_ != 0)
    logger.info(f"  Non-zero coefficients: {n_nonzero}/{len(selected_names)}")

    # Predictions stats
    y_proba = model.predict_proba(X_scaled)[:, 1]
    max_p = y_proba.max()
    logger.info(f"  Max probability: {max_p:.2f}")

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
        "total_trades": int(wf_results.get("avg_trades", 0) * 5),
        "n_nonzero_coef": int(n_nonzero),
    }

    logger.info(f"  Final: WR={wf_wr:.1f}%, PF={wf_pf:.2f}, Sharpe={sharpe:.2f}")

    # Package model with scaler and feature indices
    model_package = {
        "model": model,
        "scaler": scaler,
        "feature_indices": selected_indices,
        "feature_names": selected_names,
    }

    return model_package, metrics, selected_names


def main(train_days: int = None):
    """
    Main training function.

    Args:
        train_days: If specified, only use first N days of data for training.
                   This enables walk-forward testing (train on past, test on future).
    """
    cfg = load_config()
    conn = connect(cfg.sqlite_path)

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
        max_date = candles["ts"].max()
        min_date = candles["ts"].min()

        cutoff_date = min_date + pd.Timedelta(days=train_days)
        candles = candles[candles["ts"] < cutoff_date]
        logger.info(f"Walk-forward: using first {train_days} days (cutoff: {cutoff_date.date()})")
        logger.info(f"Training candles: {len(candles):,}")

    if len(candles) < 100_000:
        logger.warning("Low data volume! Recommend at least 100K candles.")

    # Build features
    logger.info("Building features (30 indicators)...")
    feats = build_feature_frame(candles)

    # Create TREND-FOLLOWING labels
    horizons = [(h.name, h.minutes) for h in cfg.horizons]
    logger.info("Creating trend-following labels (R:R = 2:1)...")
    labels = make_trend_following_labels(
        candles,
        horizons=horizons,
        take_pct=1.5,
        stop_pct=0.75,
        fee_bps=cfg.fee_bps,
    )

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

        ycol = f"y_trend_{name}"
        if ycol not in df.columns:
            logger.warning(f"Label {ycol} not found")
            continue

        y = df[ycol].to_numpy(dtype=int)

        model_package, metrics, selected_features = train_horizon_model(
            X, y, FEATURE_COLS, horizon=name
        )

        model_path = models_dir / f"model_trend_{name}.joblib"
        joblib.dump(model_package, model_path)

        meta[name] = {
            "type": "logreg-l1-trend-v1",
            "path": str(model_path),
            "features": selected_features,
            "all_features": FEATURE_COLS,
            "metrics": metrics,
            "trained_at": datetime.now().isoformat(),
            "data_size": len(df),
        }

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
                   f"Coef={m['n_nonzero_coef']}")
        logger.info(f"  Features: {info['features']}")

    logger.info("\nTraining complete!")
    conn.close()


if __name__ == "__main__":
    main()
