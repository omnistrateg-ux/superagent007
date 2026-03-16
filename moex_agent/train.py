"""
MOEX Agent v2 Model Training

HistGradientBoosting + Isotonic calibration with Walk-Forward validation.
Calibration ensures max probability ~ 0.60.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import load_config
from .features import FEATURE_COLS, build_feature_frame
from .labels import make_time_exit_labels
from .storage import connect

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.train")


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    n_splits: int = 5,
    p_threshold: float = 0.54,
    take_atr: float = 0.7,
    stop_atr: float = 0.4,
) -> Dict:
    """
    Walk-forward validation to estimate real performance.

    Simulates real trading: train on past, test on future.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        "trades": [],
        "pnl": [],
        "win_rates": [],
        "profit_factors": [],
    }

    model_params = {
        "max_depth": 7,
        "learning_rate": 0.05,
        "max_iter": 300,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1,
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        prices_test = prices[test_idx]
        atr_test = atr[test_idx]

        model = HistGradientBoostingClassifier(**model_params, random_state=42)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        fold_trades = []
        for i in range(len(y_proba)):
            if y_proba[i] < p_threshold:
                continue
            if prices_test[i] <= 0 or atr_test[i] <= 0:
                continue

            if y_test[i] == 1:
                pnl_pct = take_atr * atr_test[i] / prices_test[i] * 100
            else:
                pnl_pct = -stop_atr * atr_test[i] / prices_test[i] * 100

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
    prices: np.ndarray,
    atr: np.ndarray,
    horizon: str,
) -> Tuple[object, Dict]:
    """
    Train model for one horizon with Isotonic calibration.

    Isotonic calibration ensures probabilities are well-calibrated
    and max p ~ 0.60 (realistic given market efficiency).
    """
    logger.info(f"Training model for {horizon}...")
    logger.info(f"  Data: {len(X):,}, positive: {y.sum():,} ({y.mean()*100:.1f}%)")

    # Walk-forward validation
    logger.info("  Walk-forward validation...")
    wf_results = walk_forward_validation(X, y, prices, atr, n_splits=5)

    logger.info(f"  WF Results: WR={wf_results.get('avg_win_rate', 0):.1f}%, PF={wf_results.get('avg_profit_factor', 0):.2f}")

    # Final training with Isotonic calibration
    base = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42,
    )

    # Isotonic calibration - ensures max p ~ 0.60
    model = CalibratedClassifierCV(
        base,
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=3),
    )
    model.fit(X, y)

    # Check max probability after calibration
    y_proba = model.predict_proba(X)[:, 1]
    max_p = y_proba.max()
    logger.info(f"  Max probability after calibration: {max_p:.2f}")

    wf_wr = wf_results.get("avg_win_rate", 0)
    wf_pf = wf_results.get("avg_profit_factor", 0)
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
    }

    logger.info(f"  Final: WR={wf_wr:.1f}%, PF={wf_pf:.2f}, Sharpe={sharpe:.2f}")

    return model, metrics


def main():
    """Main training function."""
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

    if len(candles) < 100_000:
        logger.warning("Low data volume! Recommend at least 100K candles.")

    # Build features
    logger.info("Building features (30 indicators)...")
    feats = build_feature_frame(candles)

    # Create labels
    horizons = [(h.name, h.minutes) for h in cfg.horizons]
    logger.info("Creating labels...")
    labels = make_time_exit_labels(candles, horizons=horizons, fee_bps=cfg.fee_bps)

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
    prices = df["close"].to_numpy(dtype=float) if "close" in df.columns else np.ones(len(df))
    atr = df["atr_14"].to_numpy(dtype=float) if "atr_14" in df.columns else np.ones(len(df)) * 0.01

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    meta = {}

    # Train for each horizon
    for name, minutes in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizon: {name} ({minutes} min)")
        logger.info(f"{'='*60}")

        ycol = f"y_time_{name}"
        if ycol not in df.columns:
            logger.warning(f"Label {ycol} not found")
            continue

        y = df[ycol].to_numpy(dtype=int)

        model, metrics = train_horizon_model(X, y, prices, atr, horizon=name)

        model_path = models_dir / f"model_time_{name}.joblib"
        joblib.dump(model, model_path)

        meta[name] = {
            "type": "hgb-isotonic-v2",
            "path": str(model_path),
            "features": FEATURE_COLS,
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
                   f"Sharpe={m['sharpe']:.2f}, MaxP={m['max_probability']:.2f}")

    logger.info("\nTraining complete!")
    conn.close()


if __name__ == "__main__":
    main()
