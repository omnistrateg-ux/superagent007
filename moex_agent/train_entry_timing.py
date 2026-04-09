"""
MOEX Agent v2.5 Entry Timing Model Training

Phase 2: Train model to predict optimal entry points.

Two-level architecture:
1. Alpha Model (existing): OHLCV → direction (LONG/SHORT) at 10m/30m/1h horizons
2. Entry Timing Model (this): microstructure → entry quality (good/bad entry)

The Entry Timing Model filters Alpha signals:
- Alpha gives direction signal (high confidence LONG/SHORT)
- Entry Timing scores current microstructure
- Only enter when both signals align

Entry quality label:
- 1 = Good entry: favorable price path (low drawdown before reaching target)
- 0 = Bad entry: adverse price path (stop hit before target)

Usage:
    python -m moex_agent.train_entry_timing

    # With real microstructure data (when available)
    python -m moex_agent.train_entry_timing --use-real-data

    # Backtest entry timing impact
    python -m moex_agent.train_entry_timing --backtest
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLS, build_feature_frame
from .microstructure import MICRO_FEATURE_COLS
from .storage import get_recent_candles, connect
from .synthetic_microstructure import generate_micro_features_from_candles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not available")


def get_entry_timing_params() -> Dict:
    """CatBoost parameters for entry timing model."""
    return {
        "iterations": 200,
        "depth": 4,  # Shallower than alpha model (simpler patterns)
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_seed": 42,
        "verbose": False,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "early_stopping_rounds": 30,
    }


def create_entry_timing_labels(
    df: pd.DataFrame,
    direction_col: str = "direction",
    horizon_bars: int = 5,
    atr_col: str = "atr_14",
) -> pd.Series:
    """
    Create entry timing labels based on future price path.

    A "good entry" is when price moves favorably before hitting drawdown limit.

    Label = 1 (good entry) if:
    - For LONG: price rises 1*ATR before dropping 0.5*ATR from entry
    - For SHORT: price drops 1*ATR before rising 0.5*ATR from entry

    Args:
        df: DataFrame with OHLCV + direction + ATR
        direction_col: Column name for LONG/SHORT direction
        horizon_bars: How many bars forward to evaluate
        atr_col: Column name for ATR

    Returns:
        Series of 0/1 labels
    """
    labels = pd.Series(0, index=df.index)

    for i in range(len(df) - horizon_bars):
        idx = df.index[i]

        direction = df.loc[idx, direction_col] if direction_col in df.columns else None
        if direction not in ["LONG", "SHORT"]:
            continue

        atr = df.loc[idx, atr_col]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = df.loc[idx, "close"]

        # Look at future bars
        future_slice = df.iloc[i+1:i+1+horizon_bars]

        if len(future_slice) < horizon_bars:
            continue

        # Target and stop levels
        if direction == "LONG":
            target_price = entry_price + 1.0 * atr
            stop_price = entry_price - 0.5 * atr

            # Check if target hit before stop
            for j in range(len(future_slice)):
                bar = future_slice.iloc[j]
                # Stop hit first?
                if bar["low"] <= stop_price:
                    labels[idx] = 0  # Bad entry
                    break
                # Target hit?
                if bar["high"] >= target_price:
                    labels[idx] = 1  # Good entry
                    break

        else:  # SHORT
            target_price = entry_price - 1.0 * atr
            stop_price = entry_price + 0.5 * atr

            for j in range(len(future_slice)):
                bar = future_slice.iloc[j]
                if bar["high"] >= stop_price:
                    labels[idx] = 0
                    break
                if bar["low"] <= target_price:
                    labels[idx] = 1
                    break

    return labels


def load_alpha_directions(
    df: pd.DataFrame,
    models_dir: Path = Path("models"),
) -> pd.Series:
    """
    Load Alpha Model predictions to get direction signals.

    Uses existing trained models to predict LONG/SHORT direction.
    Only returns direction when confidence > threshold.

    Args:
        df: DataFrame with OHLCV features
        models_dir: Path to trained models

    Returns:
        Series with values 'LONG', 'SHORT', or None
    """
    from .predictor import get_registry, FEATURE_COLS

    try:
        registry = get_registry(models_dir)
        registry.ensure_loaded()
    except Exception as e:
        logger.warning(f"Could not load Alpha models: {e}")
        # Fallback: use simple momentum direction
        logger.info("Using momentum-based direction (fallback)")
        return (df["r_5m"] > 0).map({True: "LONG", False: "SHORT"})

    directions = pd.Series(index=df.index, dtype="object")

    # Use 10m horizon (best performing)
    if "10m" not in registry.horizons:
        logger.warning("10m model not available")
        return (df["r_5m"] > 0).map({True: "LONG", False: "SHORT"})

    threshold = 0.3  # Minimum probability for signal

    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = registry.predict("10m", X)

            if prob > threshold:
                directions[idx] = "LONG"
            elif prob < (1 - threshold):  # Inverted for SHORT
                directions[idx] = "SHORT"
            # else: None (no clear direction)
        except Exception:
            continue

    return directions


def train_entry_timing_model(
    df: pd.DataFrame,
    micro_df: pd.DataFrame,
    labels: pd.Series,
    train_size: float = 0.8,
) -> Tuple[object, Dict]:
    """
    Train the Entry Timing Model.

    Args:
        df: OHLCV DataFrame with features
        micro_df: Microstructure features DataFrame
        labels: Entry timing labels (0/1)
        train_size: Fraction for training

    Returns:
        (model, metrics_dict)
    """
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost required for training")

    # Combine OHLCV features with microstructure features
    # Use subset of OHLCV features that indicate volatility/regime
    ohlcv_subset = [
        "atr_14",
        "volatility_10",
        "volatility_30",
        "bb_width",
        "adx",
        "volume_sma_ratio",
    ]

    # Align indices
    common_idx = df.index.intersection(micro_df.index).intersection(labels.index)
    common_idx = labels[common_idx][labels[common_idx] != -999].index  # Remove unlabeled

    logger.info(f"Common samples: {len(common_idx)}")

    # Filter to samples with direction signal
    valid_idx = []
    for idx in common_idx:
        if idx in df.index and "direction" in df.columns:
            if df.loc[idx, "direction"] in ["LONG", "SHORT"]:
                valid_idx.append(idx)
        else:
            valid_idx.append(idx)  # If no direction column, include all

    valid_idx = pd.Index(valid_idx)
    logger.info(f"Valid samples with direction: {len(valid_idx)}")

    if len(valid_idx) < 100:
        raise ValueError(f"Not enough samples: {len(valid_idx)}")

    # Build feature matrix
    X_ohlcv = df.loc[valid_idx, ohlcv_subset].values
    X_micro = micro_df.loc[valid_idx, MICRO_FEATURE_COLS].values
    X = np.hstack([X_ohlcv, X_micro])

    y = labels.loc[valid_idx].values

    # Feature names
    feature_names = ohlcv_subset + MICRO_FEATURE_COLS

    # Handle NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    logger.info(f"After NaN removal: {len(y)} samples")
    logger.info(f"Label distribution: 0={sum(y==0)}, 1={sum(y==1)}")

    # Train/test split (time-based)
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    model = CatBoostClassifier(**get_entry_timing_params())
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=False,
    )

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = (y_pred == y_test).mean()
    precision = (y_pred & y_test).sum() / max(y_pred.sum(), 1)
    recall = (y_pred & y_test).sum() / max(y_test.sum(), 1)

    # Feature importance
    importances = model.get_feature_importance()
    top_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "top_features": [f[0] for f in top_features],
        "feature_importances": {f[0]: float(f[1]) for f in top_features},
    }

    logger.info(f"Entry Timing Model Results:")
    logger.info(f"  Accuracy: {accuracy:.1%}")
    logger.info(f"  Precision: {precision:.1%}")
    logger.info(f"  Recall: {recall:.1%}")
    logger.info(f"  Top features: {[f[0] for f in top_features[:5]]}")

    return model, metrics


def backtest_entry_timing(
    df: pd.DataFrame,
    micro_df: pd.DataFrame,
    model: object,
    threshold: float = 0.5,
) -> Dict:
    """
    Backtest entry timing model impact on trading performance.

    Compares:
    - Baseline: enter on all Alpha signals
    - Filtered: enter only when Entry Timing > threshold

    Args:
        df: OHLCV DataFrame with direction and labels
        micro_df: Microstructure features
        model: Trained Entry Timing model
        threshold: Entry timing probability threshold

    Returns:
        Comparison metrics dict
    """
    ohlcv_subset = [
        "atr_14",
        "volatility_10",
        "volatility_30",
        "bb_width",
        "adx",
        "volume_sma_ratio",
    ]

    results_baseline = []
    results_filtered = []

    common_idx = df.index.intersection(micro_df.index)

    for idx in common_idx:
        if "direction" not in df.columns:
            continue

        direction = df.loc[idx, "direction"]
        if direction not in ["LONG", "SHORT"]:
            continue

        # Get outcome (1=win, 0=loss from labels)
        if "entry_timing_label" in df.columns:
            outcome = df.loc[idx, "entry_timing_label"]
        else:
            continue

        # Baseline: take all trades
        results_baseline.append(outcome)

        # Get Entry Timing prediction
        try:
            X_ohlcv = df.loc[idx, ohlcv_subset].values.reshape(1, -1)
            X_micro = micro_df.loc[idx, MICRO_FEATURE_COLS].values.reshape(1, -1)
            X = np.hstack([X_ohlcv, X_micro])

            if not np.isnan(X).any():
                prob = model.predict_proba(X)[0, 1]

                # Filtered: only take trades with good entry timing
                if prob >= threshold:
                    results_filtered.append(outcome)
        except Exception:
            continue

    # Calculate metrics
    baseline_wr = np.mean(results_baseline) if results_baseline else 0
    filtered_wr = np.mean(results_filtered) if results_filtered else 0

    return {
        "baseline_trades": len(results_baseline),
        "baseline_win_rate": float(baseline_wr),
        "filtered_trades": len(results_filtered),
        "filtered_win_rate": float(filtered_wr),
        "improvement": float(filtered_wr - baseline_wr),
        "filter_rate": float(len(results_filtered) / max(len(results_baseline), 1)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Entry Timing Model (Phase 2)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/moex_agent.sqlite",
        help="Path to candles database"
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=30,
        help="Days of data to use for training"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real microstructure data (if available)"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest after training"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SBER",
        help="Ticker to train on"
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load candles
    logger.info(f"Loading candles for {args.ticker}...")
    conn = connect(db_path)
    candles = get_recent_candles(conn, days=args.train_days + 10, interval=1)
    conn.close()

    # Filter to ticker
    candles = candles[candles["secid"] == args.ticker].copy()

    if candles.empty:
        logger.error(f"No candles found for {args.ticker}")
        return

    logger.info(f"Loaded {len(candles)} candles")

    # Build OHLCV features
    logger.info("Building OHLCV features...")
    df = build_feature_frame(candles)
    df = df.dropna()

    logger.info(f"Features built: {len(df)} rows")

    # Set ts as index for both dataframes (align them)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    # Prepare candles for microstructure generation (needs DatetimeIndex)
    candles_for_micro = candles.copy()
    candles_for_micro["ts"] = pd.to_datetime(candles_for_micro["ts"])
    candles_for_micro = candles_for_micro.set_index("ts").sort_index()

    # Remove timezone from df index (feature builder adds UTC timezone)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Merge OHLC columns from candles into df (for label calculation)
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = candles_for_micro[col]

    # Generate microstructure features
    if args.use_real_data:
        logger.info("Loading real microstructure data...")
        from .broker import MicrostructureStorage
        storage = MicrostructureStorage(Path("data/microstructure.db"))
        # TODO: Load real data and calculate features
        raise NotImplementedError("Real microstructure data loading not implemented")
    else:
        logger.info("Generating synthetic microstructure features...")
        micro_df = generate_micro_features_from_candles(
            candles_for_micro,
            ticker=args.ticker,
            seed=42,
        )

    logger.info(f"Microstructure features: {len(micro_df)} rows")

    # Get Alpha model directions
    logger.info("Getting Alpha model directions...")
    df["direction"] = load_alpha_directions(df, models_dir)

    # Create entry timing labels
    logger.info("Creating entry timing labels...")
    labels = create_entry_timing_labels(
        df,
        direction_col="direction",
        horizon_bars=5,
    )

    df["entry_timing_label"] = labels

    positive_rate = labels.mean()
    logger.info(f"Entry timing label positive rate: {positive_rate:.1%}")

    # Filter to samples with direction signal
    has_direction = df["direction"].isin(["LONG", "SHORT"])
    logger.info(f"Samples with direction signal: {has_direction.sum()}")

    # Train model
    logger.info("Training Entry Timing Model...")
    model, metrics = train_entry_timing_model(df, micro_df, labels)

    # Save model
    model_path = models_dir / "model_entry_timing.joblib"
    joblib.dump({
        "model": model,
        "ohlcv_features": [
            "atr_14", "volatility_10", "volatility_30",
            "bb_width", "adx", "volume_sma_ratio",
        ],
        "micro_features": MICRO_FEATURE_COLS,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
    }, model_path)

    logger.info(f"Model saved: {model_path}")

    # Update meta.json
    meta_path = models_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    meta["entry_timing"] = {
        "type": "catboost-entry-timing-v2.5",
        "path": str(model_path),
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
        "ticker": args.ticker,
        "train_days": args.train_days,
        "synthetic_data": not args.use_real_data,
    }

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Updated meta.json")

    # Backtest
    if args.backtest:
        logger.info("\nRunning backtest...")
        bt_results = backtest_entry_timing(df, micro_df, model)

        logger.info(f"\nBacktest Results:")
        logger.info(f"  Baseline: {bt_results['baseline_trades']} trades, WR={bt_results['baseline_win_rate']:.1%}")
        logger.info(f"  Filtered: {bt_results['filtered_trades']} trades, WR={bt_results['filtered_win_rate']:.1%}")
        logger.info(f"  Improvement: {bt_results['improvement']:+.1%}")
        logger.info(f"  Filter rate: {bt_results['filter_rate']:.1%}")

    logger.info("\n" + "=" * 60)
    logger.info("Entry Timing Model training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
