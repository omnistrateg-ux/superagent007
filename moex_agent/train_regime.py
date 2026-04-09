"""
MOEX Agent v2.5 Regime Detector Training

Phase 3: Train per-ticker regime detector and backtest impact.

Usage:
    python -m moex_agent.train_regime

    # With specific ticker
    python -m moex_agent.train_regime --ticker SBER

    # Backtest only
    python -m moex_agent.train_regime --backtest-only
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, build_feature_frame
from .labels import make_atr_trend_labels
from .regime import (
    RegimeDetector,
    REGIME_FEATURES,
    backtest_regime_filter,
    filter_signal_by_regime,
    get_regime_position_multiplier,
)
from .storage import connect, get_recent_candles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


def load_alpha_directions(
    df: pd.DataFrame,
    models_dir: Path = Path("models"),
    threshold: float = 0.3,
) -> pd.Series:
    """
    Load Alpha Model predictions to get direction signals.

    Args:
        df: DataFrame with OHLCV features
        models_dir: Path to trained models
        threshold: Probability threshold for signal

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
        return (df["r_5m"] > 0).map({True: "LONG", False: "SHORT"})

    directions = pd.Series(index=df.index, dtype="object")

    # Use 10m horizon (best performing)
    horizon = "10m" if "10m" in registry.horizons else registry.horizons[0]

    for idx, row in df.iterrows():
        try:
            X = row[FEATURE_COLS].astype(float).values.reshape(1, -1)
            if np.isnan(X).any():
                continue

            prob = registry.predict(horizon, X)

            if prob > threshold:
                directions[idx] = "LONG"
            elif prob < (1 - threshold):
                directions[idx] = "SHORT"
        except Exception:
            continue

    return directions


def create_outcome_labels(
    df: pd.DataFrame,
    horizon_bars: int = 10,
    target_mult: float = 1.0,
    stop_mult: float = 0.5,
) -> pd.Series:
    """
    Create outcome labels based on future price path.

    Args:
        df: DataFrame with OHLCV + direction + atr_14
        horizon_bars: How many bars to evaluate
        target_mult: Target in ATR multiples
        stop_mult: Stop in ATR multiples

    Returns:
        Series of 0/1 labels (1 = profitable trade)
    """
    labels = pd.Series(index=df.index, dtype=float)

    for i in range(len(df) - horizon_bars):
        idx = df.index[i]

        direction = df.loc[idx, "direction"] if "direction" in df.columns else None
        if direction not in ["LONG", "SHORT"]:
            continue

        atr = df.loc[idx, "atr_14"]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = df.loc[idx, "close"]
        target = entry_price + (target_mult * atr if direction == "LONG" else -target_mult * atr)
        stop = entry_price + (-stop_mult * atr if direction == "LONG" else stop_mult * atr)

        future_slice = df.iloc[i+1:i+1+horizon_bars]

        # Check outcome
        for j in range(len(future_slice)):
            bar = future_slice.iloc[j]

            if direction == "LONG":
                if bar["high"] >= target:
                    labels[idx] = 1
                    break
                if bar["low"] <= stop:
                    labels[idx] = 0
                    break
            else:
                if bar["low"] <= target:
                    labels[idx] = 1
                    break
                if bar["high"] >= stop:
                    labels[idx] = 0
                    break

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Train Regime Detector (Phase 3)"
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
        "--ticker",
        type=str,
        default="SBER",
        help="Ticker to train on"
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Only run backtest, don't retrain"
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

    candles = candles[candles["secid"] == args.ticker].copy()

    if candles.empty:
        logger.error(f"No candles found for {args.ticker}")
        return

    logger.info(f"Loaded {len(candles)} candles")

    # Build features
    logger.info("Building OHLCV features...")
    df = build_feature_frame(candles)
    df = df.dropna()

    # Set index
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Merge OHLC columns
    candles["ts"] = pd.to_datetime(candles["ts"])
    candles = candles.set_index("ts").sort_index()
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = candles[col]

    logger.info(f"Features built: {len(df)} rows")

    # Load or train regime detector
    detector_path = models_dir / "regime_detector.joblib"
    detector = RegimeDetector()

    if args.backtest_only and detector_path.exists():
        logger.info("Loading existing regime detector...")
        detector.load(detector_path)
    else:
        # Train regime detector
        logger.info("Training regime detector...")
        metrics = detector.fit(df, use_ml=True)
        detector.save(detector_path)

        logger.info(f"Regime detector trained: {metrics}")

    # Get Alpha Model directions
    logger.info("Getting Alpha model directions...")
    df["direction"] = load_alpha_directions(df, models_dir)

    # Create outcome labels
    logger.info("Creating outcome labels...")
    df["label"] = create_outcome_labels(df, horizon_bars=10)

    # Show regime distribution
    logger.info("\nAnalyzing regime distribution...")
    regime_data = []
    for idx, row in df.iterrows():
        regime = detector.detect(row)
        regime_data.append({
            "regime": regime.regime.value,
            "confidence": regime.confidence,
        })

    regime_df = pd.DataFrame(regime_data)
    regime_counts = regime_df["regime"].value_counts()
    logger.info(f"\nRegime distribution:")
    for r, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        logger.info(f"  {r}: {count} ({pct:.1f}%)")

    # Backtest
    logger.info("\nRunning backtest...")
    bt_results = backtest_regime_filter(df, detector)

    logger.info(f"\nBacktest Results:")
    logger.info(f"  Baseline: {bt_results['baseline_trades']} trades, WR={bt_results['baseline_win_rate']:.1%}")
    logger.info(f"  Filtered: {bt_results['filtered_trades']} trades, WR={bt_results['filtered_win_rate']:.1%}")
    logger.info(f"  Improvement: {bt_results['improvement']:+.1%}")
    logger.info(f"  Filter rate: {bt_results['filter_rate']:.1%}")

    logger.info(f"\nPer-Regime Win Rates:")
    for regime, wr in sorted(bt_results["regime_win_rates"].items()):
        count = bt_results["regime_distribution"].get(regime, 0)
        logger.info(f"  {regime}: WR={wr:.1%} ({count} trades)")

    # Update meta.json
    meta_path = models_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    meta["regime_detector"] = {
        "type": "regime-v2.5",
        "path": str(detector_path),
        "thresholds": detector.thresholds,
        "backtest": bt_results,
        "trained_at": datetime.now().isoformat(),
        "ticker": args.ticker,
        "train_days": args.train_days,
    }

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("\nUpdated meta.json")

    # Backtest with quality filter (skip low-WR regimes)
    from .regime import filter_signal_by_regime_quality

    logger.info("\nBacktest with quality filter (skip WR < 40%)...")
    results_quality = []
    for idx, row in df.iterrows():
        if "direction" not in row or "label" not in row:
            continue

        direction = row["direction"]
        label = row["label"]

        if direction not in ["LONG", "SHORT"] or pd.isna(label):
            continue

        regime = detector.detect(row)
        allow, _ = filter_signal_by_regime_quality(
            direction, regime, bt_results["regime_win_rates"], min_wr_threshold=0.40
        )
        if allow:
            results_quality.append(label)

    if results_quality:
        quality_wr = np.mean(results_quality)
        logger.info(f"  Quality-filtered: {len(results_quality)} trades, WR={quality_wr:.1%}")
        logger.info(f"  Improvement vs baseline: {quality_wr - bt_results['baseline_win_rate']:+.1%}")
    else:
        logger.info("  No trades passed quality filter")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 - REGIME DETECTION COMPLETE")
    logger.info("=" * 60)

    improvement = bt_results['improvement']
    if improvement > 0:
        logger.info(f"Regime filtering IMPROVES win rate by {improvement:+.1%}")
        logger.info(f"Trade count reduced by {(1-bt_results['filter_rate']):.1%}")
    else:
        logger.info(f"Regime filtering does not improve win rate ({improvement:+.1%})")

    # Recommendations
    logger.info("\nRecommendations:")

    best_regime = max(
        bt_results["regime_win_rates"].items(),
        key=lambda x: x[1],
        default=("unknown", 0)
    )
    worst_regime = min(
        bt_results["regime_win_rates"].items(),
        key=lambda x: x[1],
        default=("unknown", 1)
    )

    logger.info(f"  Best regime: {best_regime[0]} (WR={best_regime[1]:.1%})")
    logger.info(f"  Worst regime: {worst_regime[0]} (WR={worst_regime[1]:.1%})")

    if worst_regime[1] < 0.4:
        logger.info(f"  Consider: Skip trading in {worst_regime[0]}")


if __name__ == "__main__":
    main()
