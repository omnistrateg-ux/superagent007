"""
MOEX Agent v2.5 OOS (Out-of-Sample) Backtest for SmartFilter

Validates SmartFilter edge on unseen data to detect overfitting.

Methodology:
1. Split data: train (70%) / test (30%)
2. Train: calibrate SmartFilter regime_win_rates
3. Test: apply SmartFilter with train calibration
4. Compare: WR, PF, Sharpe on train vs test

If test WR > baseline → edge confirmed
If test WR < baseline → overfitting detected

Usage:
    python -m moex_agent.backtest_oos
    python -m moex_agent.backtest_oos --ticker SBER --days 60
    python -m moex_agent.backtest_oos --all-tickers
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import build_feature_frame
from .regime import RegimeDetector, TickerRegime
from .calendar_features import CalendarFeatures, get_calendar_features
from .cross_asset import CrossAssetFeatures, get_cross_asset_features
from .storage import connect, get_recent_candles
from .smart_filters import SmartFilter, get_smart_filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class OOSMetrics:
    """Metrics for train/test comparison."""
    n_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    avg_pnl_atr: float
    max_drawdown: float

    def __str__(self):
        return (
            f"Trades: {self.n_trades:4d} | "
            f"WR: {self.win_rate*100:5.1f}% | "
            f"PF: {self.profit_factor:5.2f} | "
            f"Sharpe: {self.sharpe_ratio:6.2f} | "
            f"AvgPnL: {self.avg_pnl_atr:+.3f} ATR | "
            f"MaxDD: {self.max_drawdown:.2f} ATR"
        )


@dataclass
class OOSResult:
    """OOS backtest result."""
    ticker: str
    train_baseline: OOSMetrics
    train_filtered: OOSMetrics
    test_baseline: OOSMetrics
    test_filtered: OOSMetrics
    regime_wr: Dict[str, float]

    @property
    def is_overfit(self) -> bool:
        """Check if SmartFilter is overfit."""
        # Overfit if test filtered WR is worse than test baseline
        return self.test_filtered.win_rate < self.test_baseline.win_rate - 0.02

    @property
    def edge_confirmed(self) -> bool:
        """Check if SmartFilter has real edge."""
        # Edge if test filtered WR beats test baseline
        return self.test_filtered.win_rate > self.test_baseline.win_rate + 0.01


def simulate_trade(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    atr: float,
    target_mult: float = 2.0,
    stop_mult: float = 1.0,
    max_bars: int = 30,
) -> Optional[Tuple[float, int]]:
    """Simulate a single trade. Returns (pnl_atr, holding_bars) or None."""
    entry_price = df.iloc[idx]["close"]

    if direction == "LONG":
        target = entry_price + target_mult * atr
        stop = entry_price - stop_mult * atr
    else:
        target = entry_price - target_mult * atr
        stop = entry_price + stop_mult * atr

    for j in range(1, min(max_bars, len(df) - idx)):
        bar = df.iloc[idx + j]
        high, low = bar["high"], bar["low"]

        if direction == "LONG":
            if high >= target:
                return target_mult, j
            if low <= stop:
                return -stop_mult, j
        else:
            if low <= target:
                return target_mult, j
            if high >= stop:
                return -stop_mult, j

    # Exit at last bar
    exit_price = df.iloc[min(idx + max_bars, len(df) - 1)]["close"]
    pnl = (exit_price - entry_price) / atr if direction == "LONG" else (entry_price - exit_price) / atr
    return pnl, max_bars


def get_signal(row: pd.Series) -> Optional[str]:
    """Generate signal based on momentum + RSI (10m horizon)."""
    momentum = row.get("momentum_10", 0)
    rsi = row.get("rsi_14", 50)
    macd_hist = row.get("macd_hist", 0)

    if momentum > 0.002 and rsi < 70 and macd_hist > 0:
        return "LONG"
    elif momentum < -0.002 and rsi > 30 and macd_hist < 0:
        return "SHORT"
    return None


def run_backtest_period(
    df: pd.DataFrame,
    ticker: str,
    detector: RegimeDetector,
    calendar: CalendarFeatures,
    use_smart_filter: bool = False,
    smart_filter: Optional[SmartFilter] = None,
    regime_wr: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], Dict[str, Dict]]:
    """
    Run backtest on a data period.

    Returns:
        (list of pnl_atr, regime_stats dict)
    """
    pnls = []
    regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": []})

    np.random.seed(42)
    risk_sentiments = np.random.randn(len(df)) * 0.3

    i = 0
    while i < len(df) - 30:
        row = df.iloc[i]

        atr = row.get("atr_14", 0)
        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        direction = get_signal(row)
        if direction is None:
            i += 1
            continue

        # Get timestamp
        if isinstance(df.index[i], pd.Timestamp):
            ts = df.index[i].to_pydatetime()
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)
        else:
            ts = datetime.now()

        # Regime detection
        try:
            regime_state = detector.detect(row)
            regime_name = regime_state.regime.value if regime_state else "unknown"
        except Exception:
            regime_name = "unknown"
            regime_state = None

        # Calendar features
        try:
            cal_state = calendar.get_features(ts, ticker)
        except Exception:
            cal_state = None

        # SmartFilter decision
        if use_smart_filter and smart_filter:
            try:
                decision = smart_filter.should_trade(
                    ticker=ticker,
                    direction=direction,
                    regime_state=regime_state,
                    calendar_state=cal_state,
                    risk_sentiment=risk_sentiments[i],
                    regime_win_rates=regime_wr,
                )
                if not decision.allow:
                    i += 1
                    continue
            except Exception:
                pass

        # Simulate trade
        result = simulate_trade(df, i, direction, atr)
        if result is None:
            i += 1
            continue

        pnl_atr, holding_bars = result
        pnls.append(pnl_atr)

        # Track regime stats
        regime_stats[regime_name]["trades"] += 1
        regime_stats[regime_name]["pnl"].append(pnl_atr)
        if pnl_atr > 0:
            regime_stats[regime_name]["wins"] += 1

        i += holding_bars + 1

    # Compute regime win rates
    for r in regime_stats:
        if regime_stats[r]["trades"] > 0:
            regime_stats[r]["win_rate"] = regime_stats[r]["wins"] / regime_stats[r]["trades"]
            regime_stats[r]["avg_pnl"] = np.mean(regime_stats[r]["pnl"])

    return pnls, dict(regime_stats)


def compute_metrics(pnls: List[float]) -> OOSMetrics:
    """Compute metrics from PnL list."""
    if not pnls:
        return OOSMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls)
    avg_pnl = np.mean(pnls)

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe (annualized)
    if np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 6.5 * 6)  # 6 trades/hour
    else:
        sharpe = 0.0

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    return OOSMetrics(
        n_trades=len(pnls),
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        avg_pnl_atr=avg_pnl,
        max_drawdown=max_dd,
    )


def run_oos_backtest(
    ticker: str,
    days: int = 60,
    train_pct: float = 0.7,
    db_path: Path = Path("data/moex_agent.sqlite"),
) -> Optional[OOSResult]:
    """
    Run OOS backtest for a ticker.

    Args:
        ticker: Security ID
        days: Days of data to use
        train_pct: Fraction for training (default 70%)
        db_path: Path to SQLite database
    """
    logger.info(f"{'='*60}")
    logger.info(f"OOS Backtest: {ticker} ({days} days, {train_pct*100:.0f}% train)")
    logger.info(f"{'='*60}")

    # Load data
    conn = connect(db_path)
    candles = get_recent_candles(conn, days=days + 10, interval=1)
    conn.close()

    candles = candles[candles["secid"] == ticker].copy()
    if candles.empty:
        logger.error(f"No candles for {ticker}")
        return None

    logger.info(f"Loaded {len(candles)} candles")

    # Build features
    df = build_feature_frame(candles)
    df = df.dropna()

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Merge OHLC
    candles["ts"] = pd.to_datetime(candles["ts"])
    candles = candles.set_index("ts").sort_index()
    if candles.index.tz is not None:
        candles.index = candles.index.tz_localize(None)
    for col in ["open", "high", "low"]:
        if col not in df.columns and col in candles.columns:
            df[col] = candles[col]

    # Split train/test
    split_idx = int(len(df) * train_pct)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    logger.info(f"Train: {len(df_train)} bars ({df_train.index[0]} to {df_train.index[-1]})")
    logger.info(f"Test:  {len(df_test)} bars ({df_test.index[0]} to {df_test.index[-1]})")

    # Initialize components
    detector = RegimeDetector()
    calendar = get_calendar_features()
    smart_filter = SmartFilter()

    # === TRAIN PHASE ===
    logger.info("\n--- TRAIN PHASE ---")

    # Baseline (no filter)
    train_pnls_base, train_regime_stats = run_backtest_period(
        df_train, ticker, detector, calendar,
        use_smart_filter=False,
    )
    train_baseline = compute_metrics(train_pnls_base)
    logger.info(f"Train Baseline: {train_baseline}")

    # Extract regime win rates from train
    regime_wr = {r: s.get("win_rate", 0.5) for r, s in train_regime_stats.items()}
    logger.info(f"Regime WR (train): {regime_wr}")

    # With SmartFilter (calibrated on train)
    train_pnls_filt, _ = run_backtest_period(
        df_train, ticker, detector, calendar,
        use_smart_filter=True,
        smart_filter=smart_filter,
        regime_wr=regime_wr,
    )
    train_filtered = compute_metrics(train_pnls_filt)
    logger.info(f"Train Filtered: {train_filtered}")

    # === TEST PHASE (OOS) ===
    logger.info("\n--- TEST PHASE (OOS) ---")

    # Baseline (no filter)
    test_pnls_base, _ = run_backtest_period(
        df_test, ticker, detector, calendar,
        use_smart_filter=False,
    )
    test_baseline = compute_metrics(test_pnls_base)
    logger.info(f"Test Baseline:  {test_baseline}")

    # With SmartFilter (using train calibration)
    test_pnls_filt, _ = run_backtest_period(
        df_test, ticker, detector, calendar,
        use_smart_filter=True,
        smart_filter=smart_filter,
        regime_wr=regime_wr,  # Use train calibration!
    )
    test_filtered = compute_metrics(test_pnls_filt)
    logger.info(f"Test Filtered:  {test_filtered}")

    result = OOSResult(
        ticker=ticker,
        train_baseline=train_baseline,
        train_filtered=train_filtered,
        test_baseline=test_baseline,
        test_filtered=test_filtered,
        regime_wr=regime_wr,
    )

    # Summary
    logger.info("\n--- SUMMARY ---")
    train_improvement = (train_filtered.win_rate - train_baseline.win_rate) * 100
    test_improvement = (test_filtered.win_rate - test_baseline.win_rate) * 100

    logger.info(f"Train WR improvement: {train_improvement:+.1f}%")
    logger.info(f"Test WR improvement:  {test_improvement:+.1f}%")

    if result.is_overfit:
        logger.warning("OVERFIT DETECTED: Test performance worse than baseline")
    elif result.edge_confirmed:
        logger.info("EDGE CONFIRMED: Test performance beats baseline")
    else:
        logger.info("MARGINAL: No clear edge or overfit")

    return result


def main():
    parser = argparse.ArgumentParser(description="OOS Backtest for SmartFilter")
    parser.add_argument("--db-path", type=str, default="data/moex_agent.sqlite")
    parser.add_argument("--ticker", type=str, default="SBER")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--all-tickers", action="store_true", help="Run for all tickers")
    args = parser.parse_args()

    db_path = Path(args.db_path)

    if args.all_tickers:
        tickers = ["SBER", "GAZP", "BR", "MX", "RI"]
    else:
        tickers = [args.ticker]

    results = []
    for ticker in tickers:
        result = run_oos_backtest(
            ticker=ticker,
            days=args.days,
            train_pct=args.train_pct,
            db_path=db_path,
        )
        if result:
            results.append(result)

    # Final summary
    if len(results) > 1:
        print("\n" + "="*80)
        print("FINAL OOS SUMMARY")
        print("="*80)
        print(f"{'Ticker':<8} {'Train WR':>10} {'Test WR':>10} {'Delta':>8} {'Status':<15}")
        print("-"*60)

        for r in results:
            delta = (r.test_filtered.win_rate - r.test_baseline.win_rate) * 100
            status = "OVERFIT" if r.is_overfit else ("EDGE" if r.edge_confirmed else "MARGINAL")
            print(f"{r.ticker:<8} {r.train_filtered.win_rate*100:>9.1f}% {r.test_filtered.win_rate*100:>9.1f}% {delta:>+7.1f}% {status:<15}")

        # Aggregate
        n_edge = sum(1 for r in results if r.edge_confirmed)
        n_overfit = sum(1 for r in results if r.is_overfit)
        print("-"*60)
        print(f"Edge confirmed: {n_edge}/{len(results)}, Overfit: {n_overfit}/{len(results)}")


if __name__ == "__main__":
    main()
