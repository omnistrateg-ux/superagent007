"""
MOEX Agent v2.5 Phase 3 Combined Backtest

Test regime detection + cross-asset + calendar features together.

Usage:
    python -m moex_agent.backtest_phase3
    python -m moex_agent.backtest_phase3 --ticker SBER --days 30
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

from .features import FEATURE_COLS, build_feature_frame
from .regime import RegimeDetector, TickerRegime, filter_signal_by_regime
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
class TradeResult:
    """Single trade result."""
    entry_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_atr: float
    regime: str
    session_phase: str
    is_tax_period: bool
    is_expiry_week: bool
    risk_sentiment: float
    holding_bars: int


@dataclass
class BacktestResults:
    """Aggregated backtest results."""
    total_trades: int
    win_rate: float
    avg_pnl_pct: float
    avg_pnl_atr: float
    sharpe_ratio: float
    max_drawdown: float

    # By regime
    regime_stats: Dict[str, Dict]

    # By session
    session_stats: Dict[str, Dict]

    # By calendar
    tax_period_wr: float
    non_tax_period_wr: float
    expiry_week_wr: float
    non_expiry_week_wr: float

    # By risk sentiment
    risk_on_wr: float
    risk_off_wr: float


def simulate_trade(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    atr: float,
    target_mult: float = 2.0,
    stop_mult: float = 1.0,
    max_bars: int = 30,
) -> Optional[Tuple[float, float, int]]:
    """
    Simulate a single trade.

    Returns:
        (exit_price, pnl_atr, holding_bars) or None if no exit
    """
    entry_price = df.iloc[idx]["close"]

    if direction == "LONG":
        target = entry_price + target_mult * atr
        stop = entry_price - stop_mult * atr
    else:
        target = entry_price - target_mult * atr
        stop = entry_price + stop_mult * atr

    for j in range(1, min(max_bars, len(df) - idx)):
        bar = df.iloc[idx + j]
        high = bar["high"]
        low = bar["low"]

        if direction == "LONG":
            if high >= target:
                return target, target_mult, j
            if low <= stop:
                return stop, -stop_mult, j
        else:
            if low <= target:
                return target, target_mult, j
            if high >= stop:
                return stop, -stop_mult, j

    # Exit at last bar
    exit_price = df.iloc[min(idx + max_bars, len(df) - 1)]["close"]
    pnl = (exit_price - entry_price) / atr if direction == "LONG" else (entry_price - exit_price) / atr
    return exit_price, pnl, max_bars


def get_signal_direction(row: pd.Series) -> Optional[str]:
    """
    Generate signal based on momentum + RSI.

    Simple alpha: momentum positive + RSI not overbought → LONG
    """
    momentum = row.get("momentum_10", 0)
    rsi = row.get("rsi_14", 50)
    macd_hist = row.get("macd_hist", 0)

    # Momentum-based signal
    if momentum > 0.002 and rsi < 70 and macd_hist > 0:
        return "LONG"
    elif momentum < -0.002 and rsi > 30 and macd_hist < 0:
        return "SHORT"

    return None


def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    detector: RegimeDetector,
    calendar: CalendarFeatures,
    cross_asset: CrossAssetFeatures,
    use_regime_filter: bool = True,
    use_calendar_filter: bool = True,
    use_session_filter: bool = True,
) -> List[TradeResult]:
    """
    Run backtest with Phase 3 features.
    """
    trades = []

    # Ensure we have OHLC
    if "high" not in df.columns or "low" not in df.columns:
        logger.warning("Missing OHLC columns")
        return trades

    # Get returns for cross-asset
    if "r_1m" in df.columns:
        ticker_returns = df["r_1m"].dropna()
    else:
        ticker_returns = df["close"].pct_change().dropna()

    # Simulate external risk sentiment (would be real in production)
    np.random.seed(42)
    risk_sentiments = np.random.randn(len(df)) * 0.3

    i = 0
    while i < len(df) - 30:
        row = df.iloc[i]

        # Skip if missing ATR
        atr = row.get("atr_14", 0)
        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        # Get signal
        direction = get_signal_direction(row)
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

        # === Phase 3 Features ===

        # 1. Regime
        regime_state = detector.detect(row)
        regime_name = regime_state.regime.value

        # 2. Calendar
        cal_state = calendar.get_features(ts, ticker)
        session_phase = cal_state.session_phase  # Already a string
        is_tax_period = cal_state.is_tax_period
        is_expiry_week = cal_state.is_expiry_week
        event_risk_mult = cal_state.event_risk_multiplier

        # 3. Risk sentiment (simulated)
        risk_sentiment = risk_sentiments[i]

        # === Apply Filters ===

        # Regime filter
        if use_regime_filter:
            allow, reason = filter_signal_by_regime(direction, regime_state)
            if not allow:
                i += 1
                continue

        # Calendar filter: skip during high-risk events
        if use_calendar_filter:
            if event_risk_mult > 1.5:
                i += 1
                continue

        # Session filter: skip evening session and clearing periods
        if use_session_filter:
            if session_phase in ("evening_session", "evening_clearing", "clearing", "closed"):
                i += 1
                continue

        # === Execute Trade ===

        result = simulate_trade(
            df, i, direction, atr,
            target_mult=2.0,
            stop_mult=1.0,
            max_bars=30,
        )

        if result is None:
            i += 1
            continue

        exit_price, pnl_atr, holding_bars = result
        entry_price = row["close"]

        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append(TradeResult(
            entry_time=ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_atr=pnl_atr,
            regime=regime_name,
            session_phase=session_phase,
            is_tax_period=is_tax_period,
            is_expiry_week=is_expiry_week,
            risk_sentiment=risk_sentiment,
            holding_bars=holding_bars,
        ))

        # Skip holding period
        i += holding_bars + 1

    return trades


def analyze_results(trades: List[TradeResult]) -> BacktestResults:
    """Analyze trade results."""
    if not trades:
        return BacktestResults(
            total_trades=0,
            win_rate=0.0,
            avg_pnl_pct=0.0,
            avg_pnl_atr=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            regime_stats={},
            session_stats={},
            tax_period_wr=0.0,
            non_tax_period_wr=0.0,
            expiry_week_wr=0.0,
            non_expiry_week_wr=0.0,
            risk_on_wr=0.0,
            risk_off_wr=0.0,
        )

    pnls = [t.pnl_atr for t in trades]
    wins = [t for t in trades if t.pnl_atr > 0]

    # Basic stats
    win_rate = len(wins) / len(trades)
    avg_pnl_pct = np.mean([t.pnl_pct for t in trades])
    avg_pnl_atr = np.mean(pnls)

    # Sharpe
    if np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 6.5 * 60)  # Annualized
    else:
        sharpe = 0.0

    # Drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # By regime
    regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": []})
    for t in trades:
        regime_stats[t.regime]["trades"] += 1
        regime_stats[t.regime]["pnl"].append(t.pnl_atr)
        if t.pnl_atr > 0:
            regime_stats[t.regime]["wins"] += 1

    for r in regime_stats:
        regime_stats[r]["win_rate"] = regime_stats[r]["wins"] / regime_stats[r]["trades"]
        regime_stats[r]["avg_pnl"] = np.mean(regime_stats[r]["pnl"])

    # By session
    session_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
    for t in trades:
        session_stats[t.session_phase]["trades"] += 1
        if t.pnl_atr > 0:
            session_stats[t.session_phase]["wins"] += 1

    for s in session_stats:
        session_stats[s]["win_rate"] = session_stats[s]["wins"] / session_stats[s]["trades"]

    # Tax period
    tax_trades = [t for t in trades if t.is_tax_period]
    non_tax_trades = [t for t in trades if not t.is_tax_period]
    tax_period_wr = len([t for t in tax_trades if t.pnl_atr > 0]) / len(tax_trades) if tax_trades else 0.0
    non_tax_period_wr = len([t for t in non_tax_trades if t.pnl_atr > 0]) / len(non_tax_trades) if non_tax_trades else 0.0

    # Expiry week
    expiry_trades = [t for t in trades if t.is_expiry_week]
    non_expiry_trades = [t for t in trades if not t.is_expiry_week]
    expiry_week_wr = len([t for t in expiry_trades if t.pnl_atr > 0]) / len(expiry_trades) if expiry_trades else 0.0
    non_expiry_week_wr = len([t for t in non_expiry_trades if t.pnl_atr > 0]) / len(non_expiry_trades) if non_expiry_trades else 0.0

    # Risk sentiment
    risk_on_trades = [t for t in trades if t.risk_sentiment > 0]
    risk_off_trades = [t for t in trades if t.risk_sentiment <= 0]
    risk_on_wr = len([t for t in risk_on_trades if t.pnl_atr > 0]) / len(risk_on_trades) if risk_on_trades else 0.0
    risk_off_wr = len([t for t in risk_off_trades if t.pnl_atr > 0]) / len(risk_off_trades) if risk_off_trades else 0.0

    return BacktestResults(
        total_trades=len(trades),
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl_pct,
        avg_pnl_atr=avg_pnl_atr,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        regime_stats=dict(regime_stats),
        session_stats=dict(session_stats),
        tax_period_wr=tax_period_wr,
        non_tax_period_wr=non_tax_period_wr,
        expiry_week_wr=expiry_week_wr,
        non_expiry_week_wr=non_expiry_week_wr,
        risk_on_wr=risk_on_wr,
        risk_off_wr=risk_off_wr,
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Combined Backtest")
    parser.add_argument("--db-path", type=str, default="data/moex_agent.sqlite")
    parser.add_argument("--ticker", type=str, default="SBER")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    models_dir = Path("models")

    # Load candles
    logger.info(f"Loading {args.days} days of candles for {args.ticker}...")
    conn = connect(db_path)
    candles = get_recent_candles(conn, days=args.days + 10, interval=1)
    conn.close()

    candles = candles[candles["secid"] == args.ticker].copy()

    if candles.empty:
        logger.error(f"No candles found for {args.ticker}")
        return

    logger.info(f"Loaded {len(candles)} candles")

    # Build features
    logger.info("Building features...")
    df = build_feature_frame(candles)
    df = df.dropna()

    # Set index
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
        if col not in df.columns:
            df[col] = candles[col]

    logger.info(f"Features: {len(df)} rows")

    # Initialize Phase 3 components
    detector = RegimeDetector()
    detector_path = models_dir / "regime_detector.joblib"
    if detector_path.exists():
        detector.load(detector_path)
        logger.info("Loaded regime detector")
    else:
        logger.info("Training regime detector...")
        detector.fit(df, use_ml=True)

    calendar = get_calendar_features()
    cross_asset = get_cross_asset_features()

    # Simulate risk sentiment (would be real external feeds in production)
    np.random.seed(42)
    risk_sentiments = np.random.randn(len(df)) * 0.3

    # === Run Backtests ===

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST: NO FILTERS (BASELINE)")
    logger.info("=" * 60)

    trades_baseline = run_backtest(
        df, args.ticker, detector, calendar, cross_asset,
        use_regime_filter=False,
        use_calendar_filter=False,
        use_session_filter=False,
    )
    results_baseline = analyze_results(trades_baseline)

    logger.info(f"Trades: {results_baseline.total_trades}")
    logger.info(f"Win Rate: {results_baseline.win_rate:.1%}")
    logger.info(f"Avg PnL: {results_baseline.avg_pnl_atr:+.2f} ATR")
    logger.info(f"Sharpe: {results_baseline.sharpe_ratio:.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST: REGIME FILTER ONLY")
    logger.info("=" * 60)

    trades_regime = run_backtest(
        df, args.ticker, detector, calendar, cross_asset,
        use_regime_filter=True,
        use_calendar_filter=False,
        use_session_filter=False,
    )
    results_regime = analyze_results(trades_regime)

    logger.info(f"Trades: {results_regime.total_trades}")
    logger.info(f"Win Rate: {results_regime.win_rate:.1%}")
    logger.info(f"Avg PnL: {results_regime.avg_pnl_atr:+.2f} ATR")
    logger.info(f"Improvement vs baseline: {results_regime.win_rate - results_baseline.win_rate:+.1%}")

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST: ALL PHASE 3 FILTERS")
    logger.info("=" * 60)

    trades_full = run_backtest(
        df, args.ticker, detector, calendar, cross_asset,
        use_regime_filter=True,
        use_calendar_filter=True,
        use_session_filter=True,
    )
    results_full = analyze_results(trades_full)

    logger.info(f"Trades: {results_full.total_trades}")
    logger.info(f"Win Rate: {results_full.win_rate:.1%}")
    logger.info(f"Avg PnL: {results_full.avg_pnl_atr:+.2f} ATR")
    logger.info(f"Sharpe: {results_full.sharpe_ratio:.2f}")
    logger.info(f"Max DD: {results_full.max_drawdown:.2f} ATR")
    logger.info(f"Improvement vs baseline: {results_full.win_rate - results_baseline.win_rate:+.1%}")

    # === Smart Filter Backtest ===
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST: SMART FILTER (SKIP MORNING SESSION)")
    logger.info("=" * 60)

    smart_filter = get_smart_filter()

    # Get regime WR from full backtest
    regime_wr = {r: s["win_rate"] for r, s in results_full.regime_stats.items()}
    smart_filter._regime_wr_cache[args.ticker] = regime_wr

    trades_smart = []
    i = 0
    while i < len(df) - 30:
        row = df.iloc[i]
        atr = row.get("atr_14", 0)
        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        direction = get_signal_direction(row)
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

        # Get states
        regime_state = detector.detect(row)
        cal_state = calendar.get_features(ts, args.ticker)
        risk_sentiment = risk_sentiments[i] if 'risk_sentiments' in dir() else 0.0

        # Apply smart filter
        decision = smart_filter.should_trade(
            ticker=args.ticker,
            direction=direction,
            regime_state=regime_state,
            calendar_state=cal_state,
            risk_sentiment=risk_sentiment,
        )

        if not decision.allow:
            i += 1
            continue

        # Execute trade
        result = simulate_trade(df, i, direction, atr, target_mult=2.0, stop_mult=1.0, max_bars=30)
        if result is None:
            i += 1
            continue

        exit_price, pnl_atr, holding_bars = result
        entry_price = row["close"]
        pnl_pct = (exit_price - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_price) / entry_price

        trades_smart.append(TradeResult(
            entry_time=ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_atr=pnl_atr,
            regime=regime_state.regime.value,
            session_phase=cal_state.session_phase,
            is_tax_period=cal_state.is_tax_period,
            is_expiry_week=cal_state.is_expiry_week,
            risk_sentiment=risk_sentiment,
            holding_bars=holding_bars,
        ))

        i += holding_bars + 1

    results_smart = analyze_results(trades_smart)
    smart_wr = results_smart.win_rate

    logger.info(f"Trades: {results_smart.total_trades}")
    logger.info(f"Win Rate: {smart_wr:.1%}")
    logger.info(f"Avg PnL: {results_smart.avg_pnl_atr:+.2f} ATR")
    logger.info(f"Sharpe: {results_smart.sharpe_ratio:.2f}")
    logger.info(f"Improvement vs baseline: {smart_wr - results_baseline.win_rate:+.1%}")

    # Detailed breakdown
    logger.info("\n" + "-" * 40)
    logger.info("BY REGIME:")
    for regime, stats in sorted(results_full.regime_stats.items()):
        logger.info(f"  {regime}: {stats['trades']} trades, WR={stats['win_rate']:.1%}, Avg={stats['avg_pnl']:+.2f}")

    logger.info("\n" + "-" * 40)
    logger.info("BY SESSION:")
    for session, stats in sorted(results_full.session_stats.items()):
        logger.info(f"  {session}: {stats['trades']} trades, WR={stats['win_rate']:.1%}")

    logger.info("\n" + "-" * 40)
    logger.info("BY CALENDAR:")
    logger.info(f"  Tax period:     WR={results_full.tax_period_wr:.1%}")
    logger.info(f"  Non-tax period: WR={results_full.non_tax_period_wr:.1%}")
    logger.info(f"  Expiry week:    WR={results_full.expiry_week_wr:.1%}")
    logger.info(f"  Non-expiry:     WR={results_full.non_expiry_week_wr:.1%}")

    logger.info("\n" + "-" * 40)
    logger.info("BY RISK SENTIMENT:")
    logger.info(f"  Risk-on:  WR={results_full.risk_on_wr:.1%}")
    logger.info(f"  Risk-off: WR={results_full.risk_off_wr:.1%}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    baseline_wr = results_baseline.win_rate
    regime_wr = results_regime.win_rate
    full_wr = results_full.win_rate

    logger.info(f"Baseline:           {results_baseline.total_trades} trades, WR={baseline_wr:.1%}")
    logger.info(f"+ Regime filter:    {results_regime.total_trades} trades, WR={regime_wr:.1%} ({regime_wr - baseline_wr:+.1%})")
    logger.info(f"+ All Phase 3:      {results_full.total_trades} trades, WR={full_wr:.1%} ({full_wr - baseline_wr:+.1%})")
    logger.info(f"+ Smart Filter:     {results_smart.total_trades} trades, WR={smart_wr:.1%} ({smart_wr - baseline_wr:+.1%})")

    # Trade reduction
    trade_reduction = 1 - results_full.total_trades / results_baseline.total_trades if results_baseline.total_trades > 0 else 0
    logger.info(f"\nTrade reduction: {trade_reduction:.1%}")

    # Quality improvement
    if results_full.avg_pnl_atr > results_baseline.avg_pnl_atr:
        logger.info(f"Avg PnL improved: {results_baseline.avg_pnl_atr:+.2f} → {results_full.avg_pnl_atr:+.2f} ATR")

    # Recommendations
    logger.info("\n" + "-" * 40)
    logger.info("RECOMMENDATIONS:")

    # Best regime
    if results_full.regime_stats:
        best_regime = max(results_full.regime_stats.items(), key=lambda x: x[1]["win_rate"])
        worst_regime = min(results_full.regime_stats.items(), key=lambda x: x[1]["win_rate"])
        logger.info(f"  Best regime: {best_regime[0]} (WR={best_regime[1]['win_rate']:.1%})")
        logger.info(f"  Worst regime: {worst_regime[0]} (WR={worst_regime[1]['win_rate']:.1%})")
        if worst_regime[1]["win_rate"] < 0.4:
            logger.info(f"  → Consider skipping {worst_regime[0]}")

    # Tax period effect
    if results_full.tax_period_wr > 0 and results_full.non_tax_period_wr > 0:
        tax_diff = results_full.tax_period_wr - results_full.non_tax_period_wr
        if abs(tax_diff) > 0.05:
            better = "tax period" if tax_diff > 0 else "non-tax period"
            logger.info(f"  → {better} performs better ({abs(tax_diff):.1%} difference)")


if __name__ == "__main__":
    main()
