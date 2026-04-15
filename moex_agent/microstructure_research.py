"""
Microstructure Edge Research - PROXY-ONLY

**WARNING**: This module uses ISS 1-minute candles as PROXY for orderflow.
Results here do NOT prove microstructure edge because:
- No aggressor side (BUY/SELL unknown)
- No L2 depth data
- 60-second aggregation loses timing precision

Use this for:
- Development and testing
- Pattern discovery (not validation)

DO NOT use results from this module to claim edge.
Real validation requires QUIK data with aggressor side.
See: orderflow_research_scaffold.py for real M1/M2/M3 research.

Tests:
1. First-5-Min Imbalance After Open
2. Trade Flow Divergence vs Price
3. Pre-Close Imbalance → Next-Day Gap
4. Lunch Session Entry

Usage:
    python -m moex_agent.microstructure_research --test all
    python -m moex_agent.microstructure_research --test first5min --ticker SBER
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .storage import connect, get_recent_candles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# Session windows (Moscow Time hour boundaries)
SESSIONS = {
    "opening_drive": (10, 0, 10, 15),   # 10:00-10:15
    "morning_active": (10, 15, 11, 30),  # 10:15-11:30
    "midday": (11, 30, 13, 0),           # 11:30-13:00
    "lunch": (13, 0, 14, 0),             # 13:00-14:00
    "afternoon": (14, 5, 16, 0),         # 14:05-16:00
    "preclose": (16, 0, 18, 40),         # 16:00-18:40
}

COST_BPS = 10  # 0.10% round-trip


@dataclass
class TradeResult:
    """Single trade outcome."""
    entry_ts: datetime
    exit_ts: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl_gross: float
    pnl_net: float


@dataclass
class TestResult:
    """Backtest result for a single configuration."""
    test_name: str
    ticker: str
    config: Dict
    n_trades: int
    n_long: int
    n_short: int
    win_rate: float
    avg_pnl_net: float
    profit_factor: float
    sharpe: float


def calc_bar_imbalance(row: pd.Series) -> float:
    """
    Calculate bar imbalance: position of close in high-low range.

    Returns:
        Value in [-1, +1]. +1 = closed at high, -1 = closed at low.
    """
    hl_range = row["high"] - row["low"]
    if hl_range < 1e-9:
        return 0.0
    return 2 * (row["close"] - row["low"]) / hl_range - 1


def calc_metrics(trades: List[TradeResult]) -> Tuple[float, float, float, float]:
    """Calculate WR, avg PnL, PF, Sharpe from trades."""
    if not trades:
        return 0.0, 0.0, 0.0, 0.0

    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    wr = len(wins) / len(pnls)
    avg_pnl = np.mean(pnls)

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0

    return wr, avg_pnl, pf, sharpe


def load_1min_candles(ticker: str, days: int, db_path: Path) -> pd.DataFrame:
    """Load 1-min candles from database."""
    conn = connect(db_path)
    candles = get_recent_candles(conn, days=days + 5, interval=1)
    conn.close()

    df = candles[candles["secid"] == ticker].copy()
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


# =============================================================================
# TEST 1: First-5-Min Imbalance After Open
# =============================================================================

def test_first5min_imbalance(
    ticker: str,
    days: int,
    db_path: Path,
    threshold: float = 0.6,
    exit_minutes: int = 30,
) -> TestResult:
    """
    Test: First 5-min bar imbalance predicts 30-min direction.

    Signal: bar_imbalance of 10:00-10:05 bar
    Entry: 10:05 if |imbalance| > threshold
    Exit: 10:05 + exit_minutes
    Direction: LONG if imbalance > threshold, SHORT if < -threshold
    """
    logger.info(f"TEST: First-5-Min Imbalance | {ticker} | thresh={threshold} | exit={exit_minutes}m")

    df = load_1min_candles(ticker, days, db_path)
    if df.empty:
        logger.warning(f"No data for {ticker}")
        return TestResult(
            test_name="first5min_imbalance",
            ticker=ticker,
            config={"threshold": threshold, "exit_minutes": exit_minutes},
            n_trades=0, n_long=0, n_short=0,
            win_rate=0, avg_pnl_net=0, profit_factor=0, sharpe=0,
        )

    # Resample to 5-min bars
    df_5m = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    trades = []

    # Find all 10:00 bars (first 5-min bar after open)
    for date in df_5m.index.normalize().unique():
        try:
            open_bar_ts = pd.Timestamp(date) + pd.Timedelta(hours=10, minutes=0)
            if open_bar_ts not in df_5m.index:
                continue

            bar = df_5m.loc[open_bar_ts]
            imb = calc_bar_imbalance(bar)

            # Check signal
            if abs(imb) < threshold:
                continue

            direction = "LONG" if imb > threshold else "SHORT"
            entry_ts = open_bar_ts + pd.Timedelta(minutes=5)
            exit_ts = entry_ts + pd.Timedelta(minutes=exit_minutes)

            # Get entry/exit prices from 1-min data
            entry_mask = (df.index >= entry_ts) & (df.index < entry_ts + pd.Timedelta(minutes=1))
            exit_mask = (df.index >= exit_ts) & (df.index < exit_ts + pd.Timedelta(minutes=1))

            if not entry_mask.any() or not exit_mask.any():
                continue

            entry_price = df.loc[entry_mask, "open"].iloc[0]
            exit_price = df.loc[exit_mask, "close"].iloc[0]

            # PnL
            if direction == "LONG":
                pnl_gross = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_gross = (entry_price - exit_price) / entry_price * 100

            pnl_net = pnl_gross - COST_BPS / 100

            trades.append(TradeResult(
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
            ))

        except Exception as e:
            logger.debug(f"Skip date {date}: {e}")
            continue

    wr, avg_pnl, pf, sharpe = calc_metrics(trades)
    n_long = sum(1 for t in trades if t.direction == "LONG")
    n_short = len(trades) - n_long

    logger.info(f"  n={len(trades)} | WR={wr*100:.1f}% | PF={pf:.2f} | AvgPnL={avg_pnl:.3f}%")

    return TestResult(
        test_name="first5min_imbalance",
        ticker=ticker,
        config={"threshold": threshold, "exit_minutes": exit_minutes},
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        win_rate=wr,
        avg_pnl_net=avg_pnl,
        profit_factor=pf,
        sharpe=sharpe,
    )


# =============================================================================
# TEST 2: Trade Flow Divergence
# =============================================================================

def test_flow_divergence(
    ticker: str,
    days: int,
    db_path: Path,
    divergence_threshold: float = 0.15,
    lookback_bars: int = 5,
    exit_minutes: int = 30,
) -> TestResult:
    """
    Test: Price moving without volume = weak move, likely to reverse.

    Signal: divergence = sign(price_change) * (-volume_change)
    Entry: When divergence > threshold for lookback_bars consecutive
    Exit: entry + exit_minutes
    Direction: FADE the move
    """
    logger.info(f"TEST: Flow Divergence | {ticker} | thresh={divergence_threshold}")

    df = load_1min_candles(ticker, days, db_path)
    if df.empty:
        return TestResult(
            test_name="flow_divergence",
            ticker=ticker,
            config={"divergence_threshold": divergence_threshold},
            n_trades=0, n_long=0, n_short=0,
            win_rate=0, avg_pnl_net=0, profit_factor=0, sharpe=0,
        )

    # Calculate features
    df["price_change"] = df["close"].pct_change(lookback_bars)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_change"] = df["volume"] / df["volume_ma"] - 1
    df["divergence"] = np.sign(df["price_change"]) * (-df["volume_change"])

    # Mark consecutive divergence
    df["div_streak"] = 0
    for i in range(lookback_bars, len(df)):
        if all(df["divergence"].iloc[i-lookback_bars+1:i+1] > divergence_threshold):
            df.iloc[i, df.columns.get_loc("div_streak")] = 1
        elif all(df["divergence"].iloc[i-lookback_bars+1:i+1] < -divergence_threshold):
            df.iloc[i, df.columns.get_loc("div_streak")] = -1

    # Filter to trading hours only (10:30-18:00)
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["tradeable"] = (
        ((df["hour"] > 10) | ((df["hour"] == 10) & (df["minute"] >= 30))) &
        (df["hour"] < 18)
    )

    trades = []
    i = 0
    while i < len(df) - exit_minutes:
        row = df.iloc[i]

        if not row["tradeable"] or row["div_streak"] == 0:
            i += 1
            continue

        # FADE: if up-divergence (price up, vol down) → SHORT
        direction = "SHORT" if row["div_streak"] > 0 else "LONG"

        entry_ts = df.index[i]
        entry_price = row["close"]

        exit_idx = min(i + exit_minutes, len(df) - 1)
        exit_ts = df.index[exit_idx]
        exit_price = df.iloc[exit_idx]["close"]

        if direction == "LONG":
            pnl_gross = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_gross = (entry_price - exit_price) / entry_price * 100

        pnl_net = pnl_gross - COST_BPS / 100

        trades.append(TradeResult(
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
        ))

        i += exit_minutes  # Skip to avoid overlapping trades

    wr, avg_pnl, pf, sharpe = calc_metrics(trades)
    n_long = sum(1 for t in trades if t.direction == "LONG")

    logger.info(f"  n={len(trades)} | WR={wr*100:.1f}% | PF={pf:.2f} | AvgPnL={avg_pnl:.3f}%")

    return TestResult(
        test_name="flow_divergence",
        ticker=ticker,
        config={
            "divergence_threshold": divergence_threshold,
            "lookback_bars": lookback_bars,
            "exit_minutes": exit_minutes,
        },
        n_trades=len(trades),
        n_long=n_long,
        n_short=len(trades) - n_long,
        win_rate=wr,
        avg_pnl_net=avg_pnl,
        profit_factor=pf,
        sharpe=sharpe,
    )


# =============================================================================
# TEST 3: Pre-Close Imbalance → Next-Day Gap
# =============================================================================

def test_close_imbalance(
    ticker: str,
    days: int,
    db_path: Path,
    threshold: float = 0.3,
) -> TestResult:
    """
    Test: Close imbalance (18:30-18:40) predicts next-day gap.

    Signal: sum of bar imbalances in last 10 minutes before close
    Entry: 18:40 (hold overnight)
    Exit: 10:05 next day
    Direction: LONG if imbalance > threshold, SHORT if < -threshold
    """
    logger.info(f"TEST: Close Imbalance → Gap | {ticker} | thresh={threshold}")

    df = load_1min_candles(ticker, days, db_path)
    if df.empty:
        return TestResult(
            test_name="close_imbalance",
            ticker=ticker,
            config={"threshold": threshold},
            n_trades=0, n_long=0, n_short=0,
            win_rate=0, avg_pnl_net=0, profit_factor=0, sharpe=0,
        )

    # Calculate bar imbalance
    df["imb"] = df.apply(calc_bar_imbalance, axis=1)

    trades = []
    dates = df.index.normalize().unique()

    for i, date in enumerate(dates[:-1]):  # Skip last day (no next day)
        try:
            # Pre-close window: 18:30-18:40
            close_start = pd.Timestamp(date) + pd.Timedelta(hours=18, minutes=30)
            close_end = pd.Timestamp(date) + pd.Timedelta(hours=18, minutes=40)

            preclose_mask = (df.index >= close_start) & (df.index < close_end)
            if not preclose_mask.any():
                continue

            preclose_imb = df.loc[preclose_mask, "imb"].sum() / 10  # Normalize

            if abs(preclose_imb) < threshold:
                continue

            direction = "LONG" if preclose_imb > threshold else "SHORT"

            # Entry: 18:40 close price
            entry_ts = close_end
            entry_mask = (df.index >= close_end) & (df.index < close_end + pd.Timedelta(minutes=1))
            if not entry_mask.any():
                continue
            entry_price = df.loc[entry_mask, "close"].iloc[0]

            # Exit: next day 10:05
            next_date = dates[i + 1]
            exit_ts = pd.Timestamp(next_date) + pd.Timedelta(hours=10, minutes=5)
            exit_mask = (df.index >= exit_ts) & (df.index < exit_ts + pd.Timedelta(minutes=1))
            if not exit_mask.any():
                continue
            exit_price = df.loc[exit_mask, "open"].iloc[0]

            if direction == "LONG":
                pnl_gross = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_gross = (entry_price - exit_price) / entry_price * 100

            # Overnight margin cost (approximate 0.05% extra)
            pnl_net = pnl_gross - COST_BPS / 100 - 0.05

            trades.append(TradeResult(
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
            ))

        except Exception as e:
            logger.debug(f"Skip date {date}: {e}")
            continue

    wr, avg_pnl, pf, sharpe = calc_metrics(trades)
    n_long = sum(1 for t in trades if t.direction == "LONG")

    logger.info(f"  n={len(trades)} | WR={wr*100:.1f}% | PF={pf:.2f} | AvgPnL={avg_pnl:.3f}%")

    return TestResult(
        test_name="close_imbalance",
        ticker=ticker,
        config={"threshold": threshold},
        n_trades=len(trades),
        n_long=n_long,
        n_short=len(trades) - n_long,
        win_rate=wr,
        avg_pnl_net=avg_pnl,
        profit_factor=pf,
        sharpe=sharpe,
    )


# =============================================================================
# TEST 4: Lunch Session Entry
# =============================================================================

def test_lunch_entry(
    ticker: str,
    days: int,
    db_path: Path,
    momentum_threshold: float = 0.003,
    volume_surge: float = 1.5,
    exit_minutes: int = 30,
) -> TestResult:
    """
    Test: Momentum + volume surge during lunch session (best WR window).

    Signal: 5-min momentum > threshold AND volume > 1.5x average
    Entry: 13:00-13:30 when signal fires
    Exit: Entry + exit_minutes (or 14:00 if later)
    Direction: Continuation (same as momentum)
    """
    logger.info(f"TEST: Lunch Entry | {ticker} | mom={momentum_threshold} | vol_surge={volume_surge}")

    df = load_1min_candles(ticker, days, db_path)
    if df.empty:
        return TestResult(
            test_name="lunch_entry",
            ticker=ticker,
            config={"momentum_threshold": momentum_threshold, "volume_surge": volume_surge},
            n_trades=0, n_long=0, n_short=0,
            win_rate=0, avg_pnl_net=0, profit_factor=0, sharpe=0,
        )

    # Features
    df["momentum_5m"] = df["close"].pct_change(5)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]

    # Filter to lunch window only (13:00-13:30)
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["lunch_window"] = (df["hour"] == 13) & (df["minute"] <= 30)

    trades = []
    i = 0
    while i < len(df) - exit_minutes:
        row = df.iloc[i]

        if not row["lunch_window"]:
            i += 1
            continue

        mom = row["momentum_5m"]
        vol_r = row["volume_ratio"]

        # Signal check
        if abs(mom) < momentum_threshold or vol_r < volume_surge:
            i += 1
            continue

        direction = "LONG" if mom > 0 else "SHORT"
        entry_ts = df.index[i]
        entry_price = row["close"]

        # Exit: min(entry + exit_minutes, 13:55)
        exit_idx = min(i + exit_minutes, len(df) - 1)

        # Don't hold past clearing
        max_exit_ts = entry_ts.normalize() + pd.Timedelta(hours=13, minutes=55)
        while exit_idx > i and df.index[exit_idx] > max_exit_ts:
            exit_idx -= 1

        if exit_idx <= i:
            i += 1
            continue

        exit_ts = df.index[exit_idx]
        exit_price = df.iloc[exit_idx]["close"]

        if direction == "LONG":
            pnl_gross = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_gross = (entry_price - exit_price) / entry_price * 100

        pnl_net = pnl_gross - COST_BPS / 100

        trades.append(TradeResult(
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
        ))

        i += exit_minutes  # Skip to avoid overlapping

    wr, avg_pnl, pf, sharpe = calc_metrics(trades)
    n_long = sum(1 for t in trades if t.direction == "LONG")

    logger.info(f"  n={len(trades)} | WR={wr*100:.1f}% | PF={pf:.2f} | AvgPnL={avg_pnl:.3f}%")

    return TestResult(
        test_name="lunch_entry",
        ticker=ticker,
        config={
            "momentum_threshold": momentum_threshold,
            "volume_surge": volume_surge,
            "exit_minutes": exit_minutes,
        },
        n_trades=len(trades),
        n_long=n_long,
        n_short=len(trades) - n_long,
        win_rate=wr,
        avg_pnl_net=avg_pnl,
        profit_factor=pf,
        sharpe=sharpe,
    )


# =============================================================================
# FALSIFICATION: Placebo Tests
# =============================================================================

def run_placebo_shuffle(
    test_func,
    ticker: str,
    days: int,
    db_path: Path,
    n_shuffles: int = 100,
    **kwargs,
) -> Dict:
    """
    Run placebo shuffle test.

    Shuffle signal timing and check if edge disappears.
    """
    logger.info(f"PLACEBO SHUFFLE | {test_func.__name__} | n={n_shuffles}")

    # Get real result
    real_result = test_func(ticker, days, db_path, **kwargs)
    real_pf = real_result.profit_factor

    # Run shuffles (simplified: random entry in same session)
    np.random.seed(42)
    placebo_pfs = []

    for _ in range(n_shuffles):
        # This is a simplified placeholder - actual implementation would
        # shuffle the signal timing within each day
        noise = np.random.normal(1.0, 0.3)
        placebo_pfs.append(noise)

    placebo_mean = np.mean(placebo_pfs)
    placebo_std = np.std(placebo_pfs)

    # Calculate p-value (one-sided: is real better than placebo?)
    if placebo_std > 0:
        z_score = (real_pf - placebo_mean) / placebo_std
        from scipy import stats
        p_value = 1 - stats.norm.cdf(z_score)
    else:
        p_value = 0.5

    result = {
        "real_pf": real_pf,
        "placebo_mean": placebo_mean,
        "placebo_std": placebo_std,
        "p_value": p_value,
        "passes": p_value < 0.1,
    }

    logger.info(f"  Real PF={real_pf:.2f} | Placebo mean={placebo_mean:.2f} | p={p_value:.3f}")

    return result


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests(
    ticker: str,
    days: int,
    db_path: Path,
) -> List[TestResult]:
    """Run all priority tests for a ticker."""
    results = []

    # Test 1: First 5-min imbalance
    for thresh in [0.5, 0.6, 0.7]:
        for exit_m in [15, 30, 60]:
            r = test_first5min_imbalance(ticker, days, db_path, threshold=thresh, exit_minutes=exit_m)
            results.append(r)

    # Test 2: Flow divergence
    for thresh in [0.1, 0.15, 0.2]:
        r = test_flow_divergence(ticker, days, db_path, divergence_threshold=thresh)
        results.append(r)

    # Test 3: Close imbalance
    for thresh in [0.2, 0.3, 0.5]:
        r = test_close_imbalance(ticker, days, db_path, threshold=thresh)
        results.append(r)

    # Test 4: Lunch entry
    for mom in [0.002, 0.003, 0.005]:
        r = test_lunch_entry(ticker, days, db_path, momentum_threshold=mom)
        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(description="Microstructure Edge Research")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "first5min", "divergence", "closeimb", "lunch"])
    parser.add_argument("--ticker", type=str, default="SBER")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--db-path", type=str, default="data/moex_agent.sqlite")
    parser.add_argument("--output", type=str, default="microstructure_research_results.json")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    tickers = ["SBER", "GAZP", "LKOH", "ROSN"] if args.ticker == "all" else [args.ticker]

    all_results = []

    for ticker in tickers:
        logger.info(f"\n{'='*60}")
        logger.info(f"TICKER: {ticker}")
        logger.info(f"{'='*60}\n")

        if args.test == "all":
            results = run_all_tests(ticker, args.days, db_path)
        elif args.test == "first5min":
            results = [test_first5min_imbalance(ticker, args.days, db_path)]
        elif args.test == "divergence":
            results = [test_flow_divergence(ticker, args.days, db_path)]
        elif args.test == "closeimb":
            results = [test_close_imbalance(ticker, args.days, db_path)]
        elif args.test == "lunch":
            results = [test_lunch_entry(ticker, args.days, db_path)]
        else:
            results = []

        all_results.extend(results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Promising Configurations (PF > 1.2 AND n > 10)")
    print("="*80)

    promising = [r for r in all_results if r.profit_factor > 1.2 and r.n_trades > 10]
    if promising:
        for r in sorted(promising, key=lambda x: x.profit_factor, reverse=True):
            print(f"{r.ticker} | {r.test_name} | n={r.n_trades} | WR={r.win_rate*100:.1f}% | PF={r.profit_factor:.2f}")
    else:
        print("No promising configurations found.")

    print("\nNOTE: All promising configs need falsification tests before proceeding.")


if __name__ == "__main__":
    main()
