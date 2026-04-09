"""
MOEX Agent v2.5 OOS Backtest for Futures (BR, MX, RI)

Validates SmartFilter edge on futures using MOEX ISS API data.

Usage:
    python -m moex_agent.backtest_oos_futures
    python -m moex_agent.backtest_oos_futures --base BR --days 60
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)

# Futures contracts config
FUTURES_BASES = {
    "BR": {"name": "Brent", "min_dev": 0.3, "ema_period": 20},
    "MX": {"name": "MOEX Index", "min_dev": 0.15, "ema_period": 20},
    "RI": {"name": "RTS Index", "min_dev": 0.3, "ema_period": 20},
}


@dataclass
class OOSMetrics:
    """Metrics for comparison."""
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_pnl_atr: float

    def __str__(self):
        return (f"Trades: {self.n_trades:3d} | "
                f"WR: {self.win_rate*100:5.1f}% | "
                f"PF: {self.profit_factor:5.2f} | "
                f"AvgPnL: {self.avg_pnl_atr:+.3f} ATR")


def find_contracts(base: str) -> List[str]:
    """Find liquid contracts for a base symbol."""
    try:
        url = (
            "https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
            "?iss.meta=off&iss.only=marketdata&marketdata.columns=SECID,OPENPOSITION,VOLTODAY"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        rows = data.get("marketdata", {}).get("data", [])
        candidates = [(r[0], r[1] or 0, r[2] or 0) for r in rows if r[0] and r[0].startswith(base)]
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [c[0] for c in candidates[:3]]  # Top 3 by OI
    except Exception as e:
        log.warning(f"Error finding contracts for {base}: {e}")
        return []


def fetch_hourly_candles(secid: str, start_date: str, end_date: str) -> List[Dict]:
    """Fetch hourly candles from MOEX ISS."""
    candles = []
    try:
        current_start = start_date
        while current_start < end_date:
            url = (
                f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}/candles.json"
                f"?interval=60&from={current_start}&till={end_date}"
                f"&iss.meta=off&candles.columns=open,close,high,low,begin,value,volume"
            )
            data = json.loads(urllib.request.urlopen(url, timeout=20).read())
            batch = data.get("candles", {}).get("data", [])
            if not batch:
                break
            for row in batch:
                candles.append({
                    "open": row[0], "close": row[1], "high": row[2], "low": row[3],
                    "ts": row[4], "value": row[5], "volume": row[6], "secid": secid,
                })
            last_ts = batch[-1][4]
            try:
                last_dt = datetime.fromisoformat(last_ts)
                current_start = (last_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            except:
                break
            if len(batch) < 100:
                break
    except Exception as e:
        log.warning(f"Error fetching candles for {secid}: {e}")
    return candles


def calc_ema(closes: List[float], period: int = 20) -> float:
    """Calculate EMA."""
    if not closes:
        return 0.0
    ema = closes[0]
    k = 2 / (period + 1)
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def calc_atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate ATR."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period if trs else 0.0


def simulate_trade(
    candles: List[Dict],
    entry_idx: int,
    direction: str,
    atr: float,
    target_mult: float = 2.0,
    stop_mult: float = 1.0,
    max_bars: int = 20,
) -> Optional[Tuple[float, int]]:
    """Simulate trade. Returns (pnl_atr, bars_held)."""
    entry_price = candles[entry_idx]["close"]

    if direction == "LONG":
        target = entry_price + target_mult * atr
        stop = entry_price - stop_mult * atr
    else:
        target = entry_price - target_mult * atr
        stop = entry_price + stop_mult * atr

    for j in range(1, min(max_bars, len(candles) - entry_idx)):
        bar = candles[entry_idx + j]
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

    # Time exit
    if entry_idx + max_bars < len(candles):
        exit_price = candles[entry_idx + max_bars]["close"]
        pnl = (exit_price - entry_price) / atr if direction == "LONG" else (entry_price - exit_price) / atr
        return pnl, max_bars
    return None


def get_regime(candles: List[Dict], idx: int, lookback: int = 20) -> str:
    """Simple regime detection based on trend."""
    if idx < lookback:
        return "unknown"

    closes = [c["close"] for c in candles[idx-lookback:idx]]
    if len(closes) < lookback:
        return "unknown"

    trend = (closes[-1] - closes[0]) / closes[0] * 100
    volatility = np.std(np.diff(closes) / closes[:-1]) * 100

    if abs(trend) > 1.5:
        return "trend_up" if trend > 0 else "trend_down"
    elif volatility < 0.3:
        return "range_low_vol"
    else:
        return "range_high_vol"


def run_backtest_period(
    candles: List[Dict],
    base: str,
    min_dev: float,
    ema_period: int,
    use_filter: bool = False,
    regime_wr: Optional[Dict[str, float]] = None,
    min_regime_wr: float = 0.35,
) -> Tuple[List[float], Dict[str, Dict]]:
    """Run backtest on candle period."""
    pnls = []
    regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": []})

    i = ema_period + 5
    while i < len(candles) - 25:
        # Calculate EMA and deviation
        closes = [c["close"] for c in candles[i-ema_period:i]]
        ema = calc_ema(closes, ema_period)
        price = candles[i]["close"]

        if ema <= 0:
            i += 1
            continue

        dev = ((price - ema) / ema) * 100
        abs_dev = abs(dev)

        if abs_dev < min_dev:
            i += 1
            continue

        # Direction (mean reversion)
        direction = "SHORT" if dev > 0 else "LONG"

        # Get regime
        regime = get_regime(candles, i)

        # SmartFilter logic
        if use_filter and regime_wr:
            wr = regime_wr.get(regime, 0.5)
            if wr < min_regime_wr:
                i += 1
                continue

        # ATR
        atr = calc_atr(candles[max(0, i-20):i], 14)
        if atr <= 0:
            i += 1
            continue

        # Simulate trade
        result = simulate_trade(candles, i, direction, atr)
        if result is None:
            i += 1
            continue

        pnl_atr, bars_held = result
        pnls.append(pnl_atr)

        # Track regime stats
        regime_stats[regime]["trades"] += 1
        regime_stats[regime]["pnl"].append(pnl_atr)
        if pnl_atr > 0:
            regime_stats[regime]["wins"] += 1

        i += bars_held + 1

    # Compute regime WR
    for r in regime_stats:
        if regime_stats[r]["trades"] > 0:
            regime_stats[r]["win_rate"] = regime_stats[r]["wins"] / regime_stats[r]["trades"]

    return pnls, dict(regime_stats)


def compute_metrics(pnls: List[float]) -> OOSMetrics:
    """Compute metrics from PnL list."""
    if not pnls:
        return OOSMetrics(0, 0.0, 0.0, 0.0)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls)
    avg_pnl = np.mean(pnls)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    return OOSMetrics(len(pnls), win_rate, profit_factor, avg_pnl)


def run_oos_futures(base: str, days: int = 60, train_pct: float = 0.7):
    """Run OOS backtest for a futures base."""
    log.info(f"{'='*60}")
    log.info(f"OOS Futures: {base} ({days} days, {train_pct*100:.0f}% train)")
    log.info(f"{'='*60}")

    config = FUTURES_BASES.get(base)
    if not config:
        log.error(f"Unknown base: {base}")
        return None

    # Find contracts
    contracts = find_contracts(base)
    if not contracts:
        log.error(f"No contracts found for {base}")
        return None
    log.info(f"Contracts: {contracts}")

    # Fetch candles
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)

    all_candles = []
    for secid in contracts:
        candles = fetch_hourly_candles(secid, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if candles:
            all_candles.extend(candles)
            log.info(f"  {secid}: {len(candles)} candles")

    if not all_candles:
        log.error(f"No candles for {base}")
        return None

    # Sort and dedupe by timestamp
    all_candles.sort(key=lambda x: x["ts"])
    candles = []
    seen_ts = set()
    for c in all_candles:
        if c["ts"] not in seen_ts:
            seen_ts.add(c["ts"])
            candles.append(c)

    log.info(f"Total: {len(candles)} unique hourly candles")

    if len(candles) < 100:
        log.error(f"Not enough candles: {len(candles)}")
        return None

    # Split train/test
    split_idx = int(len(candles) * train_pct)
    train_candles = candles[:split_idx]
    test_candles = candles[split_idx:]

    log.info(f"Train: {len(train_candles)} candles ({train_candles[0]['ts']} to {train_candles[-1]['ts']})")
    log.info(f"Test:  {len(test_candles)} candles ({test_candles[0]['ts']} to {test_candles[-1]['ts']})")

    min_dev = config["min_dev"]
    ema_period = config["ema_period"]

    # === TRAIN ===
    log.info("\n--- TRAIN PHASE ---")

    train_pnls_base, train_regime_stats = run_backtest_period(
        train_candles, base, min_dev, ema_period, use_filter=False
    )
    train_baseline = compute_metrics(train_pnls_base)
    log.info(f"Train Baseline: {train_baseline}")

    # Extract regime WR
    regime_wr = {r: s.get("win_rate", 0.5) for r, s in train_regime_stats.items()}
    log.info(f"Regime WR (train): {regime_wr}")

    train_pnls_filt, _ = run_backtest_period(
        train_candles, base, min_dev, ema_period,
        use_filter=True, regime_wr=regime_wr
    )
    train_filtered = compute_metrics(train_pnls_filt)
    log.info(f"Train Filtered: {train_filtered}")

    # === TEST (OOS) ===
    log.info("\n--- TEST PHASE (OOS) ---")

    test_pnls_base, _ = run_backtest_period(
        test_candles, base, min_dev, ema_period, use_filter=False
    )
    test_baseline = compute_metrics(test_pnls_base)
    log.info(f"Test Baseline:  {test_baseline}")

    test_pnls_filt, _ = run_backtest_period(
        test_candles, base, min_dev, ema_period,
        use_filter=True, regime_wr=regime_wr  # Use TRAIN calibration
    )
    test_filtered = compute_metrics(test_pnls_filt)
    log.info(f"Test Filtered:  {test_filtered}")

    # Summary
    log.info("\n--- SUMMARY ---")
    train_delta = (train_filtered.win_rate - train_baseline.win_rate) * 100
    test_delta = (test_filtered.win_rate - test_baseline.win_rate) * 100

    log.info(f"Train WR improvement: {train_delta:+.1f}%")
    log.info(f"Test WR improvement:  {test_delta:+.1f}%")

    is_overfit = test_filtered.win_rate < test_baseline.win_rate - 0.02
    edge_confirmed = test_filtered.win_rate > test_baseline.win_rate + 0.01

    if is_overfit:
        log.warning("OVERFIT: Test worse than baseline")
    elif edge_confirmed:
        log.info("EDGE CONFIRMED: Test beats baseline")
    else:
        log.info("MARGINAL: No clear edge")

    return {
        "base": base,
        "train_baseline": train_baseline,
        "train_filtered": train_filtered,
        "test_baseline": test_baseline,
        "test_filtered": test_filtered,
        "is_overfit": is_overfit,
        "edge_confirmed": edge_confirmed,
    }


def main():
    parser = argparse.ArgumentParser(description="OOS Backtest for Futures")
    parser.add_argument("--base", type=str, default=None, help="Single base (BR, MX, RI)")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--train-pct", type=float, default=0.7)
    args = parser.parse_args()

    bases = [args.base] if args.base else ["BR", "MX", "RI"]

    results = []
    for base in bases:
        result = run_oos_futures(base, args.days, args.train_pct)
        if result:
            results.append(result)

    # Final summary
    if results:
        print("\n" + "="*70)
        print("FUTURES OOS SUMMARY")
        print("="*70)
        print(f"{'Base':<6} {'Train WR':>10} {'Test WR':>10} {'Delta':>8} {'Status':<12}")
        print("-"*50)

        for r in results:
            delta = (r["test_filtered"].win_rate - r["test_baseline"].win_rate) * 100
            status = "OVERFIT" if r["is_overfit"] else ("EDGE" if r["edge_confirmed"] else "MARGINAL")
            print(f"{r['base']:<6} {r['train_filtered'].win_rate*100:>9.1f}% "
                  f"{r['test_filtered'].win_rate*100:>9.1f}% {delta:>+7.1f}% {status:<12}")

        print("-"*50)
        n_edge = sum(1 for r in results if r["edge_confirmed"])
        n_overfit = sum(1 for r in results if r["is_overfit"])
        print(f"Edge: {n_edge}/{len(results)}, Overfit: {n_overfit}/{len(results)}")


if __name__ == "__main__":
    main()
