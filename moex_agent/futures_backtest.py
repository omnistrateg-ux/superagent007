#!/usr/bin/env python3
"""
Futures Backtester — Mean Reversion Strategy with Tactical Improvements

Tests the paper_futures.py strategy on historical MOEX futures data.
Includes: tiered trailing, partial take profit, kill losers fast, smart time stop.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("futures_backtest")

MSK = timezone(timedelta(hours=3))

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATION (mirrored from paper_futures.py)
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_BALANCE = 10_000_000
MAX_CONTRACTS = 10
MAX_LOSS_PER_TRADE = 15000
STOP_PCT = 2.0
TARGET_RR = 2.0

# Trailing 2.0 — "Let Winners Run"
TRAIL_TIERS = {
    15: (50, "ЗАЩИТА"),
    30: (35, "ТРЕЙЛ_1"),
    50: (25, "ТРЕЙЛ_2"),
    70: (15, "ТРЕЙЛ_3"),
    90: (8,  "ЦЕЛЬ_БЛИЗКО"),
}
TRAIL_ACTIVATE_PCT = 15

# Partial Take Profit
PARTIAL_TAKE_ENABLED = True
PARTIAL_TAKE_LEVELS = [
    (50, 30),  # 50% progress → close 30%
    (75, 30),  # 75% progress → close 30% more
]

# Kill Losers Fast
LOSER_TIERS = {
    5000:  50,
    10000: 75,
}

# Smart Time Stop
SMART_TIME_STOP = True

# Contract specs
CONTRACTS = {
    "BR": {"name": "Brent",   "lot": 10,   "tick": 0.01,  "tick_val": 6.55, "margin_pct": 15,
           "min_dev": 0.5, "side_mode": "both", "time_stop_bars": 3, "max_dev": 3.0},
    "RI": {"name": "RTS",     "lot": 1,    "tick": 10.0,  "tick_val": 10.0, "margin_pct": 12,
           "min_dev": 0.5, "side_mode": "both", "time_stop_bars": 3, "max_dev": 3.0},
    "NG": {"name": "Gas",     "lot": 100,  "tick": 0.001, "tick_val": 6.55, "margin_pct": 20,
           "min_dev": 0.5, "side_mode": "both", "time_stop_bars": 4, "max_dev": 3.0},
    "MX": {"name": "MOEX",    "lot": 1,    "tick": 1.0,   "tick_val": 1.0,  "margin_pct": 12,
           "min_dev": 0.15, "side_mode": "short_only", "time_stop_bars": 2, "max_dev": 2.0},
}


# ══════════════════════════════════════════════════════════════════════════════
# MOEX ISS DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def find_active_contracts(base: str, start_date: str, end_date: str) -> List[str]:
    """Find active contract symbols for a base within date range."""
    try:
        url = (
            f"https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
            f"?iss.meta=off&iss.only=securities"
            f"&securities.columns=SECID,SHORTNAME"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        rows = data.get("securities", {}).get("data", [])
        contracts = [r[0] for r in rows if r[0] and r[0].startswith(base)]
        return contracts[:3]  # Most recent 3 contracts
    except Exception as e:
        log.warning(f"Error finding contracts for {base}: {e}")
        return []


def fetch_hourly_candles(secid: str, start_date: str, end_date: str) -> List[Dict]:
    """Fetch hourly candles from MOEX ISS."""
    candles = []
    try:
        # MOEX ISS paginates; fetch in chunks
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
                    "open": row[0],
                    "close": row[1],
                    "high": row[2],
                    "low": row[3],
                    "ts": row[4],
                    "value": row[5],
                    "volume": row[6],
                })
            # Next page: start from last candle + 1h
            last_ts = batch[-1][4]
            try:
                last_dt = datetime.fromisoformat(last_ts)
                current_start = (last_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")  # ISO format, no spaces
            except:
                break
            if len(batch) < 100:
                break
    except Exception as e:
        log.warning(f"Error fetching candles for {secid}: {e}")
    return candles


def calc_ema(closes: List[float], period: int = 20) -> float:
    """Calculate EMA from closes."""
    if not closes:
        return 0.0
    ema = closes[0]
    k = 2 / (period + 1)
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def calc_atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate ATR from candles."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period if trs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST TRADE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    """Single backtest trade record."""
    id: int
    base: str
    secid: str
    direction: str
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_reason: str
    qty: int
    pnl_rub: float
    pnl_pct: float
    dev_at_entry: float
    bars_held: int
    partial_takes: int = 0
    loser_cuts: int = 0


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FUTURES BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

class FuturesBacktester:
    """
    Backtester for mean reversion futures strategy.

    Simulates the full strategy including:
    - EMA deviation entry
    - Tiered trailing stop
    - Partial take profit
    - Kill losers fast
    - Smart time stop
    """

    def __init__(self, mode: str = "new"):
        """
        Args:
            mode: "old" (basic trailing) or "new" (tactical improvements)
        """
        self.mode = mode
        self.balance = INITIAL_BALANCE
        self.trades: List[BacktestTrade] = []
        self.trade_id = 0
        self.equity_curve = [INITIAL_BALANCE]

    def _calc_pnl(self, spec: Dict, entry: float, exit: float, direction: str, qty: int) -> float:
        """Calculate PnL in RUB."""
        ticks = (exit - entry) / spec["tick"]
        if direction == "SHORT":
            ticks = -ticks
        return ticks * spec["tick_val"] * qty

    def _get_progress(self, entry: float, current: float, target: float, direction: str) -> float:
        """Get progress toward target (0-100%)."""
        if direction == "LONG":
            progress = (current - entry) / (target - entry) if target != entry else 0
        else:
            progress = (entry - current) / (entry - target) if target != entry else 0
        return max(0, min(1, progress)) * 100

    def _update_trailing_old(self, pos: Dict, price: float) -> Dict:
        """Old trailing logic: simple activation + fixed cushion."""
        direction = pos["direction"]
        entry = pos["entry"]
        target = pos["target"]

        # Update best price
        best = pos.get("best_price", price)
        if direction == "LONG":
            if price > best:
                pos["best_price"] = price
        else:
            if price < best:
                pos["best_price"] = price
        best = pos["best_price"]

        # Progress
        progress = self._get_progress(entry, price, target, direction)

        # Activate at 30%
        if progress >= 30 and not pos.get("trail_active"):
            pos["trail_active"] = True
            pos["stop"] = entry  # Breakeven

        if pos.get("trail_active"):
            distance = abs(best - entry)
            cushion = distance * 0.30  # Fixed 30% cushion
            if direction == "LONG":
                new_stop = best - cushion
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop
            else:
                new_stop = best + cushion
                if new_stop < pos["stop"]:
                    pos["stop"] = new_stop

        return pos

    def _update_trailing_new(self, pos: Dict, price: float) -> Dict:
        """New trailing logic: tiered cushion that tightens as profit grows."""
        direction = pos["direction"]
        entry = pos["entry"]
        target = pos["target"]

        # Update best price
        best = pos.get("best_price", price)
        if direction == "LONG":
            if price > best:
                pos["best_price"] = price
        else:
            if price < best:
                pos["best_price"] = price
        best = pos["best_price"]

        # Progress
        progress = self._get_progress(entry, price, target, direction)

        # Early breakeven at 0.3% profit
        profit_pct = abs(best - entry) / entry * 100 if entry else 0
        if profit_pct > 0.3 and not pos.get("trail_active"):
            pos["trail_active"] = True
            pos["stop"] = entry

        # Activate at TRAIL_ACTIVATE_PCT
        if progress >= TRAIL_ACTIVATE_PCT and not pos.get("trail_active"):
            pos["trail_active"] = True
            pos["stop"] = entry

        if pos.get("trail_active"):
            distance = abs(best - entry)
            target_dist = abs(target - entry)
            best_progress = (distance / target_dist * 100) if target_dist else 0

            # Find appropriate tier
            cushion_pct = 30  # Default
            for tier_progress, (tier_cushion, _) in sorted(TRAIL_TIERS.items(), reverse=True):
                if best_progress >= tier_progress:
                    cushion_pct = tier_cushion
                    break

            cushion = distance * cushion_pct / 100
            if direction == "LONG":
                new_stop = best - cushion
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop
            else:
                new_stop = best + cushion
                if new_stop < pos["stop"]:
                    pos["stop"] = new_stop

        return pos

    def _check_partial_take(self, pos: Dict, price: float, spec: Dict) -> Tuple[float, int]:
        """Check and execute partial take profit. Returns (realized_pnl, qty_closed)."""
        if not PARTIAL_TAKE_ENABLED or self.mode == "old":
            return 0.0, 0

        progress = self._get_progress(pos["entry"], price, pos["target"], pos["direction"])
        taken_levels = pos.get("partial_taken_levels", set())
        total_closed = 0
        total_pnl = 0.0

        for level_progress, close_pct in PARTIAL_TAKE_LEVELS:
            if level_progress in taken_levels:
                continue
            if progress >= level_progress:
                qty_to_close = max(1, int(pos["qty"] * close_pct / 100))
                if qty_to_close > 0 and pos["qty"] - qty_to_close > 0:
                    pnl = self._calc_pnl(spec, pos["entry"], price, pos["direction"], qty_to_close)
                    pos["qty"] -= qty_to_close
                    total_closed += qty_to_close
                    total_pnl += pnl
                    taken_levels.add(level_progress)

        pos["partial_taken_levels"] = taken_levels
        return total_pnl, total_closed

    def _check_kill_loser(self, pos: Dict, price: float, spec: Dict) -> Tuple[float, int]:
        """Check and execute loser cuts. Returns (realized_loss, qty_closed)."""
        if self.mode == "old":
            return 0.0, 0

        current_pnl = self._calc_pnl(spec, pos["entry"], price, pos["direction"], pos["qty"])
        if current_pnl >= 0:
            return 0.0, 0

        loss = abs(current_pnl)
        killed_levels = pos.get("killed_levels", set())
        total_closed = 0
        total_loss = 0.0

        for loss_threshold, total_close_pct in sorted(LOSER_TIERS.items()):
            if loss_threshold in killed_levels:
                continue
            if loss >= loss_threshold:
                prev_closed_pct = sum(LOSER_TIERS.get(l, 0) for l in killed_levels)
                close_pct = total_close_pct - prev_closed_pct
                if close_pct > 0:
                    qty_to_close = max(1, int(pos["qty"] * close_pct / 100))
                    if qty_to_close > 0 and pos["qty"] - qty_to_close > 0:
                        pnl = self._calc_pnl(spec, pos["entry"], price, pos["direction"], qty_to_close)
                        pos["qty"] -= qty_to_close
                        total_closed += qty_to_close
                        total_loss += pnl
                        killed_levels.add(loss_threshold)

        pos["killed_levels"] = killed_levels
        return total_loss, total_closed

    def _should_time_exit(self, pos: Dict, price: float, spec: Dict, bars_held: int) -> bool:
        """Check if position should exit due to time."""
        time_limit = spec.get("time_stop_bars", 4)

        if self.mode == "new" and SMART_TIME_STOP:
            current_pnl = self._calc_pnl(spec, pos["entry"], price, pos["direction"], pos["qty"])
            if current_pnl > 0:
                # Don't time-exit winners, give 50% more time
                return bars_held >= int(time_limit * 1.5)
            else:
                return bars_held >= time_limit
        else:
            return bars_held >= time_limit

    def simulate_trade(
        self,
        base: str,
        secid: str,
        direction: str,
        entry_idx: int,
        candles: List[Dict],
        ema: float,
        atr: float,
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade through candle data."""
        spec = CONTRACTS[base]
        entry_price = candles[entry_idx]["close"]

        # Calculate stop/target
        stop_dist = atr * 1.5
        target_dist = stop_dist * TARGET_RR

        if direction == "LONG":
            stop = entry_price - stop_dist
            target = entry_price + target_dist
        else:
            stop = entry_price + stop_dist
            target = entry_price - target_dist

        # Position size: based on risk
        qty = min(MAX_CONTRACTS, max(1, int(MAX_LOSS_PER_TRADE / (stop_dist * spec["tick_val"] / spec["tick"]))))

        # Dev at entry
        dev = ((entry_price - ema) / ema * 100) if ema else 0

        pos = {
            "direction": direction,
            "entry": entry_price,
            "stop": stop,
            "target": target,
            "qty": qty,
            "best_price": entry_price,
            "trail_active": False,
            "partial_taken_levels": set(),
            "killed_levels": set(),
        }

        partial_takes = 0
        loser_cuts = 0
        realized_pnl = 0.0

        time_limit = spec.get("time_stop_bars", 4)
        max_bars = min(time_limit * 2, len(candles) - entry_idx - 1)  # Safety limit

        exit_price = entry_price
        exit_reason = "TIME"
        bars_held = 0

        for i in range(1, max_bars + 1):
            if entry_idx + i >= len(candles):
                break

            bar = candles[entry_idx + i]
            price = bar["close"]
            high = bar["high"]
            low = bar["low"]
            bars_held = i

            # Update trailing
            if self.mode == "new":
                self._update_trailing_new(pos, price)
            else:
                self._update_trailing_old(pos, price)

            # Check stop
            hit_stop = False
            if direction == "LONG":
                if low <= pos["stop"]:
                    hit_stop = True
                    exit_price = pos["stop"]
            else:
                if high >= pos["stop"]:
                    hit_stop = True
                    exit_price = pos["stop"]

            if hit_stop:
                exit_reason = "STOP" if not pos.get("trail_active") else "TRAIL"
                break

            # Check target
            hit_target = False
            if direction == "LONG":
                if high >= target:
                    hit_target = True
                    exit_price = target
            else:
                if low <= target:
                    hit_target = True
                    exit_price = target

            if hit_target:
                exit_reason = "TARGET"
                break

            # Partial take profit
            if self.mode == "new":
                pnl, qty_closed = self._check_partial_take(pos, price, spec)
                if qty_closed > 0:
                    realized_pnl += pnl
                    partial_takes += 1

                # Kill loser
                loss, qty_cut = self._check_kill_loser(pos, price, spec)
                if qty_cut > 0:
                    realized_pnl += loss
                    loser_cuts += 1

            # Emergency max loss
            current_pnl = self._calc_pnl(spec, pos["entry"], price, pos["direction"], pos["qty"])
            if current_pnl < -MAX_LOSS_PER_TRADE:
                exit_price = price
                exit_reason = "MAX_LOSS"
                break

            # Time exit
            if self._should_time_exit(pos, price, spec, i):
                exit_price = price
                exit_reason = "TIME"
                break

            exit_price = price

        # Final PnL
        final_pnl = self._calc_pnl(spec, pos["entry"], exit_price, pos["direction"], pos["qty"])
        total_pnl = realized_pnl + final_pnl

        self.trade_id += 1
        return BacktestTrade(
            id=self.trade_id,
            base=base,
            secid=secid,
            direction=direction,
            entry_price=entry_price,
            entry_time=candles[entry_idx]["ts"],
            exit_price=exit_price,
            exit_time=candles[entry_idx + bars_held]["ts"] if entry_idx + bars_held < len(candles) else "",
            exit_reason=exit_reason,
            qty=qty,
            pnl_rub=total_pnl,
            pnl_pct=total_pnl / (entry_price * spec["lot"] * qty) * 100 if entry_price else 0,
            dev_at_entry=dev,
            bars_held=bars_held,
            partial_takes=partial_takes,
            loser_cuts=loser_cuts,
        )

    def run(self, days: int = 60) -> BacktestMetrics:
        """Run backtest on historical data."""
        end_date = datetime.now(MSK).date()
        start_date = end_date - timedelta(days=days)

        log.info(f"Running {self.mode.upper()} backtest: {start_date} to {end_date}")

        for base, spec in CONTRACTS.items():
            log.info(f"\nProcessing {base} ({spec['name']})...")

            # Find active contracts
            contracts = find_active_contracts(base, str(start_date), str(end_date))
            if not contracts:
                log.warning(f"No contracts found for {base}")
                continue

            # Fetch candles for each contract
            all_candles = []
            for secid in contracts:
                candles = fetch_hourly_candles(secid, str(start_date), str(end_date))
                if candles:
                    for c in candles:
                        c["secid"] = secid
                    all_candles.extend(candles)

            if not all_candles:
                log.warning(f"No candles for {base}")
                continue

            # Sort by time and deduplicate
            all_candles.sort(key=lambda x: x["ts"])
            seen_ts = set()
            candles = []
            for c in all_candles:
                if c["ts"] not in seen_ts:
                    seen_ts.add(c["ts"])
                    candles.append(c)

            log.info(f"  {len(candles)} hourly candles")

            if len(candles) < 50:
                continue

            # EMA warmup period
            ema_period = 20

            # Scan for entries
            last_trade_idx = -100  # Cooldown
            for i in range(ema_period + 5, len(candles) - 10):
                # Cooldown: no entry within 3 bars of last trade
                if i - last_trade_idx < 3:
                    continue

                # Calculate EMA
                closes = [c["close"] for c in candles[i-ema_period:i]]
                ema = calc_ema(closes, ema_period)

                # Calculate ATR
                atr = calc_atr(candles[max(0, i-20):i], 14)
                if atr <= 0:
                    continue

                price = candles[i]["close"]
                dev = ((price - ema) / ema * 100) if ema else 0

                # Entry logic: mean reversion on deviation
                min_dev = spec.get("min_dev", 0.5)
                max_dev = spec.get("max_dev", 3.0)
                side_mode = spec.get("side_mode", "both")

                direction = None
                if abs(dev) >= min_dev and abs(dev) <= max_dev:
                    if dev > 0 and side_mode in ("both", "short_only"):
                        direction = "SHORT"  # Price above EMA → revert down
                    elif dev < 0 and side_mode in ("both", "long_only"):
                        direction = "LONG"   # Price below EMA → revert up

                if direction is None:
                    continue

                # Simulate trade
                trade = self.simulate_trade(
                    base=base,
                    secid=candles[i].get("secid", ""),
                    direction=direction,
                    entry_idx=i,
                    candles=candles,
                    ema=ema,
                    atr=atr,
                )

                if trade:
                    self.trades.append(trade)
                    self.balance += trade.pnl_rub
                    self.equity_curve.append(self.balance)
                    last_trade_idx = i + trade.bars_held

        return self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics."""
        if not self.trades:
            return BacktestMetrics()

        metrics = BacktestMetrics()
        metrics.total_trades = len(self.trades)

        wins = [t for t in self.trades if t.pnl_rub > 0]
        losses = [t for t in self.trades if t.pnl_rub <= 0]

        metrics.wins = len(wins)
        metrics.losses = len(losses)
        metrics.win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        pnl_list = [t.pnl_rub for t in self.trades]
        metrics.total_pnl = sum(pnl_list)
        metrics.gross_profit = sum(t.pnl_rub for t in wins) if wins else 0
        metrics.gross_loss = abs(sum(t.pnl_rub for t in losses)) if losses else 1

        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float("inf")
        metrics.avg_win = metrics.gross_profit / len(wins) if wins else 0
        metrics.avg_loss = metrics.gross_loss / len(losses) if losses else 0

        # Sharpe
        if len(pnl_list) > 1:
            metrics.sharpe = np.mean(pnl_list) / (np.std(pnl_list) + 1e-9) * np.sqrt(252)

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        return metrics

    def print_summary(self):
        """Print backtest summary."""
        metrics = self._calculate_metrics()

        print("\n" + "=" * 70)
        print(f"FUTURES BACKTEST SUMMARY [{self.mode.upper()} MODE]")
        print("=" * 70)
        print(f"Total Trades:      {metrics.total_trades}")
        print(f"Wins/Losses:       {metrics.wins} / {metrics.losses}")
        print(f"Win Rate:          {metrics.win_rate:.1f}%")
        print(f"Total PnL:         {metrics.total_pnl:+,.0f} RUB")
        print(f"Profit Factor:     {metrics.profit_factor:.2f}")
        print(f"Sharpe Ratio:      {metrics.sharpe:.2f}")
        print(f"Max Drawdown:      {metrics.max_drawdown:,.0f} RUB")
        print(f"Avg Win:           {metrics.avg_win:+,.0f} RUB")
        print(f"Avg Loss:          {metrics.avg_loss:,.0f} RUB")
        print("=" * 70)

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        print("\nExit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(self.trades) * 100 if self.trades else 0
            pnl = sum(t.pnl_rub for t in self.trades if t.exit_reason == reason)
            print(f"  {reason:12} {count:3} ({pct:5.1f}%)  PnL: {pnl:+,.0f}")

        # By contract
        print("\nBy Contract:")
        for base in CONTRACTS:
            trades = [t for t in self.trades if t.base == base]
            if not trades:
                continue
            wins = sum(1 for t in trades if t.pnl_rub > 0)
            pnl = sum(t.pnl_rub for t in trades)
            wr = wins / len(trades) * 100 if trades else 0
            print(f"  {base:5} {len(trades):3} trades, WR {wr:5.1f}%, PnL {pnl:+,.0f}")

        if self.mode == "new":
            partial_takes = sum(t.partial_takes for t in self.trades)
            loser_cuts = sum(t.loser_cuts for t in self.trades)
            print(f"\nTactical Actions:")
            print(f"  Partial Takes:   {partial_takes}")
            print(f"  Loser Cuts:      {loser_cuts}")

        print()
        return metrics


def run_comparison(days: int = 60):
    """Run both modes and compare results."""
    print("\n" + "#" * 70)
    print("# FUTURES STRATEGY COMPARISON: OLD vs NEW")
    print("#" * 70)

    # Old mode
    old_bt = FuturesBacktester(mode="old")
    old_metrics = old_bt.run(days=days)
    old_bt.print_summary()

    # New mode
    new_bt = FuturesBacktester(mode="new")
    new_metrics = new_bt.run(days=days)
    new_bt.print_summary()

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: OLD vs NEW")
    print("=" * 70)

    def delta(old, new, fmt="+,.0f"):
        d = new - old
        return f"{d:{fmt}}" if d >= 0 else f"{d:,.0f}"

    print(f"{'Metric':<20} {'OLD':>15} {'NEW':>15} {'Delta':>15}")
    print("-" * 70)
    print(f"{'Total PnL':<20} {old_metrics.total_pnl:>15,.0f} {new_metrics.total_pnl:>15,.0f} {delta(old_metrics.total_pnl, new_metrics.total_pnl):>15}")
    print(f"{'Win Rate':<20} {old_metrics.win_rate:>14.1f}% {new_metrics.win_rate:>14.1f}% {delta(old_metrics.win_rate, new_metrics.win_rate, '+.1f'):>14}%")
    print(f"{'Profit Factor':<20} {old_metrics.profit_factor:>15.2f} {new_metrics.profit_factor:>15.2f} {delta(old_metrics.profit_factor, new_metrics.profit_factor, '+.2f'):>15}")
    print(f"{'Sharpe':<20} {old_metrics.sharpe:>15.2f} {new_metrics.sharpe:>15.2f} {delta(old_metrics.sharpe, new_metrics.sharpe, '+.2f'):>15}")
    print(f"{'Max Drawdown':<20} {old_metrics.max_drawdown:>15,.0f} {new_metrics.max_drawdown:>15,.0f} {delta(old_metrics.max_drawdown, new_metrics.max_drawdown):>15}")
    print(f"{'Avg Win':<20} {old_metrics.avg_win:>15,.0f} {new_metrics.avg_win:>15,.0f} {delta(old_metrics.avg_win, new_metrics.avg_win):>15}")
    print(f"{'Avg Loss':<20} {old_metrics.avg_loss:>15,.0f} {new_metrics.avg_loss:>15,.0f} {delta(old_metrics.avg_loss, new_metrics.avg_loss):>15}")
    print("=" * 70)

    improvement = new_metrics.total_pnl - old_metrics.total_pnl
    print(f"\nIMPROVEMENT: {improvement:+,.0f} RUB ({improvement/abs(old_metrics.total_pnl)*100 if old_metrics.total_pnl else 0:+.1f}%)")

    return old_metrics, new_metrics


if __name__ == "__main__":
    import sys

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_comparison(days=days)
