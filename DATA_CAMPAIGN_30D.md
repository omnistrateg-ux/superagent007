# 30-Day Data Collection Campaign

**Version**: 2.0
**Updated**: 2026-04-15
**Status**: PARTIALLY UNBLOCKED

> **🆕 ISS DISCOVERY**: ISS API provides BUYSELL + OPEN_POS for futures trades (~45 days).
> - **M3 and M2**: UNBLOCKED - can test now via ISS
> - **M1 and M4**: Still blocked on QUIK (need L2 depth)
> - **30-day QUIK collection**: Still needed for M1/M4 and stocks

---

## Campaign Goal

~~Collect 30+ trading days of raw microstructure data with aggressor side to enable testing of M1-M4 orderflow hypotheses.~~

**UPDATE**: ISS provides ~45 days of futures trades with BUYSELL. M3/M2 can be tested NOW.

### Immediate Goal (M3/M2 via ISS)
- Test M3 (Close Pressure) and M2 (Flow Divergence) using ISS futures trades
- No collection needed - data already available via API
- Run studies immediately

### Remaining Goal (M1/M4 via QUIK)
- Still need QUIK for L2 depth (M1, M4)
- Still need QUIK for stocks (no BUYSELL in ISS for stocks)

---

## Tickers to Collect

### Futures (FORTS) - Priority 1
| Ticker | Name | Why | Min Trades/Day |
|--------|------|-----|----------------|
| BR | Brent Crude | Most liquid, best spread | 20,000 |
| RI | RTS Index | High volume, good for imbalance | 15,000 |
| MX | MOEX Index | Lower volume, gaps study | 5,000 |

### Stocks (TQBR) - Priority 2
| Ticker | Name | Why | Min Trades/Day |
|--------|------|-----|----------------|
| SBER | Sberbank | Most liquid stock | 30,000 |
| GAZP | Gazprom | High volume, oil correlation | 15,000 |
| LKOH | Lukoil | Oil sector, gaps | 10,000 |

**Total**: 6 tickers (3 futures + 3 stocks)

---

## Sessions to Collect

### Must Collect (Priority 1)
| Session | Time (MSK) | Reason |
|---------|------------|--------|
| `opening_drive` | 10:00-10:15 | M1: Opening Imbalance |
| `lunch` | 13:00-14:00 | Best WR session |
| `preclose` | 16:00-18:40 | M3: Close Pressure |
| `close_auction` | 18:40-18:50 | Gap formation |

### Should Collect (Priority 2)
| Session | Time (MSK) | Reason |
|---------|------------|--------|
| `morning_active` | 10:15-11:30 | M2: Flow Divergence |
| `midday` | 11:30-13:00 | Baseline |
| `afternoon` | 14:05-16:00 | Post-clearing |

### Optional (Low Priority)
| Session | Time (MSK) | Reason |
|---------|------------|--------|
| `morning_forts` | 07:00-10:00 | Low WR, skip for now |
| `evening` | 19:05-23:50 | Low liquidity |

**Collection Window**: 09:50 - 18:55 MSK (main sessions)

---

## Quality Thresholds (Must-Pass)

### Per-Day Thresholds
| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| `unknown_side_pct` | < 5% | 5-10% | > 10% |
| `coverage_pct` | > 80% | 60-80% | < 60% |
| `gaps_count` | < 5 | 5-10 | > 10 |
| `max_gap_seconds` | < 60 | 60-300 | > 300 |

### Per-Campaign Thresholds
| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| `trading_days` | >= 30 | 20-29 | < 20 |
| `avg_quality_score` | > 80 | 60-80 | < 60 |
| `tickers_with_data` | 6/6 | 4-5/6 | < 4/6 |

### Blockers (Abort if any)
- `unknown_side_pct > 50%` - ISS fallback active, no real QUIK data
- `trades_per_day < 1000` for any ticker - data source broken
- `reconnect_count > 100/day` - connection unstable

---

## First 2 Event Studies (READY NOW via ISS)

### Study 1: Close Pressure → Gap (M3) 🟢 READY

**Priority**: HIGHEST (1 event/day, cleanest signal)
**Data**: ISS futures trades with BUYSELL (~45 days available)

**Hypothesis**: Large net buying/selling in final 15 minutes predicts overnight gap direction.

**Event Definition**:
```python
event_time = "18:25-18:40 MSK"  # Last 15 min before close auction
entry_time = "10:00 next day"   # Open
exit_time  = "10:15 next day"   # +15 min
```

**Signal**:
```python
# Net flow in close window
buy_volume = sum(qty for trade in close_window if side == 'B')
sell_volume = sum(qty for trade in close_window if side == 'S')
imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

# Signal: BUY next open if imbalance > +0.3, SELL if < -0.3
```

**Run Command**:
```bash
python orderflow_research_scaffold.py --hypothesis M3 --ticker BR --days 45 --source iss --run-falsification
```

**Pass Criteria**:
| Metric | Threshold |
|--------|-----------|
| n (events) | >= 30 |
| PF | > 1.2 |
| WR | > 55% |
| Placebo shuffle p | < 0.1 |
| Placebo reverse PF | < 0.9 |

**Data Required**: ISS trades with BUYSELL ✅

---

### Study 2: Flow Divergence (M2) 🟢 READY

**Priority**: HIGH (multiple events per day)
**Data**: ISS futures trades with BUYSELL (~45 days available)

**Hypothesis**: When price makes new high/low but net flow diverges, reversal likely.

**Event Definition**:
```python
# Continuous scan during trading hours
lookback = 30  # minutes
price_new_high = price > max(prices[-lookback:])
flow_negative = net_flow_30m < -0.2  # More selling

# Divergence = price up, flow down (or vice versa)
```

**Signal**:
```python
# Bullish divergence: price new low, flow positive
# Bearish divergence: price new high, flow negative
entry = at_divergence_detection
exit = entry + 60 minutes
```

**Run Command**:
```bash
python orderflow_research_scaffold.py --hypothesis M2 --ticker BR --days 45 --source iss --run-falsification
```

**Pass Criteria**:
| Metric | Threshold |
|--------|-----------|
| n (events) | >= 50 |
| PF | > 1.2 |
| WR | > 52% |
| Placebo shuffle p | < 0.1 |

**Data Required**: ISS trades with BUYSELL ✅

---

---

## Blocked Studies (Need QUIK L2)

### Study 3: Opening Imbalance (M1) 🔴 BLOCKED

**Priority**: HIGH (after QUIK setup)
**Blocker**: Needs L2 orderbook for imbalance confirmation

**Hypothesis**: First 5-minute net flow + L2 imbalance predicts direction for next 30 minutes.

**Event Definition**:
```python
event_time = "10:00-10:05 MSK"  # First 5 min
entry_time = "10:05 MSK"        # After imbalance measured
exit_time  = "10:35 MSK"        # +30 min
```

**Why L2 needed**: Trade flow alone is noisy at open. L2 imbalance confirms direction.

**Data Required**: ISS trades ✅ + QUIK L2 ❌

---

## Collection Launch Checklist

### Prerequisites (BLOCKERS)
- [ ] BCS broker account active
- [ ] QUIK terminal installed
- [ ] QUIK logged in and receiving data
- [ ] QuikPy or Lua bridge configured
- [ ] Test connection successful

### Pre-Launch Tests
```bash
# 1. Test simulated mode works
python3 -m moex_agent.microstructure_collector --mode sim --duration 60

# 2. Test ISS fallback works
python3 -m moex_agent.microstructure_collector --mode iss --tickers SBER --duration 30

# 3. Test QUIK connection (BLOCKER)
python3 -c "from moex_agent.quik_source import QUIKDataSource; q = QUIKDataSource(['BR']); print(q.connect())"
```

### Launch Commands
```bash
# Create data directory
mkdir -p data logs

# Start collector (foreground for testing)
python3 -m moex_agent.microstructure_collector \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --mode quik \
    --db-path data/microstructure.db \
    --log-level INFO

# Start collector (background for production)
nohup python3 -m moex_agent.microstructure_collector \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --mode quik \
    --db-path data/microstructure.db \
    --log-level INFO \
    > logs/collector_$(date +%Y%m%d).log 2>&1 &
```

### Daily Monitoring
```bash
# Check collector is running
ps aux | grep microstructure_collector

# Check today's stats
python3 -m moex_agent.microstructure_validate --date $(date +%Y-%m-%d)

# Check for QUIK disconnects
grep -c "RECONNECT" logs/collector_$(date +%Y%m%d).log
```

---

## Current Status

### ✅ UNBLOCKED (ISS Futures Trades)

| Hypothesis | Data Source | Days Available | Status |
|------------|-------------|----------------|--------|
| M3 (Close Pressure) | ISS `/trades.json` | ~45 days | 🟢 READY |
| M2 (Flow Divergence) | ISS `/trades.json` | ~45 days | 🟢 READY |

**No collection needed** - ISS API provides historical trades with BUYSELL.

### ❌ STILL BLOCKED (Need QUIK)

| Hypothesis | Blocker | Why |
|------------|---------|-----|
| M1 (Opening Imbalance) | No L2 depth | ISS has trades, not orderbook |
| M4 (Queue Depletion) | No L2 depth | Requires 500ms orderbook snapshots |
| Stocks (SBER, GAZP, etc.) | No BUYSELL | ISS stocks trades lack aggressor side |

### To Unblock M1/M4/Stocks

1. **Get BCS Account** - broker account at BCS
2. **Install QUIK** - Windows terminal
3. **Configure QuikPy** - `pip install quikpy`
4. **Collect L2 data** - 30+ days

---

## Timeline

### Immediate (This Week)
| Day | Action |
|-----|--------|
| Day 1 | Run M3 study via ISS (Close Pressure) |
| Day 2 | Analyze M3 results, run falsification if PF > 1.2 |
| Day 3 | Run M2 study via ISS (Flow Divergence) |
| Day 4 | Analyze M2 results |
| Day 5 | Document findings, update source of truth |

### If M3/M2 Show Edge (PF > 1.2)
| Week | Action |
|------|--------|
| Week 2 | Full falsification (walk-forward, holdout) |
| Week 3 | Paper trading setup (if passes) |

### For M1/M4 (Still Blocked)
| Week | Milestone |
|------|-----------|
| Week 0 | Unblock QUIK, test connection |
| Week 1-4 | Collect 30 days L2 data |
| Week 5 | Run M1 (Opening Imbalance) |
| Week 6 | Run M4 (Queue Depletion) |

---

## Files Reference

| File | Purpose |
|------|---------|
| `moex_agent/quik_source.py` | QUIK data source (QuikPy + Lua) |
| `moex_agent/microstructure_collector.py` | Main collector |
| `moex_agent/microstructure_storage.py` | SQLite storage |
| `moex_agent/microstructure_validate.py` | Quality validation |
| `MICROSTRUCTURE_COLLECTION_RUNBOOK.md` | Collector docs |
| `EVENT_SCHEMA.md` | M1-M4 definitions |

---

## Success Definition

**Campaign Success** = All of:
1. 30+ trading days collected
2. All 6 tickers have data
3. unknown_side_pct < 5% average
4. coverage_pct > 80% average
5. At least one study shows PF > 1.2 with n > 50

**Campaign Failure** = Any of:
1. QUIK access never obtained
2. unknown_side_pct > 10% (ISS fallback only)
3. < 20 trading days after 2 months
4. All 3 studies show PF < 1.0
