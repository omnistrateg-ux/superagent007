# 30-Day Data Collection Campaign

**Version**: 1.0
**Created**: 2026-04-15
**Status**: READY TO LAUNCH (blocked on QUIK access)

---

## Campaign Goal

Collect 30+ trading days of raw microstructure data with aggressor side to enable testing of M1-M4 orderflow hypotheses.

**Success Criteria**: Quality Score > 80/100 for all tickers, unknown_side_pct < 5%.

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

## First 3 Event Studies

After 30 days of collection, run these studies in order:

### Study 1: Close Pressure → Gap (M3)

**Priority**: HIGHEST (1 event/day, cleanest signal)

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
buy_volume = sum(qty for trade in close_window if side == 'BUY')
sell_volume = sum(qty for trade in close_window if side == 'SELL')
imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

# Signal: BUY next open if imbalance > +0.3, SELL if < -0.3
```

**Pass Criteria**:
| Metric | Threshold |
|--------|-----------|
| n (events) | >= 30 |
| PF | > 1.2 |
| WR | > 55% |
| Placebo shuffle p | < 0.1 |
| Placebo reverse PF | < 0.9 |

**Data Required**: raw_trades with side, close auction prices

---

### Study 2: Opening Imbalance (M1)

**Priority**: HIGH (1 event/day per ticker)

**Hypothesis**: First 5-minute net flow predicts direction for next 30 minutes.

**Event Definition**:
```python
event_time = "10:00-10:05 MSK"  # First 5 min
entry_time = "10:05 MSK"        # After imbalance measured
exit_time  = "10:35 MSK"        # +30 min
```

**Signal**:
```python
# Net flow in first 5 minutes
trades_5m = get_trades("10:00", "10:05")
buy_vol = sum(t.qty for t in trades_5m if t.side == 'BUY')
sell_vol = sum(t.qty for t in trades_5m if t.side == 'SELL')
imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)

# Signal: BUY if imbalance > +0.4, SELL if < -0.4
```

**Pass Criteria**:
| Metric | Threshold |
|--------|-----------|
| n (events) | >= 50 (30 days × 3 tickers / 2 filtered) |
| PF | > 1.2 |
| WR | > 52% |
| Placebo shuffle p | < 0.1 |

**Data Required**: raw_trades with side, session labels

---

### Study 3: Flow Divergence (M2)

**Priority**: MEDIUM (multiple events/day, complex)

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

**Pass Criteria**:
| Metric | Threshold |
|--------|-----------|
| n (events) | >= 100 |
| PF | > 1.15 |
| WR | > 50% |
| Walk-forward | > 60% profitable folds |

**Data Required**: raw_trades with side, OHLC for price levels

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

## Current Blockers

| Blocker | Severity | Owner | ETA |
|---------|----------|-------|-----|
| No BCS account | CRITICAL | User | ? |
| QUIK not installed | CRITICAL | User | ? |
| QuikPy not tested | HIGH | - | After QUIK |

### To Unblock

1. **Get BCS Account**
   - Open broker account at BCS
   - Enable QUIK terminal access
   - Get login credentials

2. **Install QUIK**
   - Download QUIK from BCS
   - Install on Windows (or VM)
   - Log in and verify data feed

3. **Configure Connection**
   - Option A: Install QuikPy (`pip install quikpy`)
   - Option B: Use Lua socket bridge (port 5555)
   - Test with: `python3 -c "from moex_agent.quik_source import QUIKDataSource; ..."`

4. **Start Collection**
   - Run collector in QUIK mode
   - Monitor for first day
   - Verify unknown_side_pct < 5%

---

## Timeline

| Week | Milestone |
|------|-----------|
| Week 0 | Unblock QUIK, test connection |
| Week 1 | Collect 5 days, validate quality |
| Week 2-4 | Collect remaining 25 days |
| Week 5 | Run Study 1 (Close Pressure) |
| Week 6 | Run Study 2 (Opening Imbalance) |
| Week 7 | Run Study 3 (Flow Divergence) |
| Week 8 | Falsification tests if any pass |

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
