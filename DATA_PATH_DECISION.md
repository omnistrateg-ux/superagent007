# Data Path Decision

**Date**: 2026-04-15
**Decision**: QUIK via QuikPy/Lua Bridge
**Status**: BLOCKED (no terminal access)

---

## Requirement: Trades + L2 + OI with Aggressor Side

| Data Type | Required For | Min Frequency |
|-----------|--------------|---------------|
| Trade tape | M1, M2, M3 | Tick-level |
| Aggressor side | All hypotheses | Per trade |
| L2 depth | H3, H4 | 500ms |
| Open Interest | Futures position sizing | 1s |

---

## Data Source Comparison

| Source | Trades | Aggressor | L2 | OI | Access |
|--------|--------|-----------|-----|-----|--------|
| **QUIK** | YES | **YES (flags)** | YES | YES | BCS account |
| Tinkoff API | YES | NO | NO | NO | Account only |
| Finam API | YES | NO | Partial | NO | Account only |
| ALOR API | YES | NO | Partial | NO | Account only |
| ISS API | Partial | **NO** | NO | YES | Free |

**Critical**: Only QUIK provides aggressor side via OnAllTrade flags (bit 0).

---

## Chosen Path: QUIK

### Why QUIK

1. **Only source with aggressor side** - All hypotheses require knowing who initiated the trade
2. **Full L2 depth** - getQuoteLevel2 provides 10-50 levels
3. **Real-time OI** - getParamEx("NUMCONTRACTS") for futures
4. **Already implemented** - `quik_source.py` is complete (30KB)

### Implementation Status

| Component | File | Status |
|-----------|------|--------|
| QUIK source | `moex_agent/quik_source.py` | DONE |
| Collector | `moex_agent/microstructure_collector.py` | DONE |
| Storage | `moex_agent/microstructure_storage.py` | DONE |
| Validation | `moex_agent/microstructure_validate.py` | DONE |

---

## Blockers

### Critical (Must Resolve)

| Blocker | Owner | Action Required |
|---------|-------|-----------------|
| No BCS account | User | Open broker account at BCS |
| No QUIK terminal | User | Install QUIK, enable API access |
| No QuikPy | Auto | `pip install quikpy` after QUIK works |

### Alternative Path (If QUIK Impossible)

If BCS/QUIK access cannot be obtained:

1. **ISS proxy mode** - Collect 1m candles, test proxy-only signals
2. **Limitation**: Cannot prove orderflow edge, only candle patterns
3. **Use**: Development and pattern discovery, NOT edge validation

---

## Unblock Steps

```bash
# Step 1: Get BCS Account
# - Go to bcs.ru
# - Open broker account
# - Request QUIK access
# - Get login credentials

# Step 2: Install QUIK
# - Download from BCS personal cabinet
# - Install on Windows (or VM with shared folder)
# - Log in, verify data feed working

# Step 3: Configure API
# - QUIK > Settings > Enable "Trans2QUIK" or Lua integration
# - Note port (default 34130)

# Step 4: Test Connection
pip install quikpy
python3 -c "from moex_agent.quik_source import QUIKDataSource; q = QUIKDataSource(['BR']); print(q.connect())"

# Step 5: Start Collection
python3 -m moex_agent.microstructure_collector \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --mode quik \
    --db-path data/microstructure.db
```

---

## Exact Launch Commands

### Phase 1: Connection Test (Day 0)
```bash
# 1. Test QUIK connection
python3 -c "
from moex_agent.quik_source import QUIKDataSource
q = QUIKDataSource(['BR'])
print('Connected:', q.connect())
q.disconnect()
"

# 2. Test simulated mode (verify code works)
python3 -m moex_agent.microstructure_collector \
    --mode sim --tickers BR --duration 60

# 3. Test QUIK collection (5 min)
python3 -m moex_agent.microstructure_collector \
    --mode quik --tickers BR --duration 300 \
    --db-path data/test_quik.db

# 4. Validate test data
python3 -m moex_agent.microstructure_validate \
    --db-path data/test_quik.db --verbose
```

### Phase 2: Burn-In (3 Trading Days)
```bash
# Start burn-in collector
nohup python3 -m moex_agent.microstructure_collector \
    --mode quik \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --db-path data/microstructure_burnin.db \
    > logs/burnin_$(date +%Y%m%d).log 2>&1 &
echo $! > logs/burnin.pid

# Daily validation (run at end of each day)
python3 -m moex_agent.microstructure_validate \
    --db-path data/microstructure_burnin.db \
    --date $(date +%Y-%m-%d) --verbose

# Check collector status
tail -50 logs/burnin_$(date +%Y%m%d).log
```

### Phase 3: 30-Day Production Collection
```bash
# Stop burn-in
kill $(cat logs/burnin.pid)

# Archive burn-in data
mv data/microstructure_burnin.db data/microstructure_burnin_$(date +%Y%m%d).db

# Start production collector
nohup python3 -m moex_agent.microstructure_collector \
    --mode quik \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --db-path data/microstructure.db \
    > logs/collector_$(date +%Y%m%d).log 2>&1 &
echo $! > logs/collector.pid

# Weekly validation
python3 -m moex_agent.microstructure_validate \
    --db-path data/microstructure.db \
    --start-date $(date -d "7 days ago" +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d)
```

### Phase 4: Research Studies (After 30 Days)
```bash
# M3: Close Pressure (FIRST - highest priority, 1 event/day)
python3 orderflow_research_scaffold.py \
    --hypothesis M3 --ticker BR --days 30 \
    --db-path data/microstructure.db \
    --run-falsification

# M1: Opening Imbalance (SECOND - 1 event/day/ticker)
python3 orderflow_research_scaffold.py \
    --hypothesis M1 --ticker BR --days 30 \
    --db-path data/microstructure.db

python3 orderflow_research_scaffold.py \
    --hypothesis M1 --ticker RI --days 30 \
    --db-path data/microstructure.db

# M2: Flow Divergence (THIRD - continuous signal)
python3 orderflow_research_scaffold.py \
    --hypothesis M2 --ticker BR --days 30 \
    --db-path data/microstructure.db
```

### Utility Commands
```bash
# Check if collector running
pgrep -f microstructure_collector

# Emergency stop
pkill -f microstructure_collector

# Database size
ls -lh data/microstructure*.db

# Recent logs
tail -100 logs/collector_$(date +%Y%m%d).log

# Reconnect count today
grep -c "Reconnected" logs/collector_$(date +%Y%m%d).log
```

---

## Timeline

| Week | Milestone | Blocker |
|------|-----------|---------|
| 0 | Unblock QUIK | User action |
| 1 | Collect 5 days, validate quality | - |
| 2-4 | Collect 25+ more days | - |
| 5 | Run M3 (Close Pressure) | Need 30 events |
| 6 | Run M1 (Opening Imbalance) | Need 50+ events |
| 7 | Run M2 (Flow Divergence) | Need 100+ events |
| 8 | Falsification if any PF > 1.2 | - |

---

## Rejected Alternatives

### Tinkoff Invest API
- **Rejected**: No aggressor side in streaming trades
- **Link**: https://github.com/Tinkoff/invest-python

### Finam API
- **Rejected**: No real-time aggressor side, delayed data

### ALOR API
- **Rejected**: Limited L2, no aggressor side

### ISS API
- **Rejected as primary**: No aggressor side
- **Kept as fallback**: For testing collector, proxy hypotheses

---

## Files Reference

| File | Purpose |
|------|---------|
| `moex_agent/quik_source.py` | QUIK connection (30KB, DONE) |
| `moex_agent/microstructure_collector.py` | Orchestrator (31KB, DONE) |
| `moex_agent/microstructure_storage.py` | SQLite storage (25KB, DONE) |
| `moex_agent/microstructure_validate.py` | Quality checks (17KB, DONE) |
| `orderflow_research_scaffold.py` | M1/M2/M3 research (DONE) |
| `DATA_CAMPAIGN_30D.md` | 30-day collection plan |
| `MICROSTRUCTURE_SOURCE_OF_TRUTH.md` | Hypothesis status |

---

## Summary

**Path**: QUIK (only source with aggressor side)
**Status**: Infrastructure DONE, blocked on terminal access
**Next**: User obtains BCS account + QUIK terminal
**ETA to first study**: 5 weeks after unblock (30 trading days + 1 week analysis)
