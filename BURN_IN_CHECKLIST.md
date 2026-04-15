# Burn-In Checklist

**Purpose**: 3-day smoke test before 30-day production collection
**Status**: Ready (blocked on QUIK access)

---

## Pre-Launch Prerequisites

### QUIK Access (BLOCKING)
- [ ] BCS broker account active
- [ ] QUIK terminal installed
- [ ] QUIK logged in, data feed visible
- [ ] Port 34130 accessible
- [ ] `pip install quikpy` completed

### Environment
- [ ] Python 3.9+ installed
- [ ] `data/` directory exists
- [ ] `logs/` directory exists
- [ ] Disk space > 5GB free

---

## Day 0: Connection Test (30 min)

### Step 1: Test Simulated Mode
```bash
python3 -m moex_agent.microstructure_collector \
    --mode sim \
    --tickers BR,SBER \
    --duration 60 \
    --db-path data/test_sim.db
```

**Pass criteria**:
- [ ] Runs without errors
- [ ] Creates database file
- [ ] Shows trade/quote counts in output

### Step 2: Test QUIK Connection
```bash
python3 -c "
from moex_agent.quik_source import QUIKDataSource
q = QUIKDataSource(['BR'])
connected = q.connect()
print(f'Connected: {connected}')
if connected:
    q.disconnect()
"
```

**Pass criteria**:
- [ ] `Connected: True`
- [ ] No connection errors

### Step 3: Test QUIK Collection (5 min)
```bash
python3 -m moex_agent.microstructure_collector \
    --mode quik \
    --tickers BR \
    --duration 300 \
    --db-path data/test_quik.db
```

**Pass criteria**:
- [ ] Receives trades
- [ ] Unknown side < 5%
- [ ] No reconnects

### Step 4: Validate Test Data
```bash
python3 -m moex_agent.microstructure_validate \
    --db-path data/test_quik.db \
    --verbose
```

**Pass criteria**:
- [ ] Trades count > 0
- [ ] Unknown side < 5%
- [ ] Quality verdict: GREEN or YELLOW

---

## Day 1-3: Burn-In Collection

### Launch Command
```bash
# Start burn-in (main session only: 10:00-18:45 MSK)
nohup python3 -m moex_agent.microstructure_collector \
    --mode quik \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --db-path data/microstructure_burnin.db \
    > logs/burnin_$(date +%Y%m%d).log 2>&1 &

echo $! > logs/burnin.pid
```

### Daily Validation (end of each day)
```bash
python3 -m moex_agent.microstructure_validate \
    --db-path data/microstructure_burnin.db \
    --date $(date +%Y-%m-%d) \
    --verbose
```

---

## Burn-In Pass Thresholds

### Per Day (ALL must pass for 3 consecutive days)

| Metric | GREEN | YELLOW | RED (fail) |
|--------|-------|--------|------------|
| Trades/ticker | > 5,000 | 1,000-5,000 | < 1,000 |
| Unknown side % | < 5% | 5-10% | > 10% |
| Coverage % | > 80% | 60-80% | < 60% |
| Gaps count | < 5 | 5-15 | > 15 |
| Max gap (sec) | < 60 | 60-300 | > 300 |
| Reconnects | 0 | 1-3 | > 3 |
| Quality score | >= 80 | 60-79 | < 60 |

### Aggregate (3-day total)

| Metric | Pass | Fail |
|--------|------|------|
| GREEN days | >= 2 | < 2 |
| RED days | 0 | >= 1 |
| Avg quality score | >= 75 | < 75 |
| Total trades/ticker | > 30,000 | < 30,000 |

---

## Burn-In Failure Actions

### If Unknown Side > 10%
```
BLOCKER: QUIK not providing aggressor side
1. Check QUIK OnAllTrade callback
2. Verify flags field is being parsed
3. Check quik_source.py trade handler
```

### If Coverage < 60%
```
WARNING: Connection issues
1. Check QUIK terminal is running
2. Check network stability
3. Check reconnect logs
```

### If Quality Score < 60 (RED)
```
BLOCKER: Data quality insufficient
1. Identify specific failing metrics
2. Fix root cause before proceeding
3. Restart 3-day burn-in
```

---

## Post Burn-In: Go/No-Go Decision

### GO Criteria (ALL must be true)
- [ ] 3 consecutive days with quality >= YELLOW
- [ ] At least 2 days with quality = GREEN
- [ ] Unknown side < 10% on all days
- [ ] No RED days
- [ ] Total trades > 100,000 across all tickers

### NO-GO Actions
If criteria not met:
1. Identify root cause
2. Fix issue
3. Delete burn-in database
4. Restart 3-day burn-in

---

## Transition to 30-Day Collection

### After Burn-In Pass
```bash
# Stop burn-in
kill $(cat logs/burnin.pid)

# Archive burn-in data
mv data/microstructure_burnin.db data/microstructure_burnin_$(date +%Y%m%d).db

# Start production collection
nohup python3 -m moex_agent.microstructure_collector \
    --mode quik \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --db-path data/microstructure.db \
    > logs/collector_$(date +%Y%m%d).log 2>&1 &

echo $! > logs/collector.pid
```

---

## Quick Reference

### Check Collector Status
```bash
# Is it running?
ps aux | grep microstructure_collector

# Recent log output
tail -50 logs/collector_$(date +%Y%m%d).log

# Reconnect count today
grep -c "RECONNECT" logs/collector_$(date +%Y%m%d).log
```

### Emergency Stop
```bash
kill $(cat logs/collector.pid)
# or
pkill -f microstructure_collector
```

### Database Size
```bash
ls -lh data/microstructure*.db
```

---

## Timeline Summary

| Phase | Duration | Gate |
|-------|----------|------|
| QUIK setup | 1-2 hours | Connection test passes |
| Burn-in | 3 trading days | All pass thresholds |
| Production | 30 trading days | M3 study ready |
| M3 study | 1 week | PF > 1.2 or KILL |

---

## Files Reference

| File | Purpose |
|------|---------|
| `moex_agent/microstructure_collector.py` | Main collector |
| `moex_agent/microstructure_validate.py` | Quality validation (GREEN/YELLOW/RED) |
| `moex_agent/quik_source.py` | QUIK connection |
| `moex_agent/microstructure_storage.py` | SQLite storage |
| `data/microstructure_burnin.db` | Burn-in database |
| `data/microstructure.db` | Production database |
| `logs/burnin_YYYYMMDD.log` | Burn-in logs |
| `logs/collector_YYYYMMDD.log` | Production logs |
