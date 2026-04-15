# Microstructure Data Collection Runbook

**Version**: 1.0
**Date**: 2026-04-15
**Status**: OPERATIONAL (Simulated mode only)

---

## Quick Start

```bash
# 1. Test with simulated data (no dependencies)
python -m moex_agent.microstructure_collector --tickers SBER,BR --mode sim --duration 60

# 2. Validate collected data
python -m moex_agent.microstructure_validate --db-path data/microstructure.db

# 3. Check storage summary
python -c "from moex_agent.microstructure_storage import MicrostructureStorage; from pathlib import Path; s = MicrostructureStorage(Path('data/microstructure.db')); print(s.get_summary())"
```

---

## Files Created

| File | Purpose |
|------|---------|
| `moex_agent/microstructure_storage.py` | SQLite storage layer with session labels, gap detection |
| `moex_agent/microstructure_collector.py` | Data collector with QUIK/ISS/Simulated sources |
| `moex_agent/microstructure_validate.py` | Data quality validation script |
| `data/microstructure.db` | SQLite database (created on first run) |

---

## Data Tables

### raw_trades
Trade tape with aggressor side.

| Column | Type | Description |
|--------|------|-------------|
| `ts` | TEXT | Timestamp (MSK) |
| `ticker` | TEXT | Security ID |
| `price` | REAL | Execution price |
| `qty` | INTEGER | Number of contracts/shares |
| `side` | TEXT | Aggressor: 'BUY', 'SELL', 'UNKNOWN' |
| `trade_id` | TEXT | Exchange trade ID |
| `session` | TEXT | Session label |
| `collected_ts` | TEXT | When we received this |

### raw_quotes
Best bid/ask snapshots.

| Column | Type | Description |
|--------|------|-------------|
| `ts` | TEXT | Timestamp (MSK) |
| `ticker` | TEXT | Security ID |
| `bid_price` | REAL | Best bid price |
| `bid_size` | INTEGER | Size at best bid |
| `ask_price` | REAL | Best ask price |
| `ask_size` | INTEGER | Size at best ask |
| `spread_bps` | REAL | Spread in basis points |
| `session` | TEXT | Session label |

### raw_depth
L2 orderbook depth (top N levels).

| Column | Type | Description |
|--------|------|-------------|
| `ts` | TEXT | Timestamp (MSK) |
| `ticker` | TEXT | Security ID |
| `bid_prices` | TEXT | JSON array of prices |
| `bid_sizes` | TEXT | JSON array of sizes |
| `ask_prices` | TEXT | JSON array of prices |
| `ask_sizes` | TEXT | JSON array of sizes |
| `levels` | INTEGER | Number of levels |
| `imbalance_5` | REAL | Top-5 imbalance [-1, +1] |

### raw_oi
Open Interest updates (futures only).

| Column | Type | Description |
|--------|------|-------------|
| `ts` | TEXT | Timestamp (MSK) |
| `ticker` | TEXT | Contract ID |
| `open_interest` | INTEGER | Total open contracts |
| `oi_change` | INTEGER | Change from previous |

---

## Session Labels

All timestamps are in Moscow Time (MSK/UTC+3).

| Label | Time | Trading | Description |
|-------|------|---------|-------------|
| `morning_auction` | 06:50-07:00 | No | Stocks pre-open |
| `morning_forts` | 07:00-10:00 | Yes | FORTS only, low liquidity |
| `stock_auction` | 09:50-10:00 | No | Main stock auction |
| `opening_drive` | 10:00-10:15 | Yes | First 15 min |
| `morning_active` | 10:15-11:30 | Yes | Morning active |
| `midday` | 11:30-13:00 | Yes | Midday |
| `lunch` | 13:00-14:00 | Yes | Best WR window |
| `clearing_1` | 14:00-14:05 | No | FORTS clearing |
| `afternoon` | 14:05-16:00 | Yes | Afternoon |
| `preclose` | 16:00-18:40 | Yes | Pre-close |
| `close_auction` | 18:40-18:50 | No | Stock close auction |
| `clearing_2` | 18:45-19:05 | No | FORTS clearing |
| `evening` | 19:05-23:50 | Yes | Evening FORTS |

---

## Collector Modes

### 1. Simulated (mode=sim)

**Status**: WORKING

For development and testing. Generates realistic random data with aggressor side.

```bash
python -m moex_agent.microstructure_collector \
    --tickers SBER,GAZP,BR,RI \
    --mode sim \
    --duration 300 \
    --interval-ms 100
```

**Provides**:
- Trades with aggressor side (BUY/SELL)
- Best bid/ask quotes
- L2 depth snapshots
- OI updates (futures)

### 2. ISS Fallback (mode=iss)

**Status**: WORKING (LIMITED)

Uses MOEX ISS API. **Does NOT provide aggressor side!**

```bash
python -m moex_agent.microstructure_collector \
    --tickers SBER,GAZP \
    --mode iss
```

**Provides**:
- Trades (side=UNKNOWN)
- Best bid/ask only (no depth)
- No OI

**Limitation**: All trades have `side='UNKNOWN'`. Cannot test orderflow hypotheses!

### 3. QUIK (mode=quik)

**Status**: NOT IMPLEMENTED (BLOCKER)

Requires running QUIK terminal with Lua scripts or QuikPy.

```bash
python -m moex_agent.microstructure_collector \
    --tickers BR,RI,SBER \
    --mode quik
```

**Would provide**:
- Trades with aggressor side
- Full L2 depth (10-50 levels)
- OI updates
- Sub-second timestamps

**Blocker**: QUIK protocol implementation not complete.

---

## Running the Collector

### Background Collection (Production)

```bash
# Create systemd service or use nohup
nohup python -m moex_agent.microstructure_collector \
    --tickers BR,RI,MX,SBER,GAZP,LKOH \
    --mode quik \
    --db-path data/microstructure.db \
    --log-level INFO \
    > logs/collector.log 2>&1 &

# Check status
tail -f logs/collector.log
```

### Scheduled Daily Collection

Add to crontab:
```cron
# Start collector at 06:45 MSK (before morning auction)
45 6 * * 1-5 cd /path/to/superagent007 && python -m moex_agent.microstructure_collector --mode quik >> logs/collector.log 2>&1

# Stop at 23:55 MSK (after evening session)
55 23 * * 1-5 pkill -f microstructure_collector
```

---

## Validating Data

### Daily Validation

```bash
# Validate today's data
python -m moex_agent.microstructure_validate --date $(date +%Y-%m-%d)

# Validate specific ticker
python -m moex_agent.microstructure_validate --ticker SBER --date 2026-04-15 --verbose

# Validate date range
python -m moex_agent.microstructure_validate --start-date 2026-04-01 --end-date 2026-04-15
```

### Quality Metrics

| Metric | Good | Warning | Blocker |
|--------|------|---------|---------|
| Coverage | >80% | 50-80% | <50% |
| Unknown Side | <5% | 5-10% | >50% |
| Gaps per Day | <5 | 5-10 | >10 |
| Quality Score | >80 | 50-80 | <50 |

### Example Output

```
================================================================
VALIDATION SUMMARY
================================================================

Database: data/microstructure.db
Validated: 2026-04-15T18:30:00

OVERALL:
  Days:         10
  Tickers:      6
  Total Trades: 125,432
  Total Quotes: 1,234,567
  Avg Quality:  85.3/100

BLOCKERS:
  BLOCKER: 65.2% trades have UNKNOWN side - cannot test orderflow hypotheses

PER TICKER:
  BR       |  10 days |   25,432 trades | Q=87.2
  RI       |  10 days |   18,234 trades | Q=84.5
  SBER     |  10 days |   35,123 trades | Q=86.8
```

---

## Troubleshooting

### No Data Collected

1. Check mode is correct:
   ```bash
   python -m moex_agent.microstructure_collector --mode sim --duration 10
   # Should output "[STATS] trades=X quotes=Y ..."
   ```

2. Check database exists:
   ```bash
   ls -la data/microstructure.db
   ```

3. Check for errors in log:
   ```bash
   grep -i error logs/collector.log
   ```

### High Unknown Side Percentage

**Cause**: Using ISS mode (doesn't provide aggressor side).

**Solution**: Use QUIK mode with running terminal.

### Gaps in Data

**Causes**:
- Network issues
- QUIK disconnection
- Market halts

**Check gaps**:
```sql
sqlite3 data/microstructure.db "SELECT * FROM collection_gaps ORDER BY ts DESC LIMIT 20"
```

### Database Growing Too Large

```bash
# Check size
du -h data/microstructure.db

# Vacuum if needed (after stopping collector)
sqlite3 data/microstructure.db "VACUUM"

# Estimate: ~50-100 MB per day for 6 tickers
```

---

## Current Blockers

| Blocker | Impact | Workaround | Status |
|---------|--------|------------|--------|
| QUIK not implemented | No real aggressor side | Use simulated | HIGH |
| No BCS account | Cannot connect to QUIK | Get account | HIGH |
| No QUIK terminal | Cannot collect real data | Install QUIK | HIGH |

### To Unblock QUIK Mode

1. Get active BCS broker account
2. Install QUIK terminal
3. Configure QUIK Lua scripts or QuikPy
4. Implement `QUIKDataSource._real_data_loop()` in collector
5. Test with real market data

---

## Data for Research

After 30+ days of collection with QUIK mode:

```python
from moex_agent.microstructure_storage import MicrostructureStorage
from pathlib import Path
from datetime import datetime

storage = MicrostructureStorage(Path("data/microstructure.db"))

# Get trades for a session
df = storage.get_trades(
    ticker="BR",
    start=datetime(2026, 4, 1),
    end=datetime(2026, 4, 15),
    session="opening_drive",
)

# Analyze aggressor side
print(df.groupby("side")["qty"].sum())

# Get depth snapshots
depth_df = storage.get_depth(
    ticker="BR",
    start=datetime(2026, 4, 15, 10, 0),
    end=datetime(2026, 4, 15, 10, 5),
)
```

---

## Next Steps

1. **Get QUIK Access**: BCS account + QUIK terminal
2. **Implement QUIK Protocol**: See `QUIKDataSource` in collector
3. **Run Collection**: 30+ trading days
4. **Validate Quality**: Check unknown_side_pct < 5%
5. **Start Research**: Run M1/M2/M3 hypotheses with real data

---

## Contacts

- Code: `/Users/artempobedinskij/superagent007/moex_agent/`
- Data: `/Users/artempobedinskij/superagent007/data/microstructure.db`
- Docs: `MICROSTRUCTURE_SOURCE_OF_TRUTH.md`, `DATA_COLLECTION_SPEC.md`
