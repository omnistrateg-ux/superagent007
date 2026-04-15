# Microstructure Source of Truth

**Last Updated**: 2026-04-15
**Status**: RESEARCH-ONLY (no live, no ML, no size_mult)

> **🆕 UPDATE 2026-04-15**: ISS API provides BUYSELL + OPEN_POS for futures trades (~45 days).
> **M3 and M2 are now UNBLOCKED** via ISS. M1 still blocked (needs L2 depth).

> **Note**: Microstructure hypotheses use M1-M4 naming to avoid confusion with
> old futures hypotheses H1-H4 (which are documented in stale docs marked STALE).
> See `BURN_IN_CHECKLIST.md` for data collection launch procedure.

---

## 1. EXECUTIVE SUMMARY

| Category | Count | Status |
|----------|-------|--------|
| **KILLED** | 6 | Do not resurrect |
| **UNBLOCKED** | 2 | M3, M2 - ISS futures trades with BUYSELL |
| **BLOCKED** | 2 | M1, M4 - need L2 depth (QUIK) |
| **PROXY ONLY** | 4 | ISS 1m candles, NOT real edge proof |

**Bottom Line**:
- **No paper-worthy edge yet** - all strategies either KILLED or UNTESTED
- **M3 and M2 now testable** via ISS futures trades (~45 days available)
- M1/M4 still blocked on QUIK (need L2 depth)

---

## 2. KILLED HYPOTHESES

These are PERMANENTLY DEAD. Do not revisit.

| ID | Hypothesis | Kill Date | Reason | Evidence |
|----|------------|-----------|--------|----------|
| K1 | FORTS v2.5 EMA Mean Reversion | 2026-04 | PF < 1.0 | Never profitable after costs |
| K2 | BR Residual Gap (Brent driver) | 2026-04-15 | PF = 0.83 | All configs losing, placebo p=0.499 |
| K3 | Stock Residual Gap (IMOEX driver) | 2026-04 | 3h lookahead | Stocks open 06:50, IMOEX 09:50 |
| K4 | NVTK Residual Gap | 2026-04 | Same as K3 | Timeframe misalignment artifact |
| K5 | SmartFilter on EMA base | 2026-04 | Base strategy killed | Filter on losing strategy = invalid |
| K6 | BR 1h RSI+Volume "edge" | 2026-04 | PF = 0.83 < 1.0 | PROJECT_DESCRIPTION.md claim invalid |

### Invalid Claims from Legacy Docs

| Source | Claim | Reality |
|--------|-------|---------|
| `PROJECT_DESCRIPTION.md` | "BR edge confirmed, PF 0.83" | PF < 1.0 = NET LOSING |
| `MEMORY.md (old)` | "RI +13.3% WR, PF 2.32" | SmartFilter on killed base |
| `backtest_oos_futures.py` | SmartFilter OOS positive | Filter on killed base = INCONCLUSIVE |

---

## 3. PROXY-ONLY HYPOTHESES

These use ISS 1-minute candles as **proxy** for real orderflow.

**CRITICAL**: Positive results here do NOT prove microstructure edge.
1m candles cannot capture:
- Order book imbalance (only OHLCV available)
- Trade tape direction (no aggressor side)
- Queue dynamics (no depth data)
- Sub-second execution (1m = 60,000ms aggregation)

| ID | Hypothesis | Data Source | Best PF | n | Status |
|----|------------|-------------|---------|---|--------|
| P1 | First-5-Min Imbalance | ISS 1m bar imbalance | UNTESTED | - | PROXY |
| P2 | Trade Flow Divergence | ISS 1m volume vs price | UNTESTED | - | PROXY |
| P3 | Pre-Close Imbalance | ISS 1m bar imbalance | UNTESTED | - | PROXY |
| P4 | Lunch Session Entry | ISS 1m momentum + volume | UNTESTED | - | PROXY |

### Why Proxy Results Are Not Edge Proof

```
REAL ORDERFLOW:                    PROXY (ISS 1m):
├── Trade tape                     ├── Close - Open position
│   ├── Each trade                 │   (loses direction info)
│   ├── Timestamp (ms)             ├── Aggregated over 60 sec
│   ├── Aggressor side             │   (loses timing precision)
│   └── Size                       └── No aggressor side info
│
├── Order book L2                  ├── Not available
│   ├── 10-50 levels               │
│   ├── Bid/ask depth              │
│   └── Queue changes              │
```

### Code Location

- `moex_agent/microstructure_research.py` - ISS-based proxy tests
- Tests: `test_first5min_imbalance()`, `test_flow_divergence()`, `test_close_imbalance()`, `test_lunch_entry()`

---

## 4. HYPOTHESIS STATUS

**Naming**: M1-M4 = Microstructure (distinct from old futures H1-H4).

### 4.1 UNBLOCKED (ISS Futures Trades with BUYSELL)

| ID | Hypothesis | Data Source | Days Available | Priority | Status |
|----|------------|-------------|----------------|----------|--------|
| **M3** | Close Pressure → Gap | ISS futures trades | ~45 days | **HIGHEST** | 🟢 READY TO TEST |
| **M2** | Flow Divergence | ISS futures trades | ~45 days | HIGH | 🟢 READY TO TEST |

**ISS provides**: BUYSELL (aggressor side) + OPEN_POS for futures.
**Test order**: M3 first (1 event/day, cleanest), then M2.

### 4.2 BLOCKED (Need L2 Depth)

| ID | Hypothesis | Required Data | Blocker | Priority |
|----|------------|---------------|---------|----------|
| M1 | Opening Imbalance (5min) | Trade tape + **L2 depth** | Need QUIK | HIGH |
| M4 | Queue Depletion at S/R | L2 depth changes | Need QUIK | MEDIUM |

### 4.3 OTHER

| ID | Hypothesis | Required Data | Blocker | Priority |
|----|------------|---------------|---------|----------|
| O5 | Cross-Asset Lead-Lag | ES/Brent synced with MOEX | Need timezone align | LOW |
| O6 | First Hour Reversal (candle) | More data (n=14 now) | Need 180+ days | LOW |

### Data Requirements per Hypothesis

#### M3: Close Pressure → Gap (🟢 READY - HIGHEST PRIORITY)
```
Required:
- Trade tape 18:25-18:40 MSK with aggressor side
- Next-day open (10:00+5m)
Data Source: ISS futures trades (BUYSELL field)
Available: ~45 days
Test Priority: HIGHEST (first study NOW)
Why first: 1 event/day = cleanest signal, fastest to test

Run command:
python orderflow_research_scaffold.py --hypothesis M3 --ticker BR --days 45 --run-falsification
```

#### M2: Flow Divergence (🟢 READY)
```
Required:
- Trade tape with aggressor side (continuous)
- Price extremes detection (30-min lookback)
- Entry at divergence, exit +30/60m
Data Source: ISS futures trades (BUYSELL field)
Available: ~45 days
Test Priority: HIGH (after M3)

Run command:
python orderflow_research_scaffold.py --hypothesis M2 --ticker BR --days 45 --run-falsification
```

#### M1: Opening Imbalance (🔴 BLOCKED)
```
Required:
- Trade tape 10:00-10:05 MSK with aggressor side
- L2 order book for imbalance confirmation
- Entry at 10:05, exit +15/30/60m
Data Source: QUIK (ISS has trades but no L2)
Blocker: Need L2 depth for full validation
Test Priority: HIGH (after QUIK setup)
```

#### M4: Queue Depletion at S/R
```
Required:
- L2 snapshots every 500ms
- At least 5-10 levels
- Support/resistance detection
Data Source: QUIK getQuoteLevel2
Test Priority: LOW (complex, after M1/M2/M3)
```

#### O5: Cross-Asset Lead-Lag
```
Required:
- ES close (16:00 ET = 23:00 MSK prev day)
- RI open (07:00 MSK)
- ICE Brent hourly (overnight)
Data Source: Yahoo Finance + ISS
Blocker: Timezone alignment logic not implemented
```

#### O6: First Hour Reversal
```
Current State:
- BR main 1.0%: n=14, PF=1.28
- Too few trades for statistical significance
Required:
- 180+ days of 1h candles
- n > 50 trades
- Walk-forward validation
```

---

## 5. AVAILABLE DATA SOURCES

### 5.1 ISS API - Futures Trades (🆕 DISCOVERED)

| Data | Endpoint | Fields | Days Available |
|------|----------|--------|----------------|
| **Futures Trades** | `/trades.json` | price, qty, **BUYSELL**, **OPEN_POS** | ~45 days |
| OHLCV Candles | `/candles.json` | OHLCV | 1m-1d, years |
| Current Quote | `/securities/{id}.json` | BID1, ASK1 | Real-time |

**Key discovery**: ISS `/trades.json` for FORTS provides:
- `BUYSELL` = aggressor side ('B' or 'S')
- `OPEN_POS` = open interest at trade time
- Tick-level data with timestamps

**Enables**: M3 (Close Pressure), M2 (Flow Divergence) without QUIK

### 5.2 QUIK API (Still Needed for L2)

| Data | Method | Granularity | Status |
|------|--------|-------------|--------|
| Order Book L2 | `getQuoteLevel2` | 500ms snapshots | NOT COLLECTING |
| Trade Tape | `OnAllTrade` | Tick-by-tick | ISS alternative now |
| OI Updates | `getFuturesHolding` | On change | ISS has OPEN_POS |

**Still needed for**: M1 (Opening Imbalance needs L2), M4 (Queue Depletion)

### 5.3 NOT Available

| Data | Why Needed | Alternative |
|------|------------|-------------|
| Auction Book | Pre-open imbalance | None |
| Orders Log | Cancel/add ratio | None |
| Order Lifetime | Informed flow detection | None |

---

## 6. FEATURE INVENTORY (19 Implemented, 0 Tested)

From `moex_agent/microstructure.py`:

### Order Book Features (9)
```
micro_imbalance_5        Top 5 level imbalance [-1, +1]
micro_imbalance_10       Top 10 level imbalance
micro_microprice_gap     Weighted mid vs simple mid
micro_spread_vs_median   Current spread / session median
micro_depth_ratio        Bid depth / Ask depth
micro_depth_change_10s   Depth % change 10s
micro_depth_change_30s   Depth % change 30s
micro_imbalance_trend    Slope of imbalance (OFI proxy)
micro_bid_wall           Max bid / mean bid (wall detection)
```

### Trade Tape Features (8)
```
micro_trade_imbalance_10s   (buy-sell)/total 10s window
micro_trade_imbalance_30s   (buy-sell)/total 30s window
micro_trade_imbalance_60s   (buy-sell)/total 60s window
micro_cvd                   Cumulative Volume Delta
micro_large_trade_ratio     Fraction from trades > 50 lots
micro_trade_intensity       Trades/sec vs session average
micro_avg_buy_size          Mean aggressor buy size
micro_avg_sell_size         Mean aggressor sell size
```

### Futures Features (2)
```
micro_oi_change            OI % change over 60s
micro_oi_divergence        OI vs price divergence
```

**Status**: All features implemented but UNTESTED due to no data.

---

## 7. VALIDATION REQUIREMENTS

Before any hypothesis becomes PAPER-CANDIDATE:

| Test | Required | Threshold |
|------|----------|-----------|
| Profit Factor | YES | PF > 1.2 |
| Trade Count | YES | n > 50 |
| Placebo Shuffle | YES | p < 0.1 |
| Placebo Reverse | YES | PF < 0.9 |
| Walk-Forward | YES | >60% folds profitable |
| Final Holdout | YES | PF > 1.0 on unseen data |
| Cost Shock 2x | YES | PF > 1.0 |
| Session Stability | YES | Works in 2+ sessions |

---

## 8. NEXT STEPS

### Phase 1: Run M3 Study NOW (🟢 UNBLOCKED)
```bash
# M3: Close Pressure → Gap (HIGHEST PRIORITY)
# ~45 days of ISS futures trades available
python orderflow_research_scaffold.py \
    --hypothesis M3 \
    --ticker BR \
    --days 45 \
    --source iss \
    --run-falsification

# Expected: ~30-35 events (weekdays only, some filtered)
# Pass criteria: PF > 1.2, n > 30, placebo p < 0.1
```

### Phase 2: Run M2 Study (After M3)
```bash
# M2: Flow Divergence (continuous scan)
python orderflow_research_scaffold.py \
    --hypothesis M2 \
    --ticker BR \
    --days 45 \
    --source iss \
    --run-falsification

# Expected: 50-100+ events (multiple per day possible)
# Pass criteria: PF > 1.2, n > 50, placebo p < 0.1
```

### Phase 3: QUIK Setup (For M1/M4)
```
Still blocked on QUIK for:
- M1: Opening Imbalance (needs L2 confirmation)
- M4: Queue Depletion (needs L2 depth)

See: DATA_PATH_DECISION.md for QUIK setup
```

### Phase 4: Falsification (if PF > 1.2)
```
For each CANDIDATE:
1. Placebo shuffle (p < 0.1)
2. Placebo reverse (PF < 0.9)
3. Walk-forward (>60% folds)
4. Cost shock 2x (PF > 1.0)
5. Final holdout
```

---

## 9. FILE REFERENCES

### Active Documents
| File | Purpose | Status |
|------|---------|--------|
| `BURN_IN_CHECKLIST.md` | 3-day smoke test + launch | CURRENT |
| `DATA_PATH_DECISION.md` | QUIK chosen, blockers, commands | CURRENT |
| `DATA_CAMPAIGN_30D.md` | 30-day collection plan | CURRENT |
| `orderflow_research_scaffold.py` | M1/M2/M3 research framework | IMPLEMENTED |

### Collector Infrastructure
| File | Purpose | Status |
|------|---------|--------|
| `moex_agent/microstructure_collector.py` | Main collector | DONE |
| `moex_agent/microstructure_storage.py` | SQLite storage | DONE |
| `moex_agent/microstructure_validate.py` | Quality validation (GREEN/YELLOW/RED) | DONE |
| `moex_agent/quik_source.py` | QUIK QuikPy + Lua bridge | DONE |

### Research Code
| File | Purpose | Status |
|------|---------|--------|
| `moex_agent/microstructure.py` | 19 features | IMPLEMENTED, UNTESTED |
| `moex_agent/microstructure_research.py` | ISS proxy tests | IMPLEMENTED |

### STALE (Do Not Use for Decisions)
| File | Why Stale |
|------|-----------|
| `EDGE_SOURCE_OF_TRUTH.md` | Candle strategies KILLED |
| `FUTURES_EDGE_VALIDATION.md` | Old H1-H4 (candle-based) KILLED |
| `STOCK_EDGE_VALIDATION.md` | Stock research on hold |
| `futures/FUTURES_SOURCE_OF_TRUTH.md` | Old FORTS strategies |

---

## 10. GOLDEN RULES

1. **NO EDGE CLAIMS FROM 1M CANDLE PROXIES** - ISS data cannot prove orderflow edge
2. **NO LIVE TRADING** - Until full validation passes
3. **NO ML FIRST** - Rules-based only until edge proven
4. **NO SIZE_MULT** - Removed per Phase 4 spec
5. **KILL FAST** - If PF < 1.0 or placebo p > 0.1, hypothesis is DEAD
6. **DATA FIRST** - Cannot test O1-O4 without real orderflow data
