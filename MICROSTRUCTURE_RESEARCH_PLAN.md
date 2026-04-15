# Microstructure Research Plan

> **⚠️ STALE DOCUMENT (2026-04-15)**
>
> This document is OUTDATED. Key issues:
> - Priority 1-4 tests use ISS 1m candles (PROXY-ONLY, cannot prove edge)
> - No aggressor side in ISS data = no orderflow validation
> - Naming inconsistent (H1 vs M1)
>
> **For current status see:**
> - `MICROSTRUCTURE_SOURCE_OF_TRUTH.md` - Killed/Proxy/Open hypotheses
> - `DATA_CAMPAIGN_30D.md` - Active 30-day collection plan
> - `DATA_PATH_DECISION.md` - QUIK path and blockers
>
> **Active hypotheses (M1-M4) require QUIK data collection first.**

**Last Updated**: 2026-04-15
**Goal**: Find reproducible orderbook/orderflow patterns on MOEX
**Status**: STALE - superseded by MICROSTRUCTURE_SOURCE_OF_TRUTH.md

---

## 1. SOURCE OF TRUTH

### 1.1 KILLED (Do Not Resurrect)

| Hypothesis | Kill Date | Reason |
|------------|-----------|--------|
| FORTS v2.5 EMA Mean Reversion | 2026-04 | PF<1.0, no walk-forward |
| BR Residual Gap (IMOEX driver) | 2026-04 | Timeframe lookahead bias |
| NVTK Residual Gap | 2026-04 | 3-hour lookahead artifact |
| Stock Residual Gap (all) | 2026-04 | IMOEX opens 09:50, stocks open 06:50 |
| SmartFilter on EMA base | 2026-04 | Filter on killed strategy = invalid |
| H1: BR Residual Gap (Brent) | 2026-04-15 | PF=0.83, placebo p=0.499 |

### 1.2 OPEN HYPOTHESES (Not Yet Killed)

| ID | Hypothesis | Status | Blocker |
|----|------------|--------|---------|
| M1 | Opening Auction Imbalance → First 5m | NOT TESTED | Need auction data |
| M2 | First Hour Reversal (session open) | RESEARCH | Low n (14 trades) |
| M3 | Trade Flow Imbalance → 10m prediction | NOT TESTED | Need trade tape |
| M4 | Close Pressure → Overnight Gap | NOT TESTED | Need implementation |
| M5 | Queue Depletion at Support/Resistance | NOT TESTED | Need L2 data |
| M6 | Cross-Asset Lead-Lag (ES→RI, Brent→BR) | PARTIAL | Need sync logic |

### 1.3 SURVIVED FROM PREVIOUS RESEARCH

| Finding | Source | Status |
|---------|--------|--------|
| Session quality varies (Lunch best, Morning worst) | Phase 3 | VALIDATED |
| Opening Drive (10:00-10:15) has 50-66% WR | Phase 3 | TENTATIVE |
| First Hour Reversal BR 1.0% threshold | H3 test | LOW N (n=14) |
| Trade imbalance 10s most reliable micro feature | microstructure.py | UNTESTED |

---

## 2. SESSION MAP (Moscow Time, UTC+3)

```
┌──────────────────────────────────────────────────────────────────┐
│ MOEX TRADING SESSIONS                                            │
├──────────────────────────────────────────────────────────────────┤
│ 06:50─07:00  MORNING_AUCTION (stocks)     ░░░░░ THIN             │
│ 07:00─10:00  MORNING_SESSION (FORTS only) ░░░░░ WORST WR 25-27%  │
│                                                                  │
│ 09:50─10:00  STOCK_AUCTION                ▓▓▓▓▓ KEY EVENT        │
│ 10:00─10:15  OPENING_DRIVE                ████▓ GOOD 50-66%      │
│ 10:15─11:30  MORNING_ACTIVE               ███░░ MIXED 33-37%     │
│ 11:30─13:00  MIDDAY                       ████░ GOOD 39-100%     │
│ 13:00─14:00  LUNCH                        █████ BEST 53-75%      │
│                                                                  │
│ 14:00─14:05  CLEARING_1 (FORTS)           ───── NO TRADING       │
│                                                                  │
│ 14:05─16:00  AFTERNOON                    ████░ MIXED 28-75%     │
│ 16:00─18:40  PRE_CLOSE                    ███░░ MIXED 32-50%     │
│ 18:40─18:50  CLOSING_AUCTION (stocks)     ▓▓▓▓▓ KEY EVENT        │
│                                                                  │
│ 18:45─19:05  CLEARING_2 (FORTS)           ───── NO TRADING       │
│                                                                  │
│ 19:05─23:50  EVENING_SESSION (FORTS)      ░░░░░ THIN             │
│ 23:20─23:50  EVENING_CLOSE                ░░░░░ SKIP             │
└──────────────────────────────────────────────────────────────────┘

KEY TIMES FOR MICROSTRUCTURE:
• 09:50    Stock auction open (imbalance observable)
• 10:00    Main session open (gap realization)
• 10:00    First 5-30 min (reversal/continuation)
• 13:00    Lunch session start (best WR)
• 18:30    Pre-close pressure begins
• 18:40    Closing auction (imbalance → overnight)
```

---

## 3. DATA MAP

### 3.1 AVAILABLE FROM ISS API (Public, Free)

| Data | Endpoint | Granularity | Latency |
|------|----------|-------------|---------|
| OHLCV Candles | `/candles.json` | 1m, 10m, 60m, 1d | ~1 min |
| Current Quote | `/securities/{id}.json` | LAST, BID, OFFER | Real-time* |
| Volume/Value | `/marketdata` | VOLTODAY, VALTODAY | Real-time* |
| Trade Count | `/marketdata` | NUMTRADES | Real-time* |

*Note: "Real-time" is ~500ms-1s delay via HTTP polling

**Limitations**:
- No tick-by-tick trades
- No full order book (only best bid/ask)
- No order flow direction
- No auction data

### 3.2 AVAILABLE FROM BCS QUIK API (Broker, Requires Account)

| Data | Source | Granularity | Storage |
|------|--------|-------------|---------|
| Order Book (10-50 levels) | `getQuoteLevel2` | 500ms snapshots | SQLite |
| Trade Tape | `OnAllTrade` callback | Tick-by-tick | SQLite |
| Open Interest | `getFuturesHolding` | On change | SQLite |

**Already Implemented** (`collect_microstructure.py`):
```python
# Order book snapshot every 500ms
OrderBookSnapshot:
  - bid_prices[10], bid_volumes[10]
  - ask_prices[10], ask_volumes[10]
  - timestamp, ticker

# Trade tape
Trade:
  - timestamp, ticker, price, volume, side
```

### 3.3 NOT AVAILABLE (Would Need Exchange Feed)

| Data | Why Needed | Alternative |
|------|------------|-------------|
| Full Orders Log | Event-based OFI | Estimate from trades |
| Cancel/Add Ratio | Queue dynamics | Depth change proxy |
| Order Lifetime | Informed trading | Not available |
| Auction Order Book | Pre-open imbalance | Post-auction reconstruction |

### 3.4 DATA QUALITY BLOCKERS

| Blocker | Impact | Workaround |
|---------|--------|------------|
| No auction data | Can't test M1 | Use first 1-min bar |
| QUIK data not historical | Can't backtest micro | Collect forward |
| Trade side inference | May be wrong | Use signed tick rule |
| Evening session gaps | Missing data | Mark as blocker |

---

## 4. EVENT DEFINITIONS

### E1: Opening Auction Imbalance

```
Trigger: 09:59:55 (5 sec before open)
Signal:  auction_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

Event Definition:
- Time: T = 10:00:00 MSK
- Entry: T + 1 second (first trade after open)
- Exit: T + 5m, T + 15m, T + 30m (test multiple)

Signal Logic:
- IF auction_imbalance > +0.3 THEN expect LONG momentum
- IF auction_imbalance < -0.3 THEN expect SHORT momentum
- Threshold: test 0.1, 0.2, 0.3, 0.5

BLOCKER: Auction order book not available via QUIK.
Workaround: Use first 1-min bar imbalance as proxy.
```

### E2: First Hour Reversal

```
Trigger: 11:00:00 MSK (1 hour after open)
Signal:  first_hour_return = close_11:00 / open_10:00 - 1

Event Definition:
- Time: T = 11:00:00 MSK
- Entry: T + 1 minute
- Exit: T + 1h, T + 2h, EOD, next_open

Signal Logic:
- IF first_hour_return > +1.0% THEN SHORT (reversal)
- IF first_hour_return < -1.0% THEN LONG (reversal)
- Threshold: test 0.5%, 0.7%, 1.0%, 1.5%

Current Status: BR main 1.0% threshold, n=14, PF=1.28
BLOCKER: Low n. Need 180 days data for n>50.
```

### E3: Trade Flow Imbalance (10s window)

```
Trigger: Continuous (every 10 seconds during session)
Signal:  trade_imbalance_10s = (buy_vol - sell_vol) / total_vol

Event Definition:
- Time: Any T during 10:15-18:30 (avoid open/close noise)
- Entry: When imbalance crosses threshold
- Exit: T + 5m, T + 10m

Signal Logic:
- IF trade_imbalance_10s > +0.4 AND rising THEN LONG
- IF trade_imbalance_10s < -0.4 AND falling THEN SHORT
- Confirmation: depth_change_10s same direction

BLOCKER: Need live trade tape collection first.
Current: microstructure.py has logic but no historical data.
```

### E4: Close Pressure → Overnight Gap

```
Trigger: 18:30:00 MSK (10 min before close)
Signal:  close_pressure = trade_imbalance_30s + depth_change_30s

Event Definition:
- Time: T = 18:30 - 18:40 MSK
- Entry: 18:40 (at close auction)
- Exit: Next day 10:05 (5 min after open)

Signal Logic:
- IF close_pressure > +0.3 THEN expect positive gap → LONG
- IF close_pressure < -0.3 THEN expect negative gap → SHORT

BLOCKER: Need to define "close_pressure" precisely.
Need: 30+ days of close imbalance vs next-day gap correlation.
```

### E5: Queue Depletion at Key Level

```
Trigger: Price approaching known support/resistance
Signal:  depth_change = (current_depth - prev_depth) / prev_depth

Event Definition:
- Time: When price within 0.1% of S/R level
- Entry: When depth_change_10s < -30% (queue depleting)
- Exit: T + 5m, T + 10m

Signal Logic:
- IF at support AND bid_depth depleting THEN expect breakdown → SHORT
- IF at resistance AND ask_depth depleting THEN expect breakout → LONG

BLOCKER: Need S/R level detection logic.
Need: Rolling high/low or order flow profile.
```

### E6: Cross-Asset Lead-Lag

```
Trigger: Continuous
Signal:  lead_signal = external_return - moex_return

Event Definition:
- Pairs: ES → RI, ICE Brent → BR
- Time window: 07:00-10:00 MSK (overnight move realized)
- Entry: 10:00 (MOEX main open)
- Exit: T + 1h, T + 2h, EOD

Signal Logic:
- IF ES_overnight > +0.5% AND RI_gap < +0.3% THEN LONG RI (catch-up)
- IF Brent_overnight > +1% AND BR_gap < +0.5% THEN LONG BR (catch-up)

Current Status: external_feeds.py exists but not tested.
BLOCKER: Need timezone-aligned data merge.
```

---

## 5. MICROSTRUCTURE FEATURES (19 Available)

```
ORDER BOOK (9 features):
├── micro_imbalance_5       OB imbalance top 5 levels [-1, +1]
├── micro_imbalance_10      OB imbalance top 10 levels
├── micro_microprice_gap    Weighted mid vs simple mid [-0.5, +0.5]
├── micro_spread_vs_median  Current spread / session median
├── micro_depth_ratio       Bid depth / Ask depth
├── micro_depth_change_10s  Depth % change (queue depletion proxy)
├── micro_depth_change_30s  Depth % change over 30s
├── micro_imbalance_trend   Slope of imbalance (OFI proxy)
└── micro_bid_wall          Max bid / mean bid (wall detection)

TRADE TAPE (8 features):
├── micro_trade_imbalance_10s  (buy-sell)/total 10s window  ← MOST RELIABLE
├── micro_trade_imbalance_30s  (buy-sell)/total 30s window
├── micro_trade_imbalance_60s  (buy-sell)/total 60s window
├── micro_cvd                  Cumulative Volume Delta
├── micro_large_trade_ratio    Fraction from trades > 50 lots
├── micro_trade_intensity      Trades/sec vs session average
├── micro_avg_buy_size         Mean aggressor buy size
└── micro_avg_sell_size        Mean aggressor sell size

FUTURES (2 features):
├── micro_oi_change            OI % change over 60s
└── micro_oi_divergence        OI vs price divergence [+1, 0, -1]
```

---

## 6. FALSIFICATION PROTOCOL

Every hypothesis MUST pass all tests before promotion.

### Test 1: Placebo Shuffle
```python
def placebo_shuffle(signals, returns):
    """Signal timing should matter. Shuffle should kill edge."""
    shuffled = np.random.permutation(signals)
    placebo_pf = calc_pf(shuffled, returns)
    assert placebo_pf < real_pf * 0.8, "Placebo should be worse"
    assert abs(placebo_pf - 1.0) < 0.2, "Placebo should be near 1.0"
```

### Test 2: Placebo Reverse
```python
def placebo_reverse(signals, returns):
    """Reversing signal should lose money."""
    reversed_signals = -signals
    reverse_pf = calc_pf(reversed_signals, returns)
    assert reverse_pf < 1.0, "Reversed signal should lose"
```

### Test 3: Time-of-Day Stability
```python
def tod_stability(signals, returns, sessions):
    """Edge should exist in multiple sessions, not just one."""
    session_pfs = {}
    for session in sessions:
        mask = get_session_mask(session)
        session_pfs[session] = calc_pf(signals[mask], returns[mask])
    passing = sum(1 for pf in session_pfs.values() if pf > 1.0)
    assert passing >= 2, "Edge should work in at least 2 sessions"
```

### Test 4: Parameter Plateau
```python
def parameter_plateau(param_range, run_backtest):
    """Edge should exist in neighborhood, not just optimal point."""
    results = {p: run_backtest(param=p).pf for p in param_range}
    positive = sum(1 for pf in results.values() if pf > 1.0)
    assert positive / len(results) > 0.5, "Should work in >50% of range"
```

### Test 5: Cost Shock
```python
def cost_shock(signals, returns, cost_levels=[0.1, 0.2, 0.3]):
    """Edge should survive 2x costs."""
    for cost_pct in cost_levels:
        pf = calc_pf(signals, returns, cost=cost_pct)
        if cost_pct <= 0.2:
            assert pf > 1.0, f"Should be profitable at {cost_pct}%"
        else:
            assert pf > 0.9, f"Should be near break-even at {cost_pct}%"
```

### Test 6: Walk-Forward
```python
def walk_forward(data, train_days=60, test_days=30):
    """Edge should persist forward in time."""
    results = []
    for i in range(n_folds):
        train = data[i*30 : i*30 + train_days]
        test = data[i*30 + train_days : i*30 + train_days + test_days]
        params = optimize(train)
        test_pf = backtest(test, params)
        results.append(test_pf > 1.0)
    assert sum(results) / len(results) > 0.6, ">60% folds profitable"
```

### Test 7: Final Holdout
```python
def final_holdout(data, holdout_days=30):
    """After all optimization, run on untouched data."""
    train_val = data[:-holdout_days]
    holdout = data[-holdout_days:]

    final_params = optimize(train_val)  # Freeze params
    holdout_pf = backtest(holdout, final_params)
    assert holdout_pf > 1.0, "Holdout must be profitable"
```

---

## 7. TOP 5 PRIORITY TESTS

### Priority 1: First-5-Minutes Imbalance After Open

**Why**: Opening has highest information content. Auction reveals institutional order flow.

**Exact Definition**:
```
Time:   10:00:00 - 10:05:00 MSK
Signal: bar1_imbalance = (close_1m - open_1m) / (high_1m - low_1m)
        # >+0.5 = closed near high (buying), <-0.5 = closed near low (selling)
Entry:  10:05:00 if |bar1_imbalance| > 0.6
Exit:   10:15:00, 10:30:00, 11:00:00 (test multiple)
Direction: LONG if imbalance > +0.6, SHORT if imbalance < -0.6
```

**Data Required**: ISS 1-min candles (available)
**Falsification**: Placebo, session stability, cost shock

---

### Priority 2: Trade Flow Divergence vs Price

**Why**: Price moving without volume = weak move, likely to reverse.

**Exact Definition**:
```
Time:   Any T during 10:30-18:00 (avoid open/close)
Signal: divergence = sign(price_change_5m) * (-1 * volume_change_5m)
        # Positive divergence = price up but volume down (weak bull)
Entry:  When divergence > +0.15 for 3 consecutive bars
Exit:   T + 15m, T + 30m
Direction: FADE the move (SHORT if divergence from up-move)
```

**Data Required**: ISS 1-min candles (available)
**Falsification**: Placebo, parameter plateau, walk-forward

---

### Priority 3: Pre-Close Imbalance → Next-Day Gap

**Why**: Institutional re-balancing at close predicts overnight gap.

**Exact Definition**:
```
Time:   18:30:00 - 18:40:00 MSK
Signal: close_imbalance = sum(bar_imbalance) for last 10 1-min bars
        bar_imbalance = (close - open) / (high - low + 0.0001)
Entry:  18:40 (hold overnight)
Exit:   10:05 next day
Direction: LONG if close_imbalance > +0.5, SHORT if < -0.5
```

**Data Required**: ISS 1-min candles (available)
**Falsification**: Correlation test with actual gaps, cost shock (overnight margin)

---

### Priority 4: Lunch Session Entry (Best WR Window)

**Why**: Phase 3 showed Lunch (13:00-14:00) has 53-75% WR, best of all sessions.

**Exact Definition**:
```
Time:   13:00:00 - 13:30:00 MSK
Signal: momentum_5m = close / close_5m_ago - 1
        volume_surge = volume_5m / volume_20m_avg > 1.5
Entry:  When momentum_5m > +0.3% AND volume_surge
Exit:   13:45, 14:00 (before clearing)
Direction: Continuation (same as momentum direction)
```

**Data Required**: ISS 1-min candles (available)
**Falsification**: Compare to random entry in same window, cost shock

---

### Priority 5: Depth Depletion Before Move

**Why**: Liquidity providers pull quotes before informed flow.

**Exact Definition**:
```
Time:   Continuous during 10:15-18:30
Signal: depth_drop = (current_depth - depth_10s_ago) / depth_10s_ago
        # Rapid drop in bid depth = sellers incoming
Entry:  When depth_drop < -20% in 10 seconds
Exit:   T + 1m, T + 5m
Direction: OPPOSITE to depleted side (if bid depletes → SHORT)
```

**Data Required**: QUIK order book snapshots (500ms)
**Blocker**: Need live data collection first
**Falsification**: Placebo, side symmetry

---

## 8. IMPLEMENTATION PRIORITY

### Phase 1: ISS-Based Tests (No New Data Needed)

| Test | Data Source | ETA |
|------|-------------|-----|
| First-5-Min Imbalance | ISS 1m candles | 1 day |
| Trade Flow Divergence | ISS 1m candles | 1 day |
| Pre-Close Imbalance | ISS 1m candles | 1 day |
| Lunch Session Entry | ISS 1m candles | 1 day |

### Phase 2: QUIK-Based Tests (Requires Data Collection)

| Test | Data Source | Collection Time |
|------|-------------|-----------------|
| Depth Depletion | QUIK L2 | 30+ days |
| Trade Imbalance 10s | QUIK trades | 30+ days |
| Opening Auction Imbalance | QUIK L2 | 30+ days (if available) |

### Phase 3: Cross-Asset Tests (Requires Data Sync)

| Test | Data Sources | Complexity |
|------|--------------|------------|
| ES → RI Lead-Lag | Yahoo + ISS | Medium |
| Brent → BR Lead-Lag | Yahoo + ISS | Medium |

---

## 9. SUCCESS CRITERIA

A hypothesis becomes PAPER-CANDIDATE when:

1. **Profit Factor > 1.2** (after costs)
2. **N trades > 50** (statistical significance)
3. **Placebo shuffle p < 0.1** (not random)
4. **Placebo reverse PF < 0.9** (direction matters)
5. **Walk-forward > 60% folds** (forward persistence)
6. **Final holdout PF > 1.0** (unseen data)
7. **Cost shock 2x PF > 1.0** (margin of safety)
8. **Works in 2+ sessions** (not time-specific artifact)

---

## 10. NEXT STEPS

1. **Immediate**: Implement Priority 1-4 tests using ISS data
2. **Week 1**: Run falsification protocol on each
3. **Week 2**: Start QUIK data collection for Priority 5
4. **Week 3**: If any test passes, proceed to walk-forward
5. **Week 4+**: Final holdout on survivors

**Do NOT proceed to live trading until at least one hypothesis passes ALL 8 criteria.**
