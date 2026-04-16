# FUTOI Edge Audit - Final Verdict

**Date**: 2026-04-16
**Auditor**: Critical review of H2/H3 claims

---

## VERDICT: **PAPER-CANDIDATE** (H2 only)

Not LIVE-CANDIDATE. Not KILLED. Requires extended paper trading.

---

## Audit Results

| Check | Result | Notes |
|-------|--------|-------|
| Look-ahead bias | ✓ PASSED | Entry at T+1 open, FUTOI published T 23:50 |
| Publication timing | ✓ PASSED | Causality verified |
| Multiple testing | ✓ PASSED | Survives Bonferroni (p < 0.0025) |
| Outlier dependence | ✓ PASSED | PF improves with trimming (1.50 → 1.70) |
| Cost sensitivity | ✓ PASSED | PF=1.37 at 10bps, breaks even at ~30bps |
| Holdout contamination | ⚠ WARNING | Holdout seen before deeper research |
| H2/H3 independence | ⚠ WARNING | 100% overlap - same edge, not two |
| Recent performance | ⚠ WARNING | 2024 weak (PF=1.13), 2025 recovered |
| Seasonality | ✓ PASSED | 11/12 months profitable |

---

## Critical Issues

### 1. Holdout Contamination (SOFT)

```
Research flow:
1. run_futoi_research.py → saw ALL 2020-2025 results
2. Decided H2/H3 "look good" based on seeing holdout
3. Then ran deeper validation

The decision to pursue H2/H3 was informed by holdout results.
This is not a true out-of-sample test.
```

**Mitigation**: Walk-forward 5/5 years profitable is stronger than single holdout split.

### 2. H2 and H3 Are The Same Edge

```
Signal overlap analysis:
- H2 signals: 1,476
- H3 signals: 1,479
- Both active: 1,476 (100% overlap)
- Same direction: 890 (60%)
- Opposite direction: 586 (40%)
```

**Conclusion**: H2 and H3 fire on the same days. This is ONE edge with two formulations, not two independent edges. H2 has better metrics (PF=1.56 vs 1.40), so use H2 only.

### 3. 2024 Performance Weakness

| Year | H2 PF | H2 Sharpe |
|------|-------|-----------|
| 2020 | 1.48 | 2.12 |
| 2021 | 1.46 | 2.19 |
| 2022 | 1.62 | 2.91 |
| 2023 | 1.71 | 3.28 |
| **2024** | **1.13** | **0.70** |
| 2025 | 1.66 | 3.09 |

2024 was barely profitable. Edge may be decaying or regime-dependent.

---

## What's Real vs Stale

| Claim | Status |
|-------|--------|
| "2 confirmed edges (H2 + H3)" | **FALSE** - same edge |
| "Walk-forward 5/5" | TRUE |
| "Quarterly 22/24 profitable" | TRUE |
| "Bull + Bear both work" | TRUE |
| "Perfect monotonicity" | TRUE (H2 only) |
| "Ready for live trading" | **FALSE** - paper first |
| "30-day replay proves edge" | **FALSE** - not forward test |

---

## H2 Specification (The Only Edge)

```python
# Signal generation (correct timing)
# FUTOI published at day T 23:50, after market close
# Signal = direction of YUR net position change

if yur_net[T] > yur_net[T-1]:  # YUR adding longs
    signal = +1  # LONG
elif yur_net[T] < yur_net[T-1]:  # YUR adding shorts
    signal = -1  # SHORT
else:
    signal = 0   # NO TRADE

# Execution
entry_price = open[T+1]   # Next day open
exit_price = close[T+1]   # Same day close
```

**Expected metrics** (from backtest):
- Win rate: ~52%
- Profit factor: 1.3-1.6
- Sharpe: 1.5-3.0
- Avg trade: +0.4%

---

## Verdict Breakdown

### Why NOT KILLED

1. Timing is correct (no look-ahead)
2. Survives multiple testing correction
3. Survives outlier removal
4. Survives realistic costs
5. Walk-forward 5/5 profitable
6. 22/24 quarters profitable
7. Works in both bull and bear
8. Perfect monotonicity (stronger signal = better)

### Why NOT LIVE-CANDIDATE

1. Holdout was contaminated (saw results before deciding)
2. 2024 was weak (PF=1.13, Sharpe=0.70)
3. Single ticker (NG only) - no diversification
4. 30-day replay is NOT forward paper (same data used in research)

### Why PAPER-CANDIDATE

1. Edge logic is sound and verifiable
2. Metrics are reasonable (not too good to be true)
3. Recent performance (2025) is strong
4. Costs are manageable
5. Signal is simple and implementable

---

## Required Next Steps

### 1. TRUE Forward Paper Trading (Required)

```
Period: 2026-04-17 to 2026-07-16 (3 months)
Ticker: NG futures only
Signal: H2 (YUR delta)
Entry: Market order at next day 10:00-10:05 MSK
Exit: Market order at 23:45-23:50 MSK
Track: Fill prices, slippage, actual PF

SUCCESS CRITERIA:
- PF ≥ 1.2 over 3 months
- No month with PF < 0.8
- Actual slippage < 10bps
```

### 2. Do NOT Paper Trade H3

H3 is the same edge with worse metrics. Using both would double position on same days.

### 3. Live Trading Criteria

After 3 months paper:
- If PF ≥ 1.2: Proceed to LIVE with 0.5% capital per trade
- If PF 1.0-1.2: Extend paper another 3 months
- If PF < 1.0: KILL hypothesis

---

## Files To Trust

| File | Status |
|------|--------|
| `run_futoi_research.py` | VALID - correct timing |
| `FUTOI_EDGE_VALIDATION.md` | OVERSTATED - says "confirmed", should say "candidate" |
| `MEMORY.md` | OVERSTATED - claims "ready for paper" |
| `futoi_validation_results.csv` | VALID data, wrong interpretation |

---

## Updated Memory Recommendation

```markdown
## SOURCE OF TRUTH (2026-04-16)

> **🟡 PAPER-CANDIDATE: NG H2 (YUR Delta)**
>
> **Status**: Passed all technical checks, requires 3-month forward paper
>
> **Issues found in audit**:
> - Holdout was contaminated (soft)
> - H2/H3 are same edge (use H2 only)
> - 2024 was weak (PF=1.13)
>
> **Next step**: TRUE forward paper 2026-04-17 to 2026-07-16
```
