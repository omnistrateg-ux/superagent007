# FUTOI Positioning Edge Validation

**Date**: 2026-04-16
**Status**: **CONFIRMED** EDGE ON NG (Natural Gas)

---

## Executive Summary

Tested 4 hypotheses across 5 futures (BR, RI, MX, NG, Si) using FUTOI positioning data (YUR institutional vs FIZ retail). **2 hypotheses passed all validation gates on NG futures only**. All other ticker/hypothesis combinations were killed.

| Hypothesis | BR | RI | MX | NG | Si |
|-----------|----|----|----|----|-----|
| H1: Extreme YUR → Reversal | KILLED (PF<1.2) | INSUFF | INSUFF | KILLED | INSUFF |
| H2: YUR Delta → Continuation | KILLED (holdout) | KILLED | KILLED | **PASSED** | KILLED |
| H3: YUR/FIZ Divergence → Follow YUR | KILLED (PF<1.2) | KILLED (p>0.1) | KILLED (holdout) | **PASSED** | KILLED |
| H4: Extreme Long Ratio → Reversal | KILLED | KILLED (p>0.1) | KILLED | KILLED | KILLED |

---

## Passed Hypotheses Detail

### H2: YUR Flow Delta → Continuation (NG)

**Logic**: When YUR (institutional) adds to their position (delta > 0 = adding longs, delta < 0 = adding shorts), next-day price continues in that direction.

| Metric | Discovery (2020-2023) | Holdout (2024-2025) |
|--------|----------------------|---------------------|
| Trades | 973 | 502 |
| Profit Factor | 1.56 | 1.39 |
| Sharpe | 2.59 | 1.96 |
| p-value | < 0.001 | 0.003 |
| Total Return | +489.8% | +178.7% |

**Falsification Tests**:
| Test | Result | Pass? |
|------|--------|-------|
| Reversed signal PF | 0.64 | ✅ (< 1.0) |
| Shuffle p95 | 1.15 | ✅ (original > p95) |
| With 5bps costs PF | 1.50 | ✅ (> 1.0) |

### H3: YUR/FIZ Divergence → Follow YUR (NG)

**Logic**: When YUR and FIZ disagree on direction (one net long, other net short), follow YUR direction.

| Metric | Discovery (2020-2023) | Holdout (2024-2025) |
|--------|----------------------|---------------------|
| Trades | 976 | 502 |
| Profit Factor | 1.40 | 1.27 |
| Sharpe | 1.96 | 1.42 |
| p-value | < 0.001 | 0.023 |
| Total Return | +374.1% | +129.9% |

**Falsification Tests**:
| Test | Result | Pass? |
|------|--------|-------|
| Reversed signal PF | 0.71 | ✅ (< 1.0) |
| Shuffle p95 | 1.16 | ✅ (original > p95) |
| With 5bps costs PF | 1.34 | ✅ (> 1.0) |

---

## Killed Hypotheses Summary

### By Ticker

**BR (Brent)**:
- H1: PF=1.17 < 1.2 threshold
- H2: Discovery PF=1.24 ✓, but Holdout PF=0.85 (KILLED)
- H3: PF=1.07 < 1.2 threshold
- H4: PF=0.44 < 1.2 threshold

**RI (RTS Index)**:
- H1: Insufficient data (39 trades < 50)
- H2: PF=1.03 < 1.2 threshold
- H3: PF=1.24 ✓, but p=0.178 > 0.10 threshold
- H4: PF=1.57 ✓, but p=0.185 > 0.10 threshold

**MX (MOEX Index)**:
- H1: Insufficient data (19 trades < 50)
- H2: PF=0.99 < 1.2 threshold
- H3: Discovery PF=1.35 ✓, but Holdout PF=0.89 (KILLED)
- H4: PF=0.21 < 1.2 threshold

**Si (USD/RUB)**:
- H1: Insufficient data (36 trades < 50)
- H2: PF=1.11 < 1.2 threshold
- H3: PF=0.99 < 1.2 threshold
- H4: PF=1.04 < 1.2 threshold

---

## Data Summary

| Source | Records | Period | Coverage |
|--------|---------|--------|----------|
| FUTOI | 180,928 | 2020-01-03 to 2025-12-30 | 1,504 days |
| Prices (BR) | 1,308 | 2020-2025 | Front-month continuous |
| Prices (RI) | 336 | 2020-2025 | Limited liquidity |
| Prices (MX) | 691 | 2020-2025 | Limited liquidity |
| Prices (NG) | 1,497 | 2020-2025 | Good coverage |
| Prices (Si) | 940 | 2020-2025 | Good coverage |

**FUTOI Schema**:
- `clgroup`: YUR (institutional/juridical) vs FIZ (retail/physical)
- `pos`: Net position (long - short contracts)
- `pos_long` / `pos_short`: Gross positions
- `pos_long_num` / `pos_short_num`: Number of traders on each side

---

## Methodology

### Validation Gates

1. **Discovery (2020-2023)**: PF ≥ 1.2, p-value < 0.10, trades ≥ 50
2. **Holdout (2024-2025)**: PF ≥ 1.0
3. **Reversed Signal**: PF < 1.0 (confirms signal polarity matters)
4. **Shuffle Test**: Original PF > 95th percentile of 100 shuffled runs
5. **With Costs**: PF ≥ 1.0 after 5bps round-trip costs

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `yur_net` | YUR net position (contracts) |
| `fiz_net` | FIZ net position (contracts) |
| `yur_net_delta` | Daily change in YUR net |
| `yur_net_pct` | YUR net as % of total OI |
| `yur_net_pct_zscore` | 60-day rolling z-score |
| `yur_fiz_divergence` | 1 if sign(YUR) ≠ sign(FIZ), else 0 |
| `yur_long_ratio` | YUR longs / (longs + shorts) |

### Signal Definitions

| Signal | Long (+1) | Short (-1) | Neutral (0) |
|--------|-----------|------------|-------------|
| H1 (Reversal) | z-score < -2 | z-score > 2 | Otherwise |
| H2 (Continuation) | delta > 0 | delta < 0 | delta = 0 |
| H3 (Divergence) | diverge & YUR long | diverge & YUR short | No divergence |
| H4 (Long Ratio) | z-score < -2 | z-score > 2 | Otherwise |

---

---

## Extended Validation (2026-04-16)

### Walk-Forward Analysis (6 Folds by Year)

| Hypothesis | 2021 | 2022 | 2023 | 2024 | 2025 | Pass Rate |
|------------|------|------|------|------|------|-----------|
| H2: YUR Delta | PF=1.46 ✓ | PF=1.62 ✓ | PF=1.71 ✓ | PF=1.13 ✓ | PF=1.66 ✓ | **5/5** |
| H3: Divergence | PF=1.16 ✓ | PF=1.52 ✓ | PF=1.65 ✓ | PF=1.05 ✓ | PF=1.48 ✓ | **5/5** |

**Verdict**: CONFIRMED - Both hypotheses profitable in all 5 walk-forward folds.

### Quarterly Stability (24 Quarters)

| Hypothesis | Profitable Quarters | Loss Quarters | Stability |
|------------|---------------------|---------------|-----------|
| H2: YUR Delta | 22 | 2 (2020-Q3, 2024-Q4) | **92%** |
| H3: Divergence | 21 | 3 (2021-Q1, 2024-Q1, 2024-Q4) | **88%** |

**Verdict**: CONFIRMED - Edge is not clustered in a few periods.

### Regime Independence (Bull vs Bear)

| Hypothesis | Bull Market PF | Bear Market PF | Neutral PF |
|------------|---------------|----------------|------------|
| H2: YUR Delta | 1.45 | 1.40 | 1.76 |
| H3: Divergence | 1.46 | 1.30 | 1.27 |

**Verdict**: CONFIRMED - Edge works in both rising AND falling markets.

### Signal Monotonicity (Stronger Signal = Better Returns?)

**H2: YUR Delta**
| Bucket | PF | Sharpe |
|--------|-----|--------|
| Q1 (weak) | 1.18 | 1.00 |
| Q2 | 1.45 | 2.12 |
| Q3 | 1.60 | 2.77 |
| Q4 | 1.60 | 2.85 |
| Q5 (strong) | 1.79 | 3.35 |

**Spearman correlation = 1.00** (perfect monotonicity)

**H3: Divergence**
| Bucket | PF | Sharpe |
|--------|-----|--------|
| Q1 (weak) | 1.51 | 2.47 |
| Q5 (strong) | 1.71 | 3.27 |

**Spearman correlation = 0.10** (weak but positive)

**Verdict**: CONFIRMED - H2 shows perfect monotonicity, H3 acceptable.

---

## Risk Warnings

1. **Single Ticker**: Edge found only on NG, not diversified
2. **Modest PF**: Holdout PF 1.27-1.39 leaves thin margin for error
3. **Execution Risk**: Next-day close assumed; slippage not modeled
4. **Regime Change**: 2020-2023 includes COVID volatility, may not repeat
5. **Correlation**: H2 and H3 overlap (NG signals on same days)

---

## Recommendations

### Before Live Trading

1. **Paper Trade**: Run signals for 30+ days without capital
2. **Execution Study**: Measure actual fill prices vs. daily close
3. **Position Sizing**: Start with 0.5% of capital per signal
4. **Stop Loss**: Define max drawdown threshold (suggest 10%)

### Next Research Steps

1. **Combine H2+H3**: Test if combined signal improves PF
2. **Intraday Timing**: Test entry at open vs. close vs. VWAP
3. **NG Seasonality**: Check if edge is stronger in certain months
4. **Cross-Market**: Test NG positioning vs. crude oil (BR) returns

---

## Files

| File | Description |
|------|-------------|
| `run_futoi_research.py` | Full research pipeline |
| `data/futoi_futures/futoi_all.parquet` | Raw FUTOI data |
| `data/prices_daily_cache.parquet` | Cached daily prices |
| `data/futoi_signals.csv` | Generated signals with features |
| `data/futoi_research_results.csv` | Hypothesis test results |

---

## Conclusion

**VERDICT**: **CONFIRMED** edge on NG (Natural Gas) futures.

Two institutional flow-following strategies passed ALL validation gates:

### H2: YUR Delta → Continuation
- **Logic**: Trade in direction of daily YUR (institutional) flow change
- **Walk-forward**: 5/5 folds profitable
- **Quarterly**: 22/24 quarters profitable (92%)
- **Regime**: Works in bull (PF=1.45) AND bear (PF=1.40)
- **Monotonicity**: Perfect correlation (1.00) - stronger signals = better returns

### H3: YUR/FIZ Divergence
- **Logic**: When institutions and retail disagree, follow institutions
- **Walk-forward**: 5/5 folds profitable
- **Quarterly**: 21/24 quarters profitable (88%)
- **Regime**: Works in bull (PF=1.46) AND bear (PF=1.30)
- **Monotonicity**: Positive correlation (0.10)

### Next Steps (Before Live Trading)

1. **Paper trade 30 days** - verify execution matches backtest
2. **Start small** - 0.5% of capital per signal
3. **Monitor regime** - 2024-Q4 showed weakness, watch for degradation
4. **Combine signals** - H2+H3 overlap may improve risk-adjusted returns

**EDGE IS CONFIRMED BUT NOT GUARANTEED** - market conditions can change.
