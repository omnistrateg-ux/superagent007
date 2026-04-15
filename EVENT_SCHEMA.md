# Event Schema for Microstructure Hypotheses

**Version**: 1.0
**Date**: 2026-04-15
**Status**: RESEARCH SPECIFICATION

---

## Overview

This document defines 4 microstructure hypotheses with:
- Required data
- Label horizon
- Falsification tests
- Fail conditions

---

## M1: Opening Imbalance (First 5 Minutes)

### Hypothesis

Orderflow imbalance in first 5 minutes after open predicts 30-minute direction.

**Mechanism**: Overnight orders accumulated during auction create initial momentum. Large imbalance = informed flow direction revealed.

### Event Definition

```yaml
event_id: M1_opening_imbalance
trigger_time: 10:00:00 MSK (main session open)
observation_window: 10:00:00 - 10:05:00 MSK (5 minutes)
entry_time: 10:05:00 MSK
exit_horizons: [15m, 30m, 60m]

tickers:
  futures: [BR, RI, MX]
  stocks: [SBER, GAZP, LKOH]
```

### Required Data

| Data | Granularity | Source | Priority |
|------|-------------|--------|----------|
| Trade tape with side | Tick | QUIK | CRITICAL |
| L2 orderbook | 500ms | QUIK | HIGH |
| 5-min OHLCV | 5m | ISS (backup) | PROXY ONLY |

### Signal Calculation

```python
def calc_opening_imbalance(trades: List[Trade]) -> float:
    """
    Calculate orderflow imbalance in first 5 minutes.

    Returns:
        Imbalance in [-1, +1]. Positive = net buying.
    """
    window_trades = [t for t in trades if t.ts >= OPEN and t.ts < OPEN + 5min]

    buy_volume = sum(t.qty for t in window_trades if t.side == 'BUY')
    sell_volume = sum(t.qty for t in window_trades if t.side == 'SELL')

    total = buy_volume + sell_volume
    if total == 0:
        return 0.0

    return (buy_volume - sell_volume) / total

# Signal thresholds to test
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
```

### Label Definition

```python
def calc_label(entry_price: float, future_prices: Dict[str, float]) -> Dict:
    """
    Calculate forward returns at multiple horizons.
    """
    return {
        "return_15m": (future_prices["15m"] - entry_price) / entry_price,
        "return_30m": (future_prices["30m"] - entry_price) / entry_price,
        "return_60m": (future_prices["60m"] - entry_price) / entry_price,
        "label_30m": 1 if future_prices["30m"] > entry_price else 0,
    }
```

### Falsification Tests

| Test | Method | Pass Condition |
|------|--------|----------------|
| **F1: Placebo Shuffle** | Randomly shuffle imbalance values across days | Real PF > 1.5 × Placebo PF |
| **F2: Placebo Reverse** | Reverse signal direction | Reverse PF < 0.9 |
| **F3: Side Symmetry** | Test LONG-only vs SHORT-only | Both sides PF > 1.0 |
| **F4: Threshold Plateau** | Test thresholds ±0.1 | >50% of range has PF > 1.0 |
| **F5: Session Stability** | Test by session type | Works in 2+ session types |
| **F6: Cost Shock** | Apply 2x transaction costs | PF still > 1.0 |

### Fail Conditions

```yaml
KILL if ANY:
  - PF < 1.0 on full sample
  - n < 30 after 60 days collection
  - Placebo shuffle p > 0.2
  - Reverse PF > 1.0 (direction doesn't matter)
  - Only works in one session type
  - PF < 1.0 at 2x costs

DEMOTE to LOW_PRIORITY if:
  - 1.0 < PF < 1.2 (marginal)
  - Works only LONG or only SHORT
  - High threshold sensitivity (cliff effect)
```

### ISS Proxy (Backup)

```python
def calc_opening_imbalance_proxy(bar_5m: pd.Series) -> float:
    """
    Proxy using 5-min bar position in range.

    NOT EQUIVALENT to real orderflow imbalance.
    Use only for preliminary screening.
    """
    hl_range = bar_5m["high"] - bar_5m["low"]
    if hl_range < 1e-9:
        return 0.0

    # Position of close in range: +1 = closed at high, -1 = closed at low
    return 2 * (bar_5m["close"] - bar_5m["low"]) / hl_range - 1
```

---

## M2: Trade Flow Divergence

### Hypothesis

Price moving without corresponding orderflow is a weak move and tends to reverse.

**Mechanism**: Price moves driven by thin orderbook (low volume) are noise, not signal. Institutional flow shows as price + volume together.

### Event Definition

```yaml
event_id: M2_flow_divergence
trigger_time: Continuous (10:30-18:00 MSK)
observation_window: Rolling 5-minute
entry_time: When divergence crosses threshold
exit_horizons: [15m, 30m]

exclusions:
  - First 30 min after open (noisy)
  - Last 30 min before close
  - Clearing periods
```

### Required Data

| Data | Granularity | Source | Priority |
|------|-------------|--------|----------|
| Trade tape with side | Tick | QUIK | CRITICAL |
| L2 orderbook | 500ms | QUIK | HIGH |
| 1-min OHLCV | 1m | ISS (backup) | PROXY ONLY |

### Signal Calculation

```python
def calc_flow_divergence(
    trades: List[Trade],
    prices: List[float],
    window: int = 300  # 5 minutes in seconds
) -> float:
    """
    Calculate price-volume divergence.

    Divergence = sign(price_change) × (-volume_change)

    Positive divergence = price up, volume down (weak bull)
    Negative divergence = price down, volume down (weak bear)

    Returns:
        Divergence score. High positive = fade the up-move.
    """
    price_change = (prices[-1] - prices[-window]) / prices[-window]

    recent_volume = sum(t.qty for t in trades[-window:])
    prior_volume = sum(t.qty for t in trades[-2*window:-window])

    if prior_volume == 0:
        return 0.0

    volume_change = (recent_volume - prior_volume) / prior_volume

    return np.sign(price_change) * (-volume_change)

# Signal: FADE when divergence > threshold for N consecutive periods
def check_divergence_signal(divergence_history: List[float]) -> Optional[str]:
    """
    Check for sustained divergence.

    Returns:
        "FADE_LONG" (short entry) if positive divergence
        "FADE_SHORT" (long entry) if negative divergence
        None if no signal
    """
    THRESHOLD = 0.15
    CONSECUTIVE = 3

    recent = divergence_history[-CONSECUTIVE:]

    if all(d > THRESHOLD for d in recent):
        return "FADE_LONG"  # Price went up weakly, short it
    elif all(d < -THRESHOLD for d in recent):
        return "FADE_SHORT"  # Price went down weakly, buy it

    return None
```

### Label Definition

```python
def calc_label(entry_price: float, signal: str, future_price: float) -> Dict:
    """
    Calculate if fade was correct.
    """
    if signal == "FADE_LONG":
        # We shorted, profit if price goes down
        pnl = (entry_price - future_price) / entry_price
    else:
        # We went long, profit if price goes up
        pnl = (future_price - entry_price) / entry_price

    return {
        "pnl": pnl,
        "label": 1 if pnl > 0 else 0,
    }
```

### Falsification Tests

| Test | Method | Pass Condition |
|------|--------|----------------|
| **F1: Placebo Shuffle** | Shuffle signal timing | Real PF > Placebo × 1.5 |
| **F2: Placebo Reverse** | Enter WITH momentum instead of fading | Reverse PF < 1.0 |
| **F3: Volume Confirmation** | Test: does high volume = continuation? | Opposite pattern PF < 1.0 |
| **F4: Threshold Plateau** | Test divergence thresholds | Stable across range |
| **F5: Time-of-Day** | Test by hour | Works in 3+ hours |
| **F6: Cost Shock** | Apply 2x costs | PF > 1.0 |

### Fail Conditions

```yaml
KILL if ANY:
  - PF < 1.0
  - Momentum (anti-fade) works better
  - Only works at specific threshold (overfit)
  - Session-specific only

DEMOTE if:
  - PF 1.0-1.2 (marginal)
  - High sensitivity to consecutive periods count
```

### ISS Proxy (Backup)

```python
def calc_flow_divergence_proxy(candles: pd.DataFrame) -> pd.Series:
    """
    Proxy using 1-min candle volume.

    WARNING: Volume in ISS is total, not buy/sell.
    This is NOT equivalent to real orderflow divergence.
    """
    price_change = candles["close"].pct_change(5)
    volume_ma = candles["volume"].rolling(20).mean()
    volume_change = candles["volume"] / volume_ma - 1

    return np.sign(price_change) * (-volume_change)
```

---

## M3: Close Pressure → Overnight Gap

### Hypothesis

Orderflow imbalance in final 10 minutes of trading predicts overnight gap direction.

**Mechanism**: Institutional rebalancing before close (T+0 settlement pressure, margin calls, fund flows) reveals direction for next day.

### Event Definition

```yaml
event_id: M3_close_pressure
trigger_time: 18:30:00 MSK (10 min before close)
observation_window: 18:30:00 - 18:40:00 MSK
entry_time: 18:40:00 MSK (at close)
exit_time: Next day 10:05:00 MSK (5 min after open)

exclusions:
  - Friday close (weekend gap different)
  - Pre-holiday
  - Expiry days (futures)
```

### Required Data

| Data | Granularity | Source | Priority |
|------|-------------|--------|----------|
| Trade tape with side | Tick | QUIK | CRITICAL |
| L2 orderbook | 500ms | QUIK | HIGH |
| Next-day open price | - | ISS | REQUIRED |
| 1-min OHLCV | 1m | ISS (backup) | PROXY ONLY |

### Signal Calculation

```python
def calc_close_pressure(
    trades: List[Trade],
    orderbook_snapshots: List[OrderBook],
    window_start: datetime,  # 18:30
    window_end: datetime,    # 18:40
) -> float:
    """
    Calculate close pressure from trade flow + book pressure.

    Returns:
        Pressure score in [-1, +1]. Positive = buying pressure.
    """
    # Trade imbalance (primary)
    window_trades = [t for t in trades
                     if window_start <= t.ts < window_end]

    buy_vol = sum(t.qty for t in window_trades if t.side == 'BUY')
    sell_vol = sum(t.qty for t in window_trades if t.side == 'SELL')
    total_vol = buy_vol + sell_vol

    if total_vol == 0:
        trade_imb = 0.0
    else:
        trade_imb = (buy_vol - sell_vol) / total_vol

    # Book pressure (secondary)
    last_books = [b for b in orderbook_snapshots
                  if window_start <= b.ts < window_end]

    if not last_books:
        book_pressure = 0.0
    else:
        # Average imbalance across window
        imbalances = []
        for book in last_books:
            bid_vol = sum(book.bid_sizes[:5])
            ask_vol = sum(book.ask_sizes[:5])
            if bid_vol + ask_vol > 0:
                imbalances.append((bid_vol - ask_vol) / (bid_vol + ask_vol))
        book_pressure = np.mean(imbalances) if imbalances else 0.0

    # Combined score (weight trade flow higher)
    return 0.7 * trade_imb + 0.3 * book_pressure

# Signal thresholds
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
```

### Label Definition

```python
def calc_label(close_price: float, next_open_price: float) -> Dict:
    """
    Calculate overnight gap and label.
    """
    gap = (next_open_price - close_price) / close_price

    return {
        "overnight_gap": gap,
        "gap_direction": 1 if gap > 0 else 0,
        "gap_magnitude": abs(gap),
    }
```

### Falsification Tests

| Test | Method | Pass Condition |
|------|--------|----------------|
| **F1: Correlation Test** | Calc correlation(pressure, gap) | r > 0.15, p < 0.05 |
| **F2: Placebo Shuffle** | Shuffle pressure values | Real corr > 2 × Placebo |
| **F3: Direction Accuracy** | What % of gaps predicted correctly | Accuracy > 55% |
| **F4: Magnitude Correlation** | Does higher pressure = larger gap? | Positive correlation |
| **F5: Day-of-Week** | Test by weekday | Works Mon-Thu (no Fri) |
| **F6: Cost Shock** | Include overnight margin cost | PF > 1.0 |

### Fail Conditions

```yaml
KILL if ANY:
  - Correlation < 0.10
  - Direction accuracy < 52%
  - PF < 1.0 after overnight costs
  - Only works one direction

DEMOTE if:
  - Weak correlation (0.10-0.15)
  - Works only on high-gap days
```

### ISS Proxy (Backup)

```python
def calc_close_pressure_proxy(candles_1m: pd.DataFrame) -> float:
    """
    Proxy using 1-min bar imbalances 18:30-18:40.

    WARNING: Not equivalent to orderflow pressure.
    """
    preclose = candles_1m.between_time("18:30", "18:40")

    imbalances = []
    for _, bar in preclose.iterrows():
        hl_range = bar["high"] - bar["low"]
        if hl_range > 1e-9:
            imb = 2 * (bar["close"] - bar["low"]) / hl_range - 1
            imbalances.append(imb)

    return np.mean(imbalances) if imbalances else 0.0
```

---

## H4: Queue Depletion at Key Levels

### Hypothesis

Rapid depletion of bid/ask queue at support/resistance levels predicts breakout direction.

**Mechanism**: Market makers pull quotes when informed flow detected. Queue depletion = someone knows something.

### Event Definition

```yaml
event_id: H4_queue_depletion
trigger_time: When price within 0.1% of S/R level
observation_window: 10 seconds before trigger
entry_time: When queue depletes > 30%
exit_horizons: [1m, 5m, 10m]

requirements:
  - S/R level detection
  - Real-time L2 data
  - Sub-second latency
```

### Required Data

| Data | Granularity | Source | Priority |
|------|-------------|--------|----------|
| L2 orderbook | 500ms (or faster) | QUIK | CRITICAL |
| S/R levels | - | Derived | REQUIRED |
| Trade tape | Tick | QUIK | HIGH |

**Note**: This hypothesis CANNOT be tested with ISS data. No proxy available.

### Signal Calculation

```python
def detect_sr_level(prices: pd.Series, window: int = 50) -> List[float]:
    """
    Detect support/resistance levels from price history.

    Simple method: recent swing high/low points.
    """
    highs = prices.rolling(window).max().dropna()
    lows = prices.rolling(window).min().dropna()

    resistance = highs.iloc[-1]
    support = lows.iloc[-1]

    return [support, resistance]

def calc_queue_depletion(
    current_book: OrderBook,
    prev_book: OrderBook,
    side: str,  # "bid" or "ask"
    levels: int = 5
) -> float:
    """
    Calculate queue depletion percentage.

    Returns:
        Depletion as negative percentage. -0.3 = 30% depleted.
    """
    if side == "bid":
        current_depth = sum(current_book.bid_sizes[:levels])
        prev_depth = sum(prev_book.bid_sizes[:levels])
    else:
        current_depth = sum(current_book.ask_sizes[:levels])
        prev_depth = sum(prev_book.ask_sizes[:levels])

    if prev_depth == 0:
        return 0.0

    return (current_depth - prev_depth) / prev_depth

def check_depletion_signal(
    price: float,
    sr_levels: List[float],
    bid_depletion: float,
    ask_depletion: float,
    proximity_pct: float = 0.001,  # 0.1%
    depletion_threshold: float = -0.30,  # 30% drop
) -> Optional[str]:
    """
    Check for queue depletion signal at S/R.

    Returns:
        "SHORT" if bid depleting at support (breakdown expected)
        "LONG" if ask depleting at resistance (breakout expected)
        None if no signal
    """
    support, resistance = sr_levels

    # At support?
    if abs(price - support) / support < proximity_pct:
        if bid_depletion < depletion_threshold:
            return "SHORT"  # Support breaking

    # At resistance?
    if abs(price - resistance) / resistance < proximity_pct:
        if ask_depletion < depletion_threshold:
            return "LONG"  # Resistance breaking

    return None
```

### Label Definition

```python
def calc_label(
    entry_price: float,
    signal: str,
    future_prices: Dict[str, float]
) -> Dict:
    """
    Calculate if breakout prediction was correct.
    """
    labels = {}

    for horizon, future_price in future_prices.items():
        if signal == "LONG":
            pnl = (future_price - entry_price) / entry_price
        else:
            pnl = (entry_price - future_price) / entry_price

        labels[f"pnl_{horizon}"] = pnl
        labels[f"label_{horizon}"] = 1 if pnl > 0 else 0

    return labels
```

### Falsification Tests

| Test | Method | Pass Condition |
|------|--------|----------------|
| **F1: Baseline vs Signal** | Compare signal entries vs random S/R entries | Signal PF > Random × 1.3 |
| **F2: Depletion Matters** | Test: do non-depleting S/R touches perform worse? | Yes |
| **F3: Side Symmetry** | Bid depletion vs Ask depletion | Both work |
| **F4: S/R Method Robustness** | Test different S/R detection methods | Works with 2+ methods |
| **F5: Threshold Sensitivity** | Test depletion thresholds 20-50% | Stable |
| **F6: Cost Shock** | Apply 2x spread + commission | PF > 1.0 |

### Fail Conditions

```yaml
KILL if ANY:
  - Signal entries not better than random S/R entries
  - Depletion threshold doesn't matter (random)
  - Only works at support OR resistance (not both)
  - PF < 1.0 after costs

DEMOTE if:
  - Works only with specific S/R method
  - High threshold sensitivity
```

### ISS Proxy

**NO PROXY AVAILABLE**

This hypothesis requires:
- Real-time L2 order book depth
- Sub-second snapshot frequency
- Queue size tracking

ISS only provides best bid/ask, no depth information.

**Blocker**: Must wait for QUIK data collection.

---

## Summary: Test Priority After Data Collection

| Priority | Hypothesis | Data Required | Estimated Days to n=50 |
|----------|------------|---------------|------------------------|
| **1** | M1: Opening Imbalance | Tape + L2 | 50-60 days |
| **2** | M3: Close Pressure | Tape + L2 | 30-40 days |
| **3** | M2: Flow Divergence | Tape | 20-30 days (high frequency) |
| **4** | M4: Queue Depletion | L2 (500ms+) | 60-90 days (rare events) |

---

## First 3 Tests After L2/Tape Collection

After 30 days of data collection, run in this order:

1. **M3: Close Pressure → Gap** (fastest to validate)
   - One event per day = 30 data points in 30 days
   - Simple correlation test
   - Clear success metric (gap direction)

2. **M1: Opening Imbalance** (highest potential edge)
   - One event per day per ticker
   - With 3 tickers = 90 events in 30 days
   - Clear mechanical signal

3. **M2: Flow Divergence** (continuous signal)
   - Many events per day
   - Need to define entry rules carefully
   - Higher risk of overfitting
