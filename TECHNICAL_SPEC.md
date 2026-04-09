# MOEX Agent - Technical Specification for AI Review

## Executive Summary

MOEX Agent is a quantitative trading system for Moscow Exchange (MOEX) equities. It combines anomaly detection, machine learning predictions, and risk management to generate intraday trading signals.

**Key Metrics:**
- Codebase: ~22,000 LOC Python
- Models: 5 horizon classifiers + regime detector + entry timing
- Universe: 40 liquid MOEX stocks
- Horizons: 5m, 10m, 30m, 1h, 1d
- Best Sharpe: 1.61 (1h horizon)

---

## 1. Data Pipeline

### 1.1 Data Sources

```python
# Primary: MOEX ISS API (iss.py)
- Candles: 1-minute OHLCV via /iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles
- Quotes: Real-time bid/ask/last via /iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}
- Rate limit: ~100 requests/min (handled via ThreadPoolExecutor)

# Secondary (external_feeds.py)
- Market indices: IMOEX (RTS index)
- Currency: USD/RUB
- Commodities: Brent crude
```

### 1.2 Storage (storage.py)

```sql
-- SQLite schema
CREATE TABLE candles (
    secid TEXT, board TEXT, interval INT,
    ts TEXT, open REAL, high REAL, low REAL, close REAL,
    value REAL, volume INT,
    PRIMARY KEY (secid, board, interval, ts)
);

CREATE TABLE quotes (
    secid TEXT, board TEXT, ts TEXT,
    last REAL, bid REAL, ask REAL,
    numtrades INT, voltoday INT, valtoday REAL,
    PRIMARY KEY (secid, board, ts)
);
```

### 1.3 Feature Engineering (features.py)

```python
FEATURE_COLS = [
    # Returns (5)
    "r_1m", "r_5m", "r_10m", "r_30m", "r_60m",

    # Turnover (3)
    "turn_1m", "turn_5m", "turn_10m",

    # Volatility (5)
    "atr_14", "dist_vwap_atr", "volatility_10", "volatility_30", "hl_range",

    # Momentum (10)
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "adx", "momentum_10", "momentum_30",

    # Bollinger (3)
    "bb_position", "bb_width", "obv_change",

    # Price levels (4)
    "price_sma20_ratio", "price_sma50_ratio", "sma20_sma50_ratio", "volume_sma_ratio",

    # Anomaly (5)
    "anomaly_z_ret_5m", "anomaly_z_vol_5m", "anomaly_score",
    "anomaly_volume_spike", "anomaly_direction"
]
# Total: 35 features
```

---

## 2. ML Models

### 2.1 Label Generation (labels.py)

```python
def compute_atr_labels(df, horizon_minutes, take_atr=0.8, stop_atr=0.6):
    """
    ATR-based labels with 2:1 R:R ratio.

    Label = 1 if price hits take_profit before stop_loss
    Label = 0 otherwise

    Args:
        take_atr: Take profit = entry + ATR * take_atr
        stop_atr: Stop loss = entry - ATR * stop_atr

    Returns:
        Binary labels (0/1)
    """
```

### 2.2 Training Pipeline (train.py)

```python
# Walk-forward validation with purged CV
class PurgedKFold:
    """
    Time-series cross-validation with embargo period.
    Prevents data leakage from future observations.

    embargo_days: Gap between train and test sets
    """

# Model architecture
class EnsembleClassifier:
    """
    Weighted ensemble of CatBoost + LightGBM.

    Default weights: CatBoost=0.6, LightGBM=0.4
    Isotonic calibration ensures P(class=1) in [0, 0.6]
    """

# Training parameters
CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": False,
}
```

### 2.3 Inference (predictor.py)

```python
class ModelRegistry:
    """
    Thread-safe model registry with lazy loading.

    Methods:
        predict(horizon, X) -> float  # P(success) for single horizon
        predict_all(X) -> Dict[str, float]  # All horizons
        resolve_horizons(X, strategy) -> ResolverDecision  # Conflict resolution
    """

    EXCLUDED_HORIZONS = {"1w", "entry_timing", "regime_detector"}
```

### 2.4 Horizon Conflict Resolution (horizon_resolver.py)

```python
class ResolutionStrategy(Enum):
    WEIGHTED_VOTE = "weighted_vote"        # Default: confidence-weighted
    DEFER_TO_LONGER = "defer_to_longer"    # Trust longer horizons
    REQUIRE_CONSENSUS = "require_consensus" # All must agree
    CONFIDENCE_THRESHOLD = "confidence_threshold"  # Only high-conf
    MAJORITY_VOTE = "majority_vote"        # Simple majority

@dataclass
class ResolverDecision:
    direction: Direction  # LONG, SHORT, NEUTRAL
    confidence: float     # 0.0 - 1.0
    reason: str
    horizons_agree: bool
    contributing_horizons: List[str]
    conflicts: List[str]
```

---

## 3. Signal Generation (engine.py)

### 3.1 Pipeline Architecture

```python
class PipelineEngine:
    """
    6-level signal confirmation:

    1. Anomaly Detection (z-score > 0.3)
       - compute_anomalies() returns candidates with price/volume spikes

    2. ML Prediction (p > 0.51)
       - resolve_horizons() with WEIGHTED_VOTE strategy
       - Direction must match anomaly direction

    3. Risk Gatekeeper
       - spread_bps < 200
       - turnover_5m > 1,000,000 RUB

    4. Market Context
       - Skip all trades in PANIC regime
       - Reduce position in RISK_OFF

    5. Trend Alignment
       - Multi-timeframe trend analysis
       - Log misalignment but don't block

    6. News Filter
       - Block on breaking news events
       - CBR rate decisions, sanctions, etc.
    """
```

### 3.2 Signal Dataclass

```python
@dataclass
class Signal:
    secid: str
    direction: Direction  # LONG or SHORT
    horizon: str          # "5m", "10m", etc.
    probability: float    # ML confidence
    signal_type: str      # "time-exit" or "price-exit"
    entry: float
    take: float           # Take profit level
    stop: float           # Stop loss level
    ttl_minutes: int      # Time-to-live
    anomaly_score: float
    filter_passed: bool   # Legacy rule-based filter result
    trend_aligned: bool
    created_at: datetime
```

---

## 4. Trading Execution (trader.py)

### 4.1 Trader State

```python
@dataclass
class TraderState:
    equity: float
    positions: Dict[str, Position]
    trades: List[CompletedTrade]

@dataclass
class Position:
    secid: str
    direction: str
    entry_price: float
    size: int
    take_profit: float
    stop_loss: float
    opened_at: datetime
```

### 4.2 Risk Engine (risk.py)

```python
@dataclass
class RiskState:
    daily_pnl: float
    consecutive_losses: int
    current_drawdown_pct: float

class RiskEngine:
    """
    Risk limits:
    - max_loss_per_trade_pct: 0.5%
    - max_daily_loss_pct: 2.0%
    - max_consecutive_losses: 2
    - max_drawdown_pct: 10.0%

    Circuit breaker:
    - After 2 consecutive losses: pause 30min
    - After 2% daily loss: stop trading for day
    """
```

### 4.3 Position Management

```python
def _check_exits(self, quotes: Dict) -> None:
    """
    Exit logic priority:
    1. Stop loss hit -> Close immediately
    2. Take profit hit -> Close immediately
    3. Trailing stop activated -> Adjust stop
    4. TTL expired -> Close at market
    """

def _adjust_stop_for_evening(self, stop: float, direction: str, atr: float) -> float:
    """
    Evening session (19:05-23:50 MSK):
    - Wider stops: stop_mult = 1.5x
    - Reduced position: position_mult = 0.5x
    """
```

---

## 5. Regime Detection (regime.py)

### 5.1 Per-Ticker Regimes

```python
class RegimeType(Enum):
    TREND_UP = "trend_up"         # ADX > threshold, momentum > 0
    TREND_DOWN = "trend_down"     # ADX > threshold, momentum < 0
    RANGE_LOW_VOL = "range_low_vol"   # ADX < threshold, vol < median
    RANGE_HIGH_VOL = "range_high_vol" # ADX < threshold, vol > median

class RegimeDetector:
    """
    ML-based regime detection using K-Means clustering.

    Features:
    - ADX (trend strength)
    - Momentum (direction)
    - Volatility percentile

    Thresholds learned from historical data.
    """
```

### 5.2 Trading Rules by Regime

```python
REGIME_RULES = {
    RegimeType.TREND_UP: {
        "long_allowed": True,
        "short_allowed": False,  # Don't fight the trend
        "position_mult": 1.0,
    },
    RegimeType.TREND_DOWN: {
        "long_allowed": False,
        "short_allowed": True,
        "position_mult": 1.0,
    },
    RegimeType.RANGE_LOW_VOL: {
        "long_allowed": True,
        "short_allowed": True,
        "position_mult": 0.5,  # Reduced size in range
    },
    RegimeType.RANGE_HIGH_VOL: {
        "long_allowed": False,
        "short_allowed": False,  # Skip high-vol chop
        "position_mult": 0.0,
    },
}
```

---

## 6. Smart Filters (smart_filters.py)

### 6.1 Session Quality

```python
SESSION_QUALITY_SCORES = {
    "morning_opening": 0.3,   # 10:00-10:30: Skip - noisy
    "morning_session": 0.6,   # 10:30-13:00: OK
    "lunch": 0.8,             # 13:00-14:00: Good - less noise
    "afternoon": 0.7,         # 14:00-16:00: OK
    "pre_close": 0.4,         # 16:00-17:00: Avoid
    "evening": 0.5,           # 19:05-23:50: Reduced
}
```

### 6.2 Filter Logic

```python
class SmartFilter:
    """
    Evidence-based filtering:

    1. Skip first 30min of session (morning_opening)
    2. Skip last hour before close (pre_close)
    3. Reduce position in adverse regime
    4. Block on calendar events (CBR meetings, expiry)
    5. Adjust for risk sentiment (USD/RUB spike)
    """
```

---

## 7. Fault Tolerance (fault_tolerance.py)

### 7.1 Retry Decorator

```python
@with_retry(max_attempts=3, backoff_factor=2, initial_delay=1.0)
def fetch_data():
    """Exponential backoff: 1s -> 2s -> 4s"""
    return requests.get(url)
```

### 7.2 Circuit Breaker

```python
class CircuitBreaker:
    """
    States:
    - CLOSED: Normal operation
    - OPEN: Service failing, reject requests
    - HALF_OPEN: Testing recovery

    Parameters:
    - failure_threshold: 5 failures to open
    - recovery_timeout: 60s before half-open
    """
```

---

## 8. Backtesting Results

### 8.1 Model Performance (778K samples)

| Horizon | Win Rate | Profit Factor | Sharpe | Trades |
|---------|----------|---------------|--------|--------|
| 5m      | 17.1%    | 0.67          | -0.91  | 123    |
| 10m     | 44.2%    | 3.09          | 1.25   | 144    |
| 30m     | 48.0%    | 2.93          | 0.94   | 232    |
| 1h      | 51.4%    | 5.71          | 1.61   | 334    |
| 1d      | 47.6%    | 5.83          | 0.73   | 629    |

### 8.2 Phase 3 Improvements (SBER, 30 days)

| Feature | Win Rate Delta | Trade Reduction |
|---------|----------------|-----------------|
| Regime filter | +1.7% | -9.5% |
| SmartFilter | +7.6% | -32.8% |
| Evening adj. | +0.5% | -5.0% |

---

## 9. Code Quality Indicators

### 9.1 Architecture

- **Separation of concerns**: Data/ML/Trading layers clearly separated
- **Dependency injection**: Config passed via constructors
- **Singleton pattern**: Global registries for models, monitors
- **Dataclasses**: Typed data structures throughout
- **Type hints**: Full typing with Optional, Dict, List

### 9.2 Error Handling

```python
# Graceful degradation pattern
try:
    result = api_call()
    circuit_breaker.record_success()
except Exception as e:
    circuit_breaker.record_failure()
    logger.warning(f"API failed: {e}")
    return fallback_value
```

### 9.3 Logging

```python
# Structured logging throughout
logger = logging.getLogger(__name__)
logger.info(f"Signal generated: {signal.secid} {signal.direction}")
logger.warning(f"Risk limit hit: {reason}")
logger.error(f"Critical failure: {e}")
```

---

## 10. Known Limitations

1. **No real broker integration** - broker.py is a stub
2. **Single-threaded trading** - no async execution
3. **No position sizing optimization** - fixed % of equity
4. **Limited universe** - MOEX equities only
5. **No options/futures hedging** - spot only
6. **5m horizon underperforms** - negative Sharpe, consider removing

---

## 11. Recommended Improvements

1. **Async data fetching** - aiohttp for parallel requests
2. **Position sizing** - Kelly criterion or risk parity
3. **Slippage model** - Realistic execution costs
4. **Ensemble weighting** - Dynamic based on regime
5. **Feature selection** - Remove low-importance features
6. **Online learning** - Incremental model updates
7. **Portfolio optimization** - Cross-ticker correlation

---

## 12. File Dependency Graph

```
Core Flow:
config.yaml → config.py → iss.py → storage.py → features.py → train.py
                                                      ↓
                                               predictor.py
                                                      ↓
                                               engine.py ← anomaly.py
                                                      ↓      ← regime.py
                                               trader.py ← market_context.py
                                                      ↓
                                               risk.py → telegram.py

Support Modules:
- fault_tolerance.py (retry, circuit breaker)
- sanctions_monitor.py (risk assessment)
- smart_filters.py (signal filtering)
- horizon_resolver.py (multi-horizon)
- evening_session.py (extended hours)
```

---

## 13. API Reference

### Main Entry Points

```python
# CLI
python -m moex_agent <command> [options]

# Commands
status          # System health check
paper           # Paper trading mode
margin          # Margin trading with leverage
backtest        # Historical backtest
train           # Train ML models
fetch           # Download historical data
poll            # Real-time data polling
```

### Key Classes

```python
from moex_agent.config import load_config
from moex_agent.predictor import ModelRegistry, get_registry
from moex_agent.engine import PipelineEngine
from moex_agent.trader import Trader
from moex_agent.risk import RiskEngine
from moex_agent.regime import RegimeDetector
```

---

*Document generated for AI code review. Version 2.5.*
