# MOEX Agent v2.5 - Алготрейдинг система для Московской биржи

## Обзор проекта

MOEX Agent — это ML-based торговая система для автоматического трейдинга на Московской бирже (MOEX). Система использует машинное обучение для генерации торговых сигналов на основе аномалий цены/объема, технических индикаторов и рыночного контекста.

### Ключевые характеристики

- **~22,000 строк Python кода** в 45+ модулях
- **Multi-horizon ML модели** (5m, 10m, 30m, 1h, 1d)
- **CatBoost + LightGBM** классификаторы с isotonic calibration
- **Walk-forward validation** с purged CV для предотвращения data leakage
- **ATR-based labels** с R:R 2:1 для формирования таргетов
- **Real-time торговля** через MOEX ISS API
- **Paper trading** режим для тестирования без реальных денег

---

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                        MOEX Agent v2.5                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Data Layer  │───▶│  ML Layer    │───▶│ Trading Layer│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ MOEX ISS API │    │ 5 Horizon    │    │ Risk Engine  │      │
│  │ News Feeds   │    │ Models       │    │ Position Mgmt│      │
│  │ Market Data  │    │ Regime Det.  │    │ Order Exec.  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Структура проекта

```
superagent007/
├── moex_agent/                 # Основной пакет
│   ├── __main__.py            # CLI точка входа (553 строки)
│   ├── __init__.py            # Версия и экспорты
│   │
│   ├── # === DATA LAYER ===
│   ├── iss.py                 # MOEX ISS API клиент
│   ├── storage.py             # SQLite хранилище свечей/котировок
│   ├── features.py            # 35 технических индикаторов
│   ├── labels.py              # ATR-based label generation
│   ├── external_feeds.py      # Внешние источники данных
│   │
│   ├── # === ML LAYER ===
│   ├── train.py               # Обучение моделей (walk-forward)
│   ├── predictor.py           # Model registry и inference
│   ├── anomaly.py             # Детекция аномалий цена/объем
│   ├── regime.py              # Per-ticker режим детекция
│   ├── train_regime.py        # Обучение regime detector
│   ├── train_entry_timing.py  # Entry timing model
│   ├── horizon_resolver.py    # Multi-horizon conflict resolution
│   │
│   ├── # === SIGNAL GENERATION ===
│   ├── engine.py              # Pipeline генерации сигналов
│   ├── signals.py             # Rule-based фильтры (legacy)
│   ├── smart_filters.py       # Evidence-based SmartFilter
│   ├── market_context.py      # Рыночный контекст (IMOEX, USD/RUB)
│   ├── multi_timeframe.py     # MTF анализ трендов
│   ├── news_filter.py         # Новостной фильтр
│   │
│   ├── # === TRADING LAYER ===
│   ├── trader.py              # Основной трейдер (paper/live)
│   ├── risk.py                # Risk management engine
│   ├── broker.py              # Брокер интерфейс (заглушка)
│   ├── evening_session.py     # Вечерняя сессия (19:05-23:50)
│   │
│   ├── # === BACKTESTING ===
│   ├── backtest.py            # Основной бэктестер
│   ├── backtest_phase3.py     # Phase 3 features backtest
│   ├── futures_backtest.py    # Futures стратегии
│   │
│   ├── # === MONITORING ===
│   ├── telegram.py            # Telegram алерты
│   ├── telegram_monitor.py    # Мониторинг каналов
│   ├── sanctions_monitor.py   # Санкционный риск-монитор
│   ├── fault_tolerance.py     # Circuit breaker, retry logic
│   ├── champion_challenger.py # A/B тестирование моделей
│   │
│   ├── # === SPECIAL FEATURES ===
│   ├── microstructure.py      # Микроструктурные фичи
│   ├── orderflow.py           # Order flow анализ
│   ├── cross_asset.py         # Кросс-активные сигналы
│   ├── calendar_features.py   # Календарные эффекты
│   ├── mean_reversion.py      # Mean reversion стратегия
│   ├── futures.py             # Futures trading
│   │
│   └── web.py                 # FastAPI web interface
│
├── models/                     # Обученные модели
│   ├── meta.json              # Метаданные моделей
│   ├── model_logreg_*.joblib  # CatBoost модели по горизонтам
│   ├── model_entry_timing.joblib
│   └── regime_detector.joblib
│
├── data/                       # Runtime данные
│   ├── moex_agent.sqlite      # История свечей/котировок
│   └── *_state.json           # Состояние трейдера
│
├── tests/                      # Unit тесты
│   ├── test_anomaly.py
│   ├── test_features.py
│   ├── test_risk.py
│   └── test_signals.py
│
├── config.yaml                 # Основная конфигурация
├── requirements.txt            # Python зависимости
└── .env                        # Секреты (токены)
```

---

## Ключевые компоненты

### 1. ML Pipeline

```python
# Обучение модели (walk-forward с purged CV)
train.py:
  - PurgedKFold cross-validation (embargo = 2 дня)
  - CatBoost + LightGBM ensemble
  - Isotonic calibration (max p ~ 0.60)
  - ATR-based labels (R:R 2:1)
  - 35 технических фичей
```

**Фичи модели (35 индикаторов):**
- Returns: r_1m, r_5m, r_10m, r_30m, r_60m
- Turnover: turn_1m, turn_5m, turn_10m
- Volatility: atr_14, volatility_10, volatility_30, bb_width
- Momentum: rsi_14, rsi_7, macd, macd_signal, macd_hist, stoch_k, stoch_d, adx
- Price levels: bb_position, dist_vwap_atr, price_sma20_ratio, price_sma50_ratio
- Volume: obv_change, volume_sma_ratio
- Anomaly: z_ret_5m, z_vol_5m, score, volume_spike, direction

### 2. Signal Generation Pipeline

```
Anomaly Detection → ML Prediction → Risk Gatekeeper → Market Context → Signal
      ↓                  ↓                ↓                ↓
   z-score           5 horizons      spread/turnover   IMOEX/USD
   threshold         ensemble        checks            regime
```

**6-уровневая фильтрация:**
1. Anomaly detection (z-score > 0.3)
2. ML prediction (p > 0.51)
3. Risk gatekeeper (spread < 200bps, turnover > 1M)
4. Market context (skip PANIC regime)
5. Trend alignment (MTF analysis)
6. News filter (block on breaking news)

### 3. Risk Management

```python
risk.py:
  - max_loss_per_trade_pct: 0.5%
  - max_daily_loss_pct: 2.0%
  - max_consecutive_losses: 2
  - max_drawdown_pct: 10.0%
  - Circuit breaker на серии убытков
```

### 4. Regime Detection

```python
regime.py:
  - TREND_UP: ADX > threshold, momentum > 0
  - TREND_DOWN: ADX > threshold, momentum < 0
  - RANGE_LOW_VOL: ADX < threshold, vol < median
  - RANGE_HIGH_VOL: ADX < threshold, vol > median
```

### 5. Horizon Resolver

```python
horizon_resolver.py:
  Стратегии разрешения конфликтов между горизонтами:
  - WEIGHTED_VOTE (default) - взвешенное голосование
  - DEFER_TO_LONGER - приоритет длинным горизонтам
  - REQUIRE_CONSENSUS - требуется согласие всех
  - CONFIDENCE_THRESHOLD - только высокая уверенность
  - MAJORITY_VOTE - простое большинство
```

---

## CLI Команды

```bash
# Основные команды
python -m moex_agent status          # Статус системы
python -m moex_agent paper           # Paper trading
python -m moex_agent margin          # Margin trading
python -m moex_agent backtest        # Бэктест

# Обучение
python -m moex_agent train           # Обучить ML модели
python -m moex_agent train-regime    # Обучить regime detector

# Данные
python -m moex_agent fetch           # Загрузить исторические данные
python -m moex_agent poll            # Real-time polling

# Дополнительно
python -m moex_agent web             # Web UI (FastAPI)
python -m moex_agent phase3-backtest # Phase 3 бэктест
```

---

## Результаты бэктестов

### Baseline vs Phase 3 (SBER, 30 дней)

| Конфигурация | Trades | Win Rate | Improvement |
|--------------|--------|----------|-------------|
| Baseline     | 232    | 47.8%    | -           |
| + Regime     | 210    | 49.5%    | +1.7%       |
| + All Phase3 | 180    | 52.2%    | +4.4%       |
| + SmartFilter| 156    | 55.4%    | +7.6%       |

### По горизонтам (все тикеры)

| Horizon | Win Rate | Profit Factor | Sharpe |
|---------|----------|---------------|--------|
| 5m      | 17.1%    | 0.67          | -0.91  |
| 10m     | 44.2%    | 3.09          | 1.25   |
| 30m     | 48.0%    | 2.93          | 0.94   |
| 1h      | 51.4%    | 5.71          | 1.61   |
| 1d      | 47.6%    | 5.83          | 0.73   |

---

## Технологический стек

| Категория | Технологии |
|-----------|------------|
| Language  | Python 3.11+ |
| ML        | CatBoost, LightGBM, scikit-learn |
| Data      | pandas, numpy, SQLite |
| API       | MOEX ISS REST API |
| Config    | pydantic, PyYAML |
| Web       | FastAPI, uvicorn |
| Alerts    | Telegram Bot API |
| Testing   | pytest, pytest-cov |

---

## Уникальные особенности

1. **Walk-forward validation** с purged CV предотвращает data leakage
2. **ATR-based labels** с R:R 2:1 для адаптивных таргетов
3. **Multi-horizon ensemble** с conflict resolution
4. **Per-ticker regime detection** адаптируется к каждому инструменту
5. **SmartFilter** на основе статистики session quality
6. **Sanctions monitor** для российских акций (SDN list, sector risk)
7. **Evening session handler** для торговли 19:05-23:50
8. **Circuit breaker pattern** для fault tolerance
9. **Champion/Challenger A/B testing** для production моделей

---

## Зависимости между модулями

```
config.yaml
    │
    ▼
config.py ◄──────────────────────────┐
    │                                 │
    ▼                                 │
iss.py ─────► storage.py             │
    │             │                   │
    ▼             ▼                   │
features.py ◄─ labels.py             │
    │                                 │
    ▼                                 │
train.py ────► predictor.py          │
                   │                  │
                   ▼                  │
              engine.py ◄─────────────┤
                   │                  │
    ┌──────────────┼──────────────┐   │
    ▼              ▼              ▼   │
anomaly.py   regime.py    market_context.py
    │              │              │
    └──────────────┴──────────────┘
                   │
                   ▼
              trader.py ◄── risk.py
                   │
                   ▼
            telegram.py (alerts)
```

---

## Как запустить

```bash
# 1. Установка
git clone https://github.com/omnistrateg-ux/superagent007.git
cd superagent007
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Конфигурация
cp .env.example .env
# Отредактировать .env (TELEGRAM_BOT_TOKEN, etc.)

# 3. Загрузка данных
python -m moex_agent fetch --days 90

# 4. Обучение моделей
python -m moex_agent train

# 5. Paper trading
python -m moex_agent paper --equity 1000000
```

---

## Версионирование

- **v1.0** - Базовый anomaly detector + rule-based filters
- **v2.0** - ML модели (LightGBM), walk-forward validation
- **v2.1** - CatBoost, ATR labels, multi-horizon
- **v2.5** - Phase 3: Regime detection, SmartFilter, Evening session, Sanctions monitor

---

## Лицензия

Private / Educational use only.
