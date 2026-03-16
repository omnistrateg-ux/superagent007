# MOEX Agent v2

Real-time trading signal generator for Moscow Exchange with 4-level confirmation.

## Features

- **Anomaly Detection**: MAD z-score based price/volume spike detection
- **ML Prediction**: HistGradientBoosting + Isotonic calibration (max p ~ 0.60)
- **Risk Management**: Kill-switch (2 losses → halt, 2% daily → halt, 10% DD → stop)
- **Rule-Based Filters**: RSI, MACD, Bollinger, ADX confirmation
- **30 Technical Indicators**: Consistent feature set everywhere
- **Walk-Forward Validation**: TimeSeriesSplit for honest metrics
- **45 MOEX Tickers**: SBER, GAZP, LKOH, YDEX, T, and more

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m moex_agent init-db

# Load historical data (180 days)
python -m moex_agent bootstrap --days 180

# Train ML models
python -m moex_agent train

# Run live signal loop
python -m moex_agent live

# Check status
python -m moex_agent status
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `init-db` | Initialize SQLite database |
| `bootstrap --days N` | Load N days of historical data |
| `train` | Train ML models with Isotonic calibration |
| `live` | Run live signal generation loop |
| `paper --equity N` | Paper trading mode |
| `margin --equity N` | Margin trading mode |
| `backtest` | Run backtest on historical data |
| `web --port N` | Start web dashboard |
| `status` | Show system status |

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `config.yaml` to customize:

- Tickers to monitor
- Signal thresholds
- Risk parameters
- Telegram integration

## Architecture

```
moex_agent/
├── __init__.py      # Package init, version 2.0.0
├── __main__.py      # CLI entry point
├── config.py        # Pydantic config + dotenv
├── iss.py           # MOEX ISS API client
├── storage.py       # SQLite operations
├── features.py      # 30 technical indicators
├── labels.py        # Binary labels (net of 8bps fee)
├── anomaly.py       # MAD z-score detection
├── predictor.py     # ML model registry
├── train.py         # Training with Isotonic calibration
├── risk.py          # Kill-switch + margin control
├── signals.py       # Rule-based signal filter
└── engine.py        # Pipeline orchestration
```

## 4-Level Confirmation

1. **Anomaly Detection**: MAD z-score > 0.8 on 5-min return
2. **ML Prediction**: P(success) > 0.54 after Isotonic calibration
3. **Risk Gatekeeper**: Spread < 200bps, Turnover > 1M RUB
4. **Rule Filters**: RSI, MACD, Bollinger, ADX confirmation

## Kill-Switch Rules

- 2 consecutive losses → HALT for the day
- 2% daily loss → HALT for the day
- 10% drawdown → STOP (manual reset required)

## Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f moex-agent
```

## License

MIT
