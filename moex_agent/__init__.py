"""
MOEX Agent v2 - Trading Signal Generator for Moscow Exchange

Real-time trading signal generator with 4-level confirmation:
1. Anomaly detection (MAD z-score)
2. ML prediction (HistGradientBoosting + Isotonic calibration)
3. Risk management (kill-switch, margin control)
4. Rule-based signal filter

Usage:
    python -m moex_agent init-db
    python -m moex_agent bootstrap --days 180
    python -m moex_agent train
    python -m moex_agent live
    python -m moex_agent paper
    python -m moex_agent margin
    python -m moex_agent backtest
    python -m moex_agent web --port 8000
    python -m moex_agent status
"""

__version__ = "2.0.0"

from .config import AppConfig, load_config
from .storage import connect, database
from .engine import PipelineEngine, Signal, CycleResult
from .predictor import ModelRegistry, FEATURE_COLS

__all__ = [
    "__version__",
    "AppConfig",
    "load_config",
    "connect",
    "database",
    "PipelineEngine",
    "Signal",
    "CycleResult",
    "ModelRegistry",
    "FEATURE_COLS",
]
