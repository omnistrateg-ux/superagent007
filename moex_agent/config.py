"""
MOEX Agent v2 Configuration

Pydantic-based configuration with python-dotenv support.
All secrets loaded from .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

# Load .env file
load_dotenv()


# 45 MOEX TQBR tickers
DEFAULT_TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK", "ROSN", "SNGS", "PLZL", "TATN",
    "MGNT", "ALRS", "CHMF", "NLMK", "MTSS", "IRAO", "VTBR", "MOEX", "PHOR",
    "RUAL", "POLY", "MAGN", "AFKS", "PIKK", "HYDR", "FEES", "RTKM", "AFLT",
    "TRNFP", "SBERP", "SNGSP", "TATNP", "TCSG", "OZON", "HHRU", "FIVE",
    "FIXP", "SMLT", "SGZH", "VKCO", "POSI", "YDEX", "BELU", "CBOM", "MTLR",
    "T",
]


class HorizonConfig(BaseModel):
    """Single horizon configuration."""
    name: str
    minutes: int = Field(ge=1)


class StorageConfig(BaseModel):
    """Database storage configuration."""
    sqlite_path: Path = Field(default=Path("data/moex_agent.sqlite"))

    @field_validator("sqlite_path", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v


class UniverseConfig(BaseModel):
    """Market universe configuration."""
    engine: str = "stock"
    market: str = "shares"
    board: str = "TQBR"
    tickers: List[str] = Field(default_factory=lambda: DEFAULT_TICKERS.copy())

    @field_validator("tickers", mode="before")
    @classmethod
    def filter_tickers(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return [t for t in v if isinstance(t, str) and t.strip()]
        return v


class PriceExitConfig(BaseModel):
    """Price-based exit configuration."""
    enabled: bool = True
    take_atr: float = Field(default=0.8, ge=0)
    stop_atr: float = Field(default=0.6, ge=0)


class SignalsConfig(BaseModel):
    """Signal generation configuration."""
    horizons: List[HorizonConfig] = Field(
        default=[
            HorizonConfig(name="5m", minutes=5),
            HorizonConfig(name="10m", minutes=10),
            HorizonConfig(name="30m", minutes=30),
            HorizonConfig(name="1h", minutes=60),
            HorizonConfig(name="1d", minutes=1440),
            HorizonConfig(name="1w", minutes=10080),
        ]
    )
    p_threshold: float = Field(default=0.54, ge=0, le=1)
    cooldown_minutes: int = Field(default=30, ge=1)
    top_n_anomalies: int = Field(default=10, ge=1)
    price_exit: PriceExitConfig = Field(default_factory=PriceExitConfig)


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_spread_bps: float = Field(default=200, ge=0)
    min_turnover_rub_5m: float = Field(default=1_000_000, ge=0)
    max_loss_per_trade_pct: float = Field(default=0.5, ge=0)
    max_daily_loss_pct: float = Field(default=2.0, ge=0)
    max_consecutive_losses: int = Field(default=2, ge=1)
    max_drawdown_pct: float = Field(default=10.0, ge=0)


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""
    enabled: bool = False
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    send_recommendations: List[str] = Field(
        default=["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]
    )

    @model_validator(mode="after")
    def load_from_env(self) -> "TelegramConfig":
        if not self.bot_token:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.chat_id:
            self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        return self


class AppConfig(BaseModel):
    """Main application configuration."""
    poll_seconds: int = Field(default=5, ge=1)
    cooldown_minutes: int = Field(default=30, ge=1)
    top_n_anomalies: int = Field(default=20, ge=1)  # v2.1: increased from 10
    max_workers: int = Field(default=20, ge=1, le=100)
    fee_bps: float = Field(default=8.0, ge=0)

    storage: StorageConfig = Field(default_factory=StorageConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)

    @property
    def sqlite_path(self) -> Path:
        return self.storage.sqlite_path

    @property
    def tickers(self) -> List[str]:
        return self.universe.tickers

    @property
    def engine(self) -> str:
        return self.universe.engine

    @property
    def market(self) -> str:
        return self.universe.market

    @property
    def board(self) -> str:
        return self.universe.board

    @property
    def horizons(self) -> List[HorizonConfig]:
        return self.signals.horizons

    @property
    def p_threshold(self) -> float:
        return self.signals.p_threshold

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "AppConfig":
        """Load configuration from YAML file."""
        path = Path(path)

        if not path.exists():
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

        app_section = raw.pop("app", {})
        for key in ["poll_seconds", "cooldown_minutes", "top_n_anomalies", "max_workers", "fee_bps"]:
            if key in app_section and key not in raw:
                raw[key] = app_section[key]

        return cls.model_validate(raw)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration."""
    return AppConfig.from_yaml(path)


Config = AppConfig
