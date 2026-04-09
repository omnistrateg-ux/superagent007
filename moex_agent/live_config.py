"""
Live trading конфигурация.
Отдельные лимиты для реальных денег — не зависят от paper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LIVE_CONFIG_FILE = DATA_DIR / "live_config.json"

# Дефолтные лимиты для live (консервативные)
DEFAULTS = {
    # Общие
    "max_portfolio_pct": 80,         # Максимум 80% баланса в позициях

    # Акции
    "stocks_max_per_trade_rub": 25000,  # Макс на одну сделку акций (₽)
    "stocks_max_positions": 3,          # Макс одновременных позиций
    "stocks_min_lot_rub": 500,          # Минимальная позиция (₽)

    # Фьючерсы
    "futures_max_contracts": 2,         # Макс контрактов на сделку
    "futures_max_positions": 2,         # Макс одновременных позиций
    "futures_max_margin_pct": 30,       # Макс % баланса под маржу

    # Safety
    "daily_loss_limit_rub": 15000,      # Стоп-торговля при убытке за день (₽)
    "daily_loss_limit_pct": 5,          # Или % от баланса
    "max_trades_per_day": 10,           # Макс сделок в день
}


def load_live_config() -> Dict[str, Any]:
    """Загрузить live-конфиг (файл + defaults)."""
    config = dict(DEFAULTS)
    if LIVE_CONFIG_FILE.exists():
        try:
            override = json.loads(LIVE_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(override, dict):
                config.update(override)
        except Exception:
            pass
    return config


def save_live_config(config: Dict[str, Any]) -> None:
    """Сохранить live-конфиг."""
    LIVE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIVE_CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def calc_stock_quantity(ticker: str, price: float, paper_quantity: int) -> int:
    """Пересчитать количество акций для live.
    
    Берёт min(paper_quantity, max_per_trade / price).
    """
    cfg = load_live_config()
    max_rub = cfg["stocks_max_per_trade_rub"]
    max_by_budget = int(max_rub / price) if price > 0 else 0

    # Не больше paper, не больше бюджета
    qty = min(paper_quantity, max_by_budget)

    # Не меньше минимального лота
    if qty * price < cfg["stocks_min_lot_rub"]:
        return 0  # Слишком мелкая позиция — пропускаем

    return max(1, qty)


def calc_futures_quantity(paper_quantity: int, ticker: str = "") -> int:
    """Пересчитать количество фьючерсов для live.
    
    Live = paper quantity (не больше, не меньше).
    Max ограничен конфигом.
    """
    cfg = load_live_config()
    max_contracts = cfg["futures_max_contracts"]
    return min(paper_quantity, max_contracts)
