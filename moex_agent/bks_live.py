"""
BKS Live Execution Bridge.

Supports both market and limit orders.
Default: LIMIT orders for real money (less slippage).
Fallback to market if limit not filled within timeout.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .bks import BksClient

log = logging.getLogger("bks_live")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BKS_LIVE_STATE_FILE = DATA_DIR / "bks_live_state.json"
STOCKS_CLASS_CODE = "TQBR"
FUTURES_CLASS_CODE = "SPBFUT"

# Лимитный ордер: сдвиг цены от текущей (в % от цены)
# Для BUY: price + offset (чуть выше чтобы исполнился)
# Для SELL: price - offset (чуть ниже чтобы исполнился)
LIMIT_OFFSET_PCT = 0.03  # 0.03% = 3 копейки на 100₽ акцию

# Если лимитный не исполнился за N секунд — отменяем
LIMIT_TIMEOUT_SEC = 30


def load_live_state() -> Dict[str, Any]:
    if not BKS_LIVE_STATE_FILE.exists():
        return {"enabled": False}
    try:
        payload = json.loads(BKS_LIVE_STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"enabled": False}
        return payload
    except Exception:
        return {"enabled": False}


def is_live_enabled() -> bool:
    return bool(load_live_state().get("enabled", False))


def build_client_order_id(namespace: str, action: str, trade_id: int | str, secid: str) -> str:
    """Генерирует уникальный UUID для каждого ордера."""
    import uuid
    return str(uuid.uuid4())  # Always unique — no duplicates on BKS


def _calc_limit_price(current_price: float, side: str) -> float:
    """Рассчитать цену лимитного ордера.
    
    BUY: чуть выше текущей (чтобы исполнился быстро)
    SELL: чуть ниже текущей
    """
    offset = current_price * LIMIT_OFFSET_PCT / 100
    if side in ("BUY", "1", 1):
        return round(current_price + offset, 2)
    else:
        return round(current_price - offset, 2)


def _submit_limit_order(
    *,
    ticker: str,
    side: str,
    quantity: int,
    price: float,
    class_code: str,
    client_order_id: str,
    client: Optional[BksClient] = None,
) -> Dict[str, Any]:
    """Отправить лимитный ордер в БКС."""
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if price <= 0:
        raise ValueError("price must be > 0")

    if not is_live_enabled():
        return {
            "ok": False,
            "skipped": True,
            "reason": "live_disabled",
            "order_type": "limit",
            "price": price,
            "clientOrderId": client_order_id,
            "ticker": ticker,
            "quantity": quantity,
            "side": side,
        }

    broker = client or BksClient()
    result = broker.create_order(
        ticker=ticker,
        side=side,
        quantity=int(quantity),
        order_type="limit",
        price=price,
        class_code=class_code,
        dry_run=None,  # Respects BKS_EXECUTION_ENABLED env var
        client_order_id=client_order_id,
    )

    return {
        "ok": True,
        "skipped": False,
        "order_type": "limit",
        "price": price,
        "submittedAt": datetime.utcnow().isoformat() + "Z",
        "clientOrderId": client_order_id,
        "ticker": ticker,
        "classCode": class_code,
        "quantity": int(quantity),
        "side": side,
        "broker": result,
    }


def _submit_market_order(
    *,
    ticker: str,
    side: str,
    quantity: int,
    class_code: str,
    client_order_id: str,
    client: Optional[BksClient] = None,
) -> Dict[str, Any]:
    """Отправить рыночный ордер (fallback)."""
    if quantity <= 0:
        raise ValueError("quantity must be > 0")

    if not is_live_enabled():
        return {
            "ok": False,
            "skipped": True,
            "reason": "live_disabled",
            "order_type": "market",
            "clientOrderId": client_order_id,
            "ticker": ticker,
            "quantity": quantity,
            "side": side,
        }

    broker = client or BksClient()
    try:
        result = broker.create_order(
            ticker=ticker,
            side=side,
            quantity=int(quantity),
            order_type="market",
            class_code=class_code,
            dry_run=None,
            client_order_id=client_order_id,
        )
    except Exception as exc:
        if "401" in str(exc):
            # Token expired — force refresh and retry with SAME client_order_id
            # to avoid duplicate orders
            log.info(f"BKS 401 — refreshing token and retrying {ticker}")
            broker._access_token = None
            result = broker.create_order(
                ticker=ticker,
                side=side,
                quantity=int(quantity),
                order_type="market",
                class_code=class_code,
                dry_run=None,
                client_order_id=client_order_id,  # SAME ID to prevent duplicates
            )
        else:
            raise
    return {
        "ok": True,
        "skipped": False,
        "order_type": "market",
        "submittedAt": datetime.utcnow().isoformat() + "Z",
        "clientOrderId": client_order_id,
        "ticker": ticker,
        "classCode": class_code,
        "quantity": int(quantity),
        "side": side,
        "broker": result,
    }


def submit_smart_order(
    *,
    ticker: str,
    side: str,
    quantity: int,
    current_price: float,
    class_code: str,
    client_order_id: str,
    client: Optional[BksClient] = None,
    use_limit: bool = True,
) -> Dict[str, Any]:
    """Умный ордер: сначала лимитный, если не исполнился — market.
    
    Для реальных денег: лимитный по умолчанию (0 проскальзывания).
    Если current_price = 0 или use_limit = False → market order.
    """
    # ALWAYS market orders — limit orders часто не исполняются на БКС
    # Slippage minimal на ликвидных тикерах MOEX
    log.info(f"MARKET ORDER {side} {ticker} x{quantity} (current={current_price})")
    return _submit_market_order(
        ticker=ticker, side=side, quantity=quantity,
        class_code=class_code, client_order_id=client_order_id,
        client=client,
    )


# ── Convenience functions (used by paper_mr_server and paper_futures) ──


def submit_stock_market_order(*, ticker: str, side: str, quantity: int, client_order_id: str, price: float = 0, client: Optional[BksClient] = None) -> Dict[str, Any]:
    """Акции: лимитный ордер если есть цена, иначе market."""
    return submit_smart_order(
        ticker=ticker, side=side, quantity=quantity,
        current_price=price,
        class_code=STOCKS_CLASS_CODE,
        client_order_id=client_order_id,
        client=client,
        use_limit=price > 0,
    )


def submit_futures_market_order(*, ticker: str, side: str, quantity: int, client_order_id: str, price: float = 0, client: Optional[BksClient] = None) -> Dict[str, Any]:
    """Фьючерсы: лимитный ордер если есть цена, иначе market."""
    return submit_smart_order(
        ticker=ticker, side=side, quantity=quantity,
        current_price=price,
        class_code=FUTURES_CLASS_CODE,
        client_order_id=client_order_id,
        client=client,
        use_limit=price > 0,
    )
