"""
Order Status Tracker — отслеживает исполнение лимитных ордеров на БКС.

После отправки ордера:
1. Записывает order_id в tracker
2. Через 30 сек проверяет статус
3. Если FILLED → записывает qty в live_positions
4. Если CANCELED/REJECTED → удаляет из tracker
5. При close → проверяет live_positions, не paper qty
"""
import json
import logging
import time
from pathlib import Path
from threading import Lock

log = logging.getLogger("order_tracker")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRACKER_FILE = DATA_DIR / "order_tracker.json"
LIVE_POSITIONS_FILE = DATA_DIR / "live_positions.json"

_lock = Lock()


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, default=str))


def record_order(ticker: str, side: int, quantity: int, order_id: str, asset_class: str = "stocks"):
    """Записать отправленный ордер."""
    with _lock:
        tracker = _load(TRACKER_FILE)
        tracker[order_id] = {
            "ticker": ticker,
            "side": side,  # 1=BUY, 2=SELL
            "quantity": quantity,
            "asset_class": asset_class,
            "status": "PENDING",
            "ts": time.time(),
        }
        _save(TRACKER_FILE, tracker)
        log.info(f"ORDER TRACKED: {order_id} {ticker} side={side} qty={quantity}")


def check_pending_orders():
    """Проверить все PENDING ордера через БКС portfolio.
    
    Вызывается каждые 60 сек из main loop.
    Логика: если после отправки SELL ордера позиция появилась на БКС → FILLED.
    """
    with _lock:
        tracker = _load(TRACKER_FILE)
        if not tracker:
            return
        
        pending = {k: v for k, v in tracker.items() if v.get("status") == "PENDING"}
        if not pending:
            return
        
        # Проверяем только ордера старше 30 сек
        now = time.time()
        old_pending = {k: v for k, v in pending.items() if now - v.get("ts", 0) > 30}
        if not old_pending:
            return
        
        try:
            from moex_agent.bks import BksClient
            bks = BksClient()
            portfolio = bks.portfolio_summary()
            bks_positions = {}
            for p in portfolio.get("tradeable_positions", []):
                qty = float(p.get("quantity", 0))
                if abs(qty) > 0.5:
                    bks_positions[p["ticker"]] = qty
        except Exception as exc:
            log.debug(f"ORDER TRACKER: can't fetch BKS portfolio: {exc}")
            return
        
        live_pos = _load(LIVE_POSITIONS_FILE)
        changed = False
        
        for order_id, order in list(old_pending.items()):
            ticker = order["ticker"]
            side = order["side"]
            qty = order["quantity"]
            age = now - order.get("ts", 0)
            
            bks_qty = bks_positions.get(ticker, 0)
            
            if side == 2:  # SELL (open SHORT or close LONG)
                if bks_qty < -0.5:  # Позиция SHORT есть
                    tracker[order_id]["status"] = "FILLED"
                    live_pos[ticker] = int(bks_qty)
                    log.info(f"ORDER FILLED: {order_id} {ticker} SELL {qty} → BKS has {bks_qty}")
                    changed = True
                elif age > 300:  # 5 мин — не исполнился
                    tracker[order_id]["status"] = "EXPIRED"
                    log.info(f"ORDER EXPIRED: {order_id} {ticker} SELL {qty} — no BKS position after 5 min")
                    changed = True
            
            elif side == 1:  # BUY (open LONG or close SHORT)
                if bks_qty > 0.5:  # Позиция LONG есть
                    tracker[order_id]["status"] = "FILLED"
                    live_pos[ticker] = int(bks_qty)
                    log.info(f"ORDER FILLED: {order_id} {ticker} BUY {qty} → BKS has {bks_qty}")
                    changed = True
                elif age > 300:
                    tracker[order_id]["status"] = "EXPIRED"
                    log.info(f"ORDER EXPIRED: {order_id} {ticker} BUY {qty} — no BKS position after 5 min")
                    changed = True
        
        if changed:
            _save(TRACKER_FILE, tracker)
            _save(LIVE_POSITIONS_FILE, live_pos)
        
        # Cleanup old entries (> 24h)
        cutoff = now - 86400
        tracker = {k: v for k, v in tracker.items() if v.get("ts", 0) > cutoff}
        _save(TRACKER_FILE, tracker)


def get_live_position(ticker: str) -> int:
    """Получить реальное qty на БКС для тикера."""
    live_pos = _load(LIVE_POSITIONS_FILE)
    return live_pos.get(ticker, 0)


def has_pending_order(ticker: str) -> bool:
    """Есть ли PENDING ордер для тикера."""
    tracker = _load(TRACKER_FILE)
    return any(
        v.get("ticker") == ticker and v.get("status") == "PENDING"
        for v in tracker.values()
    )
