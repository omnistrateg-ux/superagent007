"""
BKS Position Manager — единая точка правды о позициях на БКС.

Решает:
1. API дубли (4 суб-счёта) → дедупликация
2. Direction mismatch → проверка перед close
3. Orphan detection → сравнение с paper
4. Safe close → проверка после close

Использование:
    pm = BksPositionManager()
    pm.refresh()  # обновить кэш
    
    pos = pm.get_position("RIM6")  # {qty: -4, direction: "SHORT"}
    pm.safe_close("RIM6")  # закрыть с проверками
"""
import logging
import time
import uuid
from typing import Dict, Optional, Tuple

log = logging.getLogger("bks_pm")

# Наши тикеры
OUR_STOCKS = {'LKOH','YDEX','OZON','PLZL','NVTK','SBER','ROSN','TRNFP','TATN','CHMF','POSI','VTBR','GMKN','SPBE','LENT','VKCO','SIBN','TATNP','ENPG'}
FUT_PREFIXES = ['MX', 'RI', 'BR', 'NG']


class BksPositionManager:
    """Единый менеджер позиций БКС."""
    
    def __init__(self):
        self._cache: Dict[str, dict] = {}  # ticker → {qty, pnl}
        self._last_refresh = 0
        self._client = None
    
    def _get_client(self):
        if not self._client:
            from moex_agent.bks import BksClient
            self._client = BksClient()
        return self._client
    
    def refresh(self, force: bool = False) -> Dict[str, dict]:
        """Обновить кэш позиций. Дедупликация встроена."""
        now = time.time()
        if not force and (now - self._last_refresh) < 10:  # не чаще 10 сек
            return self._cache
        
        try:
            client = self._get_client()
            raw = client.get_portfolio()
            
            seen = {}
            for item in raw.get('content', []):
                ticker = item.get('ticker', '')
                qty = item.get('quantity', 0)
                
                if not ticker or abs(qty) < 0.5:
                    continue
                if ticker in ('RUB', 'USD', 'EUR') or ticker.startswith('O.'):
                    continue
                if ticker in seen:
                    continue  # ДЕДУПЛИКАЦИЯ — первый суб-счёт
                
                # Только наши тикеры
                is_ours = ticker in OUR_STOCKS or any(f in ticker for f in FUT_PREFIXES)
                if not is_ours:
                    continue
                
                seen[ticker] = {
                    'qty': int(qty),
                    'direction': 'SHORT' if qty < 0 else 'LONG',
                    'pnl': item.get('unrealizedPL', 0),
                    'price': item.get('currentPrice', 0),
                }
            
            self._cache = seen
            self._last_refresh = now
            return self._cache
            
        except Exception as exc:
            log.warning(f"BKS refresh failed: {exc}")
            return self._cache
    
    def get_position(self, ticker: str) -> Optional[dict]:
        """Получить позицию по тикеру. None если нет."""
        self.refresh()
        return self._cache.get(ticker)
    
    def get_all_positions(self) -> Dict[str, dict]:
        """Все наши позиции на БКС."""
        self.refresh()
        return dict(self._cache)
    
    def has_position(self, ticker: str) -> bool:
        """Есть ли позиция на БКС."""
        return self.get_position(ticker) is not None
    
    def safe_open(self, ticker: str, side: int, quantity: int) -> dict:
        """Открыть позицию с проверками.
        
        side: 1=BUY, 2=SELL
        """
        # Проверяем нет ли уже позиции
        existing = self.get_position(ticker)
        if existing:
            log.warning(f"SAFE_OPEN: {ticker} уже есть на БКС qty={existing['qty']}")
            # Если в том же направлении — ОК (добавление)
            # Если в противоположном — НЕ ОТКРЫВАЕМ
            if (side == 2 and existing['qty'] > 0) or (side == 1 and existing['qty'] < 0):
                log.warning(f"SAFE_OPEN: {ticker} direction conflict, skip")
                return {'ok': False, 'reason': 'direction_conflict'}
        
        class_code = 'SPBFUT' if any(f in ticker for f in FUT_PREFIXES) else 'TQBR'
        client = self._get_client()
        
        # Generate order ID ONCE to prevent duplicates on retry
        order_id = str(uuid.uuid4())
        try:
            result = client.create_order(
                ticker=ticker, class_code=class_code,
                side=side, quantity=quantity, order_type=1,
                client_order_id=order_id
            )
            log.info(f"SAFE_OPEN: {ticker} side={side} qty={quantity} → {result.get('status')}")
            return {'ok': True, 'result': result}
        except Exception as exc:
            if '401' in str(exc):
                # Token refresh + retry with SAME order_id to prevent duplicates
                client._access_token = None
                result = client.create_order(
                    ticker=ticker, class_code=class_code,
                    side=side, quantity=quantity, order_type=1,
                    client_order_id=order_id  # SAME ID
                )
                return {'ok': True, 'result': result}
            log.warning(f"SAFE_OPEN failed: {exc}")
            return {'ok': False, 'reason': str(exc)}
    
    def safe_close(self, ticker: str) -> dict:
        """Закрыть позицию с проверками.
        
        1. Проверить что позиция есть
        2. Определить direction и qty
        3. Отправить обратный ордер
        4. Проверить что позиция закрылась
        """
        # 1. Обновить кэш
        self.refresh(force=True)
        pos = self._cache.get(ticker)
        
        if not pos:
            log.info(f"SAFE_CLOSE: {ticker} нет на БКС, skip")
            return {'ok': True, 'reason': 'no_position'}
        
        qty = abs(pos['qty'])
        # SHORT → BUY, LONG → SELL
        close_side = 1 if pos['qty'] < 0 else 2
        
        class_code = 'SPBFUT' if any(f in ticker for f in FUT_PREFIXES) else 'TQBR'
        client = self._get_client()
        
        # Generate order ID ONCE to prevent duplicates on retry
        order_id = str(uuid.uuid4())
        try:
            result = client.create_order(
                ticker=ticker, class_code=class_code,
                side=close_side, quantity=qty, order_type=1,
                client_order_id=order_id
            )
            log.info(f"SAFE_CLOSE: {ticker} {pos['direction']} x{qty} → {result.get('status')}")

            # 4. Проверить через 5 сек (увеличено с 3 для надежности)
            time.sleep(5)
            self.refresh(force=True)
            remaining = self._cache.get(ticker)
            if remaining:
                log.warning(f"SAFE_CLOSE: {ticker} ещё есть после close! qty={remaining['qty']}")
                return {'ok': False, 'reason': 'still_open', 'remaining': remaining}

            return {'ok': True, 'closed_qty': qty, 'direction': pos['direction']}

        except Exception as exc:
            if '401' in str(exc):
                # Token refresh + retry with SAME order_id to prevent duplicates
                client._access_token = None
                result = client.create_order(
                    ticker=ticker, class_code=class_code,
                    side=close_side, quantity=qty, order_type=1,
                    client_order_id=order_id  # SAME ID
                )
                return {'ok': True, 'result': result}
            log.warning(f"SAFE_CLOSE failed: {exc}")
            return {'ok': False, 'reason': str(exc)}
    
    def find_orphans(self, paper_positions: dict) -> list:
        """Найти orphan позиции (БКС есть, paper нет).
        
        paper_positions: {ticker: direction}
        """
        self.refresh(force=True)
        orphans = []
        
        for ticker, pos in self._cache.items():
            if ticker not in paper_positions:
                orphans.append({
                    'ticker': ticker,
                    'qty': pos['qty'],
                    'direction': pos['direction'],
                    'pnl': pos['pnl'],
                    'type': 'orphan',
                })
            elif paper_positions[ticker] != pos['direction']:
                orphans.append({
                    'ticker': ticker,
                    'qty': pos['qty'],
                    'direction': pos['direction'],
                    'paper_direction': paper_positions[ticker],
                    'type': 'direction_mismatch',
                })
        
        return orphans
    
    def close_all_orphans(self, paper_positions: dict) -> list:
        """Найти и закрыть все orphan."""
        orphans = self.find_orphans(paper_positions)
        results = []
        
        for orphan in orphans:
            ticker = orphan['ticker']
            log.warning(f"🚨 CLOSING ORPHAN: {ticker} {orphan['direction']} x{abs(orphan['qty'])}")
            result = self.safe_close(ticker)
            results.append({**orphan, 'close_result': result})
        
        return results


# Singleton
_instance = None

def get_bks_pm() -> BksPositionManager:
    global _instance
    if _instance is None:
        _instance = BksPositionManager()
    return _instance
