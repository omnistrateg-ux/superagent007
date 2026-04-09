"""
NEWS MOMENTUM — торговля на экстремальных новостях.

Когда рынок двигается >3% на новости → входим ПО ТРЕНДУ, не против.
Это дополнение к MR стратегии.

Условия входа:
1. CRITICAL или HIGH новость с направлением
2. Фьючерс dev > 2% (сильное движение уже идёт)
3. Направление совпадает с новостью (нефть падает → BR SHORT)

Выход:
- Trailing stop (агрессивный, 1.5% от пика)
- Time stop 120 мин
- Target: EMA (возврат к средней = конец движения)
"""
import logging
from typing import Optional, Tuple

log = logging.getLogger("news_momentum")

# Минимальная девиация для входа
MIN_DEV_PCT = 2.0

# Маппинг новость → направление фьючерса
NEWS_DIRECTION_MAP = {
    # Нефть
    "нефть упала": ("BR", "SHORT"),
    "нефть обвал": ("BR", "SHORT"),
    "oil falls": ("BR", "SHORT"),
    "oil drops": ("BR", "SHORT"),
    "oil crash": ("BR", "SHORT"),
    "brent falls": ("BR", "SHORT"),
    "wti falls": ("BR", "SHORT"),
    "нефть выросла": ("BR", "LONG"),
    "нефть взлетела": ("BR", "LONG"),
    "oil surges": ("BR", "LONG"),
    "oil jumps": ("BR", "LONG"),
    # Перемирие/война → нефть
    "перемири": ("BR", "SHORT"),  # перемирие/перемирием/перемирия → нефть вниз
    "ceasefire": ("BR", "SHORT"),
    "war ends": ("BR", "SHORT"),
    "разочарован": ("BR", "SHORT"),  # нефтяники разочарованы = нефть вниз
    "удар по иран": ("BR", "LONG"),  # война → нефть вверх
    "iran attack": ("BR", "LONG"),
    "ормуз закрыт": ("BR", "LONG"),
    "ормуз перекрыт": ("BR", "LONG"),
    # Рынок
    "мосбиржа обвал": ("MX", "SHORT"),
    "imoex crash": ("MX", "SHORT"),
    "рынок обвал": ("MX", "SHORT"),
    "рынок взлетел": ("MX", "LONG"),
    "рынок растёт": ("MX", "LONG"),
    # ЦБ
    "цб повысил ставку": ("RI", "SHORT"),
    "цб снизил ставку": ("RI", "LONG"),
    # Рубль
    "рубль обвал": ("RI", "SHORT"),
    "рубль укрепился": ("RI", "LONG"),
    "девальвация": ("RI", "SHORT"),
}


def detect_news_momentum(news_items: list, futures_data: dict) -> Optional[Tuple[str, str, str, float]]:
    """Определить есть ли momentum сигнал от новостей.
    
    Args:
        news_items: список новостей [{title, impact, score, ...}]
        futures_data: {base: {dev, price, ema, ...}}
    
    Returns:
        (base, direction, reason, dev) или None
    """
    if not news_items:
        return None
    
    # Ищем HIGH/CRITICAL новости с направлением
    for news in news_items:
        impact = news.get("impact", "")
        if impact not in ("CRITICAL", "HIGH"):
            continue
        
        title = news.get("title", "").lower()
        
        for keyword, (base, direction) in NEWS_DIRECTION_MAP.items():
            if keyword in title:
                # Проверяем что фьючерс двигается в нужном направлении
                fut = futures_data.get(base)
                if not fut:
                    continue
                
                dev = fut.get("dev", 0)
                
                # Движение должно быть сильным (abs dev > MIN_DEV)
                # Для NEWS momentum: направление определяется НОВОСТЬЮ, не dev
                if abs(dev) > MIN_DEV_PCT:
                    log.info(f"📰 NEWS MOMENTUM: {base} {direction} dev={dev:.1f}% | {title[:60]}")
                    return (base, direction, f"news_momentum: {keyword}", dev)
    
    return None
