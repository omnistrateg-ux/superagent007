"""
MOEX Agent v2 News Filter

RSS-based news monitoring with keyword scoring:
- CRITICAL (≥9): остановить ВСЕ сделки
- HIGH (7-8): не открывать новые позиции
- MEDIUM (4-6): уменьшить размер на 50%
- LOW (<4): торговать нормально
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree

import requests

from .telegram import send_telegram_message

logger = logging.getLogger("moex_agent.news_filter")

# RSS feed URLs
RSS_FEEDS = [
    ("РБК", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"),
    ("Интерфакс", "https://www.interfax.ru/rss.asp"),
]

# Keyword scoring: keyword -> score
KEYWORD_SCORES: Dict[str, int] = {
    # CRITICAL (10) - ЦБ и ставка
    "ключевая ставка": 10,
    "цб повысил": 10,
    "цб повышает": 10,
    "ставка повышена": 10,
    "банк россии повысил": 10,
    "экстренное заседание цб": 10,

    # HIGH (8) - Санкции и ограничения
    "санкции": 8,
    "санкций": 8,
    "ограничения": 8,
    "блокировка активов": 8,
    "торги приостановлены": 8,
    "торги остановлены": 8,
    "дефолт": 8,
    "мобилизация": 8,

    # MEDIUM (6) - Нефть и ОПЕК
    "опек": 6,
    "opec": 6,
    "нефть": 6,
    "нефти": 6,
    "баррель": 6,
    "газпром": 6,
    "газ": 5,

    # LOW-MEDIUM (4-5) - Общие рыночные
    "волатильность": 5,
    "обвал": 5,
    "фрс": 4,
    "fed": 4,
    "инфляция": 4,
    "девальвация": 5,
    "падение рубля": 5,
    "margin call": 5,
}

# Уровни риска
LEVEL_CRITICAL = 9   # ≥9: остановить ВСЕ
LEVEL_HIGH = 7       # 7-8: не открывать новые
LEVEL_MEDIUM = 4     # 4-6: уменьшить размер

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MOEX-Agent/2.0"})

# Cache for news items
_news_cache: List[dict] = []
_cache_time: Optional[datetime] = None
_last_alert_time: Optional[datetime] = None
CACHE_TTL = timedelta(minutes=5)
ALERT_COOLDOWN = timedelta(minutes=30)  # Не спамить алертами


@dataclass
class NewsAlert:
    """Новостной алерт."""
    level: str  # CRITICAL, HIGH, MEDIUM, LOW
    score: int
    keyword: str
    title: str
    source: str

    def __repr__(self) -> str:
        return f"NewsAlert({self.level}: {self.keyword} [{self.score}])"


@dataclass
class NewsFilterResult:
    """Результат проверки новостей."""
    max_score: int = 0
    level: str = "LOW"  # CRITICAL, HIGH, MEDIUM, LOW
    should_stop: bool = False      # Остановить всё
    should_block_new: bool = False  # Не открывать новые
    size_multiplier: float = 1.0   # Множитель размера
    alerts: List[NewsAlert] = field(default_factory=list)
    news_count: int = 0

    def __repr__(self) -> str:
        if self.should_stop:
            return f"NewsFilter(STOP: score={self.max_score})"
        elif self.should_block_new:
            return f"NewsFilter(BLOCK_NEW: score={self.max_score})"
        elif self.size_multiplier < 1.0:
            return f"NewsFilter(REDUCE x{self.size_multiplier}: score={self.max_score})"
        return f"NewsFilter(OK: score={self.max_score})"


def _fetch_rss(source: str, url: str, timeout: int = 10) -> List[dict]:
    """Fetch and parse RSS feed."""
    items = []
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()

        root = ElementTree.fromstring(resp.content)

        for item in root.iter("item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            date_el = item.find("pubDate")

            news_item = {
                "title": title_el.text if title_el is not None else "",
                "description": desc_el.text if desc_el is not None else "",
                "pubDate": date_el.text if date_el is not None else "",
                "source": source,
                "url": url,
            }
            items.append(news_item)

    except Exception as e:
        logger.warning(f"Ошибка RSS {source}: {e}")

    return items


def _fetch_all_news(force: bool = False) -> List[dict]:
    """Fetch news from all RSS feeds."""
    global _news_cache, _cache_time

    now = datetime.now(timezone.utc)

    # Return cached if fresh
    if not force and _cache_time and (now - _cache_time) < CACHE_TTL:
        return _news_cache

    all_news = []
    for source, feed_url in RSS_FEEDS:
        items = _fetch_rss(source, feed_url)
        all_news.extend(items)

    _news_cache = all_news
    _cache_time = now

    logger.debug(f"Загружено {len(all_news)} новостей из {len(RSS_FEEDS)} RSS")
    return all_news


def _score_text(text: str) -> Tuple[int, str]:
    """
    Оценить текст по ключевым словам.

    Returns:
        (max_score, matched_keyword)
    """
    if not text:
        return 0, ""

    text_lower = text.lower()
    max_score = 0
    matched_keyword = ""

    for keyword, score in KEYWORD_SCORES.items():
        if keyword.lower() in text_lower:
            if score > max_score:
                max_score = score
                matched_keyword = keyword

    return max_score, matched_keyword


def _get_level(score: int) -> str:
    """Получить уровень риска по скору."""
    if score >= LEVEL_CRITICAL:
        return "CRITICAL"
    elif score >= LEVEL_HIGH:
        return "HIGH"
    elif score >= LEVEL_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _send_news_alert(alert: NewsAlert) -> None:
    """Отправить алерт в Telegram."""
    global _last_alert_time

    now = datetime.now(timezone.utc)

    # Cooldown check
    if _last_alert_time and (now - _last_alert_time) < ALERT_COOLDOWN:
        logger.debug(f"Алерт пропущен (cooldown): {alert.keyword}")
        return

    level_emoji = {
        "CRITICAL": "🚨",
        "HIGH": "⚠️",
        "MEDIUM": "📢",
        "LOW": "ℹ️",
    }.get(alert.level, "📰")

    action = {
        "CRITICAL": "⛔ Торговля остановлена",
        "HIGH": "🚫 Новые позиции заблокированы",
        "MEDIUM": "📉 Размер позиций -50%",
        "LOW": "",
    }.get(alert.level, "")

    title_short = alert.title[:80] + "..." if len(alert.title) > 80 else alert.title

    msg = (
        f"{level_emoji} НОВОСТЬ [{alert.level}]\n"
        f"Источник: {alert.source}\n"
        f'"{title_short}"\n'
    )
    if action:
        msg += f"\n{action}"

    send_telegram_message(msg)
    _last_alert_time = now
    logger.info(f"Отправлен алерт: {alert.level} - {alert.keyword}")


def check_news_filter(
    lookback_hours: int = 4,
    send_alerts: bool = True,
) -> NewsFilterResult:
    """
    Проверить новости и вернуть результат фильтрации.

    Args:
        lookback_hours: Сколько часов новостей проверять
        send_alerts: Отправлять ли алерты в Telegram

    Returns:
        NewsFilterResult с уровнем риска и действиями
    """
    result = NewsFilterResult()

    try:
        news_items = _fetch_all_news()
        result.news_count = len(news_items)

        if not news_items:
            return result

        # Check each news item
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('description', '')}"
            score, keyword = _score_text(text)

            if score > 0:
                level = _get_level(score)
                alert = NewsAlert(
                    level=level,
                    score=score,
                    keyword=keyword,
                    title=item.get("title", "")[:100],
                    source=item.get("source", ""),
                )
                result.alerts.append(alert)

                if score > result.max_score:
                    result.max_score = score
                    result.level = level

        # Определить действия по максимальному скору
        if result.max_score >= LEVEL_CRITICAL:
            result.should_stop = True
            result.should_block_new = True
            result.size_multiplier = 0.0
        elif result.max_score >= LEVEL_HIGH:
            result.should_block_new = True
            result.size_multiplier = 0.5
        elif result.max_score >= LEVEL_MEDIUM:
            result.size_multiplier = 0.5

        # Отправить алерт для CRITICAL/HIGH
        if send_alerts and result.alerts:
            top_alert = max(result.alerts, key=lambda a: a.score)
            if top_alert.score >= LEVEL_HIGH:
                _send_news_alert(top_alert)

    except Exception as e:
        logger.error(f"Ошибка news filter: {e}")
        # При ошибке — осторожность
        result.size_multiplier = 0.75

    return result


def clear_cache() -> None:
    """Очистить кэш новостей."""
    global _news_cache, _cache_time
    _news_cache = []
    _cache_time = None


def force_refresh() -> NewsFilterResult:
    """Принудительно обновить новости и проверить."""
    clear_cache()
    return check_news_filter(send_alerts=True)
