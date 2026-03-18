"""
MOEX Agent v2 News Filter

RSS-based news monitoring for risk events:
- Sanctions, rate hikes, mobilization
- Market volatility alerts
- Trading halts
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set
from xml.etree import ElementTree

import requests

logger = logging.getLogger("moex_agent.news_filter")

# RSS feed URLs
RSS_FEEDS = [
    "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "https://www.interfax.ru/rss.asp",
]

# Keywords that block all trading
DANGER_BLOCK = {
    "санкции",
    "санкций",
    "цб повысил",
    "цб повышает",
    "ключевая ставка повышена",
    "мобилизация",
    "дефолт",
    "торги приостановлены",
    "торги остановлены",
    "режим чп",
    "военное положение",
}

# Keywords that suggest caution (reduce position size)
DANGER_CAUTION = {
    "волатильность",
    "обвал",
    "фрс",
    "fed",
    "инфляция",
    "опек",
    "opec",
    "девальвация",
    "падение рубля",
    "биржевая паника",
    "массовые продажи",
    "margin call",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MOEX-Agent/2.0"})

# Cache for news items
_news_cache: List[dict] = []
_cache_time: Optional[datetime] = None
CACHE_TTL = timedelta(minutes=5)


@dataclass
class NewsFilterResult:
    """Result of news filtering."""
    should_block: bool = False
    should_reduce_size: bool = False
    size_multiplier: float = 1.0
    block_reasons: List[str] = field(default_factory=list)
    caution_reasons: List[str] = field(default_factory=list)
    news_count: int = 0

    def __repr__(self) -> str:
        if self.should_block:
            return f"NewsFilter(BLOCK: {', '.join(self.block_reasons)})"
        elif self.should_reduce_size:
            return f"NewsFilter(CAUTION x{self.size_multiplier}: {', '.join(self.caution_reasons)})"
        return "NewsFilter(OK)"


def _fetch_rss(url: str, timeout: int = 10) -> List[dict]:
    """
    Fetch and parse RSS feed.

    Args:
        url: RSS feed URL
        timeout: Request timeout

    Returns:
        List of news items with title, description, pubDate
    """
    items = []
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()

        # Parse XML
        root = ElementTree.fromstring(resp.content)

        # Find all item elements
        for item in root.iter("item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            date_el = item.find("pubDate")

            news_item = {
                "title": title_el.text if title_el is not None else "",
                "description": desc_el.text if desc_el is not None else "",
                "pubDate": date_el.text if date_el is not None else "",
                "source": url,
            }
            items.append(news_item)

    except Exception as e:
        logger.warning(f"Failed to fetch RSS {url}: {e}")

    return items


def _fetch_all_news() -> List[dict]:
    """Fetch news from all RSS feeds."""
    global _news_cache, _cache_time

    now = datetime.now(timezone.utc)

    # Return cached if fresh
    if _cache_time and (now - _cache_time) < CACHE_TTL:
        return _news_cache

    all_news = []
    for feed_url in RSS_FEEDS:
        items = _fetch_rss(feed_url)
        all_news.extend(items)

    _news_cache = all_news
    _cache_time = now

    return all_news


def _check_keywords(text: str, keywords: Set[str]) -> List[str]:
    """
    Check text for keyword matches.

    Args:
        text: Text to search
        keywords: Set of keywords to find

    Returns:
        List of matched keywords
    """
    if not text:
        return []

    text_lower = text.lower()
    matches = []

    for keyword in keywords:
        if keyword.lower() in text_lower:
            matches.append(keyword)

    return matches


def check_news_filter(
    lookback_hours: int = 4,
) -> NewsFilterResult:
    """
    Check news for trading-blocking events.

    Fetches recent news from RSS feeds and checks for:
    - DANGER_BLOCK keywords: completely block trading
    - DANGER_CAUTION keywords: reduce position size

    Args:
        lookback_hours: How many hours of news to check

    Returns:
        NewsFilterResult with blocking/caution status
    """
    result = NewsFilterResult()

    try:
        news_items = _fetch_all_news()
        result.news_count = len(news_items)

        if not news_items:
            return result

        # Check each news item
        for item in news_items:
            # Combine title and description
            text = f"{item.get('title', '')} {item.get('description', '')}"

            # Check for blocking keywords
            block_matches = _check_keywords(text, DANGER_BLOCK)
            if block_matches:
                result.should_block = True
                for kw in block_matches:
                    reason = f"[{kw}] {item.get('title', '')[:50]}..."
                    if reason not in result.block_reasons:
                        result.block_reasons.append(reason)

            # Check for caution keywords
            caution_matches = _check_keywords(text, DANGER_CAUTION)
            if caution_matches:
                result.should_reduce_size = True
                for kw in caution_matches:
                    reason = f"[{kw}] {item.get('title', '')[:50]}..."
                    if reason not in result.caution_reasons:
                        result.caution_reasons.append(reason)

        # Set size multiplier based on caution level
        if result.should_block:
            result.size_multiplier = 0.0
        elif result.should_reduce_size:
            # Reduce more with more caution signals
            caution_count = len(result.caution_reasons)
            if caution_count >= 3:
                result.size_multiplier = 0.25
            elif caution_count >= 2:
                result.size_multiplier = 0.5
            else:
                result.size_multiplier = 0.75

    except Exception as e:
        logger.error(f"News filter error: {e}")
        # On error, don't block but be cautious
        result.should_reduce_size = True
        result.size_multiplier = 0.5
        result.caution_reasons.append(f"News fetch error: {e}")

    return result


def clear_cache() -> None:
    """Clear the news cache."""
    global _news_cache, _cache_time
    _news_cache = []
    _cache_time = None
