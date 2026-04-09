"""
News Context Analyzer: reads full article, not just headline.

For CRITICAL/HIGH news:
1. Fetches full text from URL
2. Analyzes context (fact vs discussion vs denial)
3. Returns corrected impact level

Uses keyword analysis on full text — no LLM API needed, works offline.
"""
import logging
import re
import urllib.request
from typing import Optional, Tuple

log = logging.getLogger("news_context")

USER_AGENT = "Mozilla/5.0 (compatible; MoexAgent/2.0)"

# Слова-факты: если есть в тексте → подтверждают CRITICAL
FACT_WORDS = [
    "подписал указ", "вступает в силу", "принял решение", "объявил о",
    "официально", "с сегодняшнего дня", "немедленно", "приказ",
    "начинается", "уже действует", "утвердил", "одобрил",
    "signed", "effective immediately", "announced", "declared",
]

# Слова-отрицания: если есть → это НЕ факт
DENIAL_FULL = [
    "не планирует", "не обсуждает", "не рассматривает", "опроверг",
    "отрицает", "не будет", "исключил", "не исключил", "маловероятно",
    "слухи", "вопрос о", "будет ли", "возможно ли", "рассказал",
    "комментарий", "мнение эксперта", "аналитики считают",
    "рутинная операция", "ежедневно устанавливает", "курс на",
    "denied", "unlikely", "rumors", "questioned",
    # Не про РФ / не влияет на рынок
    "львов", "киев", "украин", "военком", "подозреваемый",
    "убийств", "спасал брата", "журналист глагола",
    "повысил курс", "снизил курс", "установил курс",
]

# Слова-рутина: регулярные операции ЦБ
ROUTINE_WORDS = [
    "курс доллара на", "курс евро на", "курс юаня на",
    "официальный курс", "установил курс", "курс валют",
    "ежедневн", "плановый", "очередной",
]


def fetch_article_text(url: str, max_chars: int = 3000) -> Optional[str]:
    """Fetch and extract text from news article URL."""
    if not url or not url.startswith("http"):
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        html = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", "ignore")
        # Simple text extraction: remove tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception as exc:
        log.debug(f"Failed to fetch article: {exc}")
        return None


def analyze_context(title: str, url: str = "") -> Tuple[str, str]:
    """Analyze news context — is it a fact, discussion, or routine?

    Returns: (verdict, reason)
    verdict: "FACT" | "DENIAL" | "ROUTINE" | "DISCUSSION" | "UNKNOWN"
    """
    text_lower = title.lower()
    
    # 1. Check title first
    for word in ROUTINE_WORDS:
        if word in text_lower:
            return "ROUTINE", f"routine: '{word}' in title"
    
    for word in DENIAL_FULL:
        if word in text_lower:
            return "DENIAL", f"denial: '{word}' in title"
    
    for word in FACT_WORDS:
        if word in text_lower:
            return "FACT", f"fact: '{word}' in title"
    
    # 2. If title is ambiguous, try to fetch full text
    if url:
        full_text = fetch_article_text(url)
        if full_text:
            full_lower = full_text.lower()
            
            # Count facts vs denials in full text
            fact_count = sum(1 for w in FACT_WORDS if w in full_lower)
            denial_count = sum(1 for w in DENIAL_FULL if w in full_lower)
            routine_count = sum(1 for w in ROUTINE_WORDS if w in full_lower)
            
            if routine_count > 0:
                return "ROUTINE", f"routine in article ({routine_count} matches)"
            if denial_count > fact_count:
                return "DENIAL", f"denial>{fact_count} in article ({denial_count} denials)"
            if fact_count > denial_count:
                return "FACT", f"fact>{denial_count} in article ({fact_count} facts)"
            return "DISCUSSION", f"ambiguous ({fact_count} facts, {denial_count} denials)"
    
    return "UNKNOWN", "no clear signal"


def should_downgrade_critical(title: str, url: str = "") -> Tuple[bool, str]:
    """Check if a CRITICAL news should be downgraded.
    
    Returns: (should_downgrade, reason)
    """
    verdict, reason = analyze_context(title, url)
    
    if verdict in ("DENIAL", "ROUTINE", "DISCUSSION"):
        return True, f"{verdict}: {reason}"
    
    return False, ""
