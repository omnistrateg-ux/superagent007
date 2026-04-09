#!/usr/bin/env python3
"""
News Feed Module — RSS/API парсер новостей для торгового агента.
Источники: РБК, ТАСС, Интерфакс, ЦБ РФ
Оценивает влияние новостей на рынок и фьючерсы (SI, BR, RI, GD).
"""
import json
import logging
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("moex_agent.news")

MSK = timezone(timedelta(hours=3))

# ── RSS Sources ──────────────────────────────────────────

FEEDS = {
    "РБК": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "ТАСС": "https://tass.ru/rss/v2.xml",  # Fixed: was .com (English), now .ru (Russian)
    "Интерфакс": "https://www.interfax.ru/rss.asp",
    # Faster search/commodity coverage
    "GoogleNews Iran/Oil": (
        "https://news.google.com/rss/search?"
        "q=(Iran+OR+Hormuz+OR+%22Strait+of+Hormuz%22)+(oil+OR+crude+OR+tanker+OR+energy+OR+LNG)"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
    "GoogleNews Russia Market": (
        "https://news.google.com/rss/search?"
        "q=(Russia+OR+MOEX+OR+ruble+OR+%22Central+Bank+of+Russia%22+OR+sanctions)+(market+OR+stocks+OR+rate)"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
    "GoogleNews LNG/Gas": (
        "https://news.google.com/rss/search?"
        "q=(LNG+OR+natural+gas+OR+Qatar+OR+Europe+gas)+(price+OR+supply+OR+disruption)"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
    "OilPrice": "https://www.oilprice.com/rss/main",
    # Tier-1 Russian business media
    "Ведомости": "https://www.vedomosti.ru/rss/news",
    # Investing.com Russia — price alerts and market reactions
    "InvestingRU": "https://ru.investing.com/rss/news.rss",
    # Central Bank of Russia — rate decisions, policy
    "CBR": "https://cbr.ru/rss/eventrss",
}

USER_AGENT = "Mozilla/5.0 (compatible; MoexAgent/2.0)"

# Public Telegram channels (read via t.me/s/<channel>) — fast layer without API credentials.
TELEGRAM_FAST_CHANNELS = [
    "@markettwits",
    "@banksta",
    "@smartlabnews",
    "@investfuture",
    "@russianmacro",
    "@bankrollo",
    "@cbrstocks",       # Added: oil/energy sector news
    "@oil_capital",     # Added: oil/gas industry news
]

TELEGRAM_FAST_CHANNEL_TITLES = {
    "@bankrollo": "Банки, деньги, два офшора",
}


def _infer_source_meta(source: str) -> Tuple[str, int]:
    source_lower = (source or "").lower()
    if "manual" in source_lower:
        return "manual_forward", 10
    if "bankrollo" in source_lower or "банки, деньги, два офшора" in source_lower:
        return "telegram_fast", 5
    if source_lower.startswith("tg ") or source_lower.startswith("tg:"):
        return "telegram_fast", 4
    if "oilprice" in source_lower:
        return "commodity_feed", 3
    if "googlenews" in source_lower:
        return "search_feed", 2
    return "feed", 1

# ── Impact Keywords ──────────────────────────────────────

# CRITICAL — полная остановка торговли (только то, что реально ломает рынок)
CRITICAL_KEYWORDS = [
    # ЦБ и ставка — только реальные решения/действия, не обзорные статьи
    "цб повысил ставку", "цб снизил ставку", "цб сохранил ставку",
    "цб повысил ключевую", "цб снизил ключевую",
    "банк россии повысил ставку", "банк россии снизил ставку", "банк россии сохранил ставку",
    "банк россии повысил ключевую", "банк россии снизил ключевую",
    "внеплановое заседание цб", "экстренное заседание цб",
    # Ядерка / война / мобилизация
    "ядерный удар", "ядерная война", "nuclear war", "nuclear strike",
    "военное положение в росси",
    "объявлена мобилизаци", "всеобщая мобилизаци", "частичная мобилизаци",
    "указ о мобилизаци", "начало мобилизаци", "подписал указ о мобилизаци",
    "путин объявил мобилизаци", "путин подписал мобилизаци",
    "мосбиржа приостановила", "биржа приостановила", "торги приостановлены",
    "торги остановлены", "делистинг",
]

# HIGH — уменьшение размера (×0.5) — НЕФТЬ, ГАЗ, влияние на Россию
HIGH_KEYWORDS = [
    # Нефть — обвалы и кризисы
    "нефть упала", "нефть обвал", "нефть рухнула", "oil crash", "oil plunge",
    "opec сократил", "opec+ развал", "opec снизил", "сделка opec",
    "эмбарго на нефть", "oil embargo", "нефтяное эмбарго",
    "потолок цен на нефть", "oil price cap",
    # Газ — кризисы
    "газовый кризис", "газ перекрыт", "газопровод", "nord stream", "северный поток",
    "газпром остановил", "газпром прекратил", "транзит газа",
    # Санкции против РФ нефтегаз
    "санкции против россии", "санкции против рф", "санкции против газпром",
    "санкции против роснефт", "санкции против лукойл", "санкции против новатэк",
    "sanctions against russia", "new russia sanctions",
    # Рубль / дефолт / биржа РФ
    "обвал рубля", "девальвация рубля", "дефолт россии",
    "биржа приостановила торги", "мосбиржа приостановила",
    "отключение swift", "россию отключ",
]

# MEDIUM — alert only, no trade impact (информационно)
MEDIUM_KEYWORDS = [
    # Нефть и газ — фон
    "нефть", "oil", "brent", "urals", "опек", "opec",
    "газ природный", "lng", "спг", "газпром", "новатэк", "роснефть", "лукойл",
    # Россия экономика
    "инфляция", "ввп россии", "набиуллина", "силуанов", "минфин",
    "рубль", "доллар к рублю", "ключевая ставка", "ставку цб",
    # Биржа
    "ммвб", "moex", "мосбиржа",
    "дивиденды", "dividend",
]

# Price level breakthrough keywords → should be HIGH for the relevant futures
PRICE_BREAKTHROUGH_KEYWORDS = [
    "пробила", "пробил", "обновил максимум", "обновил минимум",
    "рекордный уровень", "максимум с", "минимум с", "впервые с",
    "surged", "surpassed", "broke through", "hit record", "all-time high",
]

# De-escalation keywords → opposite bias to escalation
DEESCALATION_KEYWORDS = [
    "завершится", "завершение", "перемирие", "ceasefire", "peace",
    "мирный план", "мирное урегулирование", "peace plan", "peace deal",
    "прекращение огня", "отвод войск", "withdrawal",
    "переговоры успешны", "договорились", "соглашение достигнуто",
    "deal reached", "agreement", "de-escalation", "деэскалац",
]

# ── Futures Impact Map ───────────────────────────────────

FUTURES_KEYWORDS = {
    "SI": ["доллар", "рубль", "usd", "rub", "курс", "валют", "цб", "ставк", "санкци", "swift"],
    "BR": [
        "нефть", "oil", "brent", "urals", "опек", "opec", "сокращение добычи", "эмбарго нефт",
        "сауд", "saudi", "iran", "иран", "ирак", "iraq", "opec+", "ормуз", "hormuz",
        "strait of hormuz", "танкер", "tanker", "middle east",
    ],
    "GD": ["золото", "gold", "драгметалл", "safe haven", "геополитик"],
    "NG": [
        "газ", "gas", "газпром", "спг", "lng", "труб", "nord stream", "катар", "lng export",
        "iran", "иран", "qatar", "ормуз", "hormuz", "middle east",
    ],
    "RI": ["ртс", "rts", "фондовый", "индекс", "иностранные инвестор", "россия", "санкции против россии"],
    "MX": ["ммвб", "moex", "мосбиржа", "биржа", "торги", "российский рынок"],
}

# Stricter keywords for long Telegram multi-topic posts: must appear near the lead.
FUTURES_LEAD_KEYWORDS = {
    "SI": ["доллар", "рубль", "usd", "rub", "курс", "валют", "цб", "ставк"],
    "BR": ["нефть", "oil", "brent", "urals", "опек", "opec", "ormuz", "hormuz", "tanker", "танкер", "iran", "иран"],
    "GD": ["золото", "gold", "драгметалл", "bullion", "safe haven"],
    "NG": ["газ", "gas", "спг", "lng", "газпром", "qatar", "катар", "nord stream", "труб"],
    "RI": ["ртс", "rts", "индекс", "санкции против россии"],
    "MX": ["ммвб", "moex", "мосбиржа", "биржа", "российский рынок"],
}

# Важные сырьевые страны / регионы — если там война/катаклизм, это влияет на нефть/газ
COMMODITY_COUNTRIES = [
    "саудов", "саудовская аравия", "iran", "иран", "ирак", "qatar", "катар",
    "uae", "оаэ", "kuwait", "кувейт", "libya", "ливи", "nigeria", "нигери",
    "venezuela", "венесуэл", "kazakhstan", "казахстан", "algeria", "алжир",
    "норвег", "norway", "azerbaijan", "азербайджан",
]

COMMODITY_SHOCK_KEYWORDS = [
    "война", "war", "удар", "strike", "attack", "атака", "обстрел",
    "взрыв", "explosion", "переворот", "coup",
    "землетряс", "earthquake", "ураган", "hurricane", "шторм", "storm",
    "наводнен", "flood", "пожар", "wildfire", "катастроф", "catastrophe",
]

# Геополитическая эскалация вокруг commodity-регионов не всегда содержит слово "война",
# но всё равно двигает нефть/газ: Iran rejects peace plan, Hormuz tension, tanker threats, etc.
GEOPOLITICAL_ESCALATION_KEYWORDS = [
    "эскалац", "escalat", "напряжен", "tension", "reject", "rejected", "отверг",
    "мирн", "peace", "ceasefire", "перемири", "ультиматум", "threat", "угроз",
    "блокад", "closure", "закрыт", "закрытие", "суверенитет", "перекрыт",
]

STRATEGIC_CHOKEPOINT_KEYWORDS = [
    "ормуз", "hormuz", "strait of hormuz", "ormuz strait",
    "танкер", "tanker", "shipping lane", "морской путь",
]


# ── Data Structures ──────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    source: str
    url: str = ""
    published: Optional[datetime] = None
    impact: str = "LOW"           # CRITICAL / HIGH / MEDIUM / LOW
    score: float = 1.0            # multiplier: 0=stop, 0.5=half, 1.0=normal
    keywords_hit: List[str] = field(default_factory=list)
    futures_affected: List[str] = field(default_factory=list)
    source_type: str = "feed"
    source_priority: int = 1
    ingested_at: Optional[datetime] = None
    confidence: float = 0.0
    confirmation_count: int = 0
    confirmed_by: List[str] = field(default_factory=list)


@dataclass
class NewsDigest:
    items: List[NewsItem] = field(default_factory=list)
    overall_score: float = 1.0    # min of all scores
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    fetched_at: Optional[datetime] = None
    futures_alerts: Dict[str, str] = field(default_factory=dict)  # {SI: "санкции..."}


# ── RSS Parser ───────────────────────────────────────────

def _fetch_rss(url: str, timeout: int = 10) -> List[Dict]:
    """Fetch and parse RSS feed, return list of {title, link, pubDate}."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        data = urllib.request.urlopen(req, timeout=timeout).read()
        root = ET.fromstring(data)

        items = []
        # Standard RSS 2.0
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title:
                items.append({"title": title, "link": link, "pubDate": pub})

        # Atom fallback
        if not items:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
                pub = (entry.findtext("atom:published", namespaces=ns) or "").strip()
                if title:
                    items.append({"title": title, "link": link, "pubDate": pub})

        return items
    except Exception as e:
        log.debug(f"RSS fetch error {url}: {e}")
        return []


def _fetch_telegram_channel(channel: str, timeout: int = 10, limit: int = 12) -> List[Dict]:
    """Fetch recent posts from a public Telegram channel page."""
    username = channel.lstrip("@")
    url = f"https://t.me/s/{username}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
        soup = BeautifulSoup(html, "lxml")
        items = []
        wraps = soup.select(".tgme_widget_message_wrap")[-limit:]
        for wrap in wraps:
            text_el = wrap.select_one(".tgme_widget_message_text")
            time_el = wrap.select_one("time")
            link_el = wrap.select_one(".tgme_widget_message_date")
            title = text_el.get_text(" ", strip=True) if text_el else ""
            link = link_el.get("href", "") if link_el else ""
            pub = time_el.get("datetime", "") if time_el else ""
            if title:
                items.append({"title": title, "link": link, "pubDate": pub})
        return items
    except Exception as e:
        log.debug(f"Telegram fetch error {channel}: {e}")
        return []


def _parse_pub_date(pub_str: str) -> Optional[datetime]:
    """Try to parse various RSS date formats with timezone support."""
    pub_str = (pub_str or "").strip()
    if not pub_str:
        return None

    # Best-effort RFC822/HTTP-date parser (handles GMT/UTC correctly)
    try:
        dt = parsedate_to_datetime(pub_str)
        if dt is not None:
            return dt
    except Exception:
        pass

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d.%m.%Y %H:%M",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(pub_str, fmt)
            if fmt.endswith("Z") and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ── Keyword Matching ─────────────────────────────────────

def _match_keywords(text: str) -> Tuple[str, float, List[str]]:
    """Match text against keyword lists, return (impact, score, matched_keywords)."""
    text_lower = text.lower()
    matched = []

    # Check CRITICAL first
    # Слова-отрицания: если есть — это НЕ факт, а обсуждение/вопрос/отрицание
    DENIAL_WORDS = [
        "будет ли", "не будет", "не планирует", "не рассматривает",
        "опроверг", "отрицает", "рассказал", "заявил что не",
        "исключил", "не исключил", "обсуждение", "слухи",
        "не обсуждают", "не обсуждает", "не готовит", "не готовят",
        "не собирается", "нет планов", "не ведётся", "не ведется",
        "львов", "киев", "украин", "военком", "подозреваемый", "убийств",
        "отклонил жалобу", "отклонил иск", "кс отклонил", "суд отклонил",
        "контрактник", "жалобу на указ",
        "повысил курс", "снизил курс", "установил курс", "курс доллара на",
    ]
    has_denial = any(dw in text_lower for dw in DENIAL_WORDS)

    for kw in CRITICAL_KEYWORDS:
        if kw in text_lower:
            matched.append(kw)
    if matched:
        if has_denial:
            # Кликбейт или отрицание — понижаем до MEDIUM
            return "MEDIUM", 1.0, matched
        return "CRITICAL", 0.0, matched

    # Special cases: major commodity-producing country / choke-point geopolitical stress.
    country_hits = [kw for kw in COMMODITY_COUNTRIES if kw in text_lower]
    shock_hits = [kw for kw in COMMODITY_SHOCK_KEYWORDS if kw in text_lower]
    escalation_hits = [kw for kw in GEOPOLITICAL_ESCALATION_KEYWORDS if kw in text_lower]
    choke_hits = [kw for kw in STRATEGIC_CHOKEPOINT_KEYWORDS if kw in text_lower]
    if country_hits and shock_hits:
        return "HIGH", 0.5, country_hits[:2] + shock_hits[:2]
    if country_hits and choke_hits:
        return "HIGH", 0.5, country_hits[:2] + choke_hits[:2]
    if country_hits and escalation_hits:
        return "HIGH", 0.5, country_hits[:2] + escalation_hits[:2]

    # Check HIGH
    matched = []
    for kw in HIGH_KEYWORDS:
        if kw in text_lower:
            matched.append(kw)
    if matched:
        return "HIGH", 0.5, matched

    # Price breakthroughs → HIGH (e.g. "Brent пробила $112")
    price_hits = [kw for kw in PRICE_BREAKTHROUGH_KEYWORDS if kw in text_lower]
    commodity_hits = [kw for kw in ("brent", "нефть", "oil", "газ", "gold", "золото") if kw in text_lower]
    if price_hits and commodity_hits:
        return "HIGH", 0.5, price_hits[:1] + commodity_hits[:1]

    # Check MEDIUM
    matched = []
    for kw in MEDIUM_KEYWORDS:
        if kw in text_lower:
            matched.append(kw)
    if matched:
        return "MEDIUM", 1.0, matched

    return "LOW", 1.0, []


def _match_futures(text: str) -> List[str]:
    """Determine which futures contracts are affected by the news."""
    text_lower = text.lower()
    # Исключения: слова-ловушки, которые содержат keyword но не про этот контракт
    FALSE_POSITIVES = {
        "NG": ["газпромбанк", "газпром банк", "газпром нефть акци"],
    }
    affected = []
    for contract, keywords in FUTURES_KEYWORDS.items():
        # Сначала проверим исключения
        fp_list = FALSE_POSITIVES.get(contract, [])
        is_false = any(fp in text_lower for fp in fp_list)
        if is_false:
            continue
        for kw in keywords:
            if kw in text_lower:
                affected.append(contract)
                break
    return affected


def _title_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "after", "into", "just",
        "как", "что", "это", "при", "после", "для", "над", "под", "или", "его", "её",
        "news", "telegram", "market", "markets", "russia", "россия",
    }
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 4 and t not in stop}


def _is_digest_like_title(text: str) -> bool:
    text_lower = (text or "").lower()
    digest_patterns = (
        "обзор новостей",
        "главное за день",
        "итоги дня",
        "итоги недели",
        "главное к утру",
        "что важно знать",
        "дайджест",
        "подборка новостей",
        "news digest",
        "morning briefing",
    )
    return any(pattern in text_lower for pattern in digest_patterns)


def _telegram_contract_lead_match(item: "NewsItem", contract: str) -> bool:
    if not item.source.startswith("TG "):
        return True
    lead = re.sub(r"\s+", " ", (item.title or "").lower()).strip()[:180]
    keywords = FUTURES_LEAD_KEYWORDS.get(contract, FUTURES_KEYWORDS.get(contract, []))
    return any(keyword in lead for keyword in keywords)


def _contract_signal_relevant(item: "NewsItem", contract: str) -> bool:
    text_lower = re.sub(r"\s+", " ", (item.title or "").lower()).strip()

    if contract == "SI":
        hard_fx = ("доллар", "usd", "рубл", "валют", "fx", "forex", "курс", "девальвац", "юань", "cny")
        policy_fx = ("валютный рынок", "курс рубля", "курс доллара", "операции на валютном рынке")
        return any(term in text_lower for term in hard_fx) or any(term in text_lower for term in policy_fx)

    if contract == "RI":
        equity_terms = ("ртс", "rts", "индекс ртс", "фондовый рынок", "рынок акций", "российский рынок акций")
        sanction_terms = ("санкции против россии", "new russia sanctions")
        return any(term in text_lower for term in equity_terms + sanction_terms)

    if contract == "MX":
        market_terms = ("мосбиржа", "moex", "ммвб", "индекс мосбиржи", "торги на мосбирже", "российский рынок акций")
        return any(term in text_lower for term in market_terms)

    if contract == "GD":
        gold_terms = ("золото", "gold", "драгметалл", "bullion", "safe haven")
        return any(term in text_lower for term in gold_terms)

    return True


def _cluster_news_events(items: List["NewsItem"], contract: Optional[str] = None) -> List[Dict[str, object]]:
    def sort_key(item: "NewsItem"):
        dt = item.published or item.ingested_at or datetime.min.replace(tzinfo=MSK)
        return (dt, item.confidence, item.source_priority)

    clusters: List[Dict[str, object]] = []
    for item in sorted(items, key=sort_key, reverse=True):
        placed = False
        for cluster in clusters:
            rep = cluster["rep"]
            same_contract = True if not contract else (contract in rep.futures_affected and contract in item.futures_affected)
            if same_contract and _same_event(item, rep):
                cluster["items"].append(item)
                placed = True
                break
        if not placed:
            clusters.append({"rep": item, "items": [item]})
    return clusters


def _event_signature(item: "NewsItem", contract: str) -> str:
    text_lower = (item.title or "").lower()
    matched_contract_terms = [kw for kw in FUTURES_KEYWORDS.get(contract, []) if kw in text_lower][:2]
    bias = _infer_contract_bias(item.title, contract)
    title_tokens = sorted(_title_tokens(item.title))[:3]
    parts = [contract, item.impact, bias]
    if matched_contract_terms:
        parts.extend(sorted(set(matched_contract_terms)))
    elif item.keywords_hit:
        parts.extend(sorted(set(item.keywords_hit))[:2])
    parts.extend(title_tokens)
    return "|".join(parts)


def _same_event(a: "NewsItem", b: "NewsItem") -> bool:
    if a is b:
        return False
    shared_keywords = set(a.keywords_hit) & set(b.keywords_hit)
    shared_futures = set(a.futures_affected) & set(b.futures_affected)
    shared_tokens = _title_tokens(a.title) & _title_tokens(b.title)
    # Ослабленный matching: совпадение по 3+ токенам ИЛИ по keywords+futures ИЛИ по keywords+tokens
    if len(shared_tokens) >= 3:
        return True
    if shared_keywords and shared_futures:
        return True
    if shared_keywords and len(shared_tokens) >= 2:
        return True
    if shared_futures and len(shared_tokens) >= 2:
        return True
    return False


def _enrich_confidence(items: List["NewsItem"]) -> None:
    for item in items:
        confirmations = sorted({other.source for other in items if _same_event(item, other)})
        item.confirmed_by = confirmations
        item.confirmation_count = len(confirmations)

        base = 0.45
        if item.impact == "HIGH":
            base = 0.60
        elif item.impact == "CRITICAL":
            base = 0.80

        # Source quality bonus (authoritative sources get more)
        # Priority: manual(10) > tg_fast(4-5) > commodity_feed(3) > search_feed(2) > feed(1)
        if item.source_type == "manual_forward":
            source_bonus = 0.15
        elif item.source_type in ("feed",) and any(s in (item.source or "").lower() for s in ("рбк", "тасс", "интерфакс")):
            source_bonus = 0.12  # Tier-1 Russian media
        elif item.source_type == "telegram_fast":
            source_bonus = 0.05  # Telegram channels = less reliable
        elif item.source_type == "commodity_feed":
            source_bonus = 0.08
        else:
            source_bonus = 0.03

        # Cross-source confirmation bonus (capped)
        confirmation_bonus = min(0.20, 0.07 * item.confirmation_count)
        item.confidence = round(min(0.95, base + source_bonus + confirmation_bonus), 2)


# ── Main API ─────────────────────────────────────────────

_cache: Dict[str, any] = {"digest": None, "ts": 0}
_health: Dict[str, any] = {"last_success": 0, "last_fail_alert": 0, "feed_status": {}}
CACHE_TTL = 120  # 2 minutes between fetches
NEWS_STALE_SECONDS = 1200  # 20 min — if no successful fetch, alert
NEWS_STALE_ALERT_INTERVAL = 900  # Don't spam stale alerts more than every 15 min


def fetch_news(max_age_minutes: int = 30, force: bool = False) -> NewsDigest:
    """
    Fetch news from all sources, score impact.
    Returns NewsDigest with overall_score for trading decisions.
    Cached for 2 minutes to avoid hammering RSS feeds.
    """
    now = time.time()
    if not force and _cache["digest"] and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["digest"]

    digest = NewsDigest(fetched_at=datetime.now(MSK))
    cutoff = datetime.now(MSK) - timedelta(minutes=max_age_minutes)
    seen_titles = set()

    feeds_ok = 0
    for source, url in FEEDS.items():
        raw_items = _fetch_rss(url)
        if raw_items:
            feeds_ok += 1
            _health["feed_status"][source] = {"ok": True, "ts": now, "count": len(raw_items)}
        else:
            _health["feed_status"][source] = {"ok": False, "ts": now, "count": 0}
        for raw in raw_items[:15]:  # Last 15 per source
            title = raw["title"]
            pub_date = _parse_pub_date(raw.get("pubDate", ""))

            # Filter by age if we can parse the date
            if pub_date:
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=MSK)
                if pub_date < cutoff:
                    continue

            impact, score, keywords = _match_keywords(title)
            if impact == "LOW":
                continue  # Skip irrelevant news

            # Context check: CRITICAL → verify by reading full article
            if impact == "CRITICAL":
                try:
                    from .news_context import should_downgrade_critical
                    downgrade, ctx_reason = should_downgrade_critical(title, raw.get("link", ""))
                    if downgrade:
                        impact = "MEDIUM"
                        score = 1.0
                        log.info(f"NEWS CONTEXT: downgraded CRITICAL → MEDIUM: {ctx_reason}")
                except Exception:
                    pass

            normalized_title = re.sub(r"\s+", " ", title.lower()).strip()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)

            futures_affected = _match_futures(title)

            source_type, source_priority = _infer_source_meta(source)
            item = NewsItem(
                title=title,
                source=source,
                url=raw.get("link", ""),
                published=pub_date,
                impact=impact,
                score=score,
                keywords_hit=keywords,
                futures_affected=futures_affected,
                source_type=source_type,
                source_priority=source_priority,
                ingested_at=datetime.now(MSK),
            )
            digest.items.append(item)

            if impact == "CRITICAL":
                digest.critical_count += 1
            elif impact == "HIGH":
                digest.high_count += 1
            elif impact == "MEDIUM":
                digest.medium_count += 1

            # Track futures alerts
            for fut in futures_affected:
                if fut not in digest.futures_alerts:
                    digest.futures_alerts[fut] = f"[{impact}] {title[:80]}"

    # Fast Telegram layer via public channel pages
    for channel in TELEGRAM_FAST_CHANNELS:
        raw_items = _fetch_telegram_channel(channel)
        if raw_items:
            feeds_ok += 1
            _health["feed_status"][f"TG {channel}"] = {"ok": True, "ts": now, "count": len(raw_items)}
        else:
            _health["feed_status"][f"TG {channel}"] = {"ok": False, "ts": now, "count": 0}

        for raw in raw_items[-10:]:
            title = raw["title"]
            pub_date = _parse_pub_date(raw.get("pubDate", ""))
            if pub_date:
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=MSK)
                if pub_date.astimezone(MSK) < cutoff:
                    continue

            impact, score, keywords = _match_keywords(title)
            if impact == "LOW":
                continue

            normalized_title = re.sub(r"\s+", " ", title.lower()).strip()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)

            futures_affected = _match_futures(title)
            channel_label = TELEGRAM_FAST_CHANNEL_TITLES.get(channel, channel)
            source = f"TG {channel_label}"
            source_type, source_priority = _infer_source_meta(source)
            item = NewsItem(
                title=title,
                source=source,
                url=raw.get("link", ""),
                published=pub_date,
                impact=impact,
                score=score,
                keywords_hit=keywords,
                futures_affected=futures_affected,
                source_type=source_type,
                source_priority=source_priority,
                ingested_at=datetime.now(MSK),
            )
            digest.items.append(item)

            if impact == "CRITICAL":
                digest.critical_count += 1
            elif impact == "HIGH":
                digest.high_count += 1
            elif impact == "MEDIUM":
                digest.medium_count += 1

            for fut in futures_affected:
                if fut not in digest.futures_alerts:
                    digest.futures_alerts[fut] = f"[{impact}] {title[:80]}"

    # Merge manual/forwarded news from chat as first-class signals
    try:
        from .manual_news import load_recent_manual_news
        for raw in load_recent_manual_news(max_age_minutes=max_age_minutes):
            title = raw.get("title") or raw.get("text") or ""
            normalized_title = re.sub(r"\s+", " ", title.lower()).strip()
            if not title or normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)

            published = raw.get("published_at") or raw.get("ingested_at")
            pub_date = datetime.fromisoformat(published) if published else None
            if pub_date and pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=MSK)

            item = NewsItem(
                title=title,
                source=raw.get("source", "manual_forward"),
                url="",
                published=pub_date,
                impact=raw.get("impact", "MEDIUM"),
                score=float(raw.get("score", 1.0) or 1.0),
                keywords_hit=list(raw.get("keywords_hit") or []),
                futures_affected=list(raw.get("futures_affected") or []),
                source_type=raw.get("source_type", "manual_forward"),
                source_priority=int(raw.get("source_priority", 10) or 10),
                ingested_at=datetime.fromisoformat(raw.get("ingested_at")) if raw.get("ingested_at") else None,
            )
            digest.items.append(item)

            if item.impact == "CRITICAL":
                digest.critical_count += 1
            elif item.impact == "HIGH":
                digest.high_count += 1
            elif item.impact == "MEDIUM":
                digest.medium_count += 1

            for fut in item.futures_affected:
                if fut not in digest.futures_alerts:
                    digest.futures_alerts[fut] = f"[{item.impact}] {title[:80]}"
    except Exception as e:
        log.debug(f"Manual news merge error: {e}")

    _enrich_confidence(digest.items)

    # Overall score = minimum (most restrictive)
    if digest.items:
        digest.overall_score = min(item.score for item in digest.items)
    else:
        digest.overall_score = 1.0

    # Sort by impact first, then prefer faster/manual sources, then fresher items
    impact_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    digest.items.sort(
        key=lambda x: (
            impact_order.get(x.impact, 3),
            -int(getattr(x, "source_priority", 0) or 0),
            -(int(x.published.timestamp()) if x.published else 0),
        )
    )

    _cache["digest"] = digest
    _cache["ts"] = now

    if feeds_ok > 0:
        _health["last_success"] = now

    if digest.critical_count or digest.high_count:
        log.info(f"NEWS: {digest.critical_count} critical, {digest.high_count} high, score={digest.overall_score}")

    return digest


def news_health_check() -> Tuple[bool, str]:
    """
    Check if news feeds are healthy.
    Returns (is_healthy, diagnostic_message).
    Call periodically — will return unhealthy if no successful fetch for NEWS_STALE_SECONDS.
    """
    now = time.time()
    last_ok = _health.get("last_success", 0)

    # Never fetched yet — not stale, just starting
    if last_ok == 0:
        return True, "Новости: ещё не проверялись"

    stale_sec = now - last_ok
    if stale_sec < NEWS_STALE_SECONDS:
        return True, f"Новости: OK (обновлено {int(stale_sec)}с назад)"

    # Stale! Run diagnostics
    diag_lines = [f"⚠️ Новости не обновлялись {int(stale_sec)}с (лимит {NEWS_STALE_SECONDS}с)"]
    diag_lines.append("Диагностика фидов:")

    for source, url in FEEDS.items():
        try:
            items = _fetch_rss(url, timeout=5)
            if items:
                diag_lines.append(f"  ✅ {source}: OK ({len(items)} новостей)")
                _health["last_success"] = now  # Fixed!
            else:
                diag_lines.append(f"  ❌ {source}: пустой ответ")
        except Exception as e:
            diag_lines.append(f"  ❌ {source}: {str(e)[:60]}")

    # Check internet
    try:
        urllib.request.urlopen("https://ya.ru", timeout=5)
        diag_lines.append("  🌐 Интернет: OK")
    except Exception:
        diag_lines.append("  🌐 Интернет: НЕДОСТУПЕН")

    return False, "\n".join(diag_lines)


def _format_age_label(published: Optional[datetime]) -> str:
    if not published:
        return ""
    try:
        dt = published.astimezone(MSK) if published.tzinfo else published.replace(tzinfo=MSK)
        age_min = max(0, int((datetime.now(MSK) - dt).total_seconds() // 60))
        return f"⏱ {age_min} мин назад"
    except Exception:
        return ""


def _format_item_info(item: NewsItem) -> str:
    kw = item.keywords_hit[0] if item.keywords_hit else ""
    age = _format_age_label(item.published)
    info = f"[{item.source}] {item.title}"
    if kw:
        info = f"«{kw}» — {info}"
    if age:
        info = f"{age} • {info}"
    return info


def check_news_impact() -> Tuple[float, Optional[str]]:
    """
    Generic market-wide impact.
    Kept for compatibility: returns the most impactful item across all news.
    """
    digest = fetch_news(max_age_minutes=30)
    if not digest.items:
        return 1.0, None
    top = digest.items[0]
    return digest.overall_score, _format_item_info(top)


def get_systemic_news_impact() -> Tuple[float, Optional[str]]:
    """
    Only system-wide CRITICAL events should stop all trading.
    HIGH oil/gas news should not globally reduce every strategy.
    """
    digest = fetch_news(max_age_minutes=30)
    critical = [item for item in digest.items if item.impact == "CRITICAL"]
    if critical:
        return 0.0, _format_item_info(critical[0])
    return 1.0, None


def get_stock_news_impact() -> Tuple[float, Optional[str]]:
    """
    Stock agent should react mostly to Russia/ruble/MOEX news, not pure oil/gas noise.
    Uses cleaned contract sentiment for SI / RI / MX instead of raw digest items.
    """
    digest = fetch_news(max_age_minutes=30)
    critical = [item for item in digest.items if item.impact == "CRITICAL"]
    if critical:
        return 0.0, _format_item_info(critical[0])

    sentiment = get_futures_sentiment()
    candidates = []
    impact_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "NONE": 0}
    for code in ("SI", "RI", "MX"):
        data = sentiment.get(code, {})
        if not data or data.get("score", 1.0) >= 1.0:
            continue
        if not data.get("alerts"):
            continue
        candidates.append((
            impact_rank.get(data.get("impact", "NONE"), 0),
            float(data.get("confidence", 0.0) or 0.0),
            -(int(data.get("latest_age_minutes")) if data.get("latest_age_minutes") is not None else 10**9),
            code,
            data,
        ))

    if not candidates:
        return 1.0, None

    _, _, _, code, top = max(candidates)
    score = min(float(item[4].get("score", 1.0) or 1.0) for item in candidates)
    age = top.get("latest_age_minutes")
    age_label = f"⏱ {age} мин назад • " if age is not None else ""
    action = top.get("action") or "Нейтрально"
    headline = (top.get("alerts") or [""])[0][:220]
    return score, f"{age_label}{code} • {action} — {headline}"


def format_news_digest(digest: NewsDigest) -> str:
    """Format digest for Telegram message."""
    if not digest.items:
        return "📰 Нет значимых новостей за последние 30 мин"

    lines = [f"📰 <b>НОВОСТНОЙ ДАЙДЖЕСТ</b> ({len(digest.items)} событий)"]
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    for item in digest.items[:10]:
        if item.impact == "CRITICAL":
            emoji = "🔴"
        elif item.impact == "HIGH":
            emoji = "🟡"
        else:
            emoji = "🔵"

        fut_tag = f" [{','.join(item.futures_affected)}]" if item.futures_affected else ""
        age = _format_age_label(item.published)
        age_prefix = f"{age} • " if age else ""
        lines.append(f"{emoji} {age_prefix}[{item.source}]{fut_tag}\n   {item.title[:100]}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    score_emoji = "🟢" if digest.overall_score >= 1.0 else "🟡" if digest.overall_score >= 0.5 else "🔴"
    lines.append(f"{score_emoji} Множитель: ×{digest.overall_score}")

    if digest.futures_alerts:
        lines.append("\n⚡ <b>Затронутые фьючерсы:</b>")
        for fut, alert in digest.futures_alerts.items():
            lines.append(f"  {fut}: {alert[:60]}")

    return "\n".join(lines)


def _infer_contract_bias(text: str, contract: str) -> str:
    text_lower = (text or "").lower()

    commodity_long = [
        "ормуз", "hormuz", "strait of hormuz", "танкер", "tanker", "атака", "attack",
        "удар", "strike", "война", "war", "эскалац", "напряжен", "reject", "отверг",
        "сокращение добычи", "opec сократ", "эмбарго", "embargo",
    ]
    commodity_short = [
        "peace plan", "мирный план", "ceasefire", "перемири", "нефть упала", "oil crash",
        "oil plunge", "price cap", "потолок цен", "добычу увелич", "opec увелич",
        # De-escalation → oil down
        "завершится", "завершение", "мирное урегулирование", "peace deal",
        "прекращение огня", "отвод войск", "withdrawal", "de-escalation", "деэскалац",
        "переговоры успешны", "договорились", "соглашение достигнуто", "deal reached",
    ]
    russia_equity_short = [
        "санкции против россии", "new russia sanctions", "дефолт россии", "обвал рубля",
        "цб повысил", "ключевая ставка", "мосбиржа приостановила", "биржа приостановила",
    ]

    if contract in ("BR", "NG"):
        if any(kw in text_lower for kw in commodity_long):
            return "LONG"
        if any(kw in text_lower for kw in commodity_short):
            return "SHORT"
    if contract in ("RI", "MX"):
        if any(kw in text_lower for kw in russia_equity_short):
            return "SHORT"
    if contract == "SI":
        if any(kw in text_lower for kw in ("обвал рубля", "девальвация рубля", "sanctions against russia")):
            return "LONG"
    return "NEUTRAL"


def _action_hint(contract: str, preferred_direction: str, impact: str) -> str:
    if preferred_direction == "LONG":
        if contract in ("BR", "NG"):
            return "Приоритет LONG ↑, SHORT осторожнее"
        return "LONG bias"
    if preferred_direction == "SHORT":
        return "Приоритет SHORT ↑, LONG осторожнее"
    if impact == "CRITICAL":
        return "Критичный фон — осторожность"
    if impact == "HIGH":
        return "Повышенное внимание"
    return "Нейтрально"


def get_futures_sentiment() -> Dict[str, Dict]:
    """
    Get news sentiment per futures contract.
    Returns per-contract score, bias and UI-ready action hints.
    """
    digest = fetch_news(max_age_minutes=60)
    sentiment = {}

    impact_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "NONE": 0}

    for contract in FUTURES_KEYWORDS:
        related_raw = [item for item in digest.items if contract in item.futures_affected]
        related_filtered = [
            item for item in related_raw
            if not (item.source.startswith("TG ") and _is_digest_like_title(item.title))
            and _telegram_contract_lead_match(item, contract)
            and _contract_signal_relevant(item, contract)
        ]
        related_non_tg = [
            item for item in related_raw
            if not item.source.startswith("TG ") and _contract_signal_relevant(item, contract)
        ]
        related = related_filtered or related_non_tg
        clusters = _cluster_news_events(related, contract=contract)
        representatives = [cluster["rep"] for cluster in clusters]

        if representatives:
            top_item = max(
                representatives,
                key=lambda item: (
                    impact_rank.get(item.impact, 0),
                    item.confidence,
                    item.published or item.ingested_at or datetime.min.replace(tzinfo=MSK),
                ),
            )
            min_score = min(min(member.score for member in cluster["items"]) for cluster in clusters)

            # Compute latest_published_dt for freshness
            published = [item.published for item in representatives if item.published]
            latest_published_dt = max(published) if published else None

            # Freshness decay: old news loses impact
            # 0-15 min = full weight, 15-30 min = 80%, 30-60 min = 60%
            if latest_published_dt:
                dt = latest_published_dt.astimezone(MSK) if latest_published_dt.tzinfo else latest_published_dt.replace(tzinfo=MSK)
                age_min = max(0, (datetime.now(MSK) - dt).total_seconds() / 60)
                if age_min > 30:
                    freshness = 0.6
                elif age_min > 15:
                    freshness = 0.8
                else:
                    freshness = 1.0
                # Blend: if score was 0.5 (HIGH), with freshness 0.6 → becomes 0.7 (softer impact)
                min_score = min_score + (1.0 - min_score) * (1.0 - freshness)
            top_impact = top_item.impact
            alerts = [item.title[:80] for item in representatives[:3]]
            event_ids = [_event_signature(item, contract) for item in representatives[:3]]
            latest_published = latest_published_dt.isoformat() if latest_published_dt else None
            latest_age_minutes = None
            if latest_published_dt:
                dt = latest_published_dt.astimezone(MSK) if latest_published_dt.tzinfo else latest_published_dt.replace(tzinfo=MSK)
                latest_age_minutes = max(0, int((datetime.now(MSK) - dt).total_seconds() // 60))

            long_votes = 0
            short_votes = 0
            for item in representatives:
                bias = _infer_contract_bias(item.title, contract)
                if bias == "LONG":
                    long_votes += 1
                elif bias == "SHORT":
                    short_votes += 1
            if long_votes > short_votes:
                preferred_direction = "LONG"
            elif short_votes > long_votes:
                preferred_direction = "SHORT"
            else:
                preferred_direction = "NEUTRAL"

            confidence = max((item.confidence for item in representatives), default=0.0)
            confirmation_count = max((len(cluster["items"]) - 1 for cluster in clusters), default=0)
            sentiment[contract] = {
                "score": min_score,
                "impact": top_impact,
                "alerts": alerts,
                "event_ids": event_ids,
                "count": len(representatives),
                "raw_count": len(related_raw),
                "latest_published": latest_published,
                "latest_age_minutes": latest_age_minutes,
                "preferred_direction": preferred_direction,
                "action": _action_hint(contract, preferred_direction, top_impact),
                "confidence": confidence,
                "confirmation_count": confirmation_count,
            }
        else:
            sentiment[contract] = {
                "score": 1.0,
                "impact": "NONE",
                "alerts": [],
                "event_ids": [],
                "count": 0,
                "raw_count": 0,
                "latest_published": None,
                "latest_age_minutes": None,
                "preferred_direction": "NEUTRAL",
                "action": "Нейтрально",
                "confidence": 0.0,
                "confirmation_count": 0,
            }

    return sentiment


# ── CLI Test ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching news...")
    digest = fetch_news(max_age_minutes=60, force=True)
    print(format_news_digest(digest))
    print()

    print("Futures sentiment:")
    sentiment = get_futures_sentiment()
    for contract, data in sentiment.items():
        if data["count"]:
            print(f"  {contract}: score={data['score']}, impact={data['impact']}, {data['count']} news")
            for alert in data["alerts"]:
                print(f"    → {alert}")
