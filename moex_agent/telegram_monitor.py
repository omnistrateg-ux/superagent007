"""
MOEX Agent v2 Telegram Monitor

Monitoring Telegram channels for market-moving news:
- 10+ financial channels tracking
- 30+ keyword patterns with scoring
- 4 alert levels: INFO, WARNING, DANGER, CRITICAL
- Real-time sentiment aggregation
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("moex_agent.telegram_monitor")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = 1        # Normal market news
    WARNING = 2     # Elevated caution
    DANGER = 3      # Significant risk
    CRITICAL = 4    # Immediate action required


# Monitored Telegram channels (public financial channels)
MONITORED_CHANNELS = [
    "@markettwits",          # Market Twits - breaking news
    "@cbikirov",             # CB economics
    "@raborussia",           # Labor Russia economics
    "@economika",            # Economy news
    "@banksta",              # Banking news
    "@russianmacro",         # Russian macro
    "@faborussia",           # Finance analytics
    "@moaborussia",          # MOEX analytics
    "@investfuture",         # Investment future
    "@smartlabnews",         # SmartLab news
]


# Keyword patterns with scores and alert levels
# Format: (pattern, score, alert_level, category)
KEYWORD_PATTERNS: List[Tuple[str, int, AlertLevel, str]] = [
    # CRITICAL - Immediate trading stop
    (r"торги\s+(приостановлены|остановлены|прекращены)", 100, AlertLevel.CRITICAL, "trading_halt"),
    (r"технический\s+сбой\s+(на\s+)?бирж", 100, AlertLevel.CRITICAL, "technical"),
    (r"дефолт", 95, AlertLevel.CRITICAL, "default"),
    (r"банкротство", 90, AlertLevel.CRITICAL, "bankruptcy"),
    (r"военное\s+положение", 100, AlertLevel.CRITICAL, "geopolitics"),
    (r"мобилизация", 90, AlertLevel.CRITICAL, "geopolitics"),
    (r"ядерн", 100, AlertLevel.CRITICAL, "geopolitics"),

    # DANGER - High risk events
    (r"санкции\s+(против|в\s+отношении)", 80, AlertLevel.DANGER, "sanctions"),
    (r"новые\s+санкции", 85, AlertLevel.DANGER, "sanctions"),
    (r"цб\s+(повысил|повышает|поднял)\s+ставку", 75, AlertLevel.DANGER, "rates"),
    (r"ключевая\s+ставка\s+повышена", 75, AlertLevel.DANGER, "rates"),
    (r"экстренное\s+заседание", 80, AlertLevel.DANGER, "emergency"),
    (r"margin\s*call", 70, AlertLevel.DANGER, "margin"),
    (r"принудительное\s+закрытие", 70, AlertLevel.DANGER, "margin"),
    (r"обвал\s+(рубля|рынка|акций)", 75, AlertLevel.DANGER, "crash"),
    (r"паника\s+на\s+(рынке|бирже)", 80, AlertLevel.DANGER, "panic"),
    (r"массовые\s+продажи", 70, AlertLevel.DANGER, "selloff"),
    (r"отток\s+капитала", 65, AlertLevel.DANGER, "capital_flight"),

    # WARNING - Elevated caution
    (r"волатильность\s+(выросла|растет|высокая)", 50, AlertLevel.WARNING, "volatility"),
    (r"резкое\s+(падение|снижение|рост)", 55, AlertLevel.WARNING, "price_move"),
    (r"фрс\s+(повысила?|снизила?)", 60, AlertLevel.WARNING, "fed"),
    (r"fed\s+(raised?|cut)", 60, AlertLevel.WARNING, "fed"),
    (r"инфляция\s+(выросла|ускорилась)", 50, AlertLevel.WARNING, "inflation"),
    (r"опек\s*\+?\s*(сократил|увеличил)", 55, AlertLevel.WARNING, "opec"),
    (r"девальвация", 60, AlertLevel.WARNING, "devaluation"),
    (r"курс\s+доллара\s+(вырос|упал)", 45, AlertLevel.WARNING, "fx"),
    (r"нефть\s+(упала|обвалилась|взлетела)", 55, AlertLevel.WARNING, "oil"),
    (r"газ\s+(подорожал|подешевел)", 50, AlertLevel.WARNING, "gas"),
    (r"дивиденды\s+(отменены|снижены)", 55, AlertLevel.WARNING, "dividends"),
    (r"buyback\s+(отменен|приостановлен)", 50, AlertLevel.WARNING, "buyback"),

    # INFO - Normal market events
    (r"отчетность\s+(лучше|хуже)\s+ожиданий", 30, AlertLevel.INFO, "earnings"),
    (r"прибыль\s+(выросла|снизилась)", 25, AlertLevel.INFO, "earnings"),
    (r"выручка\s+(выросла|снизилась)", 25, AlertLevel.INFO, "revenue"),
    (r"ipo\s+состоялось", 20, AlertLevel.INFO, "ipo"),
    (r"сделка\s+(закрыта|одобрена)", 25, AlertLevel.INFO, "deal"),
    (r"рекомендация\s+(покупать|продавать)", 20, AlertLevel.INFO, "recommendation"),
    (r"целевая\s+цена", 15, AlertLevel.INFO, "target"),
    (r"консенсус-прогноз", 15, AlertLevel.INFO, "consensus"),
]

# Ticker mentions pattern
TICKER_PATTERN = re.compile(
    r'\b(SBER|GAZP|LKOH|ROSN|GMKN|VTBR|NVTK|TATN|MGNT|ALRS|'
    r'CHMF|NLMK|MTSS|IRAO|MOEX|PHOR|RUAL|POLY|MAGN|AFKS|'
    r'PIKK|HYDR|FEES|RTKM|AFLT|TCSG|OZON|HHRU|FIVE|FIXP|'
    r'SMLT|SGZH|VKCO|POSI|YDEX|BELU|CBOM|MTLR)\b',
    re.IGNORECASE
)


@dataclass
class TelegramMessage:
    """Parsed Telegram message."""
    channel: str
    text: str
    timestamp: datetime
    message_id: Optional[int] = None


@dataclass
class AlertMatch:
    """Single pattern match in a message."""
    pattern: str
    score: int
    level: AlertLevel
    category: str
    matched_text: str


@dataclass
class MessageAlert:
    """Analyzed message with alerts."""
    message: TelegramMessage
    matches: List[AlertMatch] = field(default_factory=list)
    tickers_mentioned: Set[str] = field(default_factory=set)
    total_score: int = 0
    max_level: AlertLevel = AlertLevel.INFO

    def __repr__(self) -> str:
        return (
            f"MessageAlert(score={self.total_score}, level={self.max_level.name}, "
            f"tickers={self.tickers_mentioned}, matches={len(self.matches)})"
        )


@dataclass
class MarketSentiment:
    """Aggregated market sentiment from multiple messages."""
    timestamp: datetime
    messages_analyzed: int
    total_score: int
    avg_score: float
    max_level: AlertLevel
    alerts_by_level: Dict[AlertLevel, int] = field(default_factory=dict)
    top_categories: Dict[str, int] = field(default_factory=dict)
    ticker_mentions: Dict[str, int] = field(default_factory=dict)
    should_reduce_exposure: bool = False
    should_stop_trading: bool = False

    def __repr__(self) -> str:
        return (
            f"MarketSentiment(score={self.total_score}, level={self.max_level.name}, "
            f"reduce={self.should_reduce_exposure}, stop={self.should_stop_trading})"
        )


class TelegramMonitor:
    """
    Telegram channel monitor for market signals.

    Note: This is a framework for Telegram monitoring.
    Actual message fetching requires Telethon/Pyrogram with API credentials.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        lookback_minutes: int = 60,
    ):
        self.channels = channels or MONITORED_CHANNELS
        self.lookback_minutes = lookback_minutes
        self._message_cache: List[TelegramMessage] = []
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), score, level, cat)
            for p, score, level, cat in KEYWORD_PATTERNS
        ]

    def analyze_message(self, message: TelegramMessage) -> MessageAlert:
        """
        Analyze a single message for alerts.

        Args:
            message: Telegram message to analyze

        Returns:
            MessageAlert with all matches
        """
        alert = MessageAlert(message=message)
        text = message.text.lower()

        # Find all pattern matches
        for pattern, score, level, category in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                alert.matches.append(AlertMatch(
                    pattern=pattern.pattern,
                    score=score,
                    level=level,
                    category=category,
                    matched_text=match.group(0),
                ))
                alert.total_score += score
                if level.value > alert.max_level.value:
                    alert.max_level = level

        # Find ticker mentions
        ticker_matches = TICKER_PATTERN.findall(message.text)
        alert.tickers_mentioned = set(t.upper() for t in ticker_matches)

        return alert

    def analyze_messages(
        self,
        messages: List[TelegramMessage],
    ) -> List[MessageAlert]:
        """Analyze multiple messages."""
        return [self.analyze_message(msg) for msg in messages]

    def compute_sentiment(
        self,
        alerts: List[MessageAlert],
    ) -> MarketSentiment:
        """
        Compute aggregated market sentiment from alerts.

        Args:
            alerts: List of analyzed messages

        Returns:
            MarketSentiment aggregation
        """
        if not alerts:
            return MarketSentiment(
                timestamp=datetime.now(timezone.utc),
                messages_analyzed=0,
                total_score=0,
                avg_score=0.0,
                max_level=AlertLevel.INFO,
            )

        total_score = sum(a.total_score for a in alerts)
        max_level = max((a.max_level for a in alerts), key=lambda x: x.value)

        # Count alerts by level
        alerts_by_level: Dict[AlertLevel, int] = {}
        for level in AlertLevel:
            alerts_by_level[level] = sum(
                1 for a in alerts
                for m in a.matches
                if m.level == level
            )

        # Count categories
        categories: Dict[str, int] = {}
        for alert in alerts:
            for match in alert.matches:
                categories[match.category] = categories.get(match.category, 0) + 1

        # Count ticker mentions
        ticker_counts: Dict[str, int] = {}
        for alert in alerts:
            for ticker in alert.tickers_mentioned:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        # Determine action recommendations
        should_reduce = (
            total_score >= 100 or
            alerts_by_level.get(AlertLevel.DANGER, 0) >= 2 or
            alerts_by_level.get(AlertLevel.CRITICAL, 0) >= 1
        )

        should_stop = (
            total_score >= 200 or
            alerts_by_level.get(AlertLevel.CRITICAL, 0) >= 2 or
            max_level == AlertLevel.CRITICAL
        )

        return MarketSentiment(
            timestamp=datetime.now(timezone.utc),
            messages_analyzed=len(alerts),
            total_score=total_score,
            avg_score=total_score / len(alerts) if alerts else 0,
            max_level=max_level,
            alerts_by_level=alerts_by_level,
            top_categories=dict(sorted(categories.items(), key=lambda x: -x[1])[:5]),
            ticker_mentions=dict(sorted(ticker_counts.items(), key=lambda x: -x[1])[:10]),
            should_reduce_exposure=should_reduce,
            should_stop_trading=should_stop,
        )

    def get_ticker_alerts(
        self,
        alerts: List[MessageAlert],
        ticker: str,
    ) -> List[MessageAlert]:
        """Get alerts mentioning a specific ticker."""
        return [a for a in alerts if ticker.upper() in a.tickers_mentioned]

    def filter_by_level(
        self,
        alerts: List[MessageAlert],
        min_level: AlertLevel = AlertLevel.WARNING,
    ) -> List[MessageAlert]:
        """Filter alerts by minimum severity level."""
        return [a for a in alerts if a.max_level.value >= min_level.value]

    def filter_by_time(
        self,
        alerts: List[MessageAlert],
        since: datetime,
    ) -> List[MessageAlert]:
        """Filter alerts by timestamp."""
        return [a for a in alerts if a.message.timestamp >= since]


def create_test_messages() -> List[TelegramMessage]:
    """Create test messages for demonstration."""
    now = datetime.now(timezone.utc)
    return [
        TelegramMessage(
            channel="@markettwits",
            text="SBER отчетность лучше ожиданий, прибыль выросла на 15%",
            timestamp=now - timedelta(minutes=30),
        ),
        TelegramMessage(
            channel="@banksta",
            text="ЦБ повысил ставку на 100 б.п., VTBR и SBER под давлением",
            timestamp=now - timedelta(minutes=15),
        ),
        TelegramMessage(
            channel="@russianmacro",
            text="Волатильность выросла, курс доллара вырос на 2% за день",
            timestamp=now - timedelta(minutes=10),
        ),
        TelegramMessage(
            channel="@economika",
            text="Новые санкции против GAZP, акции падают на 5%",
            timestamp=now - timedelta(minutes=5),
        ),
    ]


def demo_analysis() -> MarketSentiment:
    """Run demo analysis on test messages."""
    monitor = TelegramMonitor()
    messages = create_test_messages()
    alerts = monitor.analyze_messages(messages)
    sentiment = monitor.compute_sentiment(alerts)

    logger.info(f"Analyzed {len(messages)} messages")
    logger.info(f"Sentiment: {sentiment}")

    for alert in alerts:
        if alert.matches:
            logger.info(f"  {alert.message.channel}: {alert}")

    return sentiment
