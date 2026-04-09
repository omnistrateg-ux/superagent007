"""
MOEX Agent v2.5 Sanctions Risk Monitor

Monitors sanctions-related risks for Russian equities trading.

Risks monitored:
1. Company-specific sanctions (SDN list)
2. Sector sanctions (energy, banking, defense)
3. News-based sanctions alerts
4. Currency volatility spikes (USD/RUB)
5. Market-wide stress indicators

Actions:
- Reduce position in sanctioned companies
- Skip trading in high-risk sectors during alerts
- Emergency position closure on breaking news
- Alert via Telegram

Usage:
    from moex_agent.sanctions_monitor import SanctionsMonitor

    monitor = SanctionsMonitor()
    risk = monitor.assess_ticker_risk("SBER")
    if risk.level == RiskLevel.HIGH:
        skip_trading()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Sanctions risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SectorType(str, Enum):
    """Sector classification for sanctions."""
    BANKING = "banking"
    ENERGY = "energy"
    DEFENSE = "defense"
    METALS = "metals"
    TECH = "tech"
    CONSUMER = "consumer"
    TELECOM = "telecom"
    OTHER = "other"


# Sector mapping for tickers
TICKER_SECTORS = {
    # Banking (high sanctions risk)
    "SBER": SectorType.BANKING,
    "VTBR": SectorType.BANKING,
    "TCSG": SectorType.BANKING,
    "CBOM": SectorType.BANKING,

    # Energy (high sanctions risk)
    "GAZP": SectorType.ENERGY,
    "LKOH": SectorType.ENERGY,
    "ROSN": SectorType.ENERGY,
    "NVTK": SectorType.ENERGY,
    "TATN": SectorType.ENERGY,
    "SNGS": SectorType.ENERGY,
    "SIBN": SectorType.ENERGY,

    # Metals
    "GMKN": SectorType.METALS,
    "NLMK": SectorType.METALS,
    "CHMF": SectorType.METALS,
    "MAGN": SectorType.METALS,
    "PLZL": SectorType.METALS,
    "ALRS": SectorType.METALS,
    "RUAL": SectorType.METALS,

    # Tech
    "YNDX": SectorType.TECH,
    "OZON": SectorType.TECH,
    "VKCO": SectorType.TECH,

    # Consumer
    "MGNT": SectorType.CONSUMER,
    "FIVE": SectorType.CONSUMER,
    "X5": SectorType.CONSUMER,

    # Telecom
    "MTSS": SectorType.TELECOM,
    "RTKM": SectorType.TELECOM,
}

# Base sanctions risk by sector
SECTOR_BASE_RISK = {
    SectorType.BANKING: 0.7,   # High risk
    SectorType.ENERGY: 0.6,    # High risk
    SectorType.DEFENSE: 0.9,   # Very high risk
    SectorType.METALS: 0.4,    # Medium risk
    SectorType.TECH: 0.3,      # Medium risk
    SectorType.CONSUMER: 0.1,  # Low risk
    SectorType.TELECOM: 0.2,   # Low risk
    SectorType.OTHER: 0.2,     # Low risk
}

# Tickers with specific sanctions (SDN list, etc.)
SANCTIONED_TICKERS: Set[str] = {
    "VTBR",   # VTB Bank - SDN
    # Add more as needed
}

# Tickers with elevated risk (not sanctioned but high exposure)
ELEVATED_RISK_TICKERS: Set[str] = {
    "SBER",   # Sberbank - partial restrictions
    "GAZP",   # Gazprom - gas sanctions
    "ROSN",   # Rosneft - oil sanctions
}


@dataclass
class SanctionsRiskAssessment:
    """Assessment of sanctions risk for a ticker."""
    ticker: str
    level: RiskLevel
    score: float  # 0-1
    sector: SectorType
    reasons: List[str] = field(default_factory=list)
    position_mult: float = 1.0  # Suggested position multiplier
    should_close: bool = False   # Emergency close recommendation
    alert_message: Optional[str] = None


@dataclass
class MarketStressIndicators:
    """Market-wide stress indicators."""
    usdrub_change_pct: float = 0.0
    usdrub_level: float = 0.0
    imoex_change_pct: float = 0.0
    vix_level: float = 20.0
    news_alert_active: bool = False
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def stress_level(self) -> float:
        """Calculate overall stress level (0-1)."""
        stress = 0.0

        # USD/RUB spike (>2% = stress)
        if abs(self.usdrub_change_pct) > 2:
            stress += 0.3
        elif abs(self.usdrub_change_pct) > 1:
            stress += 0.1

        # IMOEX crash (>3% = stress)
        if self.imoex_change_pct < -3:
            stress += 0.3
        elif self.imoex_change_pct < -2:
            stress += 0.1

        # VIX high
        if self.vix_level > 30:
            stress += 0.2
        elif self.vix_level > 25:
            stress += 0.1

        # News alert
        if self.news_alert_active:
            stress += 0.3

        return min(stress, 1.0)


class SanctionsMonitor:
    """
    Monitor and assess sanctions-related risks.
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
    ):
        self.state_path = state_path or Path("data/sanctions_state.json")
        self.stress_indicators = MarketStressIndicators()
        self.active_alerts: List[str] = []
        self.alert_history: List[Dict] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load saved state."""
        if not self.state_path.exists():
            return

        try:
            data = json.loads(self.state_path.read_text())
            self.active_alerts = data.get("active_alerts", [])
        except Exception as e:
            logger.warning(f"Failed to load sanctions state: {e}")

    def _save_state(self) -> None:
        """Save state."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_alerts": self.active_alerts,
                "stress_level": self.stress_indicators.stress_level,
                "updated_at": datetime.now().isoformat(),
            }
            self.state_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save sanctions state: {e}")

    def update_market_stress(
        self,
        usdrub_change_pct: Optional[float] = None,
        usdrub_level: Optional[float] = None,
        imoex_change_pct: Optional[float] = None,
        vix_level: Optional[float] = None,
    ) -> MarketStressIndicators:
        """Update market stress indicators."""
        if usdrub_change_pct is not None:
            self.stress_indicators.usdrub_change_pct = usdrub_change_pct
        if usdrub_level is not None:
            self.stress_indicators.usdrub_level = usdrub_level
        if imoex_change_pct is not None:
            self.stress_indicators.imoex_change_pct = imoex_change_pct
        if vix_level is not None:
            self.stress_indicators.vix_level = vix_level

        self.stress_indicators.last_update = datetime.now()
        self._save_state()

        return self.stress_indicators

    def add_news_alert(self, message: str, duration_hours: int = 24) -> None:
        """Add a news-based sanctions alert."""
        alert = {
            "message": message,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
        }

        self.active_alerts.append(message)
        self.alert_history.append(alert)
        self.stress_indicators.news_alert_active = True

        logger.warning(f"Sanctions alert added: {message}")
        self._save_state()

    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear()
        self.stress_indicators.news_alert_active = False
        self._save_state()

    def assess_ticker_risk(
        self,
        ticker: str,
        include_market_stress: bool = True,
    ) -> SanctionsRiskAssessment:
        """
        Assess sanctions risk for a ticker.

        Args:
            ticker: Ticker symbol
            include_market_stress: Include market-wide stress in assessment

        Returns:
            SanctionsRiskAssessment with risk level and recommendations
        """
        reasons = []
        score = 0.0

        # Get sector
        sector = TICKER_SECTORS.get(ticker, SectorType.OTHER)

        # Base sector risk
        base_risk = SECTOR_BASE_RISK.get(sector, 0.2)
        score += base_risk
        if base_risk >= 0.5:
            reasons.append(f"high-risk sector ({sector.value})")

        # Check if directly sanctioned
        if ticker in SANCTIONED_TICKERS:
            score += 0.5
            reasons.append("ticker on sanctions list")

        # Check elevated risk
        if ticker in ELEVATED_RISK_TICKERS:
            score += 0.2
            reasons.append("elevated sanctions exposure")

        # Include market stress
        if include_market_stress:
            stress = self.stress_indicators.stress_level
            score += stress * 0.3
            if stress > 0.5:
                reasons.append(f"market stress ({stress:.1%})")

        # Check active alerts
        if self.active_alerts:
            score += 0.2
            reasons.append(f"{len(self.active_alerts)} active alert(s)")

        # Normalize score
        score = min(score, 1.0)

        # Determine risk level
        if score >= 0.8:
            level = RiskLevel.CRITICAL
            position_mult = 0.0
            should_close = True
        elif score >= 0.6:
            level = RiskLevel.HIGH
            position_mult = 0.25
            should_close = False
        elif score >= 0.4:
            level = RiskLevel.MEDIUM
            position_mult = 0.5
            should_close = False
        else:
            level = RiskLevel.LOW
            position_mult = 1.0
            should_close = False

        # Build alert message
        alert_message = None
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            alert_message = f"Sanctions risk {level.value}: {ticker} ({', '.join(reasons)})"

        return SanctionsRiskAssessment(
            ticker=ticker,
            level=level,
            score=score,
            sector=sector,
            reasons=reasons,
            position_mult=position_mult,
            should_close=should_close,
            alert_message=alert_message,
        )

    def assess_portfolio_risk(
        self,
        positions: List[str],
    ) -> Dict:
        """
        Assess sanctions risk for entire portfolio.

        Args:
            positions: List of ticker symbols

        Returns:
            Portfolio risk summary
        """
        assessments = [self.assess_ticker_risk(t) for t in positions]

        critical = [a for a in assessments if a.level == RiskLevel.CRITICAL]
        high = [a for a in assessments if a.level == RiskLevel.HIGH]
        medium = [a for a in assessments if a.level == RiskLevel.MEDIUM]

        avg_score = sum(a.score for a in assessments) / len(assessments) if assessments else 0

        return {
            "total_positions": len(positions),
            "critical_risk": len(critical),
            "high_risk": len(high),
            "medium_risk": len(medium),
            "average_score": avg_score,
            "should_close": [a.ticker for a in critical],
            "reduce_exposure": [a.ticker for a in high],
            "market_stress": self.stress_indicators.stress_level,
            "active_alerts": len(self.active_alerts),
        }

    def get_trading_sectors(
        self,
        max_risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> List[SectorType]:
        """
        Get sectors safe for trading.

        Args:
            max_risk_level: Maximum acceptable risk level

        Returns:
            List of sectors to trade
        """
        threshold = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 1.0,
        }[max_risk_level]

        safe_sectors = []
        for sector, base_risk in SECTOR_BASE_RISK.items():
            total_risk = base_risk + self.stress_indicators.stress_level * 0.3
            if total_risk <= threshold:
                safe_sectors.append(sector)

        return safe_sectors

    def get_status(self) -> Dict:
        """Get monitor status."""
        return {
            "market_stress": {
                "level": self.stress_indicators.stress_level,
                "usdrub_change": f"{self.stress_indicators.usdrub_change_pct:+.2f}%",
                "imoex_change": f"{self.stress_indicators.imoex_change_pct:+.2f}%",
                "vix": self.stress_indicators.vix_level,
            },
            "active_alerts": self.active_alerts,
            "safe_sectors": [s.value for s in self.get_trading_sectors()],
            "high_risk_tickers": list(SANCTIONED_TICKERS | ELEVATED_RISK_TICKERS),
        }


# Singleton
_monitor: Optional[SanctionsMonitor] = None


def get_sanctions_monitor() -> SanctionsMonitor:
    """Get or create global SanctionsMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = SanctionsMonitor()
    return _monitor
