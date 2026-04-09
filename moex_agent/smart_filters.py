"""
MOEX Agent v2.5 Smart Filters

Evidence-based filters derived from Phase 3 backtest analysis.

Key findings:
1. Morning session (07:00-10:00) has worst WR across tickers → SKIP
2. Lunch/midday/opening_drive have best WR → PREFER
3. Risk sentiment affects different tickers differently
4. Tax period/expiry effects are ticker-specific

Usage:
    from moex_agent.smart_filters import SmartFilter

    filt = SmartFilter()
    allow, reason = filt.should_trade(
        ticker="SBER",
        direction="LONG",
        regime_state=regime_state,
        calendar_state=calendar_state,
        risk_sentiment=0.1,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .regime import RegimeState, TickerRegime
from .calendar_features import CalendarState

logger = logging.getLogger(__name__)


# Session quality scores (higher = better, based on backtest)
# Averaged across SBER and GAZP backtests
SESSION_QUALITY = {
    "morning_auction": 0.3,    # Thin, uncertain
    "morning_session": 0.3,    # Worst WR (25-27%)
    "opening_drive": 0.7,      # Good (50-66%)
    "morning_active": 0.5,     # Mixed (33-37%)
    "midday": 0.7,             # Good (39-100%)
    "lunch": 0.8,              # Best (53-75%)
    "clearing": 0.0,           # NO TRADING
    "afternoon": 0.6,          # Mixed (28-75%)
    "pre_close": 0.5,          # Mixed (32-50%)
    "evening_clearing": 0.0,   # NO TRADING
    "evening_session": 0.4,    # Thin liquidity
    "closed": 0.0,             # MARKET CLOSED
}

# Minimum session quality to trade
MIN_SESSION_QUALITY = 0.5


@dataclass
class FilterDecision:
    """Result of filter evaluation."""
    allow: bool
    reason: str
    confidence: float  # 0-1, how confident in the filter
    position_mult: float = 1.0  # Suggested position size multiplier


class SmartFilter:
    """
    Evidence-based trading filter combining all Phase 3 insights.
    """

    def __init__(
        self,
        skip_morning_session: bool = True,
        skip_clearing: bool = True,
        skip_low_quality_regimes: bool = True,
        min_regime_wr: float = 0.35,
        use_risk_sentiment: bool = True,
    ):
        self.skip_morning_session = skip_morning_session
        self.skip_clearing = skip_clearing
        self.skip_low_quality_regimes = skip_low_quality_regimes
        self.min_regime_wr = min_regime_wr
        self.use_risk_sentiment = use_risk_sentiment

        # Per-ticker regime win rates from backtest
        # Format: {ticker: {regime: win_rate}}
        self._regime_wr_cache: Dict[str, Dict[str, float]] = {}

    def should_trade(
        self,
        ticker: str,
        direction: str,
        regime_state: Optional[RegimeState] = None,
        calendar_state: Optional[CalendarState] = None,
        risk_sentiment: float = 0.0,
        regime_win_rates: Optional[Dict[str, float]] = None,
    ) -> FilterDecision:
        """
        Evaluate whether to take a trade.

        Args:
            ticker: Ticker symbol
            direction: "LONG" or "SHORT"
            regime_state: Current regime from RegimeDetector
            calendar_state: Current calendar state
            risk_sentiment: Global risk sentiment (-1 to 1)
            regime_win_rates: Historical regime win rates for this ticker

        Returns:
            FilterDecision with allow/reason/confidence
        """
        # Store regime WR if provided
        if regime_win_rates:
            self._regime_wr_cache[ticker] = regime_win_rates

        # Default: allow with full position
        decision = FilterDecision(
            allow=True,
            reason="passed",
            confidence=1.0,
            position_mult=1.0,
        )

        # === Session Filter ===
        if calendar_state:
            session = calendar_state.session_phase
            session_quality = SESSION_QUALITY.get(session, 0.5)

            # Hard blocks
            if session in ("clearing", "closed", "evening_clearing"):
                return FilterDecision(
                    allow=False,
                    reason=f"market closed ({session})",
                    confidence=1.0,
                )

            # Skip morning session
            if self.skip_morning_session and session == "morning_session":
                return FilterDecision(
                    allow=False,
                    reason="morning_session has poor WR",
                    confidence=0.8,
                )

            # Reduce position in low-quality sessions
            if session_quality < MIN_SESSION_QUALITY:
                decision.position_mult *= 0.5
                decision.reason = f"low quality session ({session})"
                decision.confidence = 0.7

            # Boost position in high-quality sessions
            if session_quality >= 0.7:
                decision.position_mult *= 1.2
                decision.confidence = 0.9

        # === Regime Filter ===
        if regime_state and self.skip_low_quality_regimes:
            regime_name = regime_state.regime.value

            # Check historical WR
            regime_wr = self._regime_wr_cache.get(ticker, {}).get(regime_name)
            if regime_wr is not None and regime_wr < self.min_regime_wr:
                return FilterDecision(
                    allow=False,
                    reason=f"{regime_name} has WR={regime_wr:.1%} < {self.min_regime_wr:.1%}",
                    confidence=0.7,
                )

            # Range regimes need stronger confirmation
            if regime_state.regime in (TickerRegime.RANGE_LOW_VOL, TickerRegime.RANGE_HIGH_VOL):
                if regime_state.confidence < 0.6:
                    decision.position_mult *= 0.7
                    decision.reason = "uncertain range regime"

            # Counter-trend caution
            if direction == "LONG" and regime_state.regime == TickerRegime.TREND_DOWN:
                decision.position_mult *= 0.5
                decision.reason = "counter-trend LONG in downtrend"
            elif direction == "SHORT" and regime_state.regime == TickerRegime.TREND_UP:
                decision.position_mult *= 0.5
                decision.reason = "counter-trend SHORT in uptrend"

        # === Risk Sentiment Filter ===
        if self.use_risk_sentiment and abs(risk_sentiment) > 0.3:
            # Strong risk-off
            if risk_sentiment < -0.3:
                if ticker in ("SBER", "VTBR", "TCSG"):  # Banks
                    decision.position_mult *= 0.7
                    decision.reason = "risk-off environment for banks"
                elif direction == "LONG":
                    decision.position_mult *= 0.8

            # Strong risk-on
            elif risk_sentiment > 0.3:
                if direction == "LONG":
                    decision.position_mult *= 1.1
                    decision.confidence = 0.85

        # === Calendar Event Filter ===
        if calendar_state:
            # High-risk events
            if calendar_state.event_risk_multiplier > 1.5:
                decision.position_mult *= 0.5
                decision.reason = "high-risk calendar event"
                decision.confidence = 0.8

            # Tax period (ticker-specific)
            if calendar_state.is_tax_period:
                if ticker in ("GAZP", "LKOH", "ROSN"):  # Export taxes
                    decision.position_mult *= 0.8
                    decision.reason = "tax period caution for exporters"

        # Final position size bounds
        decision.position_mult = max(0.3, min(1.5, decision.position_mult))

        return decision

    def get_optimal_session_windows(self) -> Dict[str, Tuple[str, str]]:
        """
        Get optimal trading windows based on session quality.

        Returns:
            Dict mapping quality tier to (start_session, end_session)
        """
        return {
            "best": ("lunch", "midday"),       # 11:30-14:00, 52-100% WR
            "good": ("opening_drive", "afternoon"),  # 10:00-16:00
            "avoid": ("morning_session", "morning_session"),  # 07:00-10:00
        }


# Singleton
_smart_filter: Optional[SmartFilter] = None


def get_smart_filter() -> SmartFilter:
    """Get or create global SmartFilter instance."""
    global _smart_filter
    if _smart_filter is None:
        _smart_filter = SmartFilter()
    return _smart_filter
