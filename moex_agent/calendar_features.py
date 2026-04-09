"""
MOEX Agent v2.5 Calendar Features

Phase 3: Calendar effects and macro events.

Features:
1. Intraday seasonality (session phases)
2. Day of week effects
3. Month-end rebalancing
4. Tax period (20-25 monthly) → RUB strengthens → RI pressure
5. Expiration weeks
6. Dividend dates
7. Macro events (CBR, OPEC, Fed)
8. Russian holidays

Usage:
    from moex_agent.calendar_features import CalendarFeatures

    cal = CalendarFeatures()
    features = cal.get_features(datetime.now(), ticker="RI")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    """MOEX session phases."""
    MORNING_AUCTION = "morning_auction"  # 06:50-07:00
    MORNING_SESSION = "morning_session"  # 07:00-10:00 (FORTS only, thin)
    OPENING_DRIVE = "opening_drive"      # 10:00-10:15
    MORNING_ACTIVE = "morning_active"    # 10:15-11:30
    MIDDAY = "midday"                    # 11:30-13:00
    LUNCH = "lunch"                      # 13:00-14:00 (low liquidity)
    CLEARING = "clearing"                # 14:00-14:05 (NO TRADING)
    AFTERNOON = "afternoon"              # 14:05-16:00
    PRE_CLOSE = "pre_close"              # 16:00-18:40
    CLOSING_AUCTION = "closing_auction"  # 18:40-18:50 (equities)
    EVENING_CLEARING = "evening_clearing"# 18:45-19:05 (FORTS)
    EVENING_SESSION = "evening_session"  # 19:05-23:50 (FORTS only)
    CLOSED = "closed"


@dataclass
class CalendarState:
    """Container for calendar features."""
    # Session
    session_phase: str = "closed"
    time_of_day_normalized: float = 0.5  # 0=open, 1=close
    is_evening: bool = False

    # Week
    day_of_week: int = 0  # 0=Mon, 4=Fri
    is_monday: bool = False
    is_friday: bool = False

    # Month
    is_month_start: bool = False  # First 3 days
    is_month_end: bool = False    # Last 3 days
    is_quarter_end: bool = False

    # Tax period
    is_tax_period: bool = False  # 20-25 monthly
    tax_period_day: int = 0      # 0 if not, 1-6 otherwise

    # Expiration
    is_expiry_week: bool = False
    is_expiry_day: bool = False
    days_to_expiry: int = 999

    # Dividends
    days_to_divcut: int = 999
    div_yield_expected: float = 0.0
    is_post_divcut: bool = False

    # Events
    is_cbr_day: bool = False     # ЦБ РФ rate decision
    is_opec_day: bool = False    # OPEC+ meeting
    is_fed_day: bool = False     # Fed decision
    is_nfp_day: bool = False     # NonFarm Payrolls
    is_holiday_adjacent: bool = False

    # Risk adjustments
    event_risk_multiplier: float = 1.0
    liquidity_expected: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cal_session_phase": hash(self.session_phase) % 100 / 100,  # Encode as numeric
            "cal_time_normalized": self.time_of_day_normalized,
            "cal_is_evening": float(self.is_evening),
            "cal_day_of_week": self.day_of_week / 4.0,  # Normalize
            "cal_is_monday": float(self.is_monday),
            "cal_is_friday": float(self.is_friday),
            "cal_is_month_start": float(self.is_month_start),
            "cal_is_month_end": float(self.is_month_end),
            "cal_is_quarter_end": float(self.is_quarter_end),
            "cal_is_tax_period": float(self.is_tax_period),
            "cal_tax_day": self.tax_period_day / 6.0,
            "cal_is_expiry_week": float(self.is_expiry_week),
            "cal_is_expiry_day": float(self.is_expiry_day),
            "cal_days_to_expiry": min(self.days_to_expiry, 30) / 30.0,
            "cal_days_to_divcut": min(self.days_to_divcut, 30) / 30.0,
            "cal_div_yield": self.div_yield_expected,
            "cal_is_post_divcut": float(self.is_post_divcut),
            "cal_is_cbr_day": float(self.is_cbr_day),
            "cal_is_opec_day": float(self.is_opec_day),
            "cal_is_fed_day": float(self.is_fed_day),
            "cal_event_risk_mult": self.event_risk_multiplier,
            "cal_liquidity_expected": self.liquidity_expected,
        }


class CalendarFeatures:
    """
    Calculate calendar-based features.
    """

    # CBR rate decision dates 2024-2025 (Fridays, ~8 per year)
    CBR_DATES_2024 = {
        date(2024, 2, 16), date(2024, 3, 22), date(2024, 4, 26),
        date(2024, 6, 7), date(2024, 7, 26), date(2024, 9, 13),
        date(2024, 10, 25), date(2024, 12, 20),
    }
    CBR_DATES_2025 = {
        date(2025, 2, 14), date(2025, 3, 21), date(2025, 4, 25),
        date(2025, 6, 6), date(2025, 7, 25), date(2025, 9, 12),
        date(2025, 10, 24), date(2025, 12, 19),
    }
    CBR_DATES_2026 = {
        date(2026, 2, 13), date(2026, 3, 20), date(2026, 4, 24),
        date(2026, 6, 5), date(2026, 7, 24), date(2026, 9, 11),
        date(2026, 10, 23), date(2026, 12, 18),
    }
    CBR_DATES = CBR_DATES_2024 | CBR_DATES_2025 | CBR_DATES_2026

    # FOMC dates (approximate, check yearly)
    FED_DATES_2024 = {
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
        date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
        date(2024, 11, 7), date(2024, 12, 18),
    }
    FED_DATES_2025 = {
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
        date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
        date(2025, 11, 5), date(2025, 12, 17),
    }
    FED_DATES_2026 = {
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
        date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
        date(2026, 11, 4), date(2026, 12, 16),
    }
    FED_DATES = FED_DATES_2024 | FED_DATES_2025 | FED_DATES_2026

    # Russian holidays (non-trading days)
    RU_HOLIDAYS_2024 = {
        date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
        date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8),
        date(2024, 2, 23), date(2024, 3, 8), date(2024, 5, 1),
        date(2024, 5, 9), date(2024, 6, 12), date(2024, 11, 4),
    }
    RU_HOLIDAYS_2025 = {
        date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3),
        date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8),
        date(2025, 2, 24), date(2025, 3, 10), date(2025, 5, 1),
        date(2025, 5, 2), date(2025, 5, 9), date(2025, 6, 12),
        date(2025, 6, 13), date(2025, 11, 4),
    }
    RU_HOLIDAYS = RU_HOLIDAYS_2024 | RU_HOLIDAYS_2025

    # Dividend calendar (example - should be updated dynamically)
    # Format: {ticker: [(divcut_date, div_amount), ...]}
    DIV_CALENDAR: Dict[str, List[Tuple[date, float]]] = {}

    # FORTS expiration dates (third Thursday of expiry month)
    # Quarterly: March, June, September, December
    EXPIRY_MONTHS = {3, 6, 9, 12}

    def __init__(self):
        pass

    def get_features(
        self,
        timestamp: datetime,
        ticker: Optional[str] = None,
        expiry_date: Optional[date] = None,
    ) -> CalendarState:
        """
        Calculate all calendar features for a timestamp.

        Args:
            timestamp: Current datetime
            ticker: Ticker for dividend info
            expiry_date: Futures expiration date

        Returns:
            CalendarState with all features
        """
        state = CalendarState()
        d = timestamp.date()

        # Session phase
        state.session_phase = self._get_session_phase(timestamp).value
        state.time_of_day_normalized = self._time_of_day_normalized(timestamp)
        state.is_evening = self._is_evening_session(timestamp)

        # Day of week
        state.day_of_week = d.weekday()
        state.is_monday = (state.day_of_week == 0)
        state.is_friday = (state.day_of_week == 4)

        # Month effects
        state.is_month_start = (d.day <= 3)
        state.is_month_end = (d.day >= 28) or (
            d.day >= 25 and (d + timedelta(days=7)).month != d.month
        )
        state.is_quarter_end = (d.month in {3, 6, 9, 12}) and state.is_month_end

        # Tax period
        state.is_tax_period = self._is_tax_period(d)
        state.tax_period_day = self._tax_period_day(d)

        # Expiration
        if expiry_date:
            state.days_to_expiry = (expiry_date - d).days
            state.is_expiry_day = (d == expiry_date)
            state.is_expiry_week = (0 <= state.days_to_expiry <= 5)
        else:
            # Estimate next expiry
            next_expiry = self._next_expiry_date(d)
            if next_expiry:
                state.days_to_expiry = (next_expiry - d).days
                state.is_expiry_day = (d == next_expiry)
                state.is_expiry_week = (0 <= state.days_to_expiry <= 5)

        # Dividends
        if ticker and ticker in self.DIV_CALENDAR:
            divcut, div_amount = self._next_dividend(ticker, d)
            if divcut:
                state.days_to_divcut = (divcut - d).days
                state.is_post_divcut = (d > divcut) and ((d - divcut).days <= 5)
                # Would need price to calculate yield
                state.div_yield_expected = 0.0  # Placeholder

        # Events
        state.is_cbr_day = (d in self.CBR_DATES) or (d - timedelta(days=1) in self.CBR_DATES)
        state.is_fed_day = (d in self.FED_DATES)
        state.is_nfp_day = self._is_nfp_day(d)
        state.is_holiday_adjacent = self._is_holiday_adjacent(d)

        # Risk multipliers
        state.event_risk_multiplier = self._event_risk_multiplier(state)
        state.liquidity_expected = self._expected_liquidity(state)

        return state

    def _get_session_phase(self, ts: datetime) -> SessionPhase:
        """Determine current MOEX session phase."""
        # Convert to MSK if needed (assume input is MSK)
        h, m = ts.hour, ts.minute
        time_minutes = h * 60 + m

        # Morning auction: 06:50-07:00
        if 410 <= time_minutes < 420:
            return SessionPhase.MORNING_AUCTION

        # Morning session (FORTS only): 07:00-10:00
        if 420 <= time_minutes < 600:
            return SessionPhase.MORNING_SESSION

        # Opening drive: 10:00-10:15
        if 600 <= time_minutes < 615:
            return SessionPhase.OPENING_DRIVE

        # Morning active: 10:15-11:30
        if 615 <= time_minutes < 690:
            return SessionPhase.MORNING_ACTIVE

        # Midday: 11:30-13:00
        if 690 <= time_minutes < 780:
            return SessionPhase.MIDDAY

        # Lunch: 13:00-14:00
        if 780 <= time_minutes < 840:
            return SessionPhase.LUNCH

        # Clearing: 14:00-14:05
        if 840 <= time_minutes < 845:
            return SessionPhase.CLEARING

        # Afternoon: 14:05-16:00
        if 845 <= time_minutes < 960:
            return SessionPhase.AFTERNOON

        # Pre-close: 16:00-18:40
        if 960 <= time_minutes < 1120:
            return SessionPhase.PRE_CLOSE

        # Closing auction (equities): 18:40-18:50
        if 1120 <= time_minutes < 1130:
            return SessionPhase.CLOSING_AUCTION

        # Evening clearing: 18:45-19:05
        if 1125 <= time_minutes < 1145:
            return SessionPhase.EVENING_CLEARING

        # Evening session: 19:05-23:50
        if 1145 <= time_minutes < 1430:
            return SessionPhase.EVENING_SESSION

        return SessionPhase.CLOSED

    def _time_of_day_normalized(self, ts: datetime) -> float:
        """
        Normalize time within trading day [0, 1].

        0 = 10:00 (main session open)
        1 = 18:40 (main session close)
        """
        h, m = ts.hour, ts.minute
        time_minutes = h * 60 + m

        # Main session: 10:00 (600) to 18:40 (1120)
        session_start = 600
        session_end = 1120
        session_length = session_end - session_start

        if time_minutes < session_start:
            return 0.0
        if time_minutes > session_end:
            return 1.0

        return float((time_minutes - session_start) / session_length)

    def _is_evening_session(self, ts: datetime) -> bool:
        """Check if in evening session (19:05-23:50)."""
        h, m = ts.hour, ts.minute
        time_minutes = h * 60 + m
        return 1145 <= time_minutes < 1430

    def _is_tax_period(self, d: date) -> bool:
        """
        Check if in Russian tax period (20-25 of each month).

        Exporters sell USD → RUB strengthens → RI pressure down.
        """
        return 20 <= d.day <= 25

    def _tax_period_day(self, d: date) -> int:
        """Return day of tax period (1-6) or 0 if not in tax period."""
        if 20 <= d.day <= 25:
            return d.day - 19
        return 0

    def _next_expiry_date(self, d: date) -> Optional[date]:
        """Find next FORTS expiration date (3rd Thursday of quarter month)."""
        # Find next expiry month
        month = d.month
        year = d.year

        for _ in range(4):
            if month in self.EXPIRY_MONTHS:
                # Find 3rd Thursday
                first_day = date(year, month, 1)
                first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
                third_thursday = first_thursday + timedelta(weeks=2)

                if third_thursday > d:
                    return third_thursday

            month += 1
            if month > 12:
                month = 1
                year += 1

        return None

    def _next_dividend(self, ticker: str, d: date) -> Tuple[Optional[date], float]:
        """Find next dividend cutoff date for ticker."""
        if ticker not in self.DIV_CALENDAR:
            return None, 0.0

        for divcut, amount in sorted(self.DIV_CALENDAR[ticker]):
            if divcut >= d:
                return divcut, amount

        return None, 0.0

    def _is_nfp_day(self, d: date) -> bool:
        """Check if NonFarm Payrolls day (first Friday of month)."""
        if d.weekday() != 4:  # Not Friday
            return False

        # First Friday = day 1-7
        return d.day <= 7

    def _is_holiday_adjacent(self, d: date) -> bool:
        """Check if day before/after Russian holiday."""
        prev_day = d - timedelta(days=1)
        next_day = d + timedelta(days=1)

        return (prev_day in self.RU_HOLIDAYS) or (next_day in self.RU_HOLIDAYS)

    def _event_risk_multiplier(self, state: CalendarState) -> float:
        """
        Calculate position size multiplier for calendar events.

        Lower multiplier = reduce position size.
        """
        mult = 1.0

        # CBR day: high impact on banks, rates
        if state.is_cbr_day:
            mult *= 0.5

        # Fed day: global risk
        if state.is_fed_day:
            mult *= 0.7

        # OPEC: oil impact
        if state.is_opec_day:
            mult *= 0.6

        # Expiry day: high volatility
        if state.is_expiry_day:
            mult *= 0.3

        # Expiry week: elevated volatility
        elif state.is_expiry_week:
            mult *= 0.8

        # Holiday adjacent: low liquidity
        if state.is_holiday_adjacent:
            mult *= 0.7

        # Evening session: thin liquidity
        if state.is_evening:
            mult *= 0.5

        return max(0.2, mult)

    def _expected_liquidity(self, state: CalendarState) -> float:
        """
        Estimate expected liquidity (1.0 = normal).

        < 1.0 means wider spreads, harder execution.
        """
        liquidity = 1.0

        # Session phase
        if state.session_phase == SessionPhase.LUNCH.value:
            liquidity *= 0.7
        elif state.session_phase == SessionPhase.MORNING_SESSION.value:
            liquidity *= 0.5
        elif state.session_phase == SessionPhase.EVENING_SESSION.value:
            liquidity *= 0.3
        elif state.session_phase == SessionPhase.CLEARING.value:
            liquidity *= 0.0

        # Day of week
        if state.is_friday:
            liquidity *= 0.9  # Some pre-weekend reduction

        # Month effects
        if state.is_month_end:
            liquidity *= 1.1  # Rebalancing = more volume

        # Holiday adjacent
        if state.is_holiday_adjacent:
            liquidity *= 0.6

        return max(0.0, liquidity)

    # === Trading rules ===

    def should_skip_trade(
        self,
        state: CalendarState,
        ticker: str,
        direction: str,
    ) -> Tuple[bool, str]:
        """
        Check if trade should be skipped based on calendar.

        Args:
            state: Calendar state
            ticker: Ticker
            direction: "LONG" or "SHORT"

        Returns:
            (should_skip, reason)
        """
        # Never trade during clearing
        if state.session_phase == SessionPhase.CLEARING.value:
            return True, "Clearing time - no trading"

        # No new positions on expiry day
        if state.is_expiry_day:
            return True, "Expiry day - no new positions"

        # Tax period + RI LONG
        if state.is_tax_period and ticker == "RI" and direction == "LONG":
            return True, "Tax period - RI LONG blocked (RUB strengthening)"

        # CBR day + SBER
        if state.is_cbr_day and ticker in {"SBER", "VTBR"}:
            return True, "CBR rate day - bank stocks blocked"

        return False, ""


# Feature column names
CALENDAR_FEATURE_COLS = [
    "cal_session_phase",
    "cal_time_normalized",
    "cal_is_evening",
    "cal_day_of_week",
    "cal_is_monday",
    "cal_is_friday",
    "cal_is_month_start",
    "cal_is_month_end",
    "cal_is_quarter_end",
    "cal_is_tax_period",
    "cal_tax_day",
    "cal_is_expiry_week",
    "cal_is_expiry_day",
    "cal_days_to_expiry",
    "cal_days_to_divcut",
    "cal_div_yield",
    "cal_is_post_divcut",
    "cal_is_cbr_day",
    "cal_is_opec_day",
    "cal_is_fed_day",
    "cal_event_risk_mult",
    "cal_liquidity_expected",
]


# Singleton
_calendar: Optional[CalendarFeatures] = None


def get_calendar_features() -> CalendarFeatures:
    """Get or create global CalendarFeatures instance."""
    global _calendar
    if _calendar is None:
        _calendar = CalendarFeatures()
    return _calendar
