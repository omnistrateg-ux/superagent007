"""
MOEX Agent v2.5 Evening Session Handler

Evening session (FORTS): 19:05 - 23:50 MSK

Characteristics:
- Lower liquidity than main session
- Wider spreads
- Follows US/EU market moves
- RI/SI more active than equities
- Higher volatility on news

Trading rules:
1. Reduce position size (0.5x default)
2. Widen stop-loss (1.5x ATR instead of 1x)
3. Prefer futures (RI, SI, BR) over equities
4. Skip first 15 min after open (19:05-19:20)
5. Skip last 30 min before close (23:20-23:50)
6. Monitor CME/ICE for lead signals

Usage:
    from moex_agent.evening_session import EveningSessionHandler

    handler = EveningSessionHandler()
    if handler.is_evening_session():
        adjustments = handler.get_trading_adjustments(ticker)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Moscow timezone
MSK = ZoneInfo("Europe/Moscow")

# Evening session times
EVENING_SESSION_START = time(19, 5)   # 19:05 MSK
EVENING_SESSION_END = time(23, 50)    # 23:50 MSK
EVENING_CLEARING_START = time(18, 45) # 18:45 MSK
EVENING_CLEARING_END = time(19, 5)    # 19:05 MSK

# Skip periods within evening session
EVENING_OPEN_SKIP_MINUTES = 15  # Skip 19:05-19:20
EVENING_CLOSE_SKIP_MINUTES = 30  # Skip 23:20-23:50


@dataclass
class EveningAdjustments:
    """Trading adjustments for evening session."""
    position_mult: float = 1.0      # Position size multiplier
    stop_mult: float = 1.0          # Stop-loss multiplier
    take_mult: float = 1.0          # Take-profit multiplier
    max_spread_mult: float = 1.0    # Max spread multiplier
    min_turnover_mult: float = 1.0  # Min turnover multiplier
    allow_trading: bool = True
    skip_reason: Optional[str] = None
    preferred_instruments: List[str] = None

    def __post_init__(self):
        if self.preferred_instruments is None:
            self.preferred_instruments = []


# Instrument categories for evening session
EVENING_PREFERRED = {
    # Futures - most liquid in evening
    "RI": 1.0,    # RTS Index futures
    "SI": 1.0,    # USD/RUB futures
    "BR": 0.9,    # Brent futures
    "MX": 0.8,    # MOEX Index futures
    "GD": 0.7,    # Gold futures
    "NG": 0.6,    # Natural Gas futures
}

EVENING_REDUCED = {
    # Equities - less liquid in evening
    "SBER": 0.5,
    "GAZP": 0.5,
    "LKOH": 0.5,
    "ROSN": 0.4,
    "GMKN": 0.4,
    "NVTK": 0.3,
    "VTBR": 0.3,
}

# Tickers to skip entirely in evening
EVENING_SKIP = [
    "MGNT", "FIVE", "RTKM", "MTSS",  # Low liquidity
    "OZON", "VKCO",  # Tech - thin evening
]


class EveningSessionHandler:
    """
    Handler for evening session trading adjustments.
    """

    def __init__(
        self,
        position_mult: float = 0.5,
        stop_mult: float = 1.5,
        take_mult: float = 1.2,
        max_spread_mult: float = 2.0,
        min_turnover_mult: float = 0.5,
    ):
        """
        Args:
            position_mult: Default position size multiplier for evening
            stop_mult: Stop-loss distance multiplier (wider stops)
            take_mult: Take-profit multiplier
            max_spread_mult: Allow wider spreads in evening
            min_turnover_mult: Lower turnover threshold
        """
        self.position_mult = position_mult
        self.stop_mult = stop_mult
        self.take_mult = take_mult
        self.max_spread_mult = max_spread_mult
        self.min_turnover_mult = min_turnover_mult

    def get_msk_time(self, dt: Optional[datetime] = None) -> datetime:
        """Get current time in Moscow timezone."""
        if dt is None:
            dt = datetime.now(MSK)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=MSK)
        return dt

    def is_evening_session(self, dt: Optional[datetime] = None) -> bool:
        """Check if current time is within evening session."""
        msk_time = self.get_msk_time(dt).time()
        return EVENING_SESSION_START <= msk_time <= EVENING_SESSION_END

    def is_evening_clearing(self, dt: Optional[datetime] = None) -> bool:
        """Check if current time is evening clearing (no trading)."""
        msk_time = self.get_msk_time(dt).time()
        return EVENING_CLEARING_START <= msk_time < EVENING_CLEARING_END

    def is_evening_open_period(self, dt: Optional[datetime] = None) -> bool:
        """Check if within first N minutes after evening open."""
        msk_dt = self.get_msk_time(dt)
        msk_time = msk_dt.time()

        if not self.is_evening_session(dt):
            return False

        evening_start = datetime.combine(msk_dt.date(), EVENING_SESSION_START)
        evening_start = evening_start.replace(tzinfo=MSK)
        skip_end = evening_start + timedelta(minutes=EVENING_OPEN_SKIP_MINUTES)

        return msk_dt < skip_end

    def is_evening_close_period(self, dt: Optional[datetime] = None) -> bool:
        """Check if within last N minutes before evening close."""
        msk_dt = self.get_msk_time(dt)
        msk_time = msk_dt.time()

        if not self.is_evening_session(dt):
            return False

        evening_end = datetime.combine(msk_dt.date(), EVENING_SESSION_END)
        evening_end = evening_end.replace(tzinfo=MSK)
        skip_start = evening_end - timedelta(minutes=EVENING_CLOSE_SKIP_MINUTES)

        return msk_dt >= skip_start

    def get_trading_adjustments(
        self,
        ticker: str,
        dt: Optional[datetime] = None,
    ) -> EveningAdjustments:
        """
        Get trading adjustments for a ticker in evening session.

        Args:
            ticker: Instrument ticker
            dt: Optional datetime (default: now)

        Returns:
            EveningAdjustments with position sizing and rules
        """
        # Not evening session - return defaults
        if not self.is_evening_session(dt):
            return EveningAdjustments(
                allow_trading=True,
                skip_reason=None,
            )

        # Evening clearing - no trading
        if self.is_evening_clearing(dt):
            return EveningAdjustments(
                allow_trading=False,
                skip_reason="evening clearing (18:45-19:05)",
            )

        # Skip first 15 min after open
        if self.is_evening_open_period(dt):
            return EveningAdjustments(
                allow_trading=False,
                skip_reason=f"evening open period (skip first {EVENING_OPEN_SKIP_MINUTES} min)",
            )

        # Skip last 30 min before close
        if self.is_evening_close_period(dt):
            return EveningAdjustments(
                allow_trading=False,
                skip_reason=f"evening close period (skip last {EVENING_CLOSE_SKIP_MINUTES} min)",
            )

        # Skip certain tickers entirely
        if ticker in EVENING_SKIP:
            return EveningAdjustments(
                allow_trading=False,
                skip_reason=f"{ticker} has insufficient evening liquidity",
            )

        # Get ticker-specific multiplier
        ticker_mult = 1.0
        if ticker in EVENING_PREFERRED:
            ticker_mult = EVENING_PREFERRED[ticker]
        elif ticker in EVENING_REDUCED:
            ticker_mult = EVENING_REDUCED[ticker]
        else:
            ticker_mult = 0.3  # Unknown ticker - very conservative

        return EveningAdjustments(
            position_mult=self.position_mult * ticker_mult,
            stop_mult=self.stop_mult,
            take_mult=self.take_mult,
            max_spread_mult=self.max_spread_mult,
            min_turnover_mult=self.min_turnover_mult,
            allow_trading=True,
            skip_reason=None,
            preferred_instruments=list(EVENING_PREFERRED.keys()),
        )

    def get_session_info(self, dt: Optional[datetime] = None) -> Dict:
        """Get current session information."""
        msk_dt = self.get_msk_time(dt)

        return {
            "msk_time": msk_dt.strftime("%H:%M:%S"),
            "is_evening_session": self.is_evening_session(dt),
            "is_evening_clearing": self.is_evening_clearing(dt),
            "is_evening_open_period": self.is_evening_open_period(dt),
            "is_evening_close_period": self.is_evening_close_period(dt),
            "session_start": EVENING_SESSION_START.strftime("%H:%M"),
            "session_end": EVENING_SESSION_END.strftime("%H:%M"),
            "preferred_instruments": list(EVENING_PREFERRED.keys()),
        }

    def minutes_to_close(self, dt: Optional[datetime] = None) -> Optional[int]:
        """Get minutes remaining until session close."""
        if not self.is_evening_session(dt):
            return None

        msk_dt = self.get_msk_time(dt)
        evening_end = datetime.combine(msk_dt.date(), EVENING_SESSION_END)
        evening_end = evening_end.replace(tzinfo=MSK)

        delta = evening_end - msk_dt
        return int(delta.total_seconds() / 60)


# Singleton
_evening_handler: Optional[EveningSessionHandler] = None


def get_evening_handler() -> EveningSessionHandler:
    """Get or create global EveningSessionHandler instance."""
    global _evening_handler
    if _evening_handler is None:
        _evening_handler = EveningSessionHandler()
    return _evening_handler


def is_evening_session(dt: Optional[datetime] = None) -> bool:
    """Convenience function to check if evening session."""
    return get_evening_handler().is_evening_session(dt)


def get_evening_adjustments(
    ticker: str,
    dt: Optional[datetime] = None,
) -> EveningAdjustments:
    """Convenience function to get evening adjustments."""
    return get_evening_handler().get_trading_adjustments(ticker, dt)
