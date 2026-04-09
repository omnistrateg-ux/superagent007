"""
MOEX Agent v2.5 External Feeds

Phase 3: CME/ICE lead signals for MOEX trading.

Timeline (MSK):
  23:50 → MOEX FORTS closes
  00:00-07:00 → CME/ICE active trading
  07:00 → MOEX FORTS morning session
  10:00 → MOEX main session

Lead-lag: overnight moves on CME predict MOEX morning gaps.

Data sources:
  - Yahoo Finance API (free, 15-min delay) - sufficient for overnight
  - MOEX ISS API (free, real-time for indices) - USD/RUB, IMOEX

Usage:
    from moex_agent.external_feeds import ExternalFeeds

    feeds = ExternalFeeds()
    features = feeds.get_lead_features()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not available - external feeds disabled")


@dataclass
class LeadSignals:
    """Container for external lead signals."""
    # Brent
    brent_overnight_return: Optional[float] = None
    brent_5m_lead: Optional[float] = None

    # S&P 500
    sp500_overnight_return: Optional[float] = None
    vix_level: Optional[float] = None
    vix_change: Optional[float] = None

    # Natural Gas
    henry_hub_overnight_return: Optional[float] = None

    # Gold
    gold_overnight_return: Optional[float] = None

    # USD/RUB (from MOEX)
    usdrub_overnight_change: Optional[float] = None

    # Composite
    global_risk_sentiment: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "lead_brent_overnight": self.brent_overnight_return or 0.0,
            "lead_brent_5m": self.brent_5m_lead or 0.0,
            "lead_sp500_overnight": self.sp500_overnight_return or 0.0,
            "lead_vix": self.vix_level or 20.0,
            "lead_vix_change": self.vix_change or 0.0,
            "lead_ng_overnight": self.henry_hub_overnight_return or 0.0,
            "lead_gold_overnight": self.gold_overnight_return or 0.0,
            "lead_usdrub_change": self.usdrub_overnight_change or 0.0,
            "lead_risk_sentiment": self.global_risk_sentiment or 0.0,
        }


class ExternalFeeds:
    """
    Fetch external market data for lead signal calculation.

    Primary use: overnight returns from CME/ICE to predict MOEX morning gaps.
    """

    # Yahoo Finance tickers
    TICKERS = {
        "brent": "BZ=F",       # ICE Brent Crude
        "wti": "CL=F",         # NYMEX WTI
        "sp500": "ES=F",       # S&P 500 E-mini
        "vix": "^VIX",         # VIX Index
        "gold": "GC=F",        # COMEX Gold
        "nat_gas": "NG=F",     # NYMEX Natural Gas
        "dxy": "DX-Y.NYB",     # Dollar Index
    }

    # MOEX ISS endpoints
    ISS_BASE = "https://iss.moex.com/iss"

    # Cache
    _cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
    CACHE_TTL_MINUTES = 5

    def __init__(self):
        self._session = None
        if HAS_YFINANCE:
            import requests
            self._session = requests.Session()

    def get_lead_features(self) -> LeadSignals:
        """
        Get all lead signals for current moment.

        Returns:
            LeadSignals with external market data
        """
        signals = LeadSignals()

        if not HAS_YFINANCE:
            logger.warning("yfinance not available")
            return signals

        try:
            # Brent
            signals.brent_overnight_return = self._get_overnight_return("brent")

            # S&P 500
            signals.sp500_overnight_return = self._get_overnight_return("sp500")

            # VIX
            vix_data = self._get_recent_data("vix", days=2)
            if vix_data is not None and len(vix_data) >= 2:
                signals.vix_level = float(vix_data["Close"].iloc[-1])
                signals.vix_change = float(
                    (vix_data["Close"].iloc[-1] - vix_data["Close"].iloc[-2])
                    / vix_data["Close"].iloc[-2]
                )

            # Natural Gas
            signals.henry_hub_overnight_return = self._get_overnight_return("nat_gas")

            # Gold
            signals.gold_overnight_return = self._get_overnight_return("gold")

            # USD/RUB (from MOEX)
            signals.usdrub_overnight_change = self._get_usdrub_change()

            # Composite risk sentiment
            signals.global_risk_sentiment = self._calculate_risk_sentiment(signals)

        except Exception as e:
            logger.error(f"Error fetching lead signals: {e}")

        return signals

    def _get_overnight_return(self, instrument: str) -> Optional[float]:
        """
        Calculate overnight return (23:50 MSK to 09:55 MSK).

        For MOEX morning session, this captures CME/ICE moves while MOEX was closed.
        """
        data = self._get_recent_data(instrument, days=2)
        if data is None or len(data) < 2:
            return None

        try:
            # Get yesterday's close and current price
            yesterday_close = data["Close"].iloc[-2]
            current_price = data["Close"].iloc[-1]

            if yesterday_close > 0:
                return float((current_price - yesterday_close) / yesterday_close)
        except Exception as e:
            logger.debug(f"Error calculating overnight return for {instrument}: {e}")

        return None

    def _get_recent_data(
        self,
        instrument: str,
        days: int = 2,
    ) -> Optional[pd.DataFrame]:
        """Fetch recent price data with caching."""
        ticker = self.TICKERS.get(instrument)
        if not ticker:
            return None

        # Check cache
        cache_key = f"{ticker}_{days}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.CACHE_TTL_MINUTES):
                return cached_data

        try:
            data = yf.download(
                ticker,
                period=f"{days}d",
                interval="1d",
                progress=False,
            )

            if not data.empty:
                # Flatten MultiIndex columns (yfinance returns ('Close', 'TICKER'))
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                self._cache[cache_key] = (datetime.now(), data)
                return data

        except Exception as e:
            logger.debug(f"Error fetching {ticker}: {e}")

        return None

    def _get_usdrub_change(self) -> Optional[float]:
        """Get USD/RUB overnight change from MOEX ISS API."""
        try:
            import requests

            url = f"{self.ISS_BASE}/engines/currency/markets/selt/boards/CETS/securities/USD000UTSTOM.json"
            resp = requests.get(url, params={"iss.meta": "off"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            md = data.get("marketdata", {})
            cols = md.get("columns", [])
            rows = md.get("data", [])

            if not rows:
                return None

            row = dict(zip(cols, rows[0]))
            current = row.get("LAST") or row.get("WAPRICE")
            prev_close = row.get("CLOSEPRICE") or row.get("PREVPRICE")

            if current and prev_close and prev_close > 0:
                return float((current - prev_close) / prev_close)

        except Exception as e:
            logger.debug(f"Error fetching USD/RUB: {e}")

        return None

    def _calculate_risk_sentiment(self, signals: LeadSignals) -> float:
        """
        Calculate composite global risk sentiment.

        Formula: ES_return × 0.4 + VIX_change × (-0.3) + gold_change × (-0.2) + em_proxy × 0.1

        Returns:
            Score in [-1, 1]. Positive = risk-on, negative = risk-off.
        """
        score = 0.0
        count = 0

        # S&P 500 return (positive = risk-on)
        if signals.sp500_overnight_return is not None:
            score += signals.sp500_overnight_return * 100 * 0.4  # Convert to %
            count += 1

        # VIX change (positive VIX = risk-off, so negate)
        if signals.vix_change is not None:
            score -= signals.vix_change * 100 * 0.3
            count += 1

        # Gold (safe haven - positive gold = risk-off)
        if signals.gold_overnight_return is not None:
            score -= signals.gold_overnight_return * 100 * 0.2
            count += 1

        # USD/RUB (RUB weakening = risk-off for Russia)
        if signals.usdrub_overnight_change is not None:
            score -= signals.usdrub_overnight_change * 100 * 0.1
            count += 1

        if count == 0:
            return 0.0

        # Normalize to [-1, 1]
        return float(np.clip(score / 5.0, -1.0, 1.0))

    # === Specific instrument lead signals ===

    def get_brent_lead(self, br_moex_return_5m: float) -> float:
        """
        Get ICE Brent lead signal vs BR MOEX.

        If ICE moved but BR MOEX hasn't yet → signal.

        Args:
            br_moex_return_5m: BR futures return on MOEX (last 5 min)

        Returns:
            Lead signal: positive = ICE ahead (buy BR), negative = ICE lagging
        """
        ice_return = self._get_overnight_return("brent")
        if ice_return is None:
            return 0.0

        # Lead = ICE return - MOEX return
        # If ICE is up more than MOEX → MOEX will catch up
        return float(ice_return - br_moex_return_5m)

    def get_sp500_lead(self, ri_moex_return_5m: float) -> float:
        """
        Get S&P 500 lead signal vs RI MOEX.

        Args:
            ri_moex_return_5m: RI futures return on MOEX (last 5 min)

        Returns:
            Lead signal: positive = ES ahead, negative = ES lagging
        """
        es_return = self._get_overnight_return("sp500")
        if es_return is None:
            return 0.0

        return float(es_return - ri_moex_return_5m)

    def get_gold_lead(self, plzl_return_5m: float) -> float:
        """
        Get Gold lead signal vs PLZL.

        Args:
            plzl_return_5m: PLZL return (last 5 min)

        Returns:
            Lead signal
        """
        gold_return = self._get_overnight_return("gold")
        if gold_return is None:
            return 0.0

        return float(gold_return - plzl_return_5m)

    def predict_gap(self, ticker: str) -> float:
        """
        Predict opening gap for ticker based on overnight moves.

        Args:
            ticker: MOEX ticker

        Returns:
            Predicted gap (positive = gap up)
        """
        signals = self.get_lead_features()

        # Ticker-specific beta to external markets
        betas = {
            # Oil & Gas → Brent
            "BR": (signals.brent_overnight_return or 0) * 0.8,
            "LKOH": (signals.brent_overnight_return or 0) * 0.5,
            "ROSN": (signals.brent_overnight_return or 0) * 0.5,
            "TATN": (signals.brent_overnight_return or 0) * 0.4,
            "GAZP": (signals.brent_overnight_return or 0) * 0.3,

            # Index futures → S&P
            "RI": (signals.sp500_overnight_return or 0) * 0.4
                  - (signals.usdrub_overnight_change or 0) * 0.3,
            "MX": (signals.sp500_overnight_return or 0) * 0.3,

            # Banks → Risk sentiment
            "SBER": (signals.global_risk_sentiment or 0) * 0.02,

            # Gold → Gold futures
            "PLZL": (signals.gold_overnight_return or 0) * 0.6,

            # Natural Gas
            "NG": (signals.henry_hub_overnight_return or 0) * 0.7,
        }

        return float(betas.get(ticker, 0.0))


# Singleton instance
_feeds: Optional[ExternalFeeds] = None


def get_external_feeds() -> ExternalFeeds:
    """Get or create global ExternalFeeds instance."""
    global _feeds
    if _feeds is None:
        _feeds = ExternalFeeds()
    return _feeds


# Feature column names
LEAD_FEATURE_COLS = [
    "lead_brent_overnight",
    "lead_brent_5m",
    "lead_sp500_overnight",
    "lead_vix",
    "lead_vix_change",
    "lead_ng_overnight",
    "lead_gold_overnight",
    "lead_usdrub_change",
    "lead_risk_sentiment",
]
