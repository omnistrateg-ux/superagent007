"""
MOEX Agent v2 Market Context

Fetches market-wide context for trading decisions:
- IMOEX index level and change
- USD/RUB rate and change
- Brent oil price and change
- Market regime classification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger("moex_agent.market_context")

# Oil & gas tickers affected by Brent
OIL_GAS_TICKERS = {"GAZP", "LKOH", "ROSN", "SIBN", "TATN", "SNGS", "NVTK"}

# Bank tickers affected by RUB volatility
BANK_TICKERS = {"SBER", "VTBR", "TCSG", "CBOM", "SBERP"}

ISS_BASE = "https://iss.moex.com/iss"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MOEX-Agent/2.0"})


class MarketRegime(Enum):
    """Market regime classification."""
    RISK_ON = "risk_on"      # Bullish, go long
    NEUTRAL = "neutral"      # Normal conditions
    RISK_OFF = "risk_off"    # Bearish, prefer shorts
    PANIC = "panic"          # High volatility, no trading


@dataclass
class MarketContext:
    """Market-wide context data."""
    imoex_value: Optional[float] = None
    imoex_change_pct: Optional[float] = None
    usdrub_value: Optional[float] = None
    usdrub_change_pct: Optional[float] = None
    brent_value: Optional[float] = None
    brent_change_pct: Optional[float] = None
    regime: MarketRegime = MarketRegime.NEUTRAL

    def __repr__(self) -> str:
        return (
            f"MarketContext(regime={self.regime.value}, "
            f"IMOEX={self.imoex_change_pct:+.2f}%, "
            f"USD/RUB={self.usdrub_change_pct:+.2f}%, "
            f"Brent={self.brent_change_pct:+.2f}%)"
        )


def _fetch_imoex() -> tuple[Optional[float], Optional[float]]:
    """Fetch IMOEX index value and daily change."""
    try:
        url = f"{ISS_BASE}/engines/stock/markets/index/securities/IMOEX.json"
        resp = SESSION.get(url, params={"iss.meta": "off"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # marketdata contains current values
        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])

        if not rows:
            return None, None

        row = dict(zip(cols, rows[0]))
        value = row.get("CURRENTVALUE") or row.get("LASTVALUE")
        open_val = row.get("OPENVALUE")

        if value and open_val and open_val > 0:
            change_pct = (value - open_val) / open_val * 100
            return float(value), float(change_pct)

        return float(value) if value else None, None

    except Exception as e:
        logger.warning(f"Failed to fetch IMOEX: {e}")
        return None, None


def _fetch_usdrub() -> tuple[Optional[float], Optional[float]]:
    """Fetch USD/RUB rate and daily change."""
    try:
        url = f"{ISS_BASE}/engines/currency/markets/selt/boards/CETS/securities/USD000UTSTOM.json"
        resp = SESSION.get(url, params={"iss.meta": "off"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])

        if not rows:
            return None, None

        row = dict(zip(cols, rows[0]))
        value = row.get("LAST") or row.get("WAPRICE")
        open_val = row.get("OPEN")

        if value and open_val and open_val > 0:
            change_pct = (value - open_val) / open_val * 100
            return float(value), float(change_pct)

        return float(value) if value else None, None

    except Exception as e:
        logger.warning(f"Failed to fetch USD/RUB: {e}")
        return None, None


def _fetch_brent() -> tuple[Optional[float], Optional[float]]:
    """Fetch Brent futures price and daily change."""
    try:
        url = f"{ISS_BASE}/engines/futures/markets/forts/securities.json"
        resp = SESSION.get(url, params={"iss.meta": "off"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])

        # Find BR* contract (nearest month)
        secid_idx = cols.index("SECID") if "SECID" in cols else None
        if secid_idx is None:
            return None, None

        br_rows = [r for r in rows if r[secid_idx] and r[secid_idx].startswith("BR")]
        if not br_rows:
            return None, None

        # Take first (nearest) contract
        row = dict(zip(cols, br_rows[0]))
        value = row.get("LAST") or row.get("WAPRICE")
        open_val = row.get("OPEN")

        if value and open_val and open_val > 0:
            change_pct = (value - open_val) / open_val * 100
            return float(value), float(change_pct)

        return float(value) if value else None, None

    except Exception as e:
        logger.warning(f"Failed to fetch Brent: {e}")
        return None, None


def _classify_regime(
    imoex_change: Optional[float],
    usdrub_change: Optional[float],
    brent_change: Optional[float],
) -> MarketRegime:
    """
    Classify market regime based on market data.

    PANIC: IMOEX < -3% or USD/RUB > +3%
    RISK_OFF: IMOEX < -1% or USD/RUB > +1.5%
    RISK_ON: IMOEX > +1% and USD/RUB < +0.5%
    NEUTRAL: otherwise
    """
    # Handle missing data
    imoex = imoex_change or 0.0
    usdrub = usdrub_change or 0.0

    # PANIC conditions
    if imoex < -3.0 or usdrub > 3.0:
        return MarketRegime.PANIC

    # RISK_OFF conditions
    if imoex < -1.0 or usdrub > 1.5:
        return MarketRegime.RISK_OFF

    # RISK_ON conditions
    if imoex > 1.0 and usdrub < 0.5:
        return MarketRegime.RISK_ON

    return MarketRegime.NEUTRAL


def fetch_market_context() -> MarketContext:
    """
    Fetch complete market context.

    Returns:
        MarketContext with current market data and regime
    """
    imoex_val, imoex_chg = _fetch_imoex()
    usdrub_val, usdrub_chg = _fetch_usdrub()
    brent_val, brent_chg = _fetch_brent()

    regime = _classify_regime(imoex_chg, usdrub_chg, brent_chg)

    return MarketContext(
        imoex_value=imoex_val,
        imoex_change_pct=imoex_chg or 0.0,
        usdrub_value=usdrub_val,
        usdrub_change_pct=usdrub_chg or 0.0,
        brent_value=brent_val,
        brent_change_pct=brent_chg or 0.0,
        regime=regime,
    )


def should_skip_by_context(
    ticker: str,
    direction: str,
    ctx: MarketContext,
) -> tuple[bool, str]:
    """
    Check if signal should be skipped based on market context.

    Rules:
    - PANIC regime: skip all signals
    - RISK_OFF regime: only SHORT allowed
    - Brent < -1.5%: no LONG on oil & gas
    - USD/RUB > +2%: no LONG on banks

    Args:
        ticker: Security ticker
        direction: "LONG" or "SHORT"
        ctx: Current market context

    Returns:
        (should_skip, reason) tuple
    """
    direction_upper = direction.upper() if isinstance(direction, str) else direction.value.upper()

    # PANIC: block everything
    if ctx.regime == MarketRegime.PANIC:
        return True, f"PANIC regime: IMOEX {ctx.imoex_change_pct:+.1f}%, USD/RUB {ctx.usdrub_change_pct:+.1f}%"

    # RISK_OFF: only shorts
    if ctx.regime == MarketRegime.RISK_OFF and direction_upper == "LONG":
        return True, f"RISK_OFF regime: only SHORT allowed"

    # Brent crash: no long oil & gas
    if (ctx.brent_change_pct is not None and
        ctx.brent_change_pct < -1.5 and
        ticker in OIL_GAS_TICKERS and
        direction_upper == "LONG"):
        return True, f"Brent {ctx.brent_change_pct:+.1f}%: no LONG oil&gas"

    # RUB crash: no long banks
    if (ctx.usdrub_change_pct is not None and
        ctx.usdrub_change_pct > 2.0 and
        ticker in BANK_TICKERS and
        direction_upper == "LONG"):
        return True, f"USD/RUB {ctx.usdrub_change_pct:+.1f}%: no LONG banks"

    return False, ""
