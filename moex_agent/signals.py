"""
MOEX Agent v2 Rule-Based Signal Filter

Final confirmation layer after ML prediction.
Applies technical analysis rules to filter signals.

Additional filters:
- Spread: >25bps skip for 5m/10m horizons
- Liquidity: SHORT disabled for Tier 3 tickers
- Opening session: Skip first 15 minutes of trading
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .anomaly import AnomalyResult, Direction

logger = logging.getLogger("moex_agent.signals")

# Liquidity tiers for MOEX TQBR
# Tier 1: Most liquid (blue chips)
TIER_1_TICKERS: Set[str] = {
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK", "ROSN", "SNGS", "PLZL",
    "TATN", "MGNT", "ALRS", "CHMF", "NLMK", "VTBR", "MOEX",
}

# Tier 2: Liquid
TIER_2_TICKERS: Set[str] = {
    "MTSS", "IRAO", "PHOR", "RUAL", "POLY", "MAGN", "AFKS", "PIKK",
    "HYDR", "FEES", "RTKM", "AFLT", "TCSG", "OZON", "YDEX", "T",
}

# Tier 3: Less liquid (SHORT restricted)
TIER_3_TICKERS: Set[str] = {
    "TRNFP", "SBERP", "SNGSP", "TATNP", "HHRU", "FIVE", "FIXP",
    "SMLT", "SGZH", "VKCO", "POSI", "BELU", "CBOM", "MTLR",
}

# MOEX trading session start (Moscow time = UTC+3)
MOEX_OPEN_HOUR = 10
MOEX_OPEN_MINUTE = 0
OPENING_SESSION_MINUTES = 15  # Skip first 15 minutes


@dataclass
class SignalFilter:
    """Configuration for signal filtering rules."""
    min_rsi_for_long: float = 25.0
    max_rsi_for_long: float = 70.0
    min_rsi_for_short: float = 30.0
    max_rsi_for_short: float = 75.0
    min_adx: float = 20.0
    max_bb_position_long: float = 0.85
    min_bb_position_short: float = 0.15
    min_volume_spike: float = 1.2
    require_macd_confirm: bool = True

    # Spread limits by horizon
    max_spread_bps_5m: float = 25.0
    max_spread_bps_10m: float = 25.0
    max_spread_bps_30m: float = 50.0
    max_spread_bps_1h: float = 100.0

    # Opening session filter
    skip_opening_session: bool = True
    opening_session_minutes: int = 15

    # Liquidity tier restrictions
    restrict_short_tier3: bool = True


def check_rsi_condition(
    rsi: float,
    direction: Direction,
    config: SignalFilter,
) -> bool:
    """
    Check if RSI supports the trade direction.

    Long: RSI should be in recovery zone (not overbought)
    Short: RSI should be in decline zone (not oversold)
    """
    if direction == Direction.LONG:
        return config.min_rsi_for_long <= rsi <= config.max_rsi_for_long
    else:
        return config.min_rsi_for_short <= rsi <= config.max_rsi_for_short


def check_macd_condition(
    macd: float,
    macd_signal: float,
    macd_hist: float,
    direction: Direction,
) -> bool:
    """
    Check if MACD supports the trade direction.

    Long: MACD crossing above signal or positive histogram
    Short: MACD crossing below signal or negative histogram
    """
    if direction == Direction.LONG:
        return macd > macd_signal or macd_hist > 0
    else:
        return macd < macd_signal or macd_hist < 0


def check_bollinger_condition(
    bb_position: float,
    direction: Direction,
    config: SignalFilter,
) -> bool:
    """
    Check if Bollinger Band position supports the direction.

    Long: Price not at upper band (room to grow)
    Short: Price not at lower band (room to fall)
    """
    if direction == Direction.LONG:
        return bb_position <= config.max_bb_position_long
    else:
        return bb_position >= config.min_bb_position_short


def check_adx_condition(adx: float, config: SignalFilter) -> bool:
    """Check if ADX indicates sufficient trend strength."""
    return adx >= config.min_adx


def check_volume_condition(
    volume_spike: float,
    config: SignalFilter,
) -> bool:
    """Check if volume confirms the move."""
    return volume_spike >= config.min_volume_spike


def get_liquidity_tier(ticker: str) -> int:
    """
    Get liquidity tier for a ticker.

    Tier 1: Most liquid (all directions allowed)
    Tier 2: Liquid (all directions allowed)
    Tier 3: Less liquid (SHORT restricted)

    Returns:
        1, 2, or 3
    """
    if ticker in TIER_1_TICKERS:
        return 1
    elif ticker in TIER_2_TICKERS:
        return 2
    else:
        return 3


def check_spread_for_horizon(
    spread_bps: Optional[float],
    horizon: str,
    config: SignalFilter,
) -> tuple[bool, str]:
    """
    Check if spread is acceptable for the given horizon.

    Short horizons (5m, 10m) require tight spreads.

    Returns:
        (passes, reason)
    """
    if spread_bps is None:
        return True, "No spread data"

    max_spread = {
        "5m": config.max_spread_bps_5m,
        "10m": config.max_spread_bps_10m,
        "30m": config.max_spread_bps_30m,
        "1h": config.max_spread_bps_1h,
    }.get(horizon, 200.0)

    if spread_bps > max_spread:
        return False, f"Spread {spread_bps:.1f}bps > {max_spread:.0f}bps for {horizon}"

    return True, "Spread OK"


def check_liquidity_tier(
    ticker: str,
    direction: Direction,
    config: SignalFilter,
) -> tuple[bool, str]:
    """
    Check if direction is allowed for ticker's liquidity tier.

    Tier 3 tickers: SHORT is restricted (low liquidity, high slippage risk).

    Returns:
        (passes, reason)
    """
    tier = get_liquidity_tier(ticker)

    if config.restrict_short_tier3 and tier == 3 and direction == Direction.SHORT:
        return False, f"{ticker} is Tier 3 - SHORT restricted"

    return True, f"Tier {tier} OK"


def check_opening_session(
    timestamp: Optional[datetime] = None,
    config: SignalFilter = None,
) -> tuple[bool, str]:
    """
    Check if we're in the opening session (first 15 minutes).

    Opening session is typically volatile with wide spreads.

    Returns:
        (passes, reason) - False if should skip
    """
    config = config or SignalFilter()

    if not config.skip_opening_session:
        return True, "Opening filter disabled"

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Convert to Moscow time (UTC+3)
    moscow_hour = (timestamp.hour + 3) % 24
    moscow_minute = timestamp.minute

    # Check if within opening session
    session_start = MOEX_OPEN_HOUR * 60 + MOEX_OPEN_MINUTE
    current_time = moscow_hour * 60 + moscow_minute
    session_end = session_start + config.opening_session_minutes

    if session_start <= current_time < session_end:
        minutes_since_open = current_time - session_start
        return False, f"Opening session ({minutes_since_open}min since open)"

    return True, "Not in opening session"


def filter_signal(
    anomaly: AnomalyResult,
    features: Dict[str, float],
    config: Optional[SignalFilter] = None,
    horizon: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> tuple[bool, List[str]]:
    """
    Apply rule-based filters to a signal.

    Filters applied:
    1. Opening session (first 15 minutes)
    2. Spread for horizon
    3. Liquidity tier (SHORT restricted for Tier 3)
    4. RSI
    5. MACD
    6. Bollinger Bands
    7. ADX
    8. Volume

    Args:
        anomaly: Detected anomaly
        features: Dict of feature values
        config: Filter configuration
        horizon: Signal horizon (for spread check)
        timestamp: Signal timestamp (for opening session check)

    Returns:
        (passes, reasons) - whether signal passes and list of reasons
    """
    config = config or SignalFilter()
    reasons = []
    passes = True

    # 1. Opening session check
    session_ok, session_reason = check_opening_session(timestamp, config)
    if not session_ok:
        reasons.append(session_reason)
        passes = False

    # 2. Spread check for horizon
    if horizon:
        spread_ok, spread_reason = check_spread_for_horizon(
            anomaly.spread_bps, horizon, config
        )
        if not spread_ok:
            reasons.append(spread_reason)
            passes = False

    # 3. Liquidity tier check
    tier_ok, tier_reason = check_liquidity_tier(
        anomaly.secid, anomaly.direction, config
    )
    if not tier_ok:
        reasons.append(tier_reason)
        passes = False

    # 4. RSI check
    rsi = features.get("rsi_14", 50.0)
    if not check_rsi_condition(rsi, anomaly.direction, config):
        reasons.append(f"RSI={rsi:.1f} not suitable for {anomaly.direction.value}")
        passes = False

    # 5. MACD check
    if config.require_macd_confirm:
        macd = features.get("macd", 0.0)
        macd_signal = features.get("macd_signal", 0.0)
        macd_hist = features.get("macd_hist", 0.0)
        if not check_macd_condition(macd, macd_signal, macd_hist, anomaly.direction):
            reasons.append(f"MACD not confirming {anomaly.direction.value}")
            passes = False

    # 6. Bollinger check
    bb_position = features.get("bb_position", 0.5)
    if not check_bollinger_condition(bb_position, anomaly.direction, config):
        reasons.append(f"BB position={bb_position:.2f} too extreme")
        passes = False

    # 7. ADX check
    adx = features.get("adx", 0.0)
    if not check_adx_condition(adx, config):
        reasons.append(f"ADX={adx:.1f} too weak (need >= {config.min_adx})")
        passes = False

    # 8. Volume check
    if not check_volume_condition(anomaly.volume_spike, config):
        reasons.append(f"Volume spike={anomaly.volume_spike:.2f} too low")
        passes = False

    if passes:
        reasons.append("All rules passed")

    return passes, reasons


def rank_signals(
    signals: List[Dict],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Rank signals by composite score.

    Args:
        signals: List of signal dicts with features
        weights: Feature weights for scoring

    Returns:
        Sorted list of signals (best first)
    """
    weights = weights or {
        "probability": 2.0,
        "anomaly_score": 1.5,
        "volume_spike": 1.0,
        "adx": 0.5,
    }

    def compute_score(sig: Dict) -> float:
        score = 0.0
        score += weights.get("probability", 0) * sig.get("probability", 0)
        score += weights.get("anomaly_score", 0) * sig.get("anomaly_score", 0) / 3
        score += weights.get("volume_spike", 0) * min(sig.get("volume_spike", 1), 3) / 3
        score += weights.get("adx", 0) * min(sig.get("adx", 0), 50) / 50
        return score

    for sig in signals:
        sig["composite_score"] = compute_score(sig)

    return sorted(signals, key=lambda x: x.get("composite_score", 0), reverse=True)


def validate_entry_conditions(
    ticker: str,
    direction: Direction,
    features: Dict[str, float],
    anomaly: AnomalyResult,
) -> tuple[bool, str]:
    """
    Final validation before entry.

    4 levels of confirmation:
    1. Anomaly detection (already done)
    2. ML prediction (already done)
    3. Risk management (external)
    4. Rule-based filters (this function)

    Returns:
        (valid, reason)
    """
    filter_config = SignalFilter()
    passes, reasons = filter_signal(anomaly, features, filter_config)

    if not passes:
        return False, "; ".join(reasons)

    # Additional sanity checks
    close = features.get("close", 0)
    if close <= 0:
        return False, "Invalid close price"

    atr = features.get("atr_14", 0)
    if atr <= 0:
        return False, "Invalid ATR"

    # Check for extreme conditions
    rsi = features.get("rsi_14", 50)
    if rsi > 90 or rsi < 10:
        return False, f"RSI={rsi:.1f} in extreme zone"

    volatility = features.get("volatility_10", 0)
    if volatility > 0.1:
        return False, f"Volatility={volatility:.3f} too high"

    return True, "Entry conditions validated"
