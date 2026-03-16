"""
MOEX Agent v2 Rule-Based Signal Filter

Final confirmation layer after ML prediction.
Applies technical analysis rules to filter signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .anomaly import AnomalyResult, Direction

logger = logging.getLogger("moex_agent.signals")


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


def filter_signal(
    anomaly: AnomalyResult,
    features: Dict[str, float],
    config: Optional[SignalFilter] = None,
) -> tuple[bool, List[str]]:
    """
    Apply rule-based filters to a signal.

    Args:
        anomaly: Detected anomaly
        features: Dict of feature values
        config: Filter configuration

    Returns:
        (passes, reasons) - whether signal passes and list of reasons
    """
    config = config or SignalFilter()
    reasons = []
    passes = True

    # RSI check
    rsi = features.get("rsi_14", 50.0)
    if not check_rsi_condition(rsi, anomaly.direction, config):
        reasons.append(f"RSI={rsi:.1f} not suitable for {anomaly.direction.value}")
        passes = False

    # MACD check
    if config.require_macd_confirm:
        macd = features.get("macd", 0.0)
        macd_signal = features.get("macd_signal", 0.0)
        macd_hist = features.get("macd_hist", 0.0)
        if not check_macd_condition(macd, macd_signal, macd_hist, anomaly.direction):
            reasons.append(f"MACD not confirming {anomaly.direction.value}")
            passes = False

    # Bollinger check
    bb_position = features.get("bb_position", 0.5)
    if not check_bollinger_condition(bb_position, anomaly.direction, config):
        reasons.append(f"BB position={bb_position:.2f} too extreme")
        passes = False

    # ADX check
    adx = features.get("adx", 0.0)
    if not check_adx_condition(adx, config):
        reasons.append(f"ADX={adx:.1f} too weak (need >= {config.min_adx})")
        passes = False

    # Volume check
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
