"""
MOEX Agent v2.5 Risk Management

Kill-switch, margin control, dynamic position sizing.

v2.0 (RiskEngine):
- 2 consecutive losses -> HALT for the day
- 2% daily loss -> HALT for the day
- 10% drawdown -> STOP trading (manual reset required)

v2.1 (AdaptiveRiskEngine):
- No hard kill-switches
- Smooth position sizing based on: drawdown, confidence, volatility, streak
- Rolling Sharpe for performance tracking
- Position size = base * dd_mult * conf_mult * vol_mult * streak_mult

v2.5 (Phase 1 Simplification):
- streak_mult DISABLED (always 1.0) - anti-martingale doesn't improve edge
- Position size = base * dd_mult * conf_mult * vol_mult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("moex_agent.risk")


class MarketRegime(str, Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN = "UNKNOWN"


class RiskDecision(str, Enum):
    """Risk engine decision."""
    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"
    DISABLE = "DISABLE"


class DayMode(str, Enum):
    """Trading day mode."""
    NORMAL = "NORMAL"
    HALT = "HALT"
    STOP = "STOP"


@dataclass
class RiskParams:
    """Basic risk parameters."""
    max_spread_bps: float = 200.0
    min_turnover_rub_5m: float = 1_000_000.0


@dataclass
class KillSwitchConfig:
    """Kill-switch thresholds."""
    max_loss_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_consecutive_losses: int = 2
    max_drawdown_pct: float = 10.0
    cooldown_minutes: int = 60


@dataclass
class RiskState:
    """Current risk state."""
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    trades_today: int = 0
    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    day_mode: DayMode = DayMode.NORMAL
    day_start: Optional[datetime] = None

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    @property
    def daily_loss_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return -self.daily_pnl / self.equity * 100 if self.daily_pnl < 0 else 0.0


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    decision: RiskDecision
    leverage: float
    max_position_pct: float
    reason: str
    warnings: List[str] = field(default_factory=list)
    regime: MarketRegime = MarketRegime.UNKNOWN


class RiskEngine:
    """
    Risk management engine with kill-switch.

    Kill-switch rules:
    - 2 consecutive losses -> HALT
    - 2% daily loss -> HALT
    - 10% drawdown -> STOP
    """

    DISABLED_HORIZONS = {"1d", "1w"}

    HORIZON_MAX_LEVERAGE = {
        "5m": 3.0,
        "10m": 3.0,
        "30m": 2.5,
        "1h": 2.0,
    }

    def __init__(
        self,
        initial_equity: float,
        config: Optional[KillSwitchConfig] = None,
    ):
        self.config = config or KillSwitchConfig()
        now = datetime.now(timezone.utc)
        self.state = RiskState(
            equity=initial_equity,
            peak_equity=initial_equity,
            day_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
        )
        logger.info(f"RiskEngine initialized: equity={initial_equity:,.0f}")

    def check_kill_switch(self) -> Tuple[bool, Optional[str]]:
        """
        Check if kill-switch should be activated.

        Returns:
            (is_active, reason)
        """
        if self.state.kill_switch_active:
            return True, self.state.kill_switch_reason

        # 2 consecutive losses -> HALT
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            reason = f"KILL: {self.state.consecutive_losses} consecutive losses"
            self._activate_kill_switch(reason)
            return True, reason

        # 2% daily loss -> HALT
        if self.state.daily_loss_pct >= self.config.max_daily_loss_pct:
            reason = f"KILL: Daily loss {self.state.daily_loss_pct:.1f}% >= {self.config.max_daily_loss_pct}%"
            self._activate_kill_switch(reason)
            return True, reason

        # 10% drawdown -> STOP
        if self.state.current_drawdown_pct >= self.config.max_drawdown_pct:
            reason = f"STOP: Drawdown {self.state.current_drawdown_pct:.1f}% >= {self.config.max_drawdown_pct}%"
            self._activate_kill_switch(reason, permanent=True)
            return True, reason

        return False, None

    def _activate_kill_switch(self, reason: str, permanent: bool = False) -> None:
        """Activate kill-switch."""
        self.state.kill_switch_active = True
        self.state.kill_switch_reason = reason
        self.state.day_mode = DayMode.STOP if permanent else DayMode.HALT
        logger.warning(f"KILL-SWITCH: {reason}")

    def detect_regime(self, candles_df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect market regime and volatility percentile."""
        if candles_df.empty or len(candles_df) < 100:
            return MarketRegime.UNKNOWN, 50.0

        close = candles_df["close"]
        returns_20d = close.pct_change(20).iloc[-1] if len(close) > 20 else 0

        volatility = close.pct_change().rolling(20).std().iloc[-1]
        hist_vol = close.pct_change().rolling(20).std().dropna().tolist()

        if hist_vol:
            vol_percentile = sum(1 for v in hist_vol if v < volatility) / len(hist_vol) * 100
        else:
            vol_percentile = 50.0

        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) > 50 else sma_10

        if vol_percentile > 85:
            regime = MarketRegime.HIGH_VOL
        elif sma_10 > sma_50 * 1.01 and returns_20d > 0.02:
            regime = MarketRegime.BULL
        elif sma_10 < sma_50 * 0.99 and returns_20d < -0.02:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.SIDEWAYS

        return regime, vol_percentile

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        leverage: float = 1.0,
    ) -> Tuple[int, float]:
        """Calculate position size based on risk per trade."""
        if leverage <= 0 or price <= 0 or atr <= 0:
            return 0, 0.0

        max_loss_rub = self.state.equity * (self.config.max_loss_per_trade_pct / 100)
        stop_distance = 0.4 * atr
        stop_pct = stop_distance / price

        position_value = max_loss_rub / (stop_pct * leverage)
        shares = int(position_value / price)

        max_position_value = self.state.equity * 0.10
        max_shares = int(max_position_value / price)
        shares = min(shares, max_shares)

        stop_price = price - stop_distance
        return max(0, shares), round(stop_price, 2)

    def assess_trade(
        self,
        ticker: str,
        direction: str,
        horizon: str,
        model_confidence: float,
        candles_df: pd.DataFrame,
        price: float,
        atr: float,
    ) -> RiskAssessment:
        """Full risk assessment for a potential trade."""
        warnings = []

        # Check kill-switch
        kill_active, kill_reason = self.check_kill_switch()
        if kill_active:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason=kill_reason or "Kill-switch active",
                warnings=["KILL-SWITCH ACTIVE"],
            )

        # Check horizon
        if horizon in self.DISABLED_HORIZONS:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason=f"Horizon {horizon} disabled for margin trading",
            )

        # Detect regime
        regime, vol_percentile = self.detect_regime(candles_df)

        if regime == MarketRegime.UNKNOWN:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason="Market regime unknown",
                regime=regime,
            )

        # Calculate leverage
        max_lev = self.HORIZON_MAX_LEVERAGE.get(horizon, 1.0)
        leverage = min(max_lev, 1.0)  # Conservative default

        if vol_percentile > 80:
            leverage *= 0.5
            warnings.append(f"High volatility ({vol_percentile:.0f}th percentile)")

        if self.state.consecutive_losses >= 1:
            leverage *= 0.75
            warnings.append(f"Loss streak: {self.state.consecutive_losses}")

        # Position size
        shares, stop_price = self.calculate_position_size(price, atr, leverage)

        if shares == 0:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason="Position size = 0",
                regime=regime,
            )

        position_pct = (shares * price) / self.state.equity * 100

        return RiskAssessment(
            decision=RiskDecision.ALLOW if leverage >= 1.0 else RiskDecision.RESTRICT,
            leverage=leverage,
            max_position_pct=position_pct,
            reason=f"OK: lev={leverage:.1f}x, regime={regime.value}",
            warnings=warnings,
            regime=regime,
        )

    def record_trade_result(self, pnl: float, is_win: bool, update_equity: bool = True) -> None:
        """
        Record trade result and update state.

        Args:
            pnl: Profit/loss amount
            is_win: Whether the trade was profitable
            update_equity: If True, update equity (set False if caller already updated)
        """
        # Check new day BEFORE recording trade
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.state.day_start and today_start > self.state.day_start:
            self._reset_daily()

        if update_equity:
            self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.daily_pnl += pnl

        if is_win:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        self.state.trades_today += 1

        logger.info(
            f"Trade: PnL={pnl:+.0f}, Equity={self.state.equity:,.0f}, "
            f"DD={self.state.current_drawdown_pct:.1f}%"
        )

    def _reset_daily(self) -> None:
        """Reset daily counters."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        self.state.day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if self.state.day_mode == DayMode.HALT:
            self.state.day_mode = DayMode.NORMAL
            self.state.kill_switch_active = False
            self.state.kill_switch_reason = None
            logger.info("New day: HALT mode reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current risk engine status."""
        kill_active, kill_reason = self.check_kill_switch()

        return {
            "equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "drawdown_pct": round(self.state.current_drawdown_pct, 2),
            "daily_pnl": self.state.daily_pnl,
            "daily_loss_pct": round(self.state.daily_loss_pct, 2),
            "consecutive_losses": self.state.consecutive_losses,
            "trades_today": self.state.trades_today,
            "kill_switch_active": kill_active,
            "kill_switch_reason": kill_reason,
            "day_mode": self.state.day_mode.value,
            "status": "DISABLED" if kill_active else "ACTIVE",
        }

    def reset_kill_switch(self, confirm: bool = False) -> bool:
        """Manually reset kill-switch."""
        if not confirm:
            logger.warning("Kill-switch reset requires confirm=True")
            return False

        self.state.kill_switch_active = False
        self.state.kill_switch_reason = None
        self.state.consecutive_losses = 0
        self.state.day_mode = DayMode.NORMAL

        logger.warning("KILL-SWITCH MANUALLY RESET")
        return True


def spread_bps(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """Calculate spread in basis points."""
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 10000


def pass_gatekeeper(
    p: float,
    p_threshold: float,
    turnover_5m: float,
    spread: Optional[float],
    risk: RiskParams,
) -> bool:
    """Check if signal passes risk gatekeeper."""
    if p < p_threshold:
        return False
    if turnover_5m < risk.min_turnover_rub_5m:
        return False
    if spread is not None and spread > risk.max_spread_bps:
        return False
    return True


# ============================================================================
# v2.1: Adaptive Risk Management
# ============================================================================

@dataclass
class AdaptiveRiskConfig:
    """
    v2.1 Adaptive risk configuration.

    Instead of hard kill-switches, uses smooth scaling based on performance.
    """
    # Base position sizing
    base_position_pct: float = 2.0  # 2% of equity per trade
    max_position_pct: float = 5.0  # Maximum 5% per trade
    min_position_pct: float = 0.5  # Minimum 0.5% per trade

    # Drawdown scaling
    max_drawdown_pct: float = 15.0  # At this DD, position size = min
    soft_drawdown_pct: float = 5.0  # Start reducing at this DD

    # Rolling performance window
    rolling_window: int = 20  # Number of trades for rolling metrics

    # Confidence scaling
    min_confidence: float = 0.5  # Below this, skip trade
    max_confidence: float = 0.8  # Above this, full size

    # Volatility scaling
    high_vol_percentile: float = 80.0  # Reduce size above this
    vol_reduction_factor: float = 0.5  # Multiply size by this in high vol

    # Loss streak handling (soft, not hard)
    loss_streak_reduction: float = 0.2  # Reduce by 20% per consecutive loss
    max_loss_streak_reduction: float = 0.5  # Maximum 50% reduction


class AdaptiveRiskEngine:
    """
    v2.1 Adaptive Risk Engine.

    Key differences from v2.0 RiskEngine:
    1. No hard kill-switches (no HALT/STOP modes)
    2. Smooth position sizing based on multiple factors
    3. Rolling Sharpe for performance tracking
    4. Confidence-based sizing

    Position size formula:
    size = base_size * confidence_mult * drawdown_mult * vol_mult * streak_mult
    """

    def __init__(
        self,
        initial_equity: float,
        config: Optional[AdaptiveRiskConfig] = None,
    ):
        self.config = config or AdaptiveRiskConfig()
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_pnl = 0.0

        # Trade history for rolling metrics
        self.trade_history: List[Dict] = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trades_today = 0

        # Day tracking
        self.day_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        logger.info(f"AdaptiveRiskEngine v2.1 initialized: equity={initial_equity:,.0f}")

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    @property
    def rolling_sharpe(self) -> float:
        """Rolling Sharpe ratio over last N trades."""
        if len(self.trade_history) < 5:
            return 0.0

        recent = self.trade_history[-self.config.rolling_window:]
        pnl_list = [t["pnl_pct"] for t in recent]

        mean_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list)

        if std_pnl < 1e-9:
            return 0.0

        return mean_pnl / std_pnl

    @property
    def rolling_win_rate(self) -> float:
        """Rolling win rate over last N trades."""
        if len(self.trade_history) < 5:
            return 0.5  # Neutral assumption

        recent = self.trade_history[-self.config.rolling_window:]
        wins = sum(1 for t in recent if t["pnl_pct"] > 0)
        return wins / len(recent)

    def _drawdown_multiplier(self) -> float:
        """
        Calculate position size multiplier based on drawdown.

        Linear scale from 1.0 at soft_drawdown to min_ratio at max_drawdown.
        """
        dd = self.current_drawdown_pct

        if dd <= self.config.soft_drawdown_pct:
            return 1.0

        if dd >= self.config.max_drawdown_pct:
            return self.config.min_position_pct / self.config.base_position_pct

        # Linear interpolation
        range_dd = self.config.max_drawdown_pct - self.config.soft_drawdown_pct
        progress = (dd - self.config.soft_drawdown_pct) / range_dd
        min_mult = self.config.min_position_pct / self.config.base_position_pct

        return 1.0 - progress * (1.0 - min_mult)

    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Calculate position size multiplier based on model confidence.

        Linear scale from 0 at min_confidence to 1.0 at max_confidence.
        """
        if confidence <= self.config.min_confidence:
            return 0.0

        if confidence >= self.config.max_confidence:
            return 1.0

        range_conf = self.config.max_confidence - self.config.min_confidence
        return (confidence - self.config.min_confidence) / range_conf

    def _streak_multiplier(self) -> float:
        """
        Calculate position size multiplier based on loss streak.

        Phase 1: DISABLED - always returns 1.0.
        Streak-based sizing is anti-martingale and doesn't improve edge.
        Position size should depend on signal quality, not trade history.

        Original logic (kept for reference):
        - Reduced size by loss_streak_reduction per consecutive loss
        - Capped at max_loss_streak_reduction
        """
        # Phase 1: Disable streak-based sizing
        return 1.0

    def _volatility_multiplier(self, vol_percentile: float) -> float:
        """
        Calculate position size multiplier based on volatility percentile.

        Returns vol_reduction_factor if above high_vol_percentile, else 1.0.
        """
        if vol_percentile >= self.config.high_vol_percentile:
            return self.config.vol_reduction_factor
        return 1.0

    def calculate_position_size(
        self,
        confidence: float,
        vol_percentile: float = 50.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate adaptive position size.

        Args:
            confidence: Model confidence (probability)
            vol_percentile: Current volatility percentile (0-100)

        Returns:
            (position_size_pct, breakdown_dict)
        """
        # Calculate all multipliers
        dd_mult = self._drawdown_multiplier()
        conf_mult = self._confidence_multiplier(confidence)
        streak_mult = self._streak_multiplier()
        vol_mult = self._volatility_multiplier(vol_percentile)

        # Combined multiplier
        combined = dd_mult * conf_mult * streak_mult * vol_mult

        # Calculate final size
        base = self.config.base_position_pct
        size = base * combined

        # Clamp to min/max
        size = max(self.config.min_position_pct, min(self.config.max_position_pct, size))

        # If confidence too low, return 0
        if conf_mult == 0:
            size = 0.0

        breakdown = {
            "base_pct": base,
            "dd_mult": dd_mult,
            "conf_mult": conf_mult,
            "streak_mult": streak_mult,
            "vol_mult": vol_mult,
            "combined_mult": combined,
            "final_size_pct": size,
        }

        return size, breakdown

    def should_trade(self, confidence: float) -> Tuple[bool, str]:
        """
        Determine if we should take a trade.

        v2.1: Always allows trading unless confidence is too low.
        No hard kill-switches.

        Returns:
            (should_trade, reason)
        """
        # Check confidence threshold
        if confidence < self.config.min_confidence:
            return False, f"Confidence {confidence:.2f} < {self.config.min_confidence}"

        # Check extreme drawdown (soft warning, still allows trading)
        if self.current_drawdown_pct >= self.config.max_drawdown_pct:
            logger.warning(f"High drawdown: {self.current_drawdown_pct:.1f}%")

        return True, "OK"

    def record_trade(self, pnl: float, pnl_pct: float) -> None:
        """
        Record a completed trade.

        Args:
            pnl: Absolute PnL amount
            pnl_pct: PnL as percentage of position
        """
        # Check for new day
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if today_start > self.day_start:
            self._reset_daily()

        # Update equity
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_pnl += pnl
        self.trades_today += 1

        # Update streaks
        is_win = pnl > 0
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Add to history
        self.trade_history.append({
            "timestamp": now.isoformat(),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "is_win": is_win,
            "equity": self.equity,
            "drawdown_pct": self.current_drawdown_pct,
        })

        # Keep history bounded
        max_history = self.config.rolling_window * 5
        if len(self.trade_history) > max_history:
            self.trade_history = self.trade_history[-max_history:]

        logger.info(
            f"Trade recorded: PnL={pnl:+,.0f} ({pnl_pct:+.2f}%), "
            f"Equity={self.equity:,.0f}, DD={self.current_drawdown_pct:.1f}%, "
            f"Streak={'W' if is_win else 'L'}{self.consecutive_wins if is_win else self.consecutive_losses}"
        )

    def _reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.day_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        logger.info("New trading day started")

    def get_status(self) -> Dict[str, Any]:
        """Get current risk engine status."""
        return {
            "version": "2.1-adaptive",
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": round(self.current_drawdown_pct, 2),
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "rolling_sharpe": round(self.rolling_sharpe, 2),
            "rolling_win_rate": round(self.rolling_win_rate * 100, 1),
            "total_trades": len(self.trade_history),
        }
