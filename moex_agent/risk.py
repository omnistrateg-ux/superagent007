"""
MOEX Agent v2 Risk Management

Kill-switch, margin control, dynamic position sizing.

Kill-switch rules:
- 2 consecutive losses -> HALT for the day
- 2% daily loss -> HALT for the day
- 10% drawdown -> STOP trading (manual reset required)
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
