"""
MOEX Agent v2 Paper/Margin Trader

Virtual account trading with leverage support.
Uses MarginRiskEngine for position sizing and kill-switch.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anomaly import Direction
from .config import AppConfig, load_config
from .engine import PipelineEngine, Signal
from .iss import close_session
from .risk import RiskEngine, KillSwitchConfig, RiskDecision
from .storage import connect, save_trade
from .telegram import send_signal_alert, send_trade_result, send_kill_switch_alert

logger = logging.getLogger("moex_agent.trader")


@dataclass
class Position:
    """Open position."""
    secid: str
    direction: Direction
    entry_price: float
    shares: int
    stop_price: float
    take_price: float
    horizon: str
    entry_time: datetime
    ttl_minutes: int


@dataclass
class Trade:
    """Completed trade."""
    secid: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str  # 'take', 'stop', 'ttl', 'manual'


@dataclass
class TraderState:
    """Trader state for persistence."""
    equity: float
    positions: List[Position] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    daily_pnl: float = 0.0
    trades_today: int = 0


class Trader:
    """
    Paper/Margin trader with virtual account.

    Features:
    - Virtual account with configurable starting equity
    - Leverage support via RiskEngine
    - Position tracking with stop/take levels
    - Trade history and statistics
    - State persistence to JSON
    - Telegram notifications
    """

    def __init__(
        self,
        config: AppConfig,
        initial_equity: float = 1_000_000,
        leverage: float = 1.0,
        state_path: Optional[Path] = None,
    ):
        self.config = config
        self.leverage = leverage
        self.state_path = state_path or Path("data/trader_state.json")

        # Initialize risk engine
        kill_config = KillSwitchConfig(
            max_loss_per_trade_pct=config.risk.max_loss_per_trade_pct,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_consecutive_losses=config.risk.max_consecutive_losses,
            max_drawdown_pct=config.risk.max_drawdown_pct,
        )
        self.risk_engine = RiskEngine(initial_equity, kill_config)

        # Initialize pipeline engine
        self.pipeline = PipelineEngine(config)
        self.pipeline.load_models()

        # State
        self.state = TraderState(equity=initial_equity)
        self.cooldown_map: Dict[str, datetime] = defaultdict(
            lambda: datetime(1970, 1, 1, tzinfo=timezone.utc)
        )

        # Load saved state if exists
        self._load_state()

        logger.info(f"Trader initialized: equity={initial_equity:,.0f}, leverage={leverage}x")

    def _load_state(self) -> None:
        """Load state from JSON file."""
        if not self.state_path.exists():
            return

        try:
            data = json.loads(self.state_path.read_text())
            self.state.equity = data.get("equity", self.state.equity)
            self.state.daily_pnl = data.get("daily_pnl", 0.0)
            self.state.trades_today = data.get("trades_today", 0)
            self.risk_engine.state.equity = self.state.equity
            self.risk_engine.state.peak_equity = data.get("peak_equity", self.state.equity)
            logger.info(f"Loaded trader state: equity={self.state.equity:,.0f}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to JSON file."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "equity": self.state.equity,
                "peak_equity": self.risk_engine.state.peak_equity,
                "daily_pnl": self.state.daily_pnl,
                "trades_today": self.state.trades_today,
                "positions": [
                    {
                        "secid": p.secid,
                        "direction": p.direction.value,
                        "entry_price": p.entry_price,
                        "shares": p.shares,
                        "stop_price": p.stop_price,
                        "take_price": p.take_price,
                        "horizon": p.horizon,
                        "entry_time": p.entry_time.isoformat(),
                        "ttl_minutes": p.ttl_minutes,
                    }
                    for p in self.state.positions
                ],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _calculate_position_size(
        self,
        price: float,
        atr: float,
        direction: Direction,
    ) -> tuple[int, float, float]:
        """
        Calculate position size based on risk.

        Returns:
            (shares, stop_price, take_price)
        """
        # Max loss per trade
        max_loss = self.state.equity * (self.config.risk.max_loss_per_trade_pct / 100)

        # Stop distance
        stop_atr = self.config.signals.price_exit.stop_atr
        take_atr = self.config.signals.price_exit.take_atr
        stop_distance = stop_atr * atr

        # Position size
        risk_per_share = stop_distance
        if risk_per_share <= 0:
            return 0, 0, 0

        shares = int(max_loss / risk_per_share)

        # Apply leverage cap
        max_position = self.state.equity * self.leverage * 0.2  # Max 20% per position
        max_shares = int(max_position / price)
        shares = min(shares, max_shares)

        # Calculate stop/take prices
        if direction == Direction.LONG:
            stop_price = price - stop_distance
            take_price = price + take_atr * atr
        else:
            stop_price = price + stop_distance
            take_price = price - take_atr * atr

        return shares, round(stop_price, 2), round(take_price, 2)

    def _check_exit_conditions(
        self,
        position: Position,
        current_price: float,
    ) -> tuple[bool, str, float]:
        """
        Check if position should be exited.

        Returns:
            (should_exit, reason, exit_price)
        """
        now = datetime.now(timezone.utc)

        # TTL check
        time_elapsed = (now - position.entry_time).total_seconds() / 60
        if time_elapsed >= position.ttl_minutes:
            return True, "ttl", current_price

        # Stop/Take check
        if position.direction == Direction.LONG:
            if current_price <= position.stop_price:
                return True, "stop", position.stop_price
            if current_price >= position.take_price:
                return True, "take", position.take_price
        else:
            if current_price >= position.stop_price:
                return True, "stop", position.stop_price
            if current_price <= position.take_price:
                return True, "take", position.take_price

        return False, "", 0

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: str,
        conn,
    ) -> Trade:
        """Close position and record trade."""
        now = datetime.now(timezone.utc)

        # Calculate PnL
        if position.direction == Direction.LONG:
            pnl_per_share = exit_price - position.entry_price
        else:
            pnl_per_share = position.entry_price - exit_price

        pnl = pnl_per_share * position.shares
        pnl_pct = pnl_per_share / position.entry_price

        # Create trade record
        trade = Trade(
            secid=position.secid,
            direction=position.direction.value,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=position.entry_time,
            exit_time=now,
            exit_reason=exit_reason,
        )

        # Update state
        self.state.equity += pnl
        self.state.daily_pnl += pnl
        self.state.trades_today += 1
        self.state.trades.append(trade)

        # Update risk engine (equity already updated above, so update_equity=False)
        is_win = pnl > 0
        self.risk_engine.record_trade_result(pnl, is_win, update_equity=False)

        # Save to database
        save_trade(
            conn,
            secid=position.secid,
            direction=position.direction.value,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            status=exit_reason,
            horizon=position.horizon,
        )

        # Telegram notification
        if self.config.telegram.enabled:
            send_trade_result(
                self.config.telegram.bot_token or "",
                self.config.telegram.chat_id or "",
                ticker=position.secid,
                direction=position.direction.value,
                pnl=pnl,
                pnl_pct=pnl_pct,
                equity=self.state.equity,
                is_stop=(exit_reason == "stop"),
            )

        logger.info(
            f"CLOSED: {position.secid} {position.direction.value} "
            f"PnL={pnl:+,.0f} ({pnl_pct:+.2%}) reason={exit_reason}"
        )

        return trade

    def _open_position(
        self,
        signal: Signal,
        shares: int,
        stop_price: float,
        take_price: float,
    ) -> Position:
        """Open new position."""
        position = Position(
            secid=signal.secid,
            direction=signal.direction,
            entry_price=signal.entry or 0,
            shares=shares,
            stop_price=stop_price,
            take_price=take_price,
            horizon=signal.horizon,
            entry_time=datetime.now(timezone.utc),
            ttl_minutes=signal.ttl_minutes or 60,
        )

        self.state.positions.append(position)
        self.cooldown_map[signal.secid] = datetime.now(timezone.utc)

        logger.info(
            f"OPENED: {signal.secid} {signal.direction.value} "
            f"@ {signal.entry:.2f} x{shares} stop={stop_price:.2f} take={take_price:.2f}"
        )

        return position

    def run_cycle(self, conn) -> Dict[str, Any]:
        """
        Run one trading cycle.

        1. Check and close existing positions
        2. Check kill-switch
        3. Get new signals
        4. Open new positions

        Returns:
            Cycle statistics
        """
        stats = {
            "positions_checked": 0,
            "positions_closed": 0,
            "signals_received": 0,
            "positions_opened": 0,
            "kill_switch": False,
        }

        # 1. Check existing positions
        quotes = self.pipeline.fetch_quotes_parallel()

        positions_to_close = []
        for position in self.state.positions:
            stats["positions_checked"] += 1

            quote = quotes.get(position.secid, {})
            current_price = quote.get("last")

            if current_price is None:
                continue

            should_exit, reason, exit_price = self._check_exit_conditions(
                position, current_price
            )

            if should_exit:
                positions_to_close.append((position, exit_price, reason))

        # Close positions
        for position, exit_price, reason in positions_to_close:
            self._close_position(position, exit_price, reason, conn)
            self.state.positions.remove(position)
            stats["positions_closed"] += 1

        # 2. Check kill-switch
        kill_active, kill_reason = self.risk_engine.check_kill_switch()
        if kill_active:
            stats["kill_switch"] = True
            logger.warning(f"Kill-switch active: {kill_reason}")

            if self.config.telegram.enabled:
                send_kill_switch_alert(
                    self.config.telegram.bot_token or "",
                    self.config.telegram.chat_id or "",
                    reason=kill_reason or "",
                    equity=self.state.equity,
                    drawdown_pct=self.risk_engine.state.current_drawdown_pct,
                )

            self._save_state()
            return stats

        # 3. Get new signals
        result = self.pipeline.run_cycle(conn, cooldown_map=self.cooldown_map)
        stats["signals_received"] = len(result.signals)

        # 4. Open new positions
        for signal in result.signals:
            # Skip if already have position in this ticker
            if any(p.secid == signal.secid for p in self.state.positions):
                continue

            # Skip if no entry price or ATR
            if not signal.entry:
                continue

            # Get ATR from features (approximate from spread)
            atr = signal.entry * 0.01  # Fallback: 1% of price

            # Calculate position size
            shares, stop_price, take_price = self._calculate_position_size(
                signal.entry, atr, signal.direction
            )

            if shares <= 0:
                continue

            # Risk assessment
            assessment = self.risk_engine.assess_trade(
                ticker=signal.secid,
                direction=signal.direction.value,
                horizon=signal.horizon,
                model_confidence=signal.probability,
                candles_df=self.pipeline.fetch_candles_parallel(
                    (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
                    datetime.now(timezone.utc).date().isoformat(),
                ).get(signal.secid, []),
                price=signal.entry,
                atr=atr,
            )

            if assessment.decision == RiskDecision.DISABLE:
                logger.debug(f"Risk rejected {signal.secid}: {assessment.reason}")
                continue

            # Open position
            self._open_position(signal, shares, stop_price, take_price)
            stats["positions_opened"] += 1

            # Telegram notification
            if self.config.telegram.enabled:
                send_signal_alert(
                    self.config.telegram.bot_token or "",
                    self.config.telegram.chat_id or "",
                    ticker=signal.secid,
                    direction=signal.direction.value,
                    horizon=signal.horizon,
                    p=signal.probability,
                    score=signal.anomaly_score,
                    entry=signal.entry,
                    take=take_price,
                    stop=stop_price,
                    volume_spike=signal.volume_spike,
                )

        self._save_state()
        return stats

    def run(self, max_cycles: Optional[int] = None) -> None:
        """
        Run trading loop.

        Args:
            max_cycles: Optional max cycles (None = infinite)
        """
        conn = connect(self.config.sqlite_path)
        shutdown_requested = False
        cycle_count = 0

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            logger.info("Shutdown signal received...")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Trader loop started")
        logger.info(f"Equity: {self.state.equity:,.0f} | Leverage: {self.leverage}x")

        try:
            while not shutdown_requested:
                cycle_count += 1

                if max_cycles and cycle_count > max_cycles:
                    break

                try:
                    stats = self.run_cycle(conn)

                    if cycle_count % 12 == 0:  # Every ~1 min
                        logger.info(
                            f"Cycle {cycle_count}: "
                            f"positions={len(self.state.positions)} "
                            f"equity={self.state.equity:,.0f} "
                            f"daily_pnl={self.state.daily_pnl:+,.0f}"
                        )

                    if stats["kill_switch"]:
                        logger.warning("Kill-switch active, pausing...")
                        time.sleep(60)
                        continue

                    time.sleep(self.config.poll_seconds)

                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    time.sleep(10)

        finally:
            conn.close()
            close_session()
            self._save_state()
            logger.info(
                f"Trader stopped. Cycles: {cycle_count}, "
                f"Final equity: {self.state.equity:,.0f}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        trades = self.state.trades

        if not trades:
            return {"trades": 0}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "avg_win": gross_profit / len(wins) if wins else 0,
            "avg_loss": gross_loss / len(losses) if losses else 0,
            "equity": self.state.equity,
            "drawdown_pct": self.risk_engine.state.current_drawdown_pct,
        }


def run_paper_trading(
    config_path: str = "config.yaml",
    equity: float = 1_000_000,
    leverage: float = 1.0,
) -> None:
    """Run paper trading mode."""
    config = load_config(config_path)
    trader = Trader(config, initial_equity=equity, leverage=leverage)
    trader.run()


def run_margin_trading(
    config_path: str = "config.yaml",
    equity: float = 1_000_000,
    leverage: float = 3.0,
) -> None:
    """Run margin trading mode."""
    config = load_config(config_path)
    trader = Trader(
        config,
        initial_equity=equity,
        leverage=leverage,
        state_path=Path("data/margin_trader_state.json"),
    )
    trader.run()
