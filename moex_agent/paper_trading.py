"""
MOEX Agent v2 Paper Trading

Real-time signal generation with Telegram alerts.
NO actual trading - just paper tracking.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import load_config
from .iss import fetch_candles, fetch_quotes
from .mean_reversion import (
    MR_FEATURE_COLS,
    MarketRegime,
    build_mr_features,
    check_volume_filter,
    detect_market_regime,
    is_session_warmup,
)
from .storage import connect
from .telegram import send_telegram_message

logger = logging.getLogger("moex_agent.paper_trading")

# Paper trade state file
STATE_FILE = Path("data/paper_trades.json")


@dataclass
class PaperTrade:
    """Paper trade record."""
    id: str
    timestamp: str
    secid: str
    direction: str  # LONG or SHORT
    entry_price: float
    target_price: float  # VWAP
    stop_price: float
    probability: float
    z_score: float
    status: str = "OPEN"  # OPEN, WIN, LOSS, EXPIRED
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "secid": self.secid,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "probability": self.probability,
            "z_score": self.z_score,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "pnl_pct": self.pnl_pct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperTrade":
        return cls(**d)


class PaperTrader:
    """
    Paper trading engine for mean reversion strategy.

    Generates signals in real-time, sends to Telegram, tracks results.
    NO actual trading.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        tickers: Optional[List[str]] = None,
        dist_threshold: float = 1.0,
        stop_pct: float = 0.2,
        p_threshold: float = 0.3,
        max_hold_minutes: int = 480,
    ):
        self.config = load_config(config_path)
        self.tickers = tickers or self.config.tickers[:15]  # Default: first 15
        self.dist_threshold = dist_threshold
        self.stop_pct = stop_pct
        self.p_threshold = p_threshold
        self.max_hold_minutes = max_hold_minutes

        self.trades: List[PaperTrade] = []
        self.cooldowns: Dict[str, datetime] = {}
        self.model_package = None
        self.running = False

        # Load model
        self._load_model()
        self._load_state()

    def _load_model(self) -> None:
        """Load mean reversion model."""
        import joblib

        model_path = Path("models/model_mr.joblib")
        if model_path.exists():
            self.model_package = joblib.load(model_path)
            logger.info("Loaded MR model")
        else:
            logger.warning("MR model not found! Run 'python -m moex_agent mr' first.")

    def _load_state(self) -> None:
        """Load paper trades from disk."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.trades = [PaperTrade.from_dict(t) for t in data.get("trades", [])]
                logger.info(f"Loaded {len(self.trades)} paper trades from state")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save paper trades to disk."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"trades": [t.to_dict() for t in self.trades]}
        STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _fetch_recent_candles(self, secid: str, minutes: int = 200) -> pd.DataFrame:
        """Fetch recent candles for a ticker."""
        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        till_date = now.strftime("%Y-%m-%d")

        try:
            candles = fetch_candles(
                self.config.engine,
                self.config.market,
                self.config.board,
                secid,
                interval=1,
                from_date=from_date,
                till_date=till_date,
            )

            if not candles:
                return pd.DataFrame()

            df = pd.DataFrame([
                {
                    "secid": secid,
                    "ts": c.ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "value": c.value,
                }
                for c in candles
            ])

            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            return df.tail(minutes)

        except Exception as e:
            logger.warning(f"Failed to fetch candles for {secid}: {e}")
            return pd.DataFrame()

    def _check_signal(self, secid: str, candles: pd.DataFrame) -> Optional[dict]:
        """
        Check for mean reversion signal.

        Returns:
            Signal dict or None
        """
        if len(candles) < 100:
            return None

        # Build features
        features = build_mr_features(candles)
        if features.empty:
            return None

        features = features.dropna(subset=MR_FEATURE_COLS)
        if features.empty:
            return None

        row = features.iloc[-1]
        ts = row["ts"]

        # Check session warmup
        if is_session_warmup(ts):
            return None

        # Check volume
        vol_spike = row.get("volume_spike", 1.0)
        if vol_spike < 0.5:
            return None

        # Check market regime
        regime = detect_market_regime(candles)
        if regime == MarketRegime.PANIC:
            logger.info(f"{secid}: PANIC regime - no trading")
            return None

        # Check distance from VWAP
        dist_pct = row["dist_vwap_pct"]
        if pd.isna(dist_pct):
            return None

        signal_type = None
        if dist_pct < -self.dist_threshold:
            if regime != MarketRegime.RISK_OFF:  # No LONG in risk-off
                signal_type = "LONG"
        elif dist_pct > self.dist_threshold:
            signal_type = "SHORT"

        if signal_type is None:
            return None

        # Model prediction
        if self.model_package is None:
            prob = 0.5  # No model - use neutral probability
        else:
            try:
                X = row[MR_FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
                X_scaled = self.model_package["scaler"].transform(X)
                prob = self.model_package["model"].predict_proba(X_scaled)[0, 1]
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                prob = 0.5

        if prob < self.p_threshold:
            return None

        # Calculate targets
        entry_price = row["close"]
        vwap = row["vwap"]

        if signal_type == "LONG":
            target_price = vwap
            stop_price = entry_price * (1 - self.stop_pct / 100)
        else:
            target_price = vwap
            stop_price = entry_price * (1 + self.stop_pct / 100)

        return {
            "secid": secid,
            "direction": signal_type,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "probability": prob,
            "z_score": dist_pct,
            "regime": regime,
        }

    def _create_trade(self, signal: dict) -> PaperTrade:
        """Create a new paper trade from signal."""
        now = datetime.now(timezone.utc)
        trade_id = f"{signal['secid']}_{now.strftime('%Y%m%d_%H%M%S')}"

        trade = PaperTrade(
            id=trade_id,
            timestamp=now.isoformat(),
            secid=signal["secid"],
            direction=signal["direction"],
            entry_price=signal["entry_price"],
            target_price=signal["target_price"],
            stop_price=signal["stop_price"],
            probability=signal["probability"],
            z_score=signal["z_score"],
        )

        self.trades.append(trade)
        self.cooldowns[signal["secid"]] = now + timedelta(minutes=self.max_hold_minutes)

        return trade

    def _check_open_trades(self) -> None:
        """Check and update open trades."""
        now = datetime.now(timezone.utc)

        for trade in self.trades:
            if trade.status != "OPEN":
                continue

            # Check expiry
            trade_time = datetime.fromisoformat(trade.timestamp)
            if now - trade_time > timedelta(minutes=self.max_hold_minutes):
                # Fetch current price
                candles = self._fetch_recent_candles(trade.secid, minutes=5)
                if not candles.empty:
                    current_price = candles.iloc[-1]["close"]
                    self._close_trade(trade, current_price, "EXPIRED")
                continue

            # Fetch current price
            candles = self._fetch_recent_candles(trade.secid, minutes=5)
            if candles.empty:
                continue

            current_price = candles.iloc[-1]["close"]

            if trade.direction == "LONG":
                if current_price <= trade.stop_price:
                    self._close_trade(trade, current_price, "LOSS")
                elif current_price >= trade.target_price:
                    self._close_trade(trade, current_price, "WIN")
            else:  # SHORT
                if current_price >= trade.stop_price:
                    self._close_trade(trade, current_price, "LOSS")
                elif current_price <= trade.target_price:
                    self._close_trade(trade, current_price, "WIN")

    def _close_trade(self, trade: PaperTrade, exit_price: float, status: str) -> None:
        """Close a paper trade."""
        trade.status = status
        trade.exit_price = exit_price
        trade.exit_time = datetime.now(timezone.utc).isoformat()

        if trade.direction == "LONG":
            trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100

        # Remove fee
        trade.pnl_pct -= self.config.fee_bps / 100

        logger.info(f"CLOSED: {trade.secid} {trade.direction} -> {status} ({trade.pnl_pct:+.2f}%)")

        # Send Telegram notification
        emoji = "✅" if status == "WIN" else "❌" if status == "LOSS" else "⏱️"
        msg = (
            f"{emoji} *PAPER TRADE CLOSED*\n\n"
            f"*{trade.secid}* {trade.direction}\n"
            f"Entry: {trade.entry_price:.2f}\n"
            f"Exit: {exit_price:.2f}\n"
            f"Result: *{status}* ({trade.pnl_pct:+.2f}%)"
        )
        send_telegram_message(msg)

        self._save_state()

    def _send_signal_alert(self, signal: dict, trade: PaperTrade) -> None:
        """Send Telegram alert for new signal."""
        emoji = "📈" if signal["direction"] == "LONG" else "📉"
        regime_emoji = {
            MarketRegime.NORMAL: "🟢",
            MarketRegime.RISK_OFF: "🟡",
            MarketRegime.PANIC: "🔴",
        }.get(signal["regime"], "⚪")

        msg = (
            f"{emoji} *PAPER SIGNAL*\n\n"
            f"*{signal['secid']}* {signal['direction']}\n"
            f"Entry: {signal['entry_price']:.2f}\n"
            f"Target (VWAP): {signal['target_price']:.2f}\n"
            f"Stop: {signal['stop_price']:.2f}\n"
            f"Z-score: {signal['z_score']:.2f}%\n"
            f"Probability: {signal['probability']:.1%}\n"
            f"Regime: {regime_emoji} {signal['regime']}\n\n"
            f"_Paper trade - NO real execution_"
        )
        send_telegram_message(msg)

    def run_cycle(self) -> int:
        """
        Run one cycle of signal checking.

        Returns:
            Number of new signals generated
        """
        now = datetime.now(timezone.utc)
        new_signals = 0

        # Check open trades first
        self._check_open_trades()

        # Check for new signals
        for secid in self.tickers:
            # Cooldown check
            if secid in self.cooldowns and now < self.cooldowns[secid]:
                continue

            # Fetch candles
            candles = self._fetch_recent_candles(secid, minutes=200)
            if candles.empty:
                continue

            # Check for signal
            signal = self._check_signal(secid, candles)
            if signal is None:
                continue

            # Create paper trade
            trade = self._create_trade(signal)
            self._send_signal_alert(signal, trade)
            self._save_state()

            logger.info(f"NEW SIGNAL: {signal['secid']} {signal['direction']} @ {signal['entry_price']:.2f}")
            new_signals += 1

        return new_signals

    def run(self, poll_seconds: int = 60) -> None:
        """
        Run paper trading loop.

        Args:
            poll_seconds: Seconds between cycles
        """
        self.running = True

        def handle_signal(signum, frame):
            self.running = False
            logger.info("Shutdown requested...")

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logger.info(f"Starting paper trading: {len(self.tickers)} tickers, poll={poll_seconds}s")
        send_telegram_message(
            f"🚀 *Paper Trading Started*\n\n"
            f"Tickers: {len(self.tickers)}\n"
            f"Strategy: Mean Reversion to VWAP\n"
            f"Threshold: {self.dist_threshold}%\n"
            f"Stop: {self.stop_pct}%"
        )

        cycle = 0
        while self.running:
            cycle += 1

            try:
                n_signals = self.run_cycle()

                if n_signals > 0:
                    logger.info(f"Cycle {cycle}: {n_signals} new signals")
                elif cycle % 10 == 0:
                    # Status update every 10 cycles
                    open_trades = len([t for t in self.trades if t.status == "OPEN"])
                    logger.info(f"Cycle {cycle}: no signals, {open_trades} open trades")

            except Exception as e:
                logger.error(f"Cycle error: {e}")

            time.sleep(poll_seconds)

        # Final stats
        self.print_stats()
        send_telegram_message("⏹️ *Paper Trading Stopped*")

    def print_stats(self) -> None:
        """Print paper trading statistics."""
        closed = [t for t in self.trades if t.status != "OPEN"]
        open_trades = [t for t in self.trades if t.status == "OPEN"]

        if not closed:
            print("\nNo closed trades yet.")
            return

        wins = [t for t in closed if t.status == "WIN"]
        losses = [t for t in closed if t.status == "LOSS"]
        expired = [t for t in closed if t.status == "EXPIRED"]

        total_pnl = sum(t.pnl_pct or 0 for t in closed)
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        print("\n" + "=" * 60)
        print("PAPER TRADING STATS")
        print("=" * 60)
        print(f"Total Trades:  {len(closed)}")
        print(f"Open Trades:   {len(open_trades)}")
        print(f"Wins:          {len(wins)}")
        print(f"Losses:        {len(losses)}")
        print(f"Expired:       {len(expired)}")
        print(f"Win Rate:      {win_rate:.1f}%")
        print(f"Total PnL:     {total_pnl:+.2f}%")
        if wins:
            print(f"Avg Win:       {sum(t.pnl_pct for t in wins)/len(wins):+.2f}%")
        if losses:
            print(f"Avg Loss:      {sum(t.pnl_pct for t in losses)/len(losses):+.2f}%")
        print("=" * 60 + "\n")


def run_paper_trading(
    tickers: Optional[List[str]] = None,
    poll_seconds: int = 60,
):
    """Run paper trading from CLI."""
    trader = PaperTrader(tickers=tickers)
    trader.run(poll_seconds=poll_seconds)
