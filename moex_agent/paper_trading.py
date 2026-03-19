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
from .iss import fetch_candles
from .mean_reversion import (
    MR_FEATURE_COLS,
    MarketRegime,
    build_mr_features,
    check_volume_filter,
    detect_market_regime,
    is_session_warmup,
)
from .news_filter import check_news_filter, NewsFilterResult
from .storage import connect
from .telegram import send_telegram_message

logger = logging.getLogger("moex_agent.paper_trading")

# Paper trade state file
STATE_FILE = Path("data/paper_trades.json")

# Секторы для лимитирования позиций
SECTORS = {
    "BANKS": ["SBER", "VTBR", "MOEX"],
    "OIL": ["GAZP", "LKOH", "ROSN", "SIBN", "TATN", "NVTK"],
    "METALS": ["GMKN", "NLMK", "CHMF", "ALRS", "PLZL", "RUAL", "MAGN"],
}

# Тиры волатильности для порогов
TIER1 = ["SBER", "GAZP", "LKOH", "ROSN", "VTBR"]  # Порог 0.8%
TIER2 = ["NVTK", "TATN", "GMKN", "PLZL", "YDEX"]  # Порог 1.0%
# Остальные — Tier3 с порогом 1.5%

MAX_POSITIONS_PER_SECTOR = 2


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
        stop_pct: float = 0.5,  # 0.5% стоп
        p_threshold: float = 0.3,
        max_hold_minutes: int = 480,
    ):
        self.config = load_config(config_path)
        self.tickers = tickers or self.config.tickers  # Все 40 тикеров
        self.dist_threshold = dist_threshold
        self.stop_pct = stop_pct
        self.p_threshold = p_threshold
        self.max_hold_minutes = max_hold_minutes

        self.trades: List[PaperTrade] = []
        self.cooldowns: Dict[str, datetime] = {}
        self.model_package = None
        self.running = False
        self.last_daily_report: Optional[datetime] = None
        self.last_news_check: Optional[datetime] = None
        self.news_filter_result: Optional[NewsFilterResult] = None
        self.atr_cache: Dict[str, float] = {}  # ATR кэш для тикеров

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

    def _get_sector(self, secid: str) -> Optional[str]:
        """Получить сектор тикера."""
        for sector, tickers in SECTORS.items():
            if secid in tickers:
                return sector
        return None

    def _check_sector_limit(self, secid: str) -> bool:
        """Проверить лимит позиций в секторе. True = можно открывать."""
        sector = self._get_sector(secid)
        if sector is None:
            return True  # Нет сектора — нет лимита

        # Считаем открытые позиции в секторе
        open_in_sector = sum(
            1 for t in self.trades
            if t.status == "OPEN" and self._get_sector(t.secid) == sector
        )
        return open_in_sector < MAX_POSITIONS_PER_SECTOR

    def _calculate_atr(self, candles: pd.DataFrame, period: int = 14) -> float:
        """Рассчитать ATR (Average True Range)."""
        if len(candles) < period + 1:
            return 0.0

        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        close = candles["close"].astype(float)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return float(atr) if not pd.isna(atr) else 0.0

    def _get_dynamic_threshold(self, secid: str, candles: pd.DataFrame) -> float:
        """Получить динамический порог на основе ATR."""
        # Рассчитать ATR-based порог: 2 * ATR(14) / цена * 100
        atr = self._calculate_atr(candles)
        price = float(candles["close"].iloc[-1])

        if atr > 0 and price > 0:
            atr_threshold = 2 * atr / price * 100
            self.atr_cache[secid] = atr_threshold
            return atr_threshold

        # Fallback на тиры
        if secid in TIER1:
            return 0.8
        elif secid in TIER2:
            return 1.0
        else:
            return 1.5

    def _check_news(self) -> None:
        """Проверить новости каждые 5 минут."""
        now = datetime.now(timezone.utc)

        # Проверять каждые 5 минут
        if self.last_news_check and (now - self.last_news_check) < timedelta(minutes=5):
            return

        self.last_news_check = now
        self.news_filter_result = check_news_filter(send_alerts=True)

        if self.news_filter_result.max_score > 0:
            logger.info(
                f"News filter: {self.news_filter_result.level} "
                f"(score={self.news_filter_result.max_score}, "
                f"multiplier={self.news_filter_result.size_multiplier})"
            )

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

        # Проверка лимита сектора
        if not self._check_sector_limit(secid):
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
            logger.info(f"{secid}: PANIC режим — нет торговли")
            return None

        # Динамический порог на основе ATR
        threshold = self._get_dynamic_threshold(secid, candles)

        # Check distance from VWAP
        dist_pct = row["dist_vwap_pct"]
        if pd.isna(dist_pct):
            return None

        if dist_pct < -threshold:
            if regime != MarketRegime.RISK_OFF:  # No LONG in risk-off
                signal_type = "LONG"
            else:
                return None
        elif dist_pct > threshold:
            signal_type = "SHORT"
        else:
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

        status_ru = {"WIN": "ПРИБЫЛЬ", "LOSS": "СТОП", "EXPIRED": "ИСТЕКЛА"}.get(status, status)
        logger.info(f"ЗАКРЫТА: {trade.secid} {trade.direction} -> {status_ru} ({trade.pnl_pct:+.2f}%)")

        # Send Telegram notification
        if status == "WIN":
            msg = f"✅ {trade.secid} закрыта {trade.pnl_pct:+.1f}% | Вход: {trade.entry_price:.2f} → Выход: {exit_price:.2f}"
        elif status == "LOSS":
            msg = f"❌ {trade.secid} стоп {trade.pnl_pct:.1f}% | Вход: {trade.entry_price:.2f} → Выход: {exit_price:.2f}"
        else:
            msg = f"⏱️ {trade.secid} истекла {trade.pnl_pct:+.1f}% | Вход: {trade.entry_price:.2f} → Выход: {exit_price:.2f}"
        send_telegram_message(msg)

        self._save_state()

    def _send_signal_alert(self, signal: dict, trade: PaperTrade) -> None:
        """Отправка сигнала в Telegram."""
        dir_emoji = "🟢 ПОКУПКА" if signal["direction"] == "LONG" else "🔴 ПРОДАЖА"

        # Расчёт R:R
        entry = signal["entry_price"]
        target = signal["target_price"]
        stop = signal["stop_price"]
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        # Стоп в процентах
        stop_pct = (stop - entry) / entry * 100 if entry > 0 else 0

        # Номер сделки
        trade_num = len(self.trades)

        msg = (
            f"{dir_emoji} {signal['secid']} по {entry:.2f}\n"
            f"📊 VWAP: {target:.2f} | Девиация: {signal['z_score']:+.1f}%\n"
            f"🎯 Цель: {target:.2f} (возврат к VWAP)\n"
            f"🔴 Стоп: {stop:.2f} ({stop_pct:+.1f}%)\n"
            f"⚖️ R:R: {rr_ratio:.1f}:1\n"
            f"📋 Paper Trade #{trade_num}"
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

        # Проверка новостей каждые 5 минут
        self._check_news()

        # Check open trades first (всегда проверяем открытые)
        self._check_open_trades()

        # CRITICAL: остановить ВСЕ (не открываем новые)
        if self.news_filter_result and self.news_filter_result.should_stop:
            logger.warning("News filter CRITICAL — новые сделки заблокированы")
            return 0

        # HIGH: не открывать новые позиции
        if self.news_filter_result and self.news_filter_result.should_block_new:
            logger.info("News filter HIGH — новые сделки заблокированы")
            return 0

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

            dir_ru = "ПОКУПКА" if signal['direction'] == "LONG" else "ПРОДАЖА"
            logger.info(f"СИГНАЛ: {signal['secid']} {dir_ru} @ {signal['entry_price']:.2f}")
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

        logger.info(f"Paper trading запущен: {len(self.tickers)} тикеров, poll={poll_seconds}s")

        # Первая проверка новостей
        self._check_news()

        send_telegram_message(
            f"🚀 Paper Trading Запущен\n\n"
            f"Тикеров: {len(self.tickers)}\n"
            f"Стратегия: Mean Reversion к VWAP\n"
            f"Порог: динамический (2×ATR/цена)\n"
            f"Стоп: {self.stop_pct}%\n"
            f"Лимит: {MAX_POSITIONS_PER_SECTOR} поз/сектор\n"
            f"📰 News Filter: каждые 5 мин\n"
            f"Отчёт: 19:00 МСК"
        )

        cycle = 0
        while self.running:
            cycle += 1

            try:
                n_signals = self.run_cycle()

                if n_signals > 0:
                    logger.info(f"Цикл {cycle}: {n_signals} новых сигналов")
                elif cycle % 10 == 0:
                    # Status update every 10 cycles
                    open_trades = len([t for t in self.trades if t.status == "OPEN"])
                    logger.info(f"Цикл {cycle}: нет сигналов, {open_trades} открытых")

                # Проверка ежедневного отчёта в 19:00 МСК
                if self._check_daily_report_time():
                    self._send_daily_report()

            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")

            time.sleep(poll_seconds)

        # Final stats
        self.print_stats()
        send_telegram_message("⏹️ Paper Trading Остановлен")

    def _send_daily_report(self) -> None:
        """Отправка ежедневного отчёта в Telegram."""
        today = datetime.now(timezone.utc).date()
        today_str = today.strftime("%d.%m.%Y")

        # Сделки за сегодня
        today_trades = [
            t for t in self.trades
            if t.status != "OPEN" and
            datetime.fromisoformat(t.timestamp).date() == today
        ]

        wins = [t for t in today_trades if t.status == "WIN"]
        losses = [t for t in today_trades if t.status in ("LOSS", "EXPIRED")]

        total_trades = len(today_trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0

        # PnL (условный, в %)
        total_pnl_pct = sum(t.pnl_pct or 0 for t in today_trades)

        # Лучшая и худшая сделки
        best_trade = max(today_trades, key=lambda t: t.pnl_pct or -999) if today_trades else None
        worst_trade = min(today_trades, key=lambda t: t.pnl_pct or 999) if today_trades else None

        # Открытые позиции
        open_trades = [t for t in self.trades if t.status == "OPEN"]

        lines = [
            f"📊 Итоги {today_str}",
            f"Сделок: {total_trades} | Выигрыш: {win_count} | Проигрыш: {loss_count}",
            f"WR: {win_rate:.0f}% | PnL: {total_pnl_pct:+.2f}%",
        ]

        if best_trade and best_trade.pnl_pct:
            lines.append(f"Лучшая: {best_trade.secid} {best_trade.pnl_pct:+.1f}%")
        if worst_trade and worst_trade.pnl_pct:
            lines.append(f"Худшая: {worst_trade.secid} {worst_trade.pnl_pct:+.1f}%")

        if open_trades:
            lines.append("")
            lines.append("Открытые:")
            for t in open_trades[:3]:  # Макс 3 для краткости
                lines.append(f"  {t.secid} {t.direction} (вход {t.entry_price:.2f})")

        msg = "\n".join(lines)
        send_telegram_message(msg)
        logger.info("Ежедневный отчёт отправлен")

    def _check_daily_report_time(self) -> bool:
        """Проверить, пора ли отправлять ежедневный отчёт (19:00 МСК)."""
        now = datetime.now(timezone.utc)
        msk_hour = (now.hour + 3) % 24  # UTC+3

        # 19:00 МСК = 16:00 UTC
        if msk_hour == 19 and now.minute < 5:
            # Проверяем, не отправляли ли уже сегодня
            if self.last_daily_report is None or self.last_daily_report.date() != now.date():
                self.last_daily_report = now
                return True
        return False

    def print_stats(self) -> None:
        """Print paper trading statistics."""
        closed = [t for t in self.trades if t.status != "OPEN"]
        open_trades = [t for t in self.trades if t.status == "OPEN"]

        if not closed:
            print("\nНет закрытых сделок.")
            return

        wins = [t for t in closed if t.status == "WIN"]
        losses = [t for t in closed if t.status == "LOSS"]
        expired = [t for t in closed if t.status == "EXPIRED"]

        total_pnl = sum(t.pnl_pct or 0 for t in closed)
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        print("\n" + "=" * 60)
        print("PAPER TRADING СТАТИСТИКА")
        print("=" * 60)
        print(f"Всего сделок:  {len(closed)}")
        print(f"Открытых:      {len(open_trades)}")
        print(f"Выигрыш:       {len(wins)}")
        print(f"Проигрыш:      {len(losses)}")
        print(f"Истекло:       {len(expired)}")
        print(f"Win Rate:      {win_rate:.1f}%")
        print(f"Общий PnL:     {total_pnl:+.2f}%")
        if wins:
            print(f"Средний выигрыш: {sum(t.pnl_pct for t in wins)/len(wins):+.2f}%")
        if losses:
            print(f"Средний проигрыш: {sum(t.pnl_pct for t in losses)/len(losses):+.2f}%")
        print("=" * 60 + "\n")


def run_paper_trading(
    tickers: Optional[List[str]] = None,
    poll_seconds: int = 60,
):
    """Run paper trading from CLI."""
    trader = PaperTrader(tickers=tickers)
    trader.run(poll_seconds=poll_seconds)
