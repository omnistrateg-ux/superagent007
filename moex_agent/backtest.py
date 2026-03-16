"""
MOEX Agent v2 Backtester

Historical backtesting with full pipeline simulation.
Exports trades to CSV with comprehensive metrics.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .anomaly import Direction, compute_anomalies
from .config import AppConfig, load_config
from .features import FEATURE_COLS, build_feature_frame
from .labels import make_time_exit_labels
from .predictor import FEATURE_COLS, ModelRegistry
from .risk import RiskParams, pass_gatekeeper
from .signals import SignalFilter, filter_signal
from .storage import connect

logger = logging.getLogger("moex_agent.backtest")


@dataclass
class BacktestTrade:
    """Single backtest trade."""
    timestamp: str
    secid: str
    direction: str
    horizon: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    probability: float
    anomaly_score: float
    is_win: bool
    exit_reason: str


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


class Backtester:
    """
    Historical backtester using anomaly + predictor + signals pipeline.

    Simulates trading on historical data with realistic assumptions:
    - Entry at close price
    - Exit at take/stop or time-based
    - Accounts for spread and fees
    """

    def __init__(
        self,
        config: AppConfig,
        models_dir: Path = Path("./models"),
    ):
        self.config = config
        self.models = ModelRegistry(models_dir)
        self.risk_params = RiskParams(
            max_spread_bps=config.risk.max_spread_bps,
            min_turnover_rub_5m=config.risk.min_turnover_rub_5m,
        )
        self.signal_filter = SignalFilter()
        self.trades: List[BacktestTrade] = []

    def load_models(self) -> None:
        """Load ML models."""
        self.models.load()

    def _simulate_trade(
        self,
        secid: str,
        direction: Direction,
        horizon: str,
        entry_idx: int,
        entry_price: float,
        probability: float,
        anomaly_score: float,
        candles: pd.DataFrame,
        atr: float,
    ) -> Optional[BacktestTrade]:
        """
        Simulate a single trade.

        Args:
            secid: Ticker symbol
            direction: Trade direction
            horizon: Time horizon
            entry_idx: Entry index in candles
            entry_price: Entry price
            probability: Model probability
            anomaly_score: Anomaly score
            candles: Full candles DataFrame for this ticker
            atr: ATR at entry

        Returns:
            BacktestTrade or None if simulation failed
        """
        # Calculate stop/take levels
        take_atr = self.config.signals.price_exit.take_atr
        stop_atr = self.config.signals.price_exit.stop_atr

        if direction == Direction.LONG:
            take_price = entry_price + take_atr * atr
            stop_price = entry_price - stop_atr * atr
        else:
            take_price = entry_price - take_atr * atr
            stop_price = entry_price + stop_atr * atr

        # Get horizon in minutes
        horizon_minutes = next(
            (h.minutes for h in self.config.horizons if h.name == horizon),
            60,
        )

        # Simulate exit
        exit_price = entry_price
        exit_reason = "ttl"
        max_bars = min(horizon_minutes, len(candles) - entry_idx - 1)

        for i in range(1, max_bars + 1):
            if entry_idx + i >= len(candles):
                break

            bar = candles.iloc[entry_idx + i]
            high = bar["high"]
            low = bar["low"]

            if direction == Direction.LONG:
                # Check stop first
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                # Then take
                if high >= take_price:
                    exit_price = take_price
                    exit_reason = "take"
                    break
            else:
                # Check stop first
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                # Then take
                if low <= take_price:
                    exit_price = take_price
                    exit_reason = "take"
                    break

            # Time exit
            exit_price = bar["close"]

        # Calculate PnL
        if direction == Direction.LONG:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        # Apply fees
        fee_pct = self.config.fee_bps / 10000
        pnl_pct -= fee_pct

        pnl = entry_price * pnl_pct  # Per share
        is_win = pnl > 0

        return BacktestTrade(
            timestamp=str(candles.iloc[entry_idx]["ts"]),
            secid=secid,
            direction=direction.value,
            horizon=horizon,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            probability=probability,
            anomaly_score=anomaly_score,
            is_win=is_win,
            exit_reason=exit_reason,
        )

    def run(
        self,
        candles_df: pd.DataFrame,
        p_threshold: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.

        Args:
            candles_df: Historical candles DataFrame
            p_threshold: Probability threshold (default from config)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestMetrics with performance statistics
        """
        self.load_models()
        self.trades = []

        p_threshold = p_threshold or self.config.p_threshold

        # Filter by date
        df = candles_df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        if start_date:
            df = df[df["ts"] >= start_date]
        if end_date:
            df = df[df["ts"] <= end_date]

        if len(df) < 1000:
            logger.warning(f"Low data volume: {len(df)} candles")

        logger.info(f"Running backtest on {len(df):,} candles...")

        # Build features
        features_df = build_feature_frame(df)
        features_df = features_df.dropna(subset=FEATURE_COLS)

        # Process by ticker
        for secid in df["secid"].unique():
            ticker_candles = df[df["secid"] == secid].reset_index(drop=True)
            ticker_features = features_df[features_df["secid"] == secid].reset_index(drop=True)

            if len(ticker_candles) < 200:
                continue

            # Sliding window for anomaly detection
            window_size = 200  # Minimum for anomaly detection
            step_size = 60  # Check every hour

            for i in range(window_size, len(ticker_candles) - 60, step_size):
                window = ticker_candles.iloc[i - window_size:i].copy()

                # Fake quotes for anomaly detection
                last_row = window.iloc[-1]
                quotes = {
                    secid: {
                        "bid": last_row["close"] * 0.9999,
                        "ask": last_row["close"] * 1.0001,
                        "last": last_row["close"],
                    }
                }

                # Detect anomalies
                anomalies = compute_anomalies(
                    candles_1m=window[["secid", "ts", "close", "value", "volume"]],
                    quotes=quotes,
                    min_turnover_rub_5m=self.risk_params.min_turnover_rub_5m,
                    max_spread_bps=self.risk_params.max_spread_bps,
                    top_n=1,
                    min_abs_z_ret=0.8,
                )

                if not anomalies:
                    continue

                anomaly = anomalies[0]

                # Get features
                feat_idx = ticker_features[ticker_features["ts"] == last_row["ts"]]
                if feat_idx.empty:
                    continue

                try:
                    X = feat_idx[FEATURE_COLS].to_numpy(dtype=float)
                except KeyError:
                    continue

                # ML prediction
                best_h, best_p = self.models.best_horizon(X)
                if best_h is None or best_p < p_threshold:
                    continue

                # Risk gatekeeper
                if not pass_gatekeeper(
                    p=best_p,
                    p_threshold=p_threshold,
                    turnover_5m=anomaly.turnover_5m,
                    spread=anomaly.spread_bps,
                    risk=self.risk_params,
                ):
                    continue

                # Signal filter
                features_dict = {
                    col: float(feat_idx[col].iloc[0])
                    for col in FEATURE_COLS
                    if col in feat_idx.columns
                }
                features_dict["volume_spike"] = anomaly.volume_spike

                filter_passed, _ = filter_signal(anomaly, features_dict, self.signal_filter)
                if not filter_passed:
                    continue

                # Simulate trade
                entry_price = last_row["close"]
                atr = feat_idx["atr_14"].iloc[0] if "atr_14" in feat_idx.columns else entry_price * 0.01

                trade = self._simulate_trade(
                    secid=secid,
                    direction=anomaly.direction,
                    horizon=best_h,
                    entry_idx=i,
                    entry_price=entry_price,
                    probability=best_p,
                    anomaly_score=anomaly.score,
                    candles=ticker_candles,
                    atr=atr,
                )

                if trade:
                    self.trades.append(trade)

        # Calculate metrics
        return self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return BacktestMetrics()

        metrics = BacktestMetrics()
        metrics.total_trades = len(self.trades)

        wins = [t for t in self.trades if t.is_win]
        losses = [t for t in self.trades if not t.is_win]

        metrics.wins = len(wins)
        metrics.losses = len(losses)
        metrics.win_rate = len(wins) / len(self.trades) * 100

        metrics.total_pnl = sum(t.pnl for t in self.trades)
        metrics.gross_profit = sum(t.pnl for t in wins) if wins else 0
        metrics.gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        metrics.profit_factor = (
            metrics.gross_profit / metrics.gross_loss
            if metrics.gross_loss > 0
            else float("inf")
        )

        metrics.avg_win = metrics.gross_profit / len(wins) if wins else 0
        metrics.avg_loss = metrics.gross_loss / len(losses) if losses else 0

        pnl_list = [t.pnl for t in self.trades]
        metrics.max_win = max(pnl_list) if pnl_list else 0
        metrics.max_loss = min(pnl_list) if pnl_list else 0

        # Sharpe ratio (daily)
        if len(pnl_list) > 1:
            pnl_array = np.array(pnl_list)
            metrics.sharpe_ratio = (
                np.mean(pnl_array) / (np.std(pnl_array) + 1e-9) * np.sqrt(252)
            )

        # Drawdown
        cumsum = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cumsum)
        drawdown = peak - cumsum
        metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        metrics.max_drawdown_pct = (
            metrics.max_drawdown / (np.max(peak) + 1e-9) * 100
            if np.max(peak) > 0
            else 0
        )

        # Calmar ratio
        annual_return = metrics.total_pnl * 252 / len(self.trades) if self.trades else 0
        metrics.calmar_ratio = (
            annual_return / metrics.max_drawdown
            if metrics.max_drawdown > 0
            else float("inf")
        )

        # Consecutive wins/losses
        current_streak = 0
        is_winning_streak = True

        for trade in self.trades:
            if trade.is_win:
                if is_winning_streak:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning_streak = True
                metrics.max_consecutive_wins = max(
                    metrics.max_consecutive_wins, current_streak
                )
            else:
                if not is_winning_streak:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning_streak = False
                metrics.max_consecutive_losses = max(
                    metrics.max_consecutive_losses, current_streak
                )

        return metrics

    def export_trades(self, path: Path) -> None:
        """
        Export trades to CSV file.

        Args:
            path: Output CSV path
        """
        if not self.trades:
            logger.warning("No trades to export")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "timestamp",
                "secid",
                "direction",
                "horizon",
                "entry_price",
                "exit_price",
                "pnl",
                "pnl_pct",
                "probability",
                "anomaly_score",
                "is_win",
                "exit_reason",
            ])

            # Data
            for trade in self.trades:
                writer.writerow([
                    trade.timestamp,
                    trade.secid,
                    trade.direction,
                    trade.horizon,
                    f"{trade.entry_price:.4f}",
                    f"{trade.exit_price:.4f}",
                    f"{trade.pnl:.4f}",
                    f"{trade.pnl_pct:.6f}",
                    f"{trade.probability:.4f}",
                    f"{trade.anomaly_score:.4f}",
                    trade.is_win,
                    trade.exit_reason,
                ])

        logger.info(f"Exported {len(self.trades)} trades to {path}")

    def print_summary(self) -> None:
        """Print backtest summary."""
        metrics = self._calculate_metrics()

        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Total Trades:     {metrics.total_trades}")
        print(f"Wins/Losses:      {metrics.wins} / {metrics.losses}")
        print(f"Win Rate:         {metrics.win_rate:.1f}%")
        print(f"Total PnL:        {metrics.total_pnl:+,.2f}")
        print(f"Profit Factor:    {metrics.profit_factor:.2f}")
        print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown:     {metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1f}%)")
        print(f"Avg Win:          {metrics.avg_win:+,.2f}")
        print(f"Avg Loss:         {metrics.avg_loss:,.2f}")
        print(f"Max Win:          {metrics.max_win:+,.2f}")
        print(f"Max Loss:         {metrics.max_loss:,.2f}")
        print(f"Max Consec Wins:  {metrics.max_consecutive_wins}")
        print(f"Max Consec Losses:{metrics.max_consecutive_losses}")
        print("=" * 60 + "\n")


def run_backtest(
    config_path: str = "config.yaml",
    export_csv: bool = True,
) -> BacktestMetrics:
    """
    Run backtest from CLI.

    Args:
        config_path: Path to config.yaml
        export_csv: Whether to export trades to CSV

    Returns:
        BacktestMetrics
    """
    config = load_config(config_path)
    conn = connect(config.sqlite_path)

    # Load historical data
    logger.info("Loading historical data...")
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = 1
    ORDER BY secid, ts
    """
    candles = pd.read_sql_query(q, conn)
    conn.close()

    logger.info(f"Loaded {len(candles):,} candles")

    # Run backtest
    backtester = Backtester(config)
    metrics = backtester.run(candles)

    # Print summary
    backtester.print_summary()

    # Export trades
    if export_csv and backtester.trades:
        backtester.export_trades(Path("data/backtest_trades.csv"))

    return metrics
