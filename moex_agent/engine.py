"""
MOEX Agent v2 Pipeline Engine

Core signal generation pipeline:
1. Fetch candles and quotes (parallel)
2. Detect anomalies (MAD z-score)
3. Build features (30 indicators)
4. Predict probabilities (ML + Isotonic)
5. Apply risk gatekeeper
6. Apply rule-based filters
7. Return signal candidates
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .anomaly import AnomalyResult, Direction, compute_anomalies
from .config import AppConfig
from .features import FEATURE_COLS, build_feature_frame
from .iss import fetch_candles, fetch_quote
from .market_context import MarketContext, fetch_market_context, should_skip_by_context
from .multi_timeframe import analyze_trend, check_trend_alignment
from .news_filter import NewsFilterResult, check_news_filter
from .predictor import ModelRegistry
from .risk import RiskParams, pass_gatekeeper
from .signals import SignalFilter, filter_signal
from .storage import get_window, upsert_many

logger = logging.getLogger("moex_agent.engine")


@dataclass
class Signal:
    """Generated trading signal."""
    secid: str
    direction: Direction
    horizon: str
    probability: float
    signal_type: str
    entry: Optional[float] = None
    take: Optional[float] = None
    stop: Optional[float] = None
    ttl_minutes: Optional[int] = None
    anomaly_score: float = 0.0
    z_ret_5m: float = 0.0
    z_vol_5m: float = 0.0
    ret_5m: float = 0.0
    turnover_5m: float = 0.0
    spread_bps: Optional[float] = None
    volume_spike: float = 1.0
    filter_passed: bool = True
    filter_reasons: List[str] = field(default_factory=list)
    trend_aligned: bool = True
    trend_reason: str = ""
    context_skipped: bool = False
    context_reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.secid,
            "direction": self.direction.value if isinstance(self.direction, Direction) else self.direction,
            "horizon": self.horizon,
            "p": round(self.probability, 4),
            "signal_type": self.signal_type,
            "entry": self.entry,
            "take": self.take,
            "stop": self.stop,
            "ttl_minutes": self.ttl_minutes,
            "anomaly": {
                "score": round(self.anomaly_score, 3),
                "z_ret_5m": round(self.z_ret_5m, 3),
                "z_vol_5m": round(self.z_vol_5m, 3),
                "ret_5m": round(self.ret_5m, 5),
                "turnover_5m": int(self.turnover_5m),
                "spread_bps": None if self.spread_bps is None else round(self.spread_bps, 1),
                "volume_spike": round(self.volume_spike, 2),
            },
            "filter_passed": self.filter_passed,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CycleResult:
    """Result of a single pipeline cycle."""
    signals: List[Signal]
    anomalies_count: int
    candles_fetched: int
    quotes_fetched: int
    duration_ms: float
    errors: List[str] = field(default_factory=list)


class PipelineEngine:
    """
    Core signal generation engine.

    Implements 4-level confirmation:
    1. Anomaly detection
    2. ML prediction
    3. Risk gatekeeper
    4. Rule-based filters
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

    def load_models(self) -> None:
        """Pre-load ML models."""
        self.models.load()

    def fetch_candles_parallel(
        self,
        from_date: str,
        till_date: str,
    ) -> Dict[str, List]:
        """Fetch candles for all tickers in parallel."""
        results = {}

        def fetch_one(secid: str):
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
                return secid, candles, None
            except Exception as e:
                return secid, [], e

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(fetch_one, s): s for s in self.config.tickers}
            for future in as_completed(futures):
                secid, candles, err = future.result()
                if err:
                    logger.warning(f"Failed to fetch {secid}: {err}")
                results[secid] = candles

        return results

    def fetch_quotes_parallel(self) -> Dict[str, Dict]:
        """Fetch quotes for all tickers in parallel."""
        results = {}

        def fetch_one(secid: str):
            try:
                q = fetch_quote(
                    self.config.engine,
                    self.config.market,
                    self.config.board,
                    secid,
                )
                return secid, q, None
            except Exception as e:
                return secid, {"secid": secid}, e

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(fetch_one, s): s for s in self.config.tickers}
            for future in as_completed(futures):
                secid, quote, err = future.result()
                if err:
                    logger.warning(f"Failed to fetch quote {secid}: {err}")
                results[secid] = quote

        return results

    def detect_anomalies(
        self,
        candles_df: pd.DataFrame,
        quotes: Dict[str, Dict],
    ) -> List[AnomalyResult]:
        """Detect price/volume anomalies."""
        return compute_anomalies(
            candles_1m=candles_df[["secid", "ts", "close", "value", "volume"]],
            quotes=quotes,
            min_turnover_rub_5m=self.risk_params.min_turnover_rub_5m,
            max_spread_bps=self.risk_params.max_spread_bps,
            top_n=self.config.top_n_anomalies,
        )

    def generate_signals(
        self,
        anomalies: List[AnomalyResult],
        features_df: pd.DataFrame,
        quotes: Dict[str, Dict],
        cooldown_map: Optional[Dict[str, datetime]] = None,
        market_ctx: Optional[MarketContext] = None,
        candles_df: Optional[pd.DataFrame] = None,
    ) -> List[Signal]:
        """
        Generate signals from anomalies.

        4-level confirmation:
        1. Anomaly detected (already done)
        2. ML probability > threshold
        3. Risk gatekeeper (spread, turnover)
        4. Rule-based filters (RSI, MACD, etc.)
        """
        self.models.ensure_loaded()

        if cooldown_map is None:
            cooldown_map = {}

        signals = []
        now = datetime.now(timezone.utc)
        cooldown_td = timedelta(minutes=self.config.cooldown_minutes)

        latest = features_df.sort_values(["secid", "ts"]).groupby("secid").tail(1)

        # Safe minimum datetime for cooldown comparison
        _MIN_DATETIME = datetime(1970, 1, 1, tzinfo=timezone.utc)

        for anomaly in anomalies:
            secid = anomaly.secid

            # Cooldown check
            last_alert = cooldown_map.get(secid, _MIN_DATETIME)
            if (now - last_alert) < cooldown_td:
                continue

            # Get features
            row = latest[latest["secid"] == secid]
            if row.empty:
                continue

            try:
                X = row[FEATURE_COLS].to_numpy(dtype=float)
            except KeyError as e:
                logger.warning(f"Missing feature for {secid}: {e}")
                continue

            # Level 2: ML prediction
            best_h, best_p = self.models.best_horizon(X)
            if best_h is None:
                continue

            # Level 3: Risk gatekeeper
            if not pass_gatekeeper(
                p=best_p,
                p_threshold=self.config.p_threshold,
                turnover_5m=anomaly.turnover_5m,
                spread=anomaly.spread_bps,
                risk=self.risk_params,
            ):
                continue

            # Level 4: Rule-based filters
            features_dict = {col: float(row[col].iloc[0]) for col in FEATURE_COLS if col in row.columns}
            features_dict["volume_spike"] = anomaly.volume_spike
            filter_passed, filter_reasons = filter_signal(anomaly, features_dict, self.signal_filter)

            # Level 5: Market context check
            context_skipped = False
            context_reason = ""
            if market_ctx is not None:
                context_skipped, context_reason = should_skip_by_context(
                    secid, anomaly.direction.value, market_ctx
                )
                if context_skipped:
                    continue

            # Level 6: Trend alignment check
            trend_aligned = True
            trend_reason = ""
            if candles_df is not None and len(candles_df) > 0:
                ticker_candles = candles_df[candles_df["secid"] == secid]
                if len(ticker_candles) >= 50:
                    trend_state = analyze_trend(secid, ticker_candles)
                    trend_aligned, trend_reason = check_trend_alignment(
                        trend_state, anomaly.direction.value
                    )
                    # Don't skip on misalignment, but record it
                    if not trend_aligned:
                        filter_reasons.append(f"Trend: {trend_reason}")

            # Compute price targets
            last_price = quotes.get(secid, {}).get("last")
            atr = float(row["atr_14"].iloc[0]) if "atr_14" in row.columns else None

            entry = float(last_price) if last_price else None
            take = None
            stop = None
            signal_type = "time-exit"

            price_exit_cfg = self.config.signals.price_exit
            if price_exit_cfg.enabled and last_price and atr and atr > 0:
                take_atr = price_exit_cfg.take_atr
                stop_atr = price_exit_cfg.stop_atr

                if anomaly.direction == Direction.LONG:
                    take = float(last_price + take_atr * atr)
                    stop = float(last_price - stop_atr * atr)
                else:
                    take = float(last_price - take_atr * atr)
                    stop = float(last_price + stop_atr * atr)

                signal_type = "price-exit"

            ttl = next(
                (h.minutes for h in self.config.horizons if h.name == best_h),
                60,
            )

            signal = Signal(
                secid=secid,
                direction=anomaly.direction,
                horizon=best_h,
                probability=best_p,
                signal_type=signal_type,
                entry=entry,
                take=take,
                stop=stop,
                ttl_minutes=ttl,
                anomaly_score=anomaly.score,
                z_ret_5m=anomaly.z_ret_5m,
                z_vol_5m=anomaly.z_vol_5m,
                ret_5m=anomaly.ret_5m,
                turnover_5m=anomaly.turnover_5m,
                spread_bps=anomaly.spread_bps,
                volume_spike=anomaly.volume_spike,
                filter_passed=filter_passed,
                filter_reasons=filter_reasons,
                trend_aligned=trend_aligned,
                trend_reason=trend_reason,
                context_skipped=context_skipped,
                context_reason=context_reason,
            )

            # Only include signals that pass all filters
            if filter_passed:
                signals.append(signal)

        return signals

    def run_cycle(
        self,
        conn,
        candles_df: Optional[pd.DataFrame] = None,
        quotes: Optional[Dict[str, Dict]] = None,
        cooldown_map: Optional[Dict[str, datetime]] = None,
    ) -> CycleResult:
        """
        Run a single pipeline cycle.

        This is the main entry point for signal generation.
        """
        start = time.perf_counter()
        errors = []

        # Pre-flight checks: news and market context
        news_result = check_news_filter()
        if news_result.should_block:
            logger.warning(f"Trading blocked by news filter: {news_result.block_reasons}")
            return CycleResult(
                signals=[],
                anomalies_count=0,
                candles_fetched=0,
                quotes_fetched=0,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[f"News block: {', '.join(news_result.block_reasons)}"],
            )

        market_ctx = fetch_market_context()
        logger.info(f"Market context: {market_ctx}")

        today = datetime.now(timezone.utc).date()
        from_date = (today - timedelta(days=3)).isoformat()
        till_date = today.isoformat()

        candles_fetched = 0
        quotes_fetched = 0

        if candles_df is None:
            all_candles = self.fetch_candles_parallel(from_date, till_date)

            for secid, candles in all_candles.items():
                if candles:
                    candles_fetched += len(candles)
                    rows = [
                        (secid, self.config.board, 1, c.ts, c.open, c.high, c.low, c.close, c.value, c.volume)
                        for c in candles
                    ]
                    try:
                        upsert_many(
                            conn,
                            "candles",
                            ("secid", "board", "interval", "ts", "open", "high", "low", "close", "value", "volume"),
                            rows,
                        )
                    except Exception as e:
                        errors.append(f"Upsert candles {secid}: {e}")

            candles_df = get_window(conn, minutes=3 * 24 * 60, interval=1)

        if quotes is None:
            quotes = self.fetch_quotes_parallel()
            quotes_fetched = len(quotes)

            now_ts = datetime.now(timezone.utc).isoformat()
            qrows = [
                (secid, self.config.board, now_ts, q.get("last"), q.get("bid"), q.get("ask"),
                 q.get("numtrades"), q.get("voltoday"), q.get("valtoday"))
                for secid, q in quotes.items()
            ]
            try:
                upsert_many(
                    conn,
                    "quotes",
                    ("secid", "board", "ts", "last", "bid", "ask", "numtrades", "voltoday", "valtoday"),
                    qrows,
                )
            except Exception as e:
                errors.append(f"Upsert quotes: {e}")

        # Level 1: Detect anomalies
        anomalies = self.detect_anomalies(candles_df, quotes)

        # Build features
        features_df = build_feature_frame(candles_df)
        features_df = features_df.dropna()

        # Levels 2-6: Generate signals with context and trend checks
        signals = self.generate_signals(
            anomalies=anomalies,
            features_df=features_df,
            quotes=quotes,
            cooldown_map=cooldown_map,
            market_ctx=market_ctx,
            candles_df=candles_df,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        return CycleResult(
            signals=signals,
            anomalies_count=len(anomalies),
            candles_fetched=candles_fetched,
            quotes_fetched=quotes_fetched,
            duration_ms=duration_ms,
            errors=errors,
        )


def create_engine(config_path: str = "config.yaml") -> PipelineEngine:
    """Factory function to create a pipeline engine."""
    from .config import load_config

    config = load_config(config_path)
    engine = PipelineEngine(config)
    engine.load_models()
    return engine
