"""
MOEX Agent v2.5 Champion/Challenger Model Switching

A/B testing framework for models in production.

Concept:
- Champion: Current production model (proven)
- Challenger: New model being tested (experimental)
- Traffic split: X% to champion, Y% to challenger
- Track performance separately
- Promote challenger to champion if it outperforms

Usage:
    from moex_agent.champion_challenger import ModelArena

    arena = ModelArena()
    arena.set_champion("models/champion")
    arena.set_challenger("models/challenger", traffic_pct=20)

    # Route prediction
    model, model_type = arena.route_prediction(ticker)
    prediction = model.predict(X)

    # Record result
    arena.record_result(model_type, ticker, pnl)

    # Check if challenger should be promoted
    if arena.should_promote_challenger():
        arena.promote_challenger()
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Statistics for a model."""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    pnl_history: List[float] = field(default_factory=list)
    start_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trades if self.trades > 0 else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.pnl_history) < 2:
            return 0.0
        pnl_std = np.std(self.pnl_history)
        if pnl_std == 0:
            return 0.0
        return np.mean(self.pnl_history) / pnl_std * np.sqrt(252)

    def record_trade(self, pnl: float, is_win: bool) -> None:
        """Record a trade result."""
        self.trades += 1
        if is_win:
            self.wins += 1
        self.total_pnl += pnl
        self.pnl_history.append(pnl)
        self.last_trade_time = datetime.now()

        # Keep only last 100 trades for memory
        if len(self.pnl_history) > 100:
            self.pnl_history = self.pnl_history[-100:]

    def to_dict(self) -> Dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "sharpe": self.sharpe,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
        }


@dataclass
class ChallengerConfig:
    """Configuration for a challenger model."""
    model_path: Path
    traffic_pct: float  # 0-100
    min_trades: int = 50  # Minimum trades before evaluation
    min_days: int = 7  # Minimum days before evaluation
    win_rate_threshold: float = 0.02  # Must beat champion by this much
    started_at: datetime = field(default_factory=datetime.now)


class ModelArena:
    """
    Arena for champion/challenger model testing.

    Supports:
    - Traffic splitting between models
    - Per-model statistics tracking
    - Automatic promotion based on performance
    - Multiple challengers
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        default_champion_path: Optional[Path] = None,
    ):
        """
        Args:
            state_path: Path to save arena state
            default_champion_path: Default champion model path
        """
        self.state_path = state_path or Path("data/arena_state.json")
        self.champion_path: Optional[Path] = default_champion_path
        self.challengers: Dict[str, ChallengerConfig] = {}

        # Statistics
        self.champion_stats = ModelStats()
        self.challenger_stats: Dict[str, ModelStats] = {}

        # Per-ticker routing (for consistency)
        self._ticker_routing: Dict[str, str] = {}

        # Loaded models cache
        self._champion_model: Any = None
        self._challenger_models: Dict[str, Any] = {}

        # Load state if exists
        self._load_state()

    def _load_state(self) -> None:
        """Load arena state from JSON."""
        if not self.state_path.exists():
            return

        try:
            data = json.loads(self.state_path.read_text())

            if "champion_path" in data and data["champion_path"]:
                self.champion_path = Path(data["champion_path"])

            if "champion_stats" in data:
                stats = data["champion_stats"]
                self.champion_stats.trades = stats.get("trades", 0)
                self.champion_stats.wins = stats.get("wins", 0)
                self.champion_stats.total_pnl = stats.get("total_pnl", 0.0)

            if "challengers" in data:
                for name, cfg in data["challengers"].items():
                    self.challengers[name] = ChallengerConfig(
                        model_path=Path(cfg["model_path"]),
                        traffic_pct=cfg["traffic_pct"],
                        min_trades=cfg.get("min_trades", 50),
                        min_days=cfg.get("min_days", 7),
                    )
                    self.challenger_stats[name] = ModelStats()

            logger.info(f"Loaded arena state: champion={self.champion_path}, challengers={list(self.challengers.keys())}")

        except Exception as e:
            logger.warning(f"Failed to load arena state: {e}")

    def _save_state(self) -> None:
        """Save arena state to JSON."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "champion_path": str(self.champion_path) if self.champion_path else None,
                "champion_stats": self.champion_stats.to_dict(),
                "challengers": {
                    name: {
                        "model_path": str(cfg.model_path),
                        "traffic_pct": cfg.traffic_pct,
                        "min_trades": cfg.min_trades,
                        "min_days": cfg.min_days,
                        "started_at": cfg.started_at.isoformat(),
                    }
                    for name, cfg in self.challengers.items()
                },
                "challenger_stats": {
                    name: stats.to_dict()
                    for name, stats in self.challenger_stats.items()
                },
                "updated_at": datetime.now().isoformat(),
            }

            self.state_path.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.warning(f"Failed to save arena state: {e}")

    def set_champion(self, model_path: Path) -> None:
        """Set the champion model."""
        self.champion_path = Path(model_path)
        self._champion_model = None  # Clear cache
        logger.info(f"Champion set: {model_path}")
        self._save_state()

    def add_challenger(
        self,
        name: str,
        model_path: Path,
        traffic_pct: float = 20.0,
        min_trades: int = 50,
        min_days: int = 7,
        win_rate_threshold: float = 0.02,
    ) -> None:
        """
        Add a challenger model.

        Args:
            name: Unique name for challenger
            model_path: Path to model
            traffic_pct: Percentage of traffic to route to challenger
            min_trades: Minimum trades before evaluation
            min_days: Minimum days before evaluation
            win_rate_threshold: Must beat champion WR by this amount
        """
        # Validate total traffic doesn't exceed 100%
        total_challenger_traffic = sum(c.traffic_pct for c in self.challengers.values())
        if total_challenger_traffic + traffic_pct > 50:
            logger.warning(f"Total challenger traffic would exceed 50%, reducing {name} to fit")
            traffic_pct = max(0, 50 - total_challenger_traffic)

        self.challengers[name] = ChallengerConfig(
            model_path=Path(model_path),
            traffic_pct=traffic_pct,
            min_trades=min_trades,
            min_days=min_days,
            win_rate_threshold=win_rate_threshold,
        )
        self.challenger_stats[name] = ModelStats(start_time=datetime.now())
        self._challenger_models[name] = None  # Clear cache

        logger.info(f"Challenger added: {name} @ {traffic_pct}% traffic")
        self._save_state()

    def remove_challenger(self, name: str) -> None:
        """Remove a challenger model."""
        if name in self.challengers:
            del self.challengers[name]
            del self.challenger_stats[name]
            if name in self._challenger_models:
                del self._challenger_models[name]
            logger.info(f"Challenger removed: {name}")
            self._save_state()

    def route_prediction(
        self,
        ticker: str,
        use_sticky_routing: bool = True,
    ) -> Tuple[str, str]:
        """
        Route a prediction request to champion or challenger.

        Args:
            ticker: Ticker symbol
            use_sticky_routing: If True, same ticker always goes to same model

        Returns:
            (model_path, model_type) where model_type is "champion" or challenger name
        """
        if not self.challengers:
            return str(self.champion_path), "champion"

        # Sticky routing - same ticker always goes to same model
        if use_sticky_routing and ticker in self._ticker_routing:
            model_type = self._ticker_routing[ticker]
            if model_type == "champion":
                return str(self.champion_path), "champion"
            elif model_type in self.challengers:
                return str(self.challengers[model_type].model_path), model_type

        # Random routing based on traffic percentages
        rand = random.random() * 100

        cumulative = 0.0
        for name, cfg in self.challengers.items():
            cumulative += cfg.traffic_pct
            if rand < cumulative:
                if use_sticky_routing:
                    self._ticker_routing[ticker] = name
                return str(cfg.model_path), name

        # Default to champion
        if use_sticky_routing:
            self._ticker_routing[ticker] = "champion"
        return str(self.champion_path), "champion"

    def record_result(
        self,
        model_type: str,
        ticker: str,
        pnl: float,
        is_win: bool,
    ) -> None:
        """
        Record a trade result for a model.

        Args:
            model_type: "champion" or challenger name
            ticker: Ticker symbol
            pnl: Profit/loss in ATR or currency
            is_win: Whether trade was profitable
        """
        if model_type == "champion":
            self.champion_stats.record_trade(pnl, is_win)
        elif model_type in self.challenger_stats:
            self.challenger_stats[model_type].record_trade(pnl, is_win)

        self._save_state()

    def should_promote_challenger(self, name: str) -> Tuple[bool, str]:
        """
        Check if a challenger should be promoted to champion.

        Args:
            name: Challenger name

        Returns:
            (should_promote, reason)
        """
        if name not in self.challengers:
            return False, "challenger not found"

        cfg = self.challengers[name]
        stats = self.challenger_stats[name]

        # Check minimum trades
        if stats.trades < cfg.min_trades:
            return False, f"insufficient trades ({stats.trades}/{cfg.min_trades})"

        # Check minimum days
        days_running = (datetime.now() - cfg.started_at).days
        if days_running < cfg.min_days:
            return False, f"insufficient time ({days_running}/{cfg.min_days} days)"

        # Check champion has enough trades for comparison
        if self.champion_stats.trades < cfg.min_trades:
            return False, f"champion has insufficient trades ({self.champion_stats.trades})"

        # Compare win rates
        challenger_wr = stats.win_rate
        champion_wr = self.champion_stats.win_rate
        wr_diff = challenger_wr - champion_wr

        if wr_diff < cfg.win_rate_threshold:
            return False, f"WR not better enough ({challenger_wr:.1%} vs {champion_wr:.1%}, diff={wr_diff:+.1%})"

        # Check if Sharpe is also better
        if stats.sharpe < self.champion_stats.sharpe * 0.9:
            return False, f"Sharpe not good enough ({stats.sharpe:.2f} vs {self.champion_stats.sharpe:.2f})"

        return True, f"challenger outperforms: WR {challenger_wr:.1%} vs {champion_wr:.1%} (+{wr_diff:.1%})"

    def promote_challenger(self, name: str) -> bool:
        """
        Promote a challenger to champion.

        Args:
            name: Challenger name

        Returns:
            True if promotion successful
        """
        if name not in self.challengers:
            return False

        cfg = self.challengers[name]

        # Archive old champion
        old_champion = self.champion_path
        logger.info(f"Promoting challenger '{name}' to champion (old: {old_champion})")

        # Update champion
        self.champion_path = cfg.model_path
        self._champion_model = None

        # Reset stats (new champion starts fresh)
        self.champion_stats = self.challenger_stats[name]
        self.champion_stats.start_time = datetime.now()

        # Remove challenger
        self.remove_challenger(name)

        # Clear routing cache
        self._ticker_routing.clear()

        self._save_state()
        return True

    def get_status(self) -> Dict:
        """Get arena status."""
        status = {
            "champion": {
                "path": str(self.champion_path) if self.champion_path else None,
                "stats": self.champion_stats.to_dict(),
            },
            "challengers": {},
        }

        for name, cfg in self.challengers.items():
            stats = self.challenger_stats[name]
            should_promote, reason = self.should_promote_challenger(name)

            status["challengers"][name] = {
                "path": str(cfg.model_path),
                "traffic_pct": cfg.traffic_pct,
                "stats": stats.to_dict(),
                "should_promote": should_promote,
                "promotion_reason": reason,
                "days_running": (datetime.now() - cfg.started_at).days,
            }

        return status


# Singleton
_arena: Optional[ModelArena] = None


def get_model_arena() -> ModelArena:
    """Get or create global ModelArena instance."""
    global _arena
    if _arena is None:
        _arena = ModelArena()
    return _arena
