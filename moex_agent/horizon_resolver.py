"""
MOEX Agent v2.5 Multi-Horizon Conflict Resolver

Resolves conflicts when different horizon models give opposing signals.

Problem:
    10m model: LONG (p=0.65)
    1h model: SHORT (p=0.58)
    → What to do?

Resolution strategies:
1. DEFER_TO_LONGER: Trust longer horizon (more reliable trend)
2. REQUIRE_CONSENSUS: Only trade when all horizons agree
3. WEIGHTED_VOTE: Weighted average by horizon reliability
4. CONFIDENCE_THRESHOLD: Take highest confidence if above threshold

Usage:
    from moex_agent.horizon_resolver import HorizonResolver

    resolver = HorizonResolver(strategy="weighted_vote")
    decision = resolver.resolve(predictions)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ResolutionStrategy(str, Enum):
    """Strategy for resolving horizon conflicts."""
    DEFER_TO_LONGER = "defer_to_longer"
    REQUIRE_CONSENSUS = "require_consensus"
    WEIGHTED_VOTE = "weighted_vote"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    MAJORITY_VOTE = "majority_vote"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class HorizonPrediction:
    """Single horizon prediction."""
    horizon: str
    direction: Direction
    probability: float  # 0.5 = neutral, >0.5 = LONG, <0.5 = SHORT
    confidence: float   # |p - 0.5| * 2, normalized to [0, 1]

    @classmethod
    def from_probability(cls, horizon: str, prob: float) -> "HorizonPrediction":
        """Create from raw probability (assumes >0.5 = LONG)."""
        if prob > 0.5:
            direction = Direction.LONG
            confidence = (prob - 0.5) * 2
        elif prob < 0.5:
            direction = Direction.SHORT
            confidence = (0.5 - prob) * 2
        else:
            direction = Direction.NEUTRAL
            confidence = 0.0

        return cls(
            horizon=horizon,
            direction=direction,
            probability=prob,
            confidence=confidence,
        )


@dataclass
class ResolverDecision:
    """Final decision from resolver."""
    direction: Direction
    confidence: float
    reason: str
    horizons_agree: bool
    contributing_horizons: List[str]
    conflicts: List[Tuple[str, str]]  # [(horizon1, horizon2), ...]


# Horizon weights based on typical reliability
# Longer horizons are generally more reliable for trend direction
HORIZON_WEIGHTS = {
    "5m": 0.5,
    "10m": 0.7,
    "30m": 0.85,
    "1h": 1.0,
    "4h": 1.1,
    "1d": 1.2,
}

# Horizon order (shortest to longest)
HORIZON_ORDER = ["5m", "10m", "30m", "1h", "4h", "1d"]


class HorizonResolver:
    """
    Resolves conflicts between multi-horizon predictions.
    """

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.WEIGHTED_VOTE,
        min_confidence: float = 0.1,
        consensus_threshold: float = 0.6,
        conflict_penalty: float = 0.3,
    ):
        """
        Args:
            strategy: Resolution strategy to use
            min_confidence: Minimum confidence to consider a prediction
            consensus_threshold: Required agreement for consensus strategy
            conflict_penalty: Confidence reduction when conflicts exist
        """
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.consensus_threshold = consensus_threshold
        self.conflict_penalty = conflict_penalty

    def resolve(
        self,
        predictions: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> ResolverDecision:
        """
        Resolve multiple horizon predictions into a single decision.

        Args:
            predictions: Dict of {horizon: probability}
                         probability > 0.5 = LONG, < 0.5 = SHORT
            weights: Optional custom weights per horizon

        Returns:
            ResolverDecision with final direction and confidence
        """
        if not predictions:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="no predictions",
                horizons_agree=True,
                contributing_horizons=[],
                conflicts=[],
            )

        weights = weights or HORIZON_WEIGHTS

        # Convert to HorizonPrediction objects
        horizon_preds = []
        for horizon, prob in predictions.items():
            pred = HorizonPrediction.from_probability(horizon, prob)
            if pred.confidence >= self.min_confidence:
                horizon_preds.append(pred)

        if not horizon_preds:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="all predictions below confidence threshold",
                horizons_agree=True,
                contributing_horizons=[],
                conflicts=[],
            )

        # Detect conflicts
        conflicts = self._detect_conflicts(horizon_preds)
        horizons_agree = len(conflicts) == 0

        # Apply resolution strategy
        if self.strategy == ResolutionStrategy.DEFER_TO_LONGER:
            return self._resolve_defer_to_longer(horizon_preds, conflicts)

        elif self.strategy == ResolutionStrategy.REQUIRE_CONSENSUS:
            return self._resolve_require_consensus(horizon_preds, conflicts)

        elif self.strategy == ResolutionStrategy.WEIGHTED_VOTE:
            return self._resolve_weighted_vote(horizon_preds, weights, conflicts)

        elif self.strategy == ResolutionStrategy.CONFIDENCE_THRESHOLD:
            return self._resolve_confidence_threshold(horizon_preds, conflicts)

        elif self.strategy == ResolutionStrategy.MAJORITY_VOTE:
            return self._resolve_majority_vote(horizon_preds, conflicts)

        else:
            # Default to weighted vote
            return self._resolve_weighted_vote(horizon_preds, weights, conflicts)

    def _detect_conflicts(
        self,
        predictions: List[HorizonPrediction],
    ) -> List[Tuple[str, str]]:
        """Detect conflicting predictions between horizons."""
        conflicts = []

        for i, pred1 in enumerate(predictions):
            for pred2 in predictions[i + 1:]:
                # LONG vs SHORT is a conflict
                if (pred1.direction == Direction.LONG and pred2.direction == Direction.SHORT) or \
                   (pred1.direction == Direction.SHORT and pred2.direction == Direction.LONG):
                    conflicts.append((pred1.horizon, pred2.horizon))

        return conflicts

    def _resolve_defer_to_longer(
        self,
        predictions: List[HorizonPrediction],
        conflicts: List[Tuple[str, str]],
    ) -> ResolverDecision:
        """
        Defer to the longest horizon prediction.

        Rationale: Longer horizons capture more reliable trends.
        """
        # Sort by horizon order (longest first)
        sorted_preds = sorted(
            predictions,
            key=lambda p: HORIZON_ORDER.index(p.horizon) if p.horizon in HORIZON_ORDER else 999,
            reverse=True,
        )

        longest = sorted_preds[0]

        # Reduce confidence if conflicts exist
        confidence = longest.confidence
        if conflicts:
            confidence *= (1 - self.conflict_penalty)

        return ResolverDecision(
            direction=longest.direction,
            confidence=confidence,
            reason=f"defer to {longest.horizon} (longest horizon)",
            horizons_agree=len(conflicts) == 0,
            contributing_horizons=[longest.horizon],
            conflicts=conflicts,
        )

    def _resolve_require_consensus(
        self,
        predictions: List[HorizonPrediction],
        conflicts: List[Tuple[str, str]],
    ) -> ResolverDecision:
        """
        Only trade when all horizons agree.

        Rationale: High conviction trades only.
        """
        if conflicts:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=f"no consensus: {len(conflicts)} conflicts",
                horizons_agree=False,
                contributing_horizons=[],
                conflicts=conflicts,
            )

        # All agree - take average confidence
        directions = set(p.direction for p in predictions)
        if len(directions) > 1:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="mixed directions",
                horizons_agree=False,
                contributing_horizons=[],
                conflicts=conflicts,
            )

        direction = predictions[0].direction
        confidence = np.mean([p.confidence for p in predictions])

        return ResolverDecision(
            direction=direction,
            confidence=confidence,
            reason=f"consensus across {len(predictions)} horizons",
            horizons_agree=True,
            contributing_horizons=[p.horizon for p in predictions],
            conflicts=conflicts,
        )

    def _resolve_weighted_vote(
        self,
        predictions: List[HorizonPrediction],
        weights: Dict[str, float],
        conflicts: List[Tuple[str, str]],
    ) -> ResolverDecision:
        """
        Weighted voting by horizon reliability.

        Rationale: Balance all signals with appropriate weights.
        """
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0

        for pred in predictions:
            weight = weights.get(pred.horizon, 1.0)
            weighted_conf = pred.confidence * weight

            if pred.direction == Direction.LONG:
                long_score += weighted_conf
            elif pred.direction == Direction.SHORT:
                short_score += weighted_conf

            total_weight += weight

        if total_weight == 0:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="no weighted predictions",
                horizons_agree=True,
                contributing_horizons=[],
                conflicts=conflicts,
            )

        # Normalize scores
        long_score /= total_weight
        short_score /= total_weight

        # Determine direction
        if long_score > short_score:
            direction = Direction.LONG
            confidence = long_score - short_score
            contributing = [p.horizon for p in predictions if p.direction == Direction.LONG]
        elif short_score > long_score:
            direction = Direction.SHORT
            confidence = short_score - long_score
            contributing = [p.horizon for p in predictions if p.direction == Direction.SHORT]
        else:
            direction = Direction.NEUTRAL
            confidence = 0.0
            contributing = []

        # Apply conflict penalty
        if conflicts:
            confidence *= (1 - self.conflict_penalty)

        return ResolverDecision(
            direction=direction,
            confidence=confidence,
            reason=f"weighted vote: LONG={long_score:.2f} SHORT={short_score:.2f}",
            horizons_agree=len(conflicts) == 0,
            contributing_horizons=contributing,
            conflicts=conflicts,
        )

    def _resolve_confidence_threshold(
        self,
        predictions: List[HorizonPrediction],
        conflicts: List[Tuple[str, str]],
    ) -> ResolverDecision:
        """
        Take the highest confidence prediction if above threshold.

        Rationale: Only take strong signals.
        """
        # Sort by confidence (highest first)
        sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
        highest = sorted_preds[0]

        # Check if above consensus threshold
        if highest.confidence < self.consensus_threshold:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=f"highest confidence {highest.confidence:.2f} below threshold {self.consensus_threshold}",
                horizons_agree=len(conflicts) == 0,
                contributing_horizons=[],
                conflicts=conflicts,
            )

        # Apply conflict penalty
        confidence = highest.confidence
        if conflicts:
            confidence *= (1 - self.conflict_penalty)

        return ResolverDecision(
            direction=highest.direction,
            confidence=confidence,
            reason=f"highest confidence: {highest.horizon} ({highest.confidence:.2f})",
            horizons_agree=len(conflicts) == 0,
            contributing_horizons=[highest.horizon],
            conflicts=conflicts,
        )

    def _resolve_majority_vote(
        self,
        predictions: List[HorizonPrediction],
        conflicts: List[Tuple[str, str]],
    ) -> ResolverDecision:
        """
        Simple majority vote (each horizon = 1 vote).

        Rationale: Democratic approach.
        """
        long_votes = sum(1 for p in predictions if p.direction == Direction.LONG)
        short_votes = sum(1 for p in predictions if p.direction == Direction.SHORT)
        total_votes = long_votes + short_votes

        if total_votes == 0:
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="no votes",
                horizons_agree=True,
                contributing_horizons=[],
                conflicts=conflicts,
            )

        if long_votes > short_votes:
            direction = Direction.LONG
            confidence = long_votes / total_votes
            contributing = [p.horizon for p in predictions if p.direction == Direction.LONG]
        elif short_votes > long_votes:
            direction = Direction.SHORT
            confidence = short_votes / total_votes
            contributing = [p.horizon for p in predictions if p.direction == Direction.SHORT]
        else:
            # Tie - neutral
            return ResolverDecision(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=f"tie: {long_votes} LONG vs {short_votes} SHORT",
                horizons_agree=False,
                contributing_horizons=[],
                conflicts=conflicts,
            )

        # Apply conflict penalty
        if conflicts:
            confidence *= (1 - self.conflict_penalty)

        return ResolverDecision(
            direction=direction,
            confidence=confidence,
            reason=f"majority: {long_votes} LONG vs {short_votes} SHORT",
            horizons_agree=len(conflicts) == 0,
            contributing_horizons=contributing,
            conflicts=conflicts,
        )


# Singleton
_resolver: Optional[HorizonResolver] = None


def get_horizon_resolver(
    strategy: ResolutionStrategy = ResolutionStrategy.WEIGHTED_VOTE,
) -> HorizonResolver:
    """Get or create global HorizonResolver instance."""
    global _resolver
    if _resolver is None or _resolver.strategy != strategy:
        _resolver = HorizonResolver(strategy=strategy)
    return _resolver


def resolve_horizons(
    predictions: Dict[str, float],
    strategy: ResolutionStrategy = ResolutionStrategy.WEIGHTED_VOTE,
) -> ResolverDecision:
    """
    Convenience function to resolve horizon conflicts.

    Args:
        predictions: Dict of {horizon: probability}
        strategy: Resolution strategy

    Returns:
        ResolverDecision
    """
    resolver = get_horizon_resolver(strategy)
    return resolver.resolve(predictions)
