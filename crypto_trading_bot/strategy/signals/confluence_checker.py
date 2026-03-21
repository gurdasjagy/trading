"""Confluence checker — scores alignment of multiple signal sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger


@dataclass
class ConfluenceResult:
    """Result of a confluence check."""

    score: float
    max_score: float
    should_trade: bool
    position_multiplier: float
    factor_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


class ConfluenceChecker:
    """Checks whether multiple signal sources align before permitting a trade.

    Scoring table
    -------------
    ========================  ========
    Factor                    Score
    ========================  ========
    Technical indicator       +1.0
    Sentiment                 +1.0
    On-chain metrics          +1.0
    News                      +1.0
    Market regime             +1.0
    Volume confirmation       +0.5
    Multi-timeframe alignment +1.0
    Funding rate              +0.5
    Fear & greed index        +0.5
    Whale activity            +0.5
    ========================  ========

    Maximum score: **8.0**
    Minimum to trade: **5.0** (configurable).
    """

    FACTOR_WEIGHTS: Dict[str, float] = {
        "technical": 1.0,
        "sentiment": 1.0,
        "onchain": 1.0,
        "news": 1.0,
        "regime": 1.0,
        "volume": 0.5,
        "multi_tf": 1.0,
        "funding": 0.5,
        "fear_greed": 0.5,
        "whale": 0.5,
    }
    MAX_SCORE: float = sum(FACTOR_WEIGHTS.values())  # 7.0

    def __init__(self, min_score: float = 5.0) -> None:
        self._min_score = min_score

    def check_confluence(self, signals: Dict[str, Optional[str]]) -> ConfluenceResult:
        """Evaluate confluence across all signal factors.

        Parameters
        ----------
        signals:
            Mapping of factor name → direction (``"long"``, ``"short"``,
            ``"neutral"``, or ``None``).  The factor names should match
            :attr:`FACTOR_WEIGHTS` keys.

        Returns
        -------
        :class:`ConfluenceResult`
        """
        # Determine the dominant direction from non-neutral signals
        directions = [v for v in signals.values() if v and v.lower() not in ("neutral", "none")]
        if not directions:
            return ConfluenceResult(
                score=0.0,
                max_score=self.MAX_SCORE,
                should_trade=False,
                position_multiplier=0.0,
                reasoning="No directional signals present",
            )

        from collections import Counter

        dominant = Counter(directions).most_common(1)[0][0]

        factor_scores: Dict[str, float] = {}
        for factor, weight in self.FACTOR_WEIGHTS.items():
            val = signals.get(factor)
            if val and val.lower() == dominant.lower():
                factor_scores[factor] = weight
            else:
                factor_scores[factor] = 0.0

        score = self._calculate_score(factor_scores)
        should_trade = score >= self._min_score
        multiplier = self._get_position_multiplier(score)

        reasons = [f"{k}(+{v})" for k, v in factor_scores.items() if v > 0]
        reasoning = (
            f"Score {score:.1f}/{self.MAX_SCORE} — {', '.join(reasons)}"
            if reasons
            else f"Score {score:.1f}/{self.MAX_SCORE} — no factors aligned"
        )

        logger.debug(
            f"Confluence: score={score:.1f}/{self.MAX_SCORE}, "
            f"trade={should_trade}, direction={dominant}"
        )
        return ConfluenceResult(
            score=score,
            max_score=self.MAX_SCORE,
            should_trade=should_trade,
            position_multiplier=multiplier,
            factor_scores=factor_scores,
            reasoning=reasoning,
        )

    def _calculate_score(self, factor_scores: Dict[str, float]) -> float:
        """Sum all factor scores."""
        return sum(factor_scores.values())

    def _get_position_multiplier(self, score: float) -> float:
        """Return a position size multiplier based on confluence strength.

        ============  ===========
        Score range   Multiplier
        ============  ===========
        < 5.0         0.0 (no trade)
        5.0 – 5.9     0.5
        6.0 – 6.4     0.75
        ≥ 6.5         1.0
        ============  ===========
        """
        if score < self._min_score:
            return 0.0
        if score < 6.0:
            return 0.5
        if score < 6.5:
            return 0.75
        return 1.0
