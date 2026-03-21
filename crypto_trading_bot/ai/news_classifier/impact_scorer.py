"""News impact scorer: estimates price-move magnitude from news classifications."""

from typing import Dict

from loguru import logger

from ai.news_classifier.categories import Direction, ImpactLevel, NewsCategory, TimeHorizon

# ---------------------------------------------------------------------------
# Historical average absolute price-move estimates per category × direction
# These are heuristic values derived from observed crypto market behaviour.
# ---------------------------------------------------------------------------

_HISTORICAL_IMPACT: Dict[NewsCategory, Dict[str, float]] = {
    NewsCategory.REGULATORY: {
        Direction.BULLISH: 0.08,
        Direction.BEARISH: 0.12,
        Direction.NEUTRAL: 0.03,
    },
    NewsCategory.SECURITY: {
        Direction.BULLISH: 0.02,
        Direction.BEARISH: 0.15,
        Direction.NEUTRAL: 0.05,
    },
    NewsCategory.MARKET: {
        Direction.BULLISH: 0.05,
        Direction.BEARISH: 0.06,
        Direction.NEUTRAL: 0.02,
    },
    NewsCategory.MACRO: {Direction.BULLISH: 0.04, Direction.BEARISH: 0.05, Direction.NEUTRAL: 0.02},
    NewsCategory.ADOPTION: {
        Direction.BULLISH: 0.07,
        Direction.BEARISH: 0.02,
        Direction.NEUTRAL: 0.02,
    },
    NewsCategory.TECHNICAL: {
        Direction.BULLISH: 0.04,
        Direction.BEARISH: 0.04,
        Direction.NEUTRAL: 0.01,
    },
    NewsCategory.PARTNERSHIP: {
        Direction.BULLISH: 0.05,
        Direction.BEARISH: 0.01,
        Direction.NEUTRAL: 0.02,
    },
    NewsCategory.DEVELOPMENT: {
        Direction.BULLISH: 0.04,
        Direction.BEARISH: 0.03,
        Direction.NEUTRAL: 0.01,
    },
    NewsCategory.UNKNOWN: {
        Direction.BULLISH: 0.02,
        Direction.BEARISH: 0.02,
        Direction.NEUTRAL: 0.01,
    },
}

# Impact-level multipliers applied on top of the base estimate
_IMPACT_MULTIPLIERS: Dict[ImpactLevel, float] = {
    ImpactLevel.CRITICAL: 2.5,
    ImpactLevel.HIGH: 1.5,
    ImpactLevel.MEDIUM: 1.0,
    ImpactLevel.LOW: 0.5,
    ImpactLevel.NOISE: 0.1,
}

# Time-horizon decay (longer horizon → smaller immediate move)
_HORIZON_DECAY: Dict[TimeHorizon, float] = {
    TimeHorizon.IMMEDIATE: 1.0,
    TimeHorizon.SHORT: 0.8,
    TimeHorizon.MEDIUM: 0.5,
    TimeHorizon.LONG: 0.3,
}


class ImpactScorer:
    """Predicts the price-impact magnitude of a news classification.

    Combines category base rates, impact-level multipliers, sentiment
    direction, time-horizon decay, and optionally a sentiment score to
    estimate how much a piece of news is likely to move prices.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_impact(
        self,
        news_item: Dict,
        classification,
    ) -> float:
        """Return a normalized impact score in [0, 1] for *news_item*.

        Args:
            news_item:      Dict with at least ``"title"`` (and optionally
                            ``"sentiment_score"``).
            classification: A :class:`~ai.news_classifier.categories.NewsClassification`
                            (or the pydantic ``NewsClassification`` from classifier.py).

        Returns:
            Float in [0.0, 1.0]; higher means larger expected price move.
        """
        try:
            category = _coerce_category(getattr(classification, "category", None))
            direction = _coerce_direction(
                getattr(classification, "direction", None)
                or getattr(classification, "impact_level", None)
            )
            impact_level = _coerce_impact(
                getattr(classification, "impact_level", None)
                or getattr(classification, "impact", None)
            )
            time_horizon = _coerce_horizon(getattr(classification, "time_horizon", None))
            confidence = float(getattr(classification, "confidence", 0.5))
            sentiment_score: float = float(news_item.get("sentiment_score", 0.0))

            base = self._get_historical_impact(category, direction)
            impact_mult = _IMPACT_MULTIPLIERS.get(impact_level, 1.0)
            horizon_decay = _HORIZON_DECAY.get(time_horizon, 0.8)

            # Sentiment score amplifies or attenuates
            sentiment_mult = 1.0 + abs(sentiment_score) * 0.5

            raw = base * impact_mult * horizon_decay * confidence * sentiment_mult
            return max(0.0, min(1.0, raw))
        except Exception as exc:
            logger.warning(f"ImpactScorer.score_impact error: {exc}")
            return 0.1

    def estimate_price_move(self, classification) -> Dict[str, float]:
        """Estimate the expected price-move percentage from a classification.

        Args:
            classification: A classification object with ``category``,
                            ``impact_level``, ``direction``, ``time_horizon``,
                            and ``confidence`` attributes.

        Returns:
            Dict with keys:
            - ``"expected_move_pct"``: signed expected move (negative = down)
            - ``"min_move_pct"``:  conservative lower bound
            - ``"max_move_pct"``:  optimistic upper bound
        """
        try:
            category = _coerce_category(getattr(classification, "category", None))
            direction = _coerce_direction(getattr(classification, "direction", None))
            impact_level = _coerce_impact(
                getattr(classification, "impact_level", None)
                or getattr(classification, "impact", None)
            )
            time_horizon = _coerce_horizon(getattr(classification, "time_horizon", None))
            confidence = float(getattr(classification, "confidence", 0.5))

            base_pct = self._get_historical_impact(category, direction) * 100.0
            impact_mult = _IMPACT_MULTIPLIERS.get(impact_level, 1.0)
            horizon_decay = _HORIZON_DECAY.get(time_horizon, 0.8)

            expected = base_pct * impact_mult * horizon_decay * confidence
            sign = -1.0 if direction == Direction.BEARISH else 1.0

            return {
                "expected_move_pct": round(sign * expected, 2),
                "min_move_pct": round(sign * expected * 0.5, 2),
                "max_move_pct": round(sign * expected * 2.0, 2),
            }
        except Exception as exc:
            logger.warning(f"ImpactScorer.estimate_price_move error: {exc}")
            return {"expected_move_pct": 0.0, "min_move_pct": 0.0, "max_move_pct": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_historical_impact(category: NewsCategory, direction: Direction) -> float:
        """Return the historical base absolute-move rate for *category* × *direction*."""
        cat_data = _HISTORICAL_IMPACT.get(category, _HISTORICAL_IMPACT[NewsCategory.UNKNOWN])
        return cat_data.get(direction, 0.02)


# ---------------------------------------------------------------------------
# Coercion helpers (handle both pydantic and dataclass classification objects)
# ---------------------------------------------------------------------------


def _coerce_category(value) -> NewsCategory:
    if isinstance(value, NewsCategory):
        return value
    try:
        return NewsCategory(str(value))
    except Exception:
        return NewsCategory.UNKNOWN


def _coerce_direction(value) -> Direction:
    if isinstance(value, Direction):
        return value
    try:
        return Direction(str(value))
    except Exception:
        return Direction.NEUTRAL


def _coerce_impact(value) -> ImpactLevel:
    if isinstance(value, ImpactLevel):
        return value
    try:
        return ImpactLevel(str(value))
    except Exception:
        return ImpactLevel.LOW


def _coerce_horizon(value) -> TimeHorizon:
    if isinstance(value, TimeHorizon):
        return value
    if isinstance(value, str):
        try:
            return TimeHorizon(value.upper())
        except Exception:
            pass
    return TimeHorizon.SHORT
