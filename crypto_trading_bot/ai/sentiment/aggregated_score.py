"""Sentiment score aggregator: weighted combination of multiple analyzer outputs."""

from typing import List, Optional

from loguru import logger

from ai.sentiment.analyzer import _LABEL_THRESHOLDS, SentimentResult

# Source-type normalisation constants (some sources naturally skew)
_SOURCE_BIAS: dict = {
    "twitter": 0.0,
    "reddit": 0.05,  # Reddit tends slightly bullish
    "news": 0.0,
    "vader": 0.0,
    "transformer": 0.0,
    "crypto_lexicon": 0.0,
    "llm": 0.0,
    "general": 0.0,
}


def _score_to_label(score: float) -> str:
    for low, high, label in _LABEL_THRESHOLDS:
        if low <= score < high:
            return label
    return "neutral"


class SentimentAggregator:
    """Combines multiple sentiment scores using configurable weighted averaging.

    Provides normalisation helpers to correct for systematic bias introduced by
    different source types or analysis methods.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, scores: List[float], weights: Optional[List[float]] = None) -> float:
        """Compute a weighted average of raw sentiment *scores*.

        Args:
            scores:  List of sentiment floats, each in [-1, 1].
            weights: Optional list of non-negative weights (same length as
                     *scores*).  When ``None``, equal weights are used.

        Returns:
            Weighted average score clamped to [-1.0, 1.0].
        """
        if not scores:
            return 0.0
        if weights is None:
            weights = [1.0] * len(scores)
        if len(weights) != len(scores):
            logger.warning(
                "SentimentAggregator.aggregate: weights/scores length mismatch — using equal weights"
            )
            weights = [1.0] * len(scores)

        total_weight = sum(w for w in weights if w > 0)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        result = weighted_sum / total_weight
        return max(-1.0, min(1.0, result))

    def aggregate_results(
        self,
        results: List[SentimentResult],
        weights: Optional[List[float]] = None,
    ) -> SentimentResult:
        """Combine a list of :class:`SentimentResult` objects into a single result.

        Args:
            results: List of :class:`SentimentResult` objects to aggregate.
            weights: Optional list of non-negative weights (same length as
                     *results*).  Defaults to equal weights.

        Returns:
            A new :class:`SentimentResult` representing the combined opinion.
        """
        if not results:
            return SentimentResult(score=0.0, label="neutral", confidence=0.0, breakdown={})

        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]

        # Use confidence-based weights when no explicit weights are provided
        effective_weights = weights if weights is not None else confidences

        combined_score = self.aggregate(scores, effective_weights)
        avg_confidence = sum(confidences) / len(confidences)

        # Merge all breakdowns
        merged_breakdown: dict = {}
        for result in results:
            merged_breakdown.update(result.breakdown)

        label = _score_to_label(combined_score)

        return SentimentResult(
            score=combined_score,
            label=label,
            confidence=min(1.0, avg_confidence),
            breakdown=merged_breakdown,
        )

    @staticmethod
    def normalize_score(score: float, source_type: str = "general") -> float:
        """Normalize *score* by correcting for known source bias.

        Args:
            score:       Raw sentiment score in [-1, 1].
            source_type: Source identifier (e.g. ``"twitter"``, ``"reddit"``).

        Returns:
            Bias-corrected score clamped to [-1.0, 1.0].
        """
        bias = _SOURCE_BIAS.get(source_type.lower(), 0.0)
        return max(-1.0, min(1.0, score - bias))
