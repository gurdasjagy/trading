"""VADER-based sentiment analyzer for social media text."""

from typing import List

from loguru import logger

from ai.sentiment.analyzer import _LABEL_THRESHOLDS, SentimentResult


def _score_to_label(score: float) -> str:
    for low, high, label in _LABEL_THRESHOLDS:
        if low <= score < high:
            return label
    return "neutral"


class VaderSentimentAnalyzer:
    """Rule-based sentiment analyzer using VADER.

    Optimized for short, informal social media text such as tweets and Reddit
    posts.  Falls back to a neutral result when vaderSentiment is not installed.
    """

    def __init__(self) -> None:
        self._vader = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Load VADER model on first use."""
        if self._initialized:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            self._vader = SentimentIntensityAnalyzer()
            logger.info("VaderSentimentAnalyzer initialized")
        except ImportError:
            logger.warning("vaderSentiment not installed — VaderSentimentAnalyzer unavailable")
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        """Analyze *text* with VADER and return a :class:`SentimentResult`.

        Args:
            text: Raw text to analyze.

        Returns:
            :class:`SentimentResult` with score in [-1, 1].
        """
        self._initialize()
        if not self._vader:
            return SentimentResult(score=0.0, label="neutral", confidence=0.1, breakdown={})
        try:
            scores = self._vader.polarity_scores(text)
            compound: float = float(scores["compound"])
            label = _score_to_label(compound)
            confidence = min(1.0, abs(compound) + 0.3)
            return SentimentResult(
                score=compound,
                label=label,
                confidence=confidence,
                breakdown={"vader": compound},
            )
        except Exception as exc:
            logger.warning(f"VaderSentimentAnalyzer.analyze error: {exc}")
            return SentimentResult(score=0.0, label="neutral", confidence=0.0, breakdown={})

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts and return results in the same order.

        Args:
            texts: List of raw text strings.

        Returns:
            List of :class:`SentimentResult` objects.
        """
        return [self.analyze(t) for t in texts]
