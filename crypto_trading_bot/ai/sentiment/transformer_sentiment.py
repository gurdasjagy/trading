"""HuggingFace Transformer-based sentiment analyzer."""

from typing import List, Optional

from loguru import logger

from ai.sentiment.analyzer import _LABEL_THRESHOLDS, SentimentResult

_DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Maps model label strings to numeric scores
_LABEL_MAP = {
    "LABEL_0": -1.0,  # negative
    "LABEL_1": 0.0,  # neutral
    "LABEL_2": 1.0,  # positive
    # Some models use explicit names
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}


def _score_to_label(score: float) -> str:
    for low, high, label in _LABEL_THRESHOLDS:
        if low <= score < high:
            return label
    return "neutral"


class TransformerSentimentAnalyzer:
    """Sentiment analyzer backed by a HuggingFace transformers pipeline.

    The model is loaded lazily on first use so that importing this module
    does not incur the startup cost when the feature is disabled.

    Falls back to a neutral result when the *transformers* package is not
    installed or model loading fails.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._pipeline = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Load the transformer pipeline on first use."""
        if self._initialized:
            return
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                truncation=True,
                max_length=512,
            )
            logger.info(f"TransformerSentimentAnalyzer initialized with model '{self._model_name}'")
        except Exception as exc:
            logger.warning(f"TransformerSentimentAnalyzer not available: {exc}")
        self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, text: str) -> Optional[SentimentResult]:
        """Run a single inference; returns ``None`` on error."""
        try:
            result = self._pipeline(text[:512])[0]  # type: ignore[index]
            raw_label: str = result["label"]
            raw_score: float = float(result["score"])
            numeric = _LABEL_MAP.get(raw_label, 0.0) * raw_score
            label = _score_to_label(numeric)
            confidence = min(1.0, raw_score * 0.9 + 0.1)
            return SentimentResult(
                score=numeric,
                label=label,
                confidence=confidence,
                breakdown={"transformer": numeric},
            )
        except Exception as exc:
            logger.debug(f"TransformerSentimentAnalyzer inference error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        """Analyze *text* with the transformer model.

        Args:
            text: Raw text to analyze.

        Returns:
            :class:`SentimentResult` with score in [-1, 1].
        """
        self._initialize()
        if not self._pipeline:
            return SentimentResult(score=0.0, label="neutral", confidence=0.1, breakdown={})
        result = self._infer(text)
        return result or SentimentResult(score=0.0, label="neutral", confidence=0.0, breakdown={})

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts; falls back per-item on error.

        Args:
            texts: List of raw text strings.

        Returns:
            List of :class:`SentimentResult` objects (same order as input).
        """
        self._initialize()
        if not self._pipeline:
            return [
                SentimentResult(score=0.0, label="neutral", confidence=0.1, breakdown={})
                for _ in texts
            ]
        results: List[SentimentResult] = []
        for text in texts:
            result = self._infer(text)
            results.append(
                result or SentimentResult(score=0.0, label="neutral", confidence=0.0, breakdown={})
            )
        return results
