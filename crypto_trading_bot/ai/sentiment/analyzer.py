"""Master sentiment analyzer combining VADER, crypto lexicon, and optional transformers."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel


class SentimentResult(BaseModel):
    """Sentiment result for a single piece of text."""

    score: float  # -1.0 to 1.0
    label: str  # very_bearish / bearish / neutral / bullish / very_bullish
    confidence: float  # 0.0 to 1.0
    breakdown: Dict[str, float] = {}  # per-source scores


class MarketSentiment(BaseModel):
    """Aggregated market sentiment for a trading symbol."""

    symbol: str
    overall_score: float
    label: str
    confidence: float
    twitter_score: Optional[float] = None
    reddit_score: Optional[float] = None
    news_score: Optional[float] = None
    fear_greed_normalized: Optional[float] = None
    timestamp: str = ""


# Crypto-specific bullish / bearish lexicon
_BULLISH_TERMS: Tuple[str, ...] = (
    "moon",
    "pump",
    "bullish",
    "breakout",
    "ath",
    "accumulate",
    "hodl",
    "rekt shorts",
    "short squeeze",
    "green",
)
_BEARISH_TERMS: Tuple[str, ...] = (
    "dump",
    "crash",
    "bearish",
    "rug pull",
    "hack",
    "exploit",
    "ban",
    "regulation",
    "rekt longs",
    "red",
    "capitulation",
)

# Score boundaries for label mapping (inclusive lower, exclusive upper)
_LABEL_THRESHOLDS: Tuple[Tuple[float, float, str], ...] = (
    (-1.01, -0.6, "very_bearish"),
    (-0.6, -0.2, "bearish"),
    (-0.2, 0.2, "neutral"),
    (0.2, 0.6, "bullish"),
    (0.6, 1.01, "very_bullish"),
)


class SentimentAnalyzer:
    """Combines multiple sentiment analysis approaches.

    Pipeline (in order of availability):
    1. VADER — rule-based, fast, always attempted.
    2. Crypto lexicon — domain-specific score adjustment.
    3. HuggingFace Transformer — optional, more accurate for short texts.
    4. LLM synthesis — used externally via :class:`~ai.brain.AIBrain`.
    """

    def __init__(
        self,
        llm_client=None,
        use_transformers: bool = False,
    ) -> None:
        self._llm = llm_client
        self._use_transformers = use_transformers
        self._vader = None
        self._transformer_pipeline = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load sentiment models (called lazily on first use)."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        except ImportError:
            logger.warning("vaderSentiment not installed — VADER unavailable")

        if self._use_transformers:
            try:
                from transformers import pipeline  # type: ignore

                self._transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    truncation=True,
                    max_length=512,
                )
                logger.info("Transformer sentiment pipeline initialized")
            except Exception as exc:
                logger.warning(f"Transformer sentiment not available: {exc}")

        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, text: str, source_type: str = "general") -> SentimentResult:
        """Analyze the sentiment of *text* using all available methods.

        Args:
            text:        Raw text to analyze.
            source_type: Hint about where the text came from (unused by default
                         but available for subclasses to weight sources).

        Returns:
            :class:`SentimentResult` with an aggregated score in [-1, 1].
        """
        if not self._initialized:
            await self.initialize()

        scores: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        # VADER
        if self._vader:
            vader_scores = self._vader.polarity_scores(text)
            scores["vader"] = float(vader_scores["compound"])
            weights["vader"] = 1.0

        # Transformer
        if self._transformer_pipeline:
            try:
                tr_result = self._transformer_pipeline(text[:512])[0]
                label_map = {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
                tr_score = label_map.get(tr_result["label"], 0.0) * float(tr_result["score"])
                scores["transformer"] = tr_score
                weights["transformer"] = 1.5  # weight transformer higher
            except Exception as exc:
                logger.debug(f"Transformer inference failed: {exc}")

        # Crypto lexicon adjustment
        lexicon_score = self._crypto_lexicon_score(text)
        if lexicon_score != 0.0:
            scores["crypto_lexicon"] = lexicon_score
            weights["crypto_lexicon"] = 0.5

        if not scores:
            return SentimentResult(score=0.0, label="neutral", confidence=0.3, breakdown={})

        total_weight = sum(weights.get(k, 1.0) for k in scores)
        weighted_score = sum(scores[k] * weights.get(k, 1.0) for k in scores) / total_weight
        weighted_score = max(-1.0, min(1.0, weighted_score))

        label = self._score_to_label(weighted_score)
        confidence = min(1.0, len(scores) * 0.3 + abs(weighted_score) * 0.4)

        return SentimentResult(
            score=weighted_score,
            label=label,
            confidence=confidence,
            breakdown=scores,
        )

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts concurrently.

        Args:
            texts: List of raw text strings.

        Returns:
            List of :class:`SentimentResult` objects (same order as input).
        """
        return list(await asyncio.gather(*[self.analyze(t) for t in texts]))

    async def get_market_sentiment(
        self,
        symbol: str,
        data_items: List[Dict],
    ) -> MarketSentiment:
        """Aggregate sentiment across multiple data items for *symbol*.

        Args:
            symbol:     Trading symbol (e.g. ``"BTC/USDT"``).
            data_items: List of dicts with at least a ``"content"`` key.

        Returns:
            :class:`MarketSentiment` with averaged scores.
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        if not data_items:
            return MarketSentiment(
                symbol=symbol,
                overall_score=0.0,
                label="neutral",
                confidence=0.3,
                timestamp=now_iso,
            )

        texts = [item.get("content", "") for item in data_items if item.get("content")]
        if not texts:
            return MarketSentiment(
                symbol=symbol,
                overall_score=0.0,
                label="neutral",
                confidence=0.3,
                timestamp=now_iso,
            )

        results = await self.analyze_batch(texts[:20])
        if not results:
            return MarketSentiment(
                symbol=symbol,
                overall_score=0.0,
                label="neutral",
                confidence=0.3,
                timestamp=now_iso,
            )

        avg_score = sum(r.score for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)

        return MarketSentiment(
            symbol=symbol,
            overall_score=avg_score,
            label=self._score_to_label(avg_score),
            confidence=avg_confidence,
            timestamp=now_iso,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Map a numeric score in [-1, 1] to a human-readable label."""
        for low, high, label in _LABEL_THRESHOLDS:
            if low <= score < high:
                return label
        return "neutral"

    @staticmethod
    def _crypto_lexicon_score(text: str) -> float:
        """Return a lexicon-based score in [-1, 1] using crypto-specific terms."""
        text_lower = text.lower()
        bull_count = sum(1 for term in _BULLISH_TERMS if term in text_lower)
        bear_count = sum(1 for term in _BEARISH_TERMS if term in text_lower)
        if bull_count == 0 and bear_count == 0:
            return 0.0
        return (bull_count - bear_count) / (bull_count + bear_count)
