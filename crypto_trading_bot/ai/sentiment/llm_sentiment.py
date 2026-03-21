"""LLM-powered sentiment analyzer using structured JSON prompts.

.. warning:: **HOT-PATH USAGE IS PROHIBITED.**

    This module makes synchronous or async calls to an external LLM API.
    It MUST NOT be called from:
    - ``BaseStrategy.generate_signal()`` or any method that runs per-tick
    - ``StrategyEngine.evaluate()`` or any method called in the Rust hot path
    - Any WebSocket message handler or orderbook-update callback

    Use this module ONLY from the :class:`~ai.regime_service.RegimeService`
    background loop (runs every ``REGIME_INTERVAL_SECONDS``, default 5 min).
    The strategy engine reads pre-computed regime state from
    ``/dev/shm/regime_state.json`` instead.
"""

import asyncio
from typing import List, Optional

from loguru import logger

from ai.sentiment.analyzer import _LABEL_THRESHOLDS, SentimentResult

_SYSTEM_PROMPT = (
    "You are a financial sentiment analyst specializing in cryptocurrency markets. "
    "Analyze the sentiment of text and respond ONLY with valid JSON."
)

_PROMPT_TEMPLATE = """\
Analyze the sentiment of the following text and return a JSON object with these exact keys:
- "score": float between -1.0 (very bearish) and 1.0 (very bullish)
- "label": one of "very_bearish", "bearish", "neutral", "bullish", "very_bullish"
- "confidence": float between 0.0 and 1.0
- "reasoning": brief one-sentence explanation

Text to analyze:
\"\"\"{text}\"\"\"

{context_block}Respond ONLY with valid JSON."""


def _score_to_label(score: float) -> str:
    for low, high, label in _LABEL_THRESHOLDS:
        if low <= score < high:
            return label
    return "neutral"


class LLMSentimentAnalyzer:
    """Sentiment analyzer that delegates to an LLM via structured JSON prompts.

    This provides the most nuanced understanding of crypto-specific language,
    sarcasm, and context, at the cost of higher latency and token usage.

    Falls back to a neutral result when no LLM client is provided or when the
    LLM call fails.
    """

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build the classification prompt."""
        context_block = f"Additional context: {context}\n\n" if context else ""
        return _PROMPT_TEMPLATE.format(text=text[:1000], context_block=context_block)

    def _parse_result(self, raw: dict) -> SentimentResult:
        """Convert an LLM JSON dict to a :class:`SentimentResult`."""
        score = max(-1.0, min(1.0, float(raw.get("score", 0.0))))
        label = raw.get("label") or _score_to_label(score)
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        return SentimentResult(
            score=score,
            label=label,
            confidence=confidence,
            breakdown={"llm": score},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, text: str, context: Optional[str] = None) -> SentimentResult:
        """Analyze *text* using the LLM.

        Args:
            text:    Raw text to analyze.
            context: Optional additional context to include in the prompt.

        Returns:
            :class:`SentimentResult` with score in [-1, 1].
        """
        if not self._llm:
            return SentimentResult(score=0.0, label="neutral", confidence=0.1, breakdown={})
        try:
            prompt = self._build_prompt(text, context)
            raw = await self._llm.query_json(prompt, _SYSTEM_PROMPT)
            if raw and "error" not in raw:
                return self._parse_result(raw)
        except Exception as exc:
            logger.warning(f"LLMSentimentAnalyzer.analyze error: {exc}")
        return SentimentResult(score=0.0, label="neutral", confidence=0.0, breakdown={})

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts concurrently.

        Args:
            texts: List of raw text strings.

        Returns:
            List of :class:`SentimentResult` objects (same order as input).
        """
        return list(await asyncio.gather(*[self.analyze(t) for t in texts]))
