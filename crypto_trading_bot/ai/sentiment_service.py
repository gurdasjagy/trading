"""AI Sentiment Service — cold-path LLM integration for market sentiment.

**Issue 4**: Runs as an isolated background service that queries LLM APIs
for market sentiment every 15 minutes.  Completely decoupled from the Rust
hot path — communicates results via :class:`~ai.regime_computer.SentimentInput`.

Key guarantees:
    - **Hard timeout**: Every LLM call has a 60-second hard timeout.
    - **Non-fatal**: All failures return neutral defaults (score=0, confidence=0).
    - **No hot-path impact**: Even if every LLM provider is down, the Rust
      engine continues unaffected with the last-known-good regime weights.

Architecture::

    SentimentService.analyze()
        ├─ _fetch_fear_greed_index()   (alternative.me API, 10s timeout)
        ├─ _fetch_headlines()           (news sources, 15s timeout)
        ├─ _query_llm_sentiment()       (LLM provider chain, 60s timeout)
        └─ _parse_llm_response()        (JSON parsing + fallback)
             ↓
        SentimentResult
             ↓
        RegimeComputer.update_sentiment()
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════
# SentimentResult
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SentimentResult:
    """Output of a single sentiment analysis cycle.

    All numeric fields default to neutral / zero so that a failed analysis
    is harmless when fed into :class:`~ai.regime_computer.RegimeComputer`.
    """

    score: float = 0.0                  # -1.0 (bearish) to +1.0 (bullish)
    confidence: float = 0.0             # 0.0 to 1.0
    fear_greed_index: int = 50          # 0 (extreme fear) to 100 (extreme greed)
    news_impact_score: float = 0.0      # 0.0 to 1.0 (high = disruptive news)
    btc_dominance_trend: str = "flat"   # flat / rising / falling
    funding_rate_bias: str = "neutral"  # neutral / long_crowded / short_crowded
    headlines: List[str] = field(default_factory=list)
    llm_provider: str = "none"          # Which LLM provider was used
    analysis_duration_ms: int = 0       # How long the analysis took
    timestamp_ms: int = 0               # When this result was generated
    error: Optional[str] = None         # Error message if analysis failed

    @classmethod
    def neutral_default(cls) -> "SentimentResult":
        """Return a neutral-sentiment result (safe default)."""
        return cls(
            score=0.0,
            confidence=0.0,
            fear_greed_index=50,
            news_impact_score=0.0,
            btc_dominance_trend="flat",
            funding_rate_bias="neutral",
            timestamp_ms=int(time.time() * 1000),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SentimentService
# ═══════════════════════════════════════════════════════════════════════════


class SentimentService:
    """Queries LLM APIs and news sources for market sentiment.

    Parameters
    ----------
    llm_client :
        :class:`~ai.llm_client.LLMClient` instance.  If ``None``, the
        service returns neutral defaults without attempting any LLM call.
    news_sources :
        List of news-source objects that have an async ``fetch()`` method.
    llm_timeout_seconds :
        Hard timeout for LLM calls (default: 60).
    fear_greed_timeout_seconds :
        Timeout for the Fear & Greed Index API (default: 10).
    headlines_timeout_seconds :
        Timeout for fetching news headlines (default: 15).
    """

    _SENTIMENT_PROMPT_TEMPLATE = """You are a crypto market analyst. Analyze the current market conditions and provide a sentiment assessment.

Current market data:
- Fear & Greed Index: {fear_greed}
- Recent headlines: {headlines}
- Market context: {context}

Respond in STRICT JSON format only (no markdown, no explanation outside JSON):
{{
    "sentiment_score": <float between -1.0 (very bearish) and 1.0 (very bullish)>,
    "confidence": <float between 0.0 and 1.0>,
    "news_impact": <float between 0.0 (no impact) and 1.0 (high impact)>,
    "btc_dominance_trend": "<flat|rising|falling>",
    "funding_bias": "<neutral|long_crowded|short_crowded>",
    "key_factors": [<list of key factor strings>],
    "reasoning": "<brief reasoning>"
}}"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        news_sources: Optional[List[Any]] = None,
        llm_timeout_seconds: float = 60.0,
        fear_greed_timeout_seconds: float = 10.0,
        headlines_timeout_seconds: float = 15.0,
    ) -> None:
        self._llm = llm_client
        self._news_sources = news_sources or []
        self._llm_timeout = llm_timeout_seconds
        self._fear_greed_timeout = fear_greed_timeout_seconds
        self._headlines_timeout = headlines_timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> SentimentResult:
        """Run a complete sentiment analysis cycle.

        Fetches the Fear & Greed Index, gathers news headlines, queries
        an LLM for sentiment assessment, and returns a
        :class:`SentimentResult`.

        **Every sub-step has a hard timeout.**  If all steps fail, a
        neutral default is returned.

        Parameters
        ----------
        market_context :
            Optional dict with live market data to include in the LLM prompt
            (e.g. regime, total PnL, top symbols).

        Returns
        -------
        SentimentResult
            Always returns a valid result — never raises.
        """
        start_ms = int(time.time() * 1000)
        context = market_context or {}

        try:
            # 1. Fetch Fear & Greed Index (parallel with headlines)
            fear_greed_task = asyncio.create_task(self._fetch_fear_greed_index())
            headlines_task = asyncio.create_task(self._fetch_headlines())

            fear_greed = await fear_greed_task
            headlines = await headlines_task

            # 2. Query LLM for sentiment (with hard timeout)
            if self._llm is not None:
                llm_result = await self._query_llm_sentiment(
                    fear_greed, headlines, context
                )
            else:
                llm_result = None

            # 3. Build result
            result = self._build_result(fear_greed, headlines, llm_result)
            result.analysis_duration_ms = int(time.time() * 1000) - start_ms
            result.timestamp_ms = int(time.time() * 1000)

            logger.info(
                "SentimentService: score={:.2f} confidence={:.2f} "
                "fear_greed={} news_impact={:.2f} provider={} ({}ms)",
                result.score,
                result.confidence,
                result.fear_greed_index,
                result.news_impact_score,
                result.llm_provider,
                result.analysis_duration_ms,
            )

            return result

        except Exception as exc:
            logger.error("SentimentService.analyze() failed: {}", exc, exc_info=True)
            result = SentimentResult.neutral_default()
            result.error = str(exc)
            result.analysis_duration_ms = int(time.time() * 1000) - start_ms
            return result

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_fear_greed_index(self) -> int:
        """Fetch the Fear & Greed Index from alternative.me API."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=self._fear_greed_timeout),
                ) as resp:
                    data = await resp.json()
                    value = int(data["data"][0]["value"])
                    logger.debug("SentimentService: Fear & Greed Index = {}", value)
                    return max(0, min(100, value))
        except ImportError:
            logger.debug("SentimentService: aiohttp not available, using default F&G=50")
            return 50
        except asyncio.TimeoutError:
            logger.warning("SentimentService: Fear & Greed API timeout ({}s)", self._fear_greed_timeout)
            return 50
        except Exception as exc:
            logger.warning("SentimentService: Fear & Greed fetch failed: {}", exc)
            return 50

    async def _fetch_headlines(self) -> List[str]:
        """Gather recent news headlines from configured news sources."""
        if not self._news_sources:
            return []

        headlines: List[str] = []
        for source in self._news_sources:
            try:
                items = await asyncio.wait_for(
                    source.fetch(),
                    timeout=self._headlines_timeout,
                )
                for item in (items or []):
                    title = item.get("title", "") if isinstance(item, dict) else str(item)
                    if title:
                        headlines.append(title[:200])  # Truncate very long titles
            except asyncio.TimeoutError:
                logger.debug("SentimentService: news source timeout")
            except Exception as exc:
                logger.debug("SentimentService: news source error: {}", exc)

        # Limit to most recent headlines
        return headlines[:20]

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    async def _query_llm_sentiment(
        self,
        fear_greed: int,
        headlines: List[str],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Query the LLM for a structured sentiment assessment.

        Returns the parsed JSON dict, or None on failure/timeout.
        """
        if self._llm is None:
            return None

        prompt = self._build_prompt(fear_greed, headlines, context)

        try:
            result = await asyncio.wait_for(
                self._llm.analyze_market(prompt),
                timeout=self._llm_timeout,
            )
            if result and isinstance(result, dict) and "error" not in result:
                return result

            # Try string response parsing
            if result and isinstance(result, str):
                return self._parse_llm_response(result)

        except asyncio.TimeoutError:
            logger.warning(
                "SentimentService: LLM timeout ({}s) — using neutral defaults",
                self._llm_timeout,
            )
        except Exception as exc:
            logger.warning("SentimentService: LLM query failed: {}", exc)

        return None

    def _build_prompt(
        self,
        fear_greed: int,
        headlines: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Build the LLM prompt with current market data."""
        headlines_str = "\n".join(f"- {h}" for h in headlines[:10]) or "No recent headlines available"
        context_str = json.dumps(context, default=str)[:500] if context else "No additional context"

        return self._SENTIMENT_PROMPT_TEMPLATE.format(
            fear_greed=fear_greed,
            headlines=headlines_str,
            context=context_str,
        )

    @staticmethod
    def _parse_llm_response(raw: Any) -> Optional[Dict[str, Any]]:
        """Parse an LLM response into a structured dict.

        Handles JSON within markdown code blocks and bare JSON strings.
        Returns None if parsing fails.
        """
        if raw is None:
            return None

        text = str(raw).strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.debug("SentimentService: could not parse LLM response as JSON")
        return None

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(
        self,
        fear_greed: int,
        headlines: List[str],
        llm_result: Optional[Dict[str, Any]],
    ) -> SentimentResult:
        """Combine all data sources into a single SentimentResult."""
        result = SentimentResult(
            fear_greed_index=fear_greed,
            headlines=headlines,
        )

        # Derive basic sentiment from Fear & Greed
        # F&G < 25 = extreme fear (-0.5), 25-45 = fear (-0.25), 55-75 = greed (+0.25), > 75 = extreme greed (+0.5)
        if fear_greed < 25:
            result.score = -0.5
            result.confidence = 0.3
        elif fear_greed < 45:
            result.score = -0.25
            result.confidence = 0.2
        elif fear_greed > 75:
            result.score = 0.5
            result.confidence = 0.3
        elif fear_greed > 55:
            result.score = 0.25
            result.confidence = 0.2
        else:
            result.score = 0.0
            result.confidence = 0.1

        # Override with LLM results if available (LLM is more nuanced)
        if llm_result is not None:
            result.llm_provider = llm_result.get("provider", "llm")

            # Sentiment score
            llm_score = llm_result.get("sentiment_score")
            if llm_score is not None:
                try:
                    llm_score_f = float(llm_score)
                    result.score = max(-1.0, min(1.0, llm_score_f))
                except (ValueError, TypeError):
                    pass

            # Confidence
            llm_conf = llm_result.get("confidence")
            if llm_conf is not None:
                try:
                    result.confidence = max(0.0, min(1.0, float(llm_conf)))
                except (ValueError, TypeError):
                    pass

            # News impact
            llm_impact = llm_result.get("news_impact")
            if llm_impact is not None:
                try:
                    result.news_impact_score = max(0.0, min(1.0, float(llm_impact)))
                except (ValueError, TypeError):
                    pass

            # BTC dominance trend
            dom = llm_result.get("btc_dominance_trend", "")
            if dom in ("flat", "rising", "falling"):
                result.btc_dominance_trend = dom

            # Funding bias
            bias = llm_result.get("funding_bias", "")
            if bias in ("neutral", "long_crowded", "short_crowded"):
                result.funding_rate_bias = bias

        # News impact boost: many headlines = higher impact score
        if len(headlines) > 10 and result.news_impact_score < 0.5:
            result.news_impact_score = min(1.0, result.news_impact_score + 0.2)

        return result

