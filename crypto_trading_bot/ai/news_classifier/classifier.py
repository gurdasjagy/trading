"""News impact classifier: rule-based fast path with optional LLM refinement."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List

from loguru import logger
from pydantic import BaseModel


class NewsCategory(str, Enum):
    REGULATORY = "REGULATORY"
    TECHNICAL = "TECHNICAL"
    ADOPTION = "ADOPTION"
    MARKET = "MARKET"
    MACRO = "MACRO"
    SECURITY = "SECURITY"
    PARTNERSHIP = "PARTNERSHIP"
    DEVELOPMENT = "DEVELOPMENT"
    UNKNOWN = "UNKNOWN"


class ImpactLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOISE = "NOISE"


class NewsDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class NewsClassification(BaseModel):
    """Structured result of a news-article classification."""

    title: str
    category: NewsCategory
    impact_level: ImpactLevel
    direction: NewsDirection
    affected_assets: List[str] = []
    time_horizon: str = "SHORT"  # IMMEDIATE / SHORT / MEDIUM / LONG
    confidence: float = 0.5
    summary: str = ""
    is_fake: bool = False
    classified_at: datetime = None  # type: ignore[assignment]

    def __init__(self, **data) -> None:  # type: ignore[override]
        if not data.get("classified_at"):
            data["classified_at"] = datetime.now(tz=timezone.utc)
        super().__init__(**data)


# ---------------------------------------------------------------------------
# Keyword sets used for rule-based classification
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[NewsCategory, List[str]] = {
    NewsCategory.REGULATORY: ["sec", "regulation", "ban", "legal", "government", "cftc", "fbi"],
    NewsCategory.SECURITY: ["hack", "exploit", "rug pull", "scam", "phishing", "stolen"],
    NewsCategory.TECHNICAL: ["upgrade", "fork", "protocol", "mainnet", "testnet", "github"],
    NewsCategory.PARTNERSHIP: ["partnership", "integrate", "collaboration"],
    NewsCategory.MARKET: ["etf", "listing", "exchange", "coinbase", "binance"],
    NewsCategory.MACRO: ["inflation", "fomc", "interest rate", "cpi", "gdp", "fed"],
    NewsCategory.ADOPTION: ["adopt", "accept", "corporate", "institutional", "nation"],
}

_BULLISH_KEYWORDS = [
    "approved",
    "bullish",
    "surge",
    "all-time high",
    "institutional buying",
    "accumulate",
    "rally",
]
_BEARISH_KEYWORDS = ["banned", "hack", "crash", "dump", "bearish", "rejected", "lawsuit", "seized"]

_HIGH_IMPACT_KEYWORDS = [
    "hack",
    "billion",
    "crash",
    "ban",
    "emergency",
    "etf approved",
    "etf rejected",
]
_CRITICAL_REGULATORY_KEYWORDS = ["sec", "fomc", "all-time high"]
_NOISE_KEYWORDS = ["analysis", "prediction", "opinion", "could", "might"]
_FAKE_KEYWORDS = ["100x guaranteed", "elon musk gives away", "free bitcoin"]


class NewsClassifier:
    """Classifies news articles by impact level, market direction, and category.

    Uses a fast rule-based first pass.  When the LLM client is provided and the
    rule-based pass flags HIGH or CRITICAL impact, a more nuanced LLM pass is
    performed to improve accuracy.
    """

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client
        self._classification_cache: Dict[int, NewsClassification] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(
        self,
        title: str,
        content: str = "",
        source: str = "",
    ) -> NewsClassification:
        """Classify a single news article.

        Args:
            title:   Article headline.
            content: Full or partial article body.
            source:  Publication / feed name.

        Returns:
            :class:`NewsClassification` with category, impact, direction, etc.
        """
        cache_key = hash(title + content[:100])
        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        result = self._rule_based_classify(title, content, source)

        # Upgrade with LLM for high-priority items where accuracy matters most
        if self._llm and result.impact_level in (ImpactLevel.CRITICAL, ImpactLevel.HIGH):
            try:
                from ai.prompt_engine import PromptEngine

                prompt = PromptEngine().build_news_classification_prompt(title, content, source)
                llm_result = await self._llm.query_json(
                    prompt,
                    "You are a crypto news classifier. Respond with JSON only.",
                )
                if llm_result and "error" not in llm_result:
                    result = self._parse_llm_result(title, llm_result)
            except Exception as exc:
                logger.warning(f"LLM classification failed, keeping rule-based result: {exc}")

        self._classification_cache[cache_key] = result
        return result

    async def batch_classify(
        self,
        news_items: List[Dict],
    ) -> List[NewsClassification]:
        """Classify a list of news dicts (each with ``title``, ``content``, ``source`` keys).

        Items that fail classification are silently skipped.
        """
        results: List[NewsClassification] = []
        for item in news_items:
            try:
                result = await self.classify(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    source=item.get("source", ""),
                )
                results.append(result)
            except Exception as exc:
                logger.warning(f"Failed to classify news item '{item.get('title', '')}': {exc}")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rule_based_classify(
        self,
        title: str,
        content: str,
        source: str,
    ) -> NewsClassification:
        """Assign category, direction, impact, and fake-flag via keyword rules."""
        text = f"{title} {content}".lower()

        # Category — first matching set wins
        category = NewsCategory.UNKNOWN
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                category = cat
                break

        # Direction
        direction = NewsDirection.NEUTRAL
        if any(kw in text for kw in _BULLISH_KEYWORDS):
            direction = NewsDirection.BULLISH
        elif any(kw in text for kw in _BEARISH_KEYWORDS):
            direction = NewsDirection.BEARISH

        # Impact
        impact = ImpactLevel.MEDIUM
        if any(kw in text for kw in _NOISE_KEYWORDS):
            impact = ImpactLevel.NOISE
        elif any(kw in text for kw in _HIGH_IMPACT_KEYWORDS):
            impact = ImpactLevel.HIGH
        elif any(kw in text for kw in _CRITICAL_REGULATORY_KEYWORDS):
            impact = (
                ImpactLevel.CRITICAL if category == NewsCategory.REGULATORY else ImpactLevel.HIGH
            )

        # Fake-news heuristic
        is_fake = any(kw in text for kw in _FAKE_KEYWORDS)

        # Asset extraction via shared base-source helper
        assets = self._extract_assets(title + " " + content)

        return NewsClassification(
            title=title,
            category=category,
            impact_level=impact,
            direction=direction,
            affected_assets=assets,
            confidence=0.6,
            is_fake=is_fake,
            summary=title[:100],
        )

    def _parse_llm_result(self, title: str, result: Dict) -> NewsClassification:
        """Convert a validated LLM JSON dict into a :class:`NewsClassification`."""
        try:
            return NewsClassification(
                title=title,
                category=NewsCategory(result.get("category", "UNKNOWN")),
                impact_level=ImpactLevel(result.get("impact_level", "LOW")),
                direction=NewsDirection(result.get("direction", "NEUTRAL")),
                affected_assets=result.get("affected_assets", []),
                time_horizon=result.get("time_horizon", "SHORT"),
                confidence=float(result.get("confidence", 0.5)),
                summary=result.get("summary", title[:100]),
                is_fake=bool(result.get("is_fake_or_recycled", False)),
            )
        except Exception as exc:
            logger.warning(f"Failed to parse LLM classification result: {exc}")
            return self._rule_based_classify(title, "", "")

    @staticmethod
    def _extract_assets(text: str) -> List[str]:
        """Extract mentioned asset tickers from *text* using the shared keyword map."""
        from data.sources.base_source import BaseSource  # local import to avoid circular deps

        return BaseSource._extract_mentioned_assets(BaseSource.__new__(BaseSource), text)
