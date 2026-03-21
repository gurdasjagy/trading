"""Fake-news and duplicate-news detector for crypto news feeds."""

import hashlib
import re
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Source credibility configuration
# ---------------------------------------------------------------------------

# Whitelisted high-credibility sources → credibility score
_WHITELISTED_SOURCES: Dict[str, float] = {
    "coindesk": 0.92,
    "cointelegraph": 0.88,
    "the block": 0.91,
    "bloomberg": 0.95,
    "reuters": 0.96,
    "wsj": 0.95,
    "wall street journal": 0.95,
    "ft": 0.94,
    "financial times": 0.94,
    "decrypt": 0.85,
    "blockworks": 0.87,
    "axios": 0.90,
    "bbc": 0.92,
    "cnbc": 0.88,
    "forbes": 0.80,
    "coinbase blog": 0.85,
    "binance blog": 0.78,
    "messari": 0.88,
    "glassnode": 0.89,
    "delphi digital": 0.87,
    "nansen": 0.86,
}

# Known low-credibility / spam source patterns
_BLACKLISTED_PATTERNS: Tuple[str, ...] = (
    "cryptopump",
    "moonshot",
    "100x",
    "getrichquick",
    "freecrypto",
    "airdrop-",
    "giveaway-",
)

# Phrases that strongly indicate fake / clickbait news
_FAKE_PHRASES: Tuple[str, ...] = (
    "100x guaranteed",
    "guaranteed profit",
    "elon musk gives away",
    "free bitcoin",
    "double your bitcoin",
    "click here to claim",
    "limited time offer",
    "secret method",
    "risk-free investment",
    "get rich quick",
    "not financial advice but definitely buy",
)

# Minimum similarity ratio to flag a pair as duplicate
_DUPLICATE_THRESHOLD: float = 0.80


class FakeNewsDetector:
    """Filters misinformation and duplicate/recycled news from crypto feeds.

    Uses a combination of:
    - Hard-coded fake-phrase detection
    - Source whitelisting / blacklisting
    - Fuzzy title similarity for duplicate detection
    """

    def __init__(self) -> None:
        self._seen_hashes: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_fake(self, news_item: Dict) -> bool:
        """Return ``True`` if *news_item* appears to be fake or misleading.

        Args:
            news_item: Dict with at least ``"title"`` and optionally
                       ``"content"`` and ``"source"`` keys.

        Returns:
            ``True`` if the item is flagged as fake.
        """
        try:
            title: str = news_item.get("title", "")
            content: str = news_item.get("content", "")
            source: str = news_item.get("source", "")

            text = f"{title} {content}".lower()

            # Hard-coded phrase check
            if any(phrase in text for phrase in _FAKE_PHRASES):
                logger.debug(f"Fake-news phrase detected in: '{title[:80]}'")
                return True

            # Blacklisted source domains
            source_lower = source.lower()
            if any(pattern in source_lower for pattern in _BLACKLISTED_PATTERNS):
                logger.debug(f"Blacklisted source: '{source}'")
                return True

            # Very low credibility
            if self.calculate_source_credibility(source) < 0.3:
                logger.debug(f"Low-credibility source flagged: '{source}'")
                return True

            return False
        except Exception as exc:
            logger.warning(f"FakeNewsDetector.is_fake error: {exc}")
            return False

    def is_duplicate(self, news_item: Dict, recent_news: List[Dict]) -> bool:
        """Return ``True`` if *news_item* is a near-duplicate of an item in *recent_news*.

        Uses a content hash for exact duplicates and fuzzy title matching for
        recycled / slightly-reworded articles.

        Args:
            news_item:   The candidate article dict (``title``, ``content``).
            recent_news: List of previously seen article dicts to compare against.

        Returns:
            ``True`` if a duplicate is detected.
        """
        try:
            title = news_item.get("title", "")
            content = news_item.get("content", "")

            # Exact content hash check
            fingerprint = self._make_hash(title, content)
            if fingerprint in self._seen_hashes:
                logger.debug(f"Exact duplicate detected: '{title[:80]}'")
                return True
            self._seen_hashes.add(fingerprint)

            # Fuzzy title similarity
            title_norm = _normalize(title)
            for other in recent_news:
                other_title_norm = _normalize(other.get("title", ""))
                if not other_title_norm:
                    continue
                ratio = SequenceMatcher(None, title_norm, other_title_norm).ratio()
                if ratio >= _DUPLICATE_THRESHOLD:
                    logger.debug(f"Near-duplicate detected (sim={ratio:.2f}): '{title[:60]}'")
                    return True

            return False
        except Exception as exc:
            logger.warning(f"FakeNewsDetector.is_duplicate error: {exc}")
            return False

    def calculate_source_credibility(self, source: str) -> float:
        """Return a credibility score in [0, 1] for *source*.

        Whitelisted sources get their configured score; unknown sources receive
        a moderate default of 0.5; blacklisted patterns receive 0.1.

        Args:
            source: Source name or URL.

        Returns:
            Credibility score in [0.0, 1.0].
        """
        if not source:
            return 0.5  # Unknown source — neutral

        source_lower = source.lower()

        # Blacklist check
        if any(pattern in source_lower for pattern in _BLACKLISTED_PATTERNS):
            return 0.1

        return self._check_source_whitelist(source)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_source_whitelist(self, source: str) -> float:
        """Return the credibility score from the whitelist, or 0.5 for unknowns.

        Args:
            source: Source name or URL string.

        Returns:
            Float in [0.0, 1.0].
        """
        source_lower = source.lower()
        for name, score in _WHITELISTED_SOURCES.items():
            if name in source_lower:
                return score
        return 0.5

    @staticmethod
    def _make_hash(title: str, content: str) -> str:
        """Create a short SHA-256 fingerprint from title + first 200 chars of content."""
        raw = f"{title.strip()}{content[:200].strip()}".encode("utf-8", errors="replace")
        return hashlib.sha256(raw).hexdigest()[:16]


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
