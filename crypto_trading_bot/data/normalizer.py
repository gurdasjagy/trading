"""Data normalization pipeline for standardizing data from all sources."""

import re
from datetime import datetime, timedelta, timezone
from typing import List

from .sources.base_source import DataItem


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DataNormalizer:
    """Normalizes and cleans data items from all sources."""

    _ASSETS = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth"],
        "SOL": ["solana", "sol"],
        "BNB": ["bnb"],
        "XRP": ["ripple", "xrp"],
        "ADA": ["cardano", "ada"],
        "DOGE": ["dogecoin", "doge"],
    }

    def __init__(self) -> None:
        self._processed_count = 0

    def normalize_item(self, item: DataItem) -> DataItem:
        """Clean and normalize a single data item in-place and return it."""
        item.content = self._clean_text(item.content)
        # Strip timezone info to ensure UTC-naive datetimes throughout
        if item.timestamp.tzinfo is not None:
            item.timestamp = item.timestamp.replace(tzinfo=None)
        # Populate mentioned assets if missing
        if not item.mentioned_assets and item.content:
            item.mentioned_assets = self._extract_assets(item.content)
        # Clamp scores to [0, 1]
        item.relevance_score = max(0.0, min(1.0, item.relevance_score))
        item.urgency_score = max(0.0, min(1.0, item.urgency_score))
        self._processed_count += 1
        return item

    def normalize_batch(self, items: List[DataItem]) -> List[DataItem]:
        """Normalize a list of data items."""
        return [self.normalize_item(item) for item in items]

    def filter_stale(self, items: List[DataItem], max_age_minutes: int = 60) -> List[DataItem]:
        """Discard items older than *max_age_minutes*."""
        cutoff = _utcnow() - timedelta(minutes=max_age_minutes)
        return [item for item in items if item.timestamp >= cutoff]

    def deduplicate(self, items: List[DataItem]) -> List[DataItem]:
        """Remove duplicate items by URL, then by leading content hash."""
        seen_urls: set = set()
        seen_content: set = set()
        unique: List[DataItem] = []
        for item in items:
            if item.url and item.url in seen_urls:
                continue
            content_hash = hash(item.content[:100])
            if content_hash in seen_content:
                continue
            if item.url:
                seen_urls.add(item.url)
            seen_content.add(content_hash)
            unique.append(item)
        return unique

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"<[^>]+>", "", text)
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
        )
        return text[:2000]  # Cap length

    def _extract_assets(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [
            asset
            for asset, keywords in self._ASSETS.items()
            if any(kw in text_lower for kw in keywords)
        ]
