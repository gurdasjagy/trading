"""Aggregates data from all configured data sources."""

import asyncio
from typing import Dict, List

from loguru import logger

from .cache import CacheManager
from .normalizer import DataNormalizer
from .sources.base_source import BaseSource, DataItem


class DataAggregator:
    """Collects, normalises, and aggregates data from all registered sources."""

    def __init__(self) -> None:
        self._sources: Dict[str, BaseSource] = {}
        self._normalizer = DataNormalizer()
        self._cache = CacheManager.get_instance()
        self._aggregated_items: List[DataItem] = []
        self._lock = asyncio.Lock()

    def register_source(self, source: BaseSource) -> None:
        """Register a data source by name."""
        self._sources[source.name] = source
        logger.info(f"Registered data source: {source.name}")

    async def start_all_sources(self) -> None:
        """Start background monitoring for every enabled source."""
        tasks = [
            asyncio.create_task(source.start_monitoring())
            for source in self._sources.values()
            if source.enabled
        ]
        logger.info(f"Started {len(tasks)} data source(s)")

    async def stop_all_sources(self) -> None:
        """Stop all running sources."""
        for source in self._sources.values():
            await source.stop_monitoring()

    async def collect_latest(
        self,
        max_age_minutes: int = 60,
        limit_per_source: int = 50,
    ) -> List[DataItem]:
        """
        Pull the latest items from every enabled source, normalise them,
        remove stale/duplicate entries, and sort by combined urgency + relevance.
        """
        all_items: List[DataItem] = []
        for source in self._sources.values():
            if not source.enabled:
                continue
            try:
                items = await source.fetch_latest(limit=limit_per_source)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Failed to collect from {source.name}: {e}")

        normalized = self._normalizer.normalize_batch(all_items)
        filtered = self._normalizer.filter_stale(normalized, max_age_minutes)
        unique = self._normalizer.deduplicate(filtered)
        unique.sort(key=lambda x: x.urgency_score + x.relevance_score, reverse=True)

        async with self._lock:
            self._aggregated_items = unique[:500]

        return unique

    async def get_items_for_symbol(self, symbol: str, limit: int = 20) -> List[DataItem]:
        """Return recently aggregated items that mention *symbol*."""
        base = symbol.replace("/USDT", "").replace("USDT", "")
        return [item for item in self._aggregated_items if base in item.mentioned_assets][:limit]

    def get_cached_items(self, limit: int = 50) -> List[DataItem]:
        """Return the most recently aggregated items without triggering a new fetch."""
        return self._aggregated_items[:limit]

    async def get_source_status(self) -> Dict[str, dict]:
        return {name: source.status for name, source in self._sources.items()}

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def enabled_count(self) -> int:
        return sum(1 for s in self._sources.values() if s.enabled)
