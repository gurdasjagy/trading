"""Tests for data source components."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.normalizer import DataNormalizer
from data.sources.base_source import BaseSource, DataItem, DataSourceType
from data.sources.fear_greed_monitor import FearGreedMonitor
from data.sources.funding_rate_monitor import FundingRateMonitor
from data.sources.news_rss_monitor import NewsRSSMonitor

# ── BaseSource abstract ───────────────────────────────────────────────────


class TestBaseSourceAbstract:
    def test_cannot_instantiate_directly(self):
        """BaseSource cannot be instantiated because it has abstract methods."""
        with pytest.raises(TypeError):
            BaseSource(name="test", source_type=DataSourceType.REST_API)  # type: ignore[abstract]

    def test_concrete_subclass_can_be_created(self):
        """A concrete subclass that implements all abstract methods can be created."""

        class ConcreteSource(BaseSource):
            async def start_monitoring(self) -> None: ...
            async def stop_monitoring(self) -> None: ...
            async def fetch_latest(self, limit: int = 50):
                return []

        src = ConcreteSource(name="test", source_type=DataSourceType.REST_API)
        assert src.name == "test"
        assert src.enabled is True

    def test_status_property(self):
        """status property returns expected keys."""

        class ConcreteSource(BaseSource):
            async def start_monitoring(self) -> None: ...
            async def stop_monitoring(self) -> None: ...
            async def fetch_latest(self, limit: int = 50):
                return []

        src = ConcreteSource(name="test-src", source_type=DataSourceType.REST_API)
        status = src.status
        assert "name" in status
        assert "enabled" in status
        assert "running" in status

    def test_extract_mentioned_assets(self):
        """_extract_mentioned_assets correctly finds BTC and ETH in text."""

        class ConcreteSource(BaseSource):
            async def start_monitoring(self): ...
            async def stop_monitoring(self): ...
            async def fetch_latest(self, limit=50):
                return []

        src = ConcreteSource("test", DataSourceType.REST_API)
        assets = src._extract_mentioned_assets("Bitcoin and ethereum are rising")
        assert "BTC" in assets
        assert "ETH" in assets


# ── FearGreedMonitor ──────────────────────────────────────────────────────


class TestFearGreedMonitorInit:
    def test_initializes_without_error(self):
        """FearGreedMonitor can be instantiated without raising."""
        monitor = FearGreedMonitor()
        assert monitor is not None

    def test_is_base_source_subclass(self):
        """FearGreedMonitor is a subclass of BaseSource."""
        monitor = FearGreedMonitor()
        assert isinstance(monitor, BaseSource)

    def test_default_enabled(self):
        """FearGreedMonitor is enabled by default."""
        monitor = FearGreedMonitor()
        assert monitor.enabled is True


# ── NewsRSSMonitor ─────────────────────────────────────────────────────────


class TestNewsRSSMonitorInit:
    def test_initializes_without_error(self):
        """NewsRSSMonitor can be instantiated without raising."""
        monitor = NewsRSSMonitor(feed_urls=["https://example.com/rss"])
        assert monitor is not None

    def test_is_base_source_subclass(self):
        """NewsRSSMonitor is a subclass of BaseSource."""
        monitor = NewsRSSMonitor(feed_urls=[])
        assert isinstance(monitor, BaseSource)


# ── FundingRateMonitor ────────────────────────────────────────────────────


class TestFundingRateMonitorInit:
    def test_initializes_without_error(self):
        """FundingRateMonitor can be instantiated without raising."""
        monitor = FundingRateMonitor()
        assert monitor is not None

    def test_is_base_source_subclass(self):
        """FundingRateMonitor is a subclass of BaseSource."""
        monitor = FundingRateMonitor()
        assert isinstance(monitor, BaseSource)


# ── DataNormalizer ────────────────────────────────────────────────────────


class TestDataNormalizer:
    def _make_item(self, content: str = "BTC is pumping") -> DataItem:
        return DataItem(
            source_type=DataSourceType.RSS_FEED,
            source_name="test_feed",
            content=content,
            timestamp=datetime.now(timezone.utc),
            relevance_score=0.7,
            urgency_score=0.5,
        )

    def test_normalize_returns_data_item(self):
        """normalize_item returns a DataItem instance."""
        normalizer = DataNormalizer()
        item = self._make_item("Bitcoin is surging today!")
        result = normalizer.normalize_item(item)
        assert isinstance(result, DataItem)

    def test_scores_clamped(self):
        """normalize_item clamps relevance/urgency scores to [0, 1]."""
        normalizer = DataNormalizer()
        item = self._make_item()
        item.relevance_score = 5.0
        item.urgency_score = -3.0
        normalized = normalizer.normalize_item(item)
        assert 0.0 <= normalized.relevance_score <= 1.0
        assert 0.0 <= normalized.urgency_score <= 1.0

    def test_mentioned_assets_extracted(self):
        """normalize_item extracts mentioned assets from content."""
        normalizer = DataNormalizer()
        item = self._make_item("Ethereum and bitcoin are both rising")
        item.mentioned_assets = []
        normalized = normalizer.normalize_item(item)
        assert len(normalized.mentioned_assets) >= 1

    def test_batch_normalization(self):
        """normalize_batch processes all items and returns same count."""
        normalizer = DataNormalizer()
        items = [self._make_item(f"item {i}") for i in range(5)]
        results = normalizer.normalize_batch(items)
        assert len(results) == 5

    def test_timezone_stripped(self):
        """normalize_item strips timezone info from timestamps."""
        normalizer = DataNormalizer()
        item = self._make_item("test")
        item.timestamp = datetime.now(timezone.utc)
        normalized = normalizer.normalize_item(item)
        assert normalized.timestamp.tzinfo is None
