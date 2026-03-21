"""Crypto Fear & Greed Index monitor."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FearGreedMonitor(BaseSource):
    """Tracks the Crypto Fear & Greed Index from alternative.me API."""

    API_URL = "https://api.alternative.me/fng/?limit=10&format=json"

    def __init__(self):
        super().__init__("fear_greed", DataSourceType.REST_API)
        self._current_value: Optional[int] = None
        self._current_label: Optional[str] = None
        self._history: List[dict] = []
        self._polling_interval = 3600  # 1 hour (index updates once daily)

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Fear & Greed Monitor started")
        while self._running:
            try:
                await self._fetch_and_update()
                await asyncio.sleep(self._polling_interval)
            except Exception as e:
                logger.error(f"Fear & Greed fetch error: {e}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 10) -> List[DataItem]:
        await self._fetch_and_update()
        if self._current_value is None:
            return []
        return [
            DataItem(
                source_type=self.source_type,
                source_name=self.name,
                content=f"Fear & Greed Index: {self._current_value} ({self._current_label})",
                timestamp=_utcnow(),
                metadata={
                    "value": self._current_value,
                    "label": self._current_label,
                    "history": self._history[:10],
                },
                relevance_score=0.7,
                urgency_score=0.3 if 25 <= self._current_value <= 75 else 0.7,
            )
        ]

    async def _fetch_and_update(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.API_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json(content_type=None)
        items = data.get("data", [])
        if items:
            self._current_value = int(items[0]["value"])
            self._current_label = items[0]["value_classification"]
            self._history = [
                {
                    "value": int(i["value"]),
                    "label": i["value_classification"],
                    "timestamp": i["timestamp"],
                }
                for i in items
            ]
            self._last_update = _utcnow()
            self._items_collected += 1
            logger.debug(f"Fear & Greed: {self._current_value} ({self._current_label})")

    def get_signal(self) -> str:
        """Convert index value to a descriptive trading signal."""
        if self._current_value is None:
            return "neutral"
        if self._current_value <= 20:
            return "extreme_fear_buy"  # Potential bottom
        elif self._current_value <= 40:
            return "fear_accumulate"
        elif self._current_value <= 60:
            return "neutral"
        elif self._current_value <= 80:
            return "greed_caution"
        else:
            return "extreme_greed_sell"  # Potential top

    @property
    def current_index(self) -> Optional[int]:
        return self._current_value
