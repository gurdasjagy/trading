"""Mempool Monitor — monitors Bitcoin mempool congestion."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MempoolMonitor(BaseSource):
    """Monitors Bitcoin mempool congestion via mempool.space API."""

    MEMPOOL_FEES_API = "https://mempool.space/api/v1/fees/recommended"
    MEMPOOL_API = "https://mempool.space/api/mempool"

    def __init__(self, polling_interval: int = 600):  # 10 minutes
        super().__init__("mempool", DataSourceType.REST_API)
        self._polling_interval = polling_interval
        self._items: List[DataItem] = []
        self._prev_congestion: Optional[str] = None

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Mempool Monitor started")
        while self._running:
            try:
                item = await self.fetch_latest()
                if item:
                    self._items.extend(item)
                    if len(self._items) > 500:
                        self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Mempool monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 1) -> List[DataItem]:
        """Fetch current mempool status and fee rates."""
        items: List[DataItem] = []

        try:
            async with aiohttp.ClientSession() as session:
                # Fetch fee recommendations
                async with session.get(
                    self.MEMPOOL_FEES_API,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Mempool fees API status {resp.status}")
                        return items
                    fees_data = await resp.json()

                # Fetch mempool stats
                async with session.get(
                    self.MEMPOOL_API,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Mempool API status {resp.status}")
                        return items
                    mempool_data = await resp.json()

            # Extract metrics
            fastest_fee = fees_data.get("fastestFee", 0)
            half_hour_fee = fees_data.get("halfHourFee", 0)
            hour_fee = fees_data.get("hourFee", 0)
            economy_fee = fees_data.get("economyFee", 0)

            mempool_size = mempool_data.get("count", 0)
            mempool_vsize = mempool_data.get("vsize", 0)

            # Determine congestion level
            if fastest_fee > 100:
                congestion_level = "extreme"
                urgency = 0.9
            elif fastest_fee > 50:
                congestion_level = "high"
                urgency = 0.7
            elif fastest_fee > 20:
                congestion_level = "medium"
                urgency = 0.5
            else:
                congestion_level = "low"
                urgency = 0.3

            # Only create item if congestion changed or is significant
            if congestion_level != self._prev_congestion or congestion_level in ["high", "extreme"]:
                content = (
                    f"Bitcoin mempool: {congestion_level} congestion "
                    f"({mempool_size:,} txs, {mempool_vsize/1_000_000:.1f} MB) – "
                    f"Fees: {fastest_fee} sat/vB (fast), {hour_fee} sat/vB (1h)"
                )

                item = DataItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    content=content,
                    timestamp=_utcnow(),
                    metadata={
                        "mempool_size": mempool_size,
                        "mempool_vsize": mempool_vsize,
                        "fastest_fee": fastest_fee,
                        "half_hour_fee": half_hour_fee,
                        "hour_fee": hour_fee,
                        "economy_fee": economy_fee,
                        "congestion_level": congestion_level,
                    },
                    relevance_score=0.6 if congestion_level in ["high", "extreme"] else 0.4,
                    urgency_score=urgency,
                    mentioned_assets=["BTC"],
                )
                items.append(item)
                self._items_collected += 1
                self._prev_congestion = congestion_level
                logger.debug(f"Mempool: {congestion_level} congestion, {fastest_fee} sat/vB")

            self._last_update = _utcnow()

        except Exception as exc:
            logger.warning(f"Mempool fetch error: {exc}")
            self._errors += 1

        return items
