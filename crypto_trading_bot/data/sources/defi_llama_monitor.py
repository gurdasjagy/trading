"""DeFiLlama TVL and protocol data monitor."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DefiLlamaMonitor(BaseSource):
    """Tracks DeFi Total Value Locked (TVL) and protocol data via the DeFiLlama API."""

    DEFILLAMA_API = "https://api.llama.fi"

    # Percentage change threshold to flag as significant
    TVL_CHANGE_THRESHOLD_PCT = 10.0

    def __init__(
        self,
        polling_interval: int = 900,  # 15 minutes
        top_protocols: int = 50,
    ):
        super().__init__("defillama", DataSourceType.REST_API)
        self._polling_interval = polling_interval
        self._top_protocols = top_protocols
        self._prev_tvl: Dict[str, float] = {}  # slug -> tvl USD
        self._items: List[DataItem] = []
        self._protocols_cache: List[dict] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("DeFiLlama Monitor started")
        while self._running:
            try:
                await self.track_tvl_changes()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"DeFiLlama monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(120)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self.track_tvl_changes()
        return self._items[-limit:]

    async def fetch_tvl(self) -> float:
        """Fetch total DeFi TVL across all protocols."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.DEFILLAMA_API}/v2/historicalChainTvl",
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            return float(data[-1].get("tvl", 0))
        except Exception as exc:
            logger.warning(f"DeFiLlama fetch_tvl error: {exc}")
        return 0.0

    async def fetch_protocols(self) -> List[dict]:
        """Fetch list of protocols with current TVL."""
        protocols: List[dict] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.DEFILLAMA_API}/protocols",
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        protocols = await resp.json()
                        self._protocols_cache = protocols
                        self._last_update = _utcnow()
        except Exception as exc:
            logger.warning(f"DeFiLlama fetch_protocols error: {exc}")
            self._errors += 1
        return protocols

    async def track_tvl_changes(self) -> List[DataItem]:
        """Detect significant TVL changes across top protocols."""
        new_items: List[DataItem] = []
        protocols = await self.fetch_protocols()
        # Sort by TVL descending and take top N
        protocols_sorted = sorted(protocols, key=lambda p: float(p.get("tvl") or 0), reverse=True)[
            : self._top_protocols
        ]

        for protocol in protocols_sorted:
            slug = protocol.get("slug") or protocol.get("name", "unknown")
            current_tvl = float(protocol.get("tvl") or 0)
            prev_tvl = self._prev_tvl.get(slug)
            if prev_tvl is not None and prev_tvl > 0 and current_tvl > 0:
                change_pct = ((current_tvl - prev_tvl) / prev_tvl) * 100
                if abs(change_pct) >= self.TVL_CHANGE_THRESHOLD_PCT:
                    item = self._calculate_signal(protocol, change_pct)
                    if item:
                        new_items.append(item)
                        self._items_collected += 1
            self._prev_tvl[slug] = current_tvl

        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        return new_items

    def _calculate_signal(self, protocol: dict, tvl_change: float) -> Optional[DataItem]:
        """Convert a TVL movement into a DataItem signal."""
        name = protocol.get("name", "Unknown")
        slug = protocol.get("slug") or protocol.get("name", "")
        chain = protocol.get("chain") or protocol.get("chains", [""])[0]
        current_tvl = float(protocol.get("tvl") or 0)
        category = protocol.get("category", "DeFi")
        direction = "increased" if tvl_change > 0 else "decreased"
        signal = "inflow_bullish" if tvl_change > 0 else "outflow_bearish"

        content = (
            f"DeFiLlama: {name} ({category}/{chain}) TVL {direction} "
            f"{tvl_change:+.1f}% to ${current_tvl:,.0f}. Signal: {signal}"
        )
        assets = self._extract_mentioned_assets(f"{name} {chain}")
        magnitude = min(1.0, abs(tvl_change) / 50.0)
        return DataItem(
            source_type=self.source_type,
            source_name=f"defillama/{slug}",
            content=content,
            url=f"https://defillama.com/protocol/{slug}",
            timestamp=_utcnow(),
            raw_data=protocol,
            metadata={
                "protocol": name,
                "chain": chain,
                "category": category,
                "tvl_usd": current_tvl,
                "tvl_change_pct": tvl_change,
                "signal": signal,
            },
            relevance_score=min(1.0, 0.5 + magnitude * 0.5),
            urgency_score=min(1.0, 0.3 + magnitude * 0.7),
            mentioned_assets=assets,
        )
