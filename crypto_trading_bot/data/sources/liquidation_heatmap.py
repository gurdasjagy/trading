"""Liquidation heatmap data source — fetches liquidation cluster data from Coinglass API."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LiquidationHeatmapSource(BaseSource):
    """Fetches liquidation cluster data from Coinglass API.
    
    Identifies price levels with concentrated liquidations that may act as
    price magnets during volatile market conditions.
    """

    API_URL = "https://open-api.coinglass.com/public/v2/liquidation_map"
    
    def __init__(
        self,
        polling_interval: int = 600,  # 10 minutes
        min_cluster_size_usd: float = 50_000_000.0,  # $50M minimum
    ) -> None:
        super().__init__("liquidation_heatmap", DataSourceType.REST_API, enabled=True)
        self._polling_interval = polling_interval
        self._min_cluster_size = min_cluster_size_usd
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Liquidation Heatmap Monitor started (poll every {}s)", self._polling_interval)
        while self._running:
            try:
                await self._fetch_and_cache()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error("Liquidation heatmap fetch error: {}", exc)
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_and_cache()
        return self._items[-limit:]

    async def fetch_liquidation_clusters(self, symbol: str) -> List[dict]:
        """Fetch liquidation cluster data for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g. "BTC", "ETH")
            
        Returns:
            List of cluster dicts with keys: price, size_usd, side
        """
        try:
            params = {"symbol": symbol.replace("/USDT", "").replace("/", "")}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.API_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Coinglass API returned status {}", resp.status)
                        return []
                    data = await resp.json(content_type=None)
            
            # Parse liquidation map data
            clusters = []
            if isinstance(data, dict) and "data" in data:
                liq_data = data["data"]
                if isinstance(liq_data, list):
                    for entry in liq_data:
                        price = float(entry.get("price", 0))
                        size_usd = float(entry.get("amount", 0))
                        side = entry.get("side", "").lower()
                        
                        if size_usd >= self._min_cluster_size:
                            clusters.append({
                                "price": price,
                                "size_usd": size_usd,
                                "side": side,
                            })
            
            logger.debug("Fetched {} liquidation clusters for {}", len(clusters), symbol)
            return clusters
        except Exception as exc:
            logger.error("Failed to fetch liquidation clusters for {}: {}", symbol, exc)
            self._errors += 1
            return []

    async def _fetch_and_cache(self) -> None:
        """Fetch liquidation data for major symbols and cache as DataItems."""
        symbols = ["BTC", "ETH", "SOL", "BNB"]
        new_items = []
        
        for symbol in symbols:
            clusters = await self.fetch_liquidation_clusters(symbol)
            if clusters:
                # Create summary item
                total_size = sum(c["size_usd"] for c in clusters)
                max_cluster = max(clusters, key=lambda c: c["size_usd"])
                
                content = (
                    f"Liquidation clusters for {symbol}: "
                    f"{len(clusters)} clusters totaling ${total_size/1e9:.2f}B. "
                    f"Largest cluster: ${max_cluster['size_usd']/1e6:.0f}M at "
                    f"${max_cluster['price']:.2f} ({max_cluster['side']})"
                )
                
                # Relevance scales with total cluster size
                relevance = min(1.0, total_size / 500_000_000.0)
                # Urgency scales with largest cluster size
                urgency = min(1.0, max_cluster["size_usd"] / 100_000_000.0)
                
                item = DataItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    content=content,
                    timestamp=_utcnow(),
                    metadata={
                        "symbol": symbol,
                        "clusters": clusters,
                        "total_size_usd": total_size,
                        "max_cluster": max_cluster,
                    },
                    relevance_score=round(relevance, 3),
                    urgency_score=round(urgency, 3),
                    mentioned_assets=[symbol],
                )
                new_items.append(item)
        
        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        self._last_update = _utcnow()
        logger.debug("Liquidation heatmap: cached {} items", len(new_items))
