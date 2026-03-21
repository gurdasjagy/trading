"""Stablecoin Flow Monitor — tracks USDT/USDC supply changes via CoinGecko API."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class StablecoinFlowMonitor(BaseSource):
    """Monitors USDT/USDC supply changes to detect large mints/burns.
    
    Large mints (>$100M) indicate capital inflow → bullish signal
    Large burns indicate capital outflow → bearish signal
    """

    COINGECKO_API = "https://api.coingecko.com/api/v3/coins/{coin_id}"
    
    def __init__(
        self,
        polling_interval: int = 1800,  # 30 minutes
        large_flow_threshold_usd: float = 100_000_000.0,  # $100M
    ) -> None:
        super().__init__("stablecoin_flow", DataSourceType.REST_API, enabled=True)
        self._polling_interval = polling_interval
        self._large_flow_threshold = large_flow_threshold_usd
        self._items: List[DataItem] = []
        self._previous_supply: dict = {}  # coin_id -> supply

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Stablecoin Flow Monitor started (poll every {}s)", self._polling_interval)
        while self._running:
            try:
                await self._fetch_and_cache()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error("Stablecoin flow fetch error: {}", exc)
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_and_cache()
        return self._items[-limit:]

    async def _fetch_and_cache(self) -> None:
        """Fetch USDT/USDC supply data and detect large flows."""
        stablecoins = [
            ("tether", "USDT"),
            ("usd-coin", "USDC"),
        ]
        
        new_items = []
        
        for coin_id, symbol in stablecoins:
            try:
                data = await self._get_coin_data(coin_id)
                if data is None:
                    continue
                
                # Extract current supply
                market_data = data.get("market_data", {})
                current_supply = float(market_data.get("circulating_supply", 0))
                
                if current_supply <= 0:
                    continue
                
                # Check for supply change
                if coin_id in self._previous_supply:
                    prev_supply = self._previous_supply[coin_id]
                    supply_change = current_supply - prev_supply
                    supply_change_usd = supply_change  # 1:1 for stablecoins
                    
                    if abs(supply_change_usd) >= self._large_flow_threshold:
                        # Large mint or burn detected
                        flow_type = "mint" if supply_change > 0 else "burn"
                        direction = "bullish" if supply_change > 0 else "bearish"
                        
                        content = (
                            f"Large {symbol} {flow_type}: "
                            f"${abs(supply_change_usd)/1e6:.0f}M "
                            f"({direction} signal for crypto markets)"
                        )
                        
                        # Relevance and urgency scale with flow size
                        relevance = min(1.0, abs(supply_change_usd) / 500_000_000.0)
                        urgency = min(1.0, abs(supply_change_usd) / 200_000_000.0)
                        
                        item = DataItem(
                            source_type=self.source_type,
                            source_name=self.name,
                            content=content,
                            timestamp=_utcnow(),
                            metadata={
                                "coin_id": coin_id,
                                "symbol": symbol,
                                "supply_change_usd": supply_change_usd,
                                "current_supply": current_supply,
                                "flow_type": flow_type,
                                "direction": direction,
                            },
                            relevance_score=round(relevance, 3),
                            urgency_score=round(urgency, 3),
                            mentioned_assets=["BTC", "ETH"],  # Affects all crypto
                        )
                        new_items.append(item)
                        logger.info(
                            "Stablecoin flow: {} {} ${:.0f}M",
                            symbol, flow_type, abs(supply_change_usd)/1e6
                        )
                
                # Update previous supply
                self._previous_supply[coin_id] = current_supply
                
            except Exception as exc:
                logger.error("Failed to fetch {} data: {}", coin_id, exc)
                self._errors += 1
        
        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        self._last_update = _utcnow()
        logger.debug("Stablecoin flow: cached {} items", len(new_items))

    async def _get_coin_data(self, coin_id: str) -> Optional[dict]:
        """Fetch coin data from CoinGecko API."""
        try:
            url = self.COINGECKO_API.format(coin_id=coin_id)
            params = {
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 429:
                        logger.warning("CoinGecko rate limit hit")
                        return None
                    if resp.status != 200:
                        logger.warning("CoinGecko API returned status {}", resp.status)
                        return None
                    return await resp.json(content_type=None)
        except Exception as exc:
            logger.error("CoinGecko API error for {}: {}", coin_id, exc)
            return None
