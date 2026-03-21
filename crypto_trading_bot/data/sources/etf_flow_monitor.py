"""ETF Flow Monitor — tracks BTC/ETH spot ETF daily flows via SoSoValue API."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ETFFlowMonitor(BaseSource):
    """Monitors BTC/ETH spot ETF daily flow data from SoSoValue API.
    
    Large inflows (>$200M/day) are strong bullish signals.
    Large outflows are bearish signals.
    """

    # SoSoValue API endpoint (public data)
    API_URL = "https://api.sosovalue.com/api/v1/etf/flows"
    
    def __init__(
        self,
        polling_interval: int = 3600,  # 1 hour
        large_flow_threshold_usd: float = 200_000_000.0,  # $200M
    ) -> None:
        super().__init__("etf_flow", DataSourceType.REST_API, enabled=True)
        self._polling_interval = polling_interval
        self._large_flow_threshold = large_flow_threshold_usd
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("ETF Flow Monitor started (poll every {}s)", self._polling_interval)
        while self._running:
            try:
                await self._fetch_and_cache()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error("ETF flow fetch error: {}", exc)
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_and_cache()
        return self._items[-limit:]

    async def fetch_etf_flows(self) -> dict:
        """Fetch ETF flow data from SoSoValue API.
        
        Returns:
            Dict with keys: btc_inflow, btc_outflow, eth_inflow, eth_outflow (all in USD)
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.API_URL,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("SoSoValue API returned status {}", resp.status)
                        return {}
                    data = await resp.json(content_type=None)
            
            # Parse flow data
            flows = {
                "btc_inflow": 0.0,
                "btc_outflow": 0.0,
                "eth_inflow": 0.0,
                "eth_outflow": 0.0,
            }
            
            if isinstance(data, dict) and "data" in data:
                etf_data = data["data"]
                
                # BTC ETF flows
                if "btc" in etf_data:
                    btc = etf_data["btc"]
                    flows["btc_inflow"] = float(btc.get("inflow", 0))
                    flows["btc_outflow"] = float(btc.get("outflow", 0))
                
                # ETH ETF flows
                if "eth" in etf_data:
                    eth = etf_data["eth"]
                    flows["eth_inflow"] = float(eth.get("inflow", 0))
                    flows["eth_outflow"] = float(eth.get("outflow", 0))
            
            logger.debug(
                "ETF flows: BTC net=${:.0f}M, ETH net=${:.0f}M",
                (flows["btc_inflow"] - flows["btc_outflow"]) / 1e6,
                (flows["eth_inflow"] - flows["eth_outflow"]) / 1e6,
            )
            return flows
        except Exception as exc:
            logger.error("Failed to fetch ETF flows: {}", exc)
            self._errors += 1
            return {}

    async def _fetch_and_cache(self) -> None:
        """Fetch ETF flow data and create DataItems for large flows."""
        flows = await self.fetch_etf_flows()
        if not flows:
            return
        
        new_items = []
        
        # Check BTC flows
        btc_net_flow = flows["btc_inflow"] - flows["btc_outflow"]
        if abs(btc_net_flow) >= self._large_flow_threshold:
            flow_type = "inflow" if btc_net_flow > 0 else "outflow"
            direction = "bullish" if btc_net_flow > 0 else "bearish"
            
            content = (
                f"Large BTC ETF {flow_type}: "
                f"${abs(btc_net_flow)/1e6:.0f}M net "
                f"({direction} signal)"
            )
            
            relevance = min(1.0, abs(btc_net_flow) / 500_000_000.0)
            urgency = min(1.0, abs(btc_net_flow) / 300_000_000.0)
            
            item = DataItem(
                source_type=self.source_type,
                source_name=self.name,
                content=content,
                timestamp=_utcnow(),
                metadata={
                    "asset": "BTC",
                    "net_flow_usd": btc_net_flow,
                    "inflow": flows["btc_inflow"],
                    "outflow": flows["btc_outflow"],
                    "flow_type": flow_type,
                    "direction": direction,
                },
                relevance_score=round(relevance, 3),
                urgency_score=round(urgency, 3),
                mentioned_assets=["BTC"],
            )
            new_items.append(item)
            logger.info("ETF flow: BTC {} ${:.0f}M", flow_type, abs(btc_net_flow)/1e6)
        
        # Check ETH flows
        eth_net_flow = flows["eth_inflow"] - flows["eth_outflow"]
        if abs(eth_net_flow) >= self._large_flow_threshold:
            flow_type = "inflow" if eth_net_flow > 0 else "outflow"
            direction = "bullish" if eth_net_flow > 0 else "bearish"
            
            content = (
                f"Large ETH ETF {flow_type}: "
                f"${abs(eth_net_flow)/1e6:.0f}M net "
                f"({direction} signal)"
            )
            
            relevance = min(1.0, abs(eth_net_flow) / 500_000_000.0)
            urgency = min(1.0, abs(eth_net_flow) / 300_000_000.0)
            
            item = DataItem(
                source_type=self.source_type,
                source_name=self.name,
                content=content,
                timestamp=_utcnow(),
                metadata={
                    "asset": "ETH",
                    "net_flow_usd": eth_net_flow,
                    "inflow": flows["eth_inflow"],
                    "outflow": flows["eth_outflow"],
                    "flow_type": flow_type,
                    "direction": direction,
                },
                relevance_score=round(relevance, 3),
                urgency_score=round(urgency, 3),
                mentioned_assets=["ETH"],
            )
            new_items.append(item)
            logger.info("ETF flow: ETH {} ${:.0f}M", flow_type, abs(eth_net_flow)/1e6)
        
        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        self._last_update = _utcnow()
        logger.debug("ETF flow: cached {} items", len(new_items))
