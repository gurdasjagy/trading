"""ETH Gas Monitor — monitors Ethereum gas prices."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EthGasMonitor(BaseSource):
    """Monitors ETH gas prices via Etherscan API."""

    ETHERSCAN_API = "https://api.etherscan.io/api"

    def __init__(
        self,
        api_key: str = "",
        polling_interval: int = 300,  # 5 minutes
    ):
        super().__init__("eth_gas", DataSourceType.REST_API)
        self._api_key = api_key
        self._polling_interval = polling_interval
        self._items: List[DataItem] = []
        self._prev_gas_level: Optional[str] = None

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("EthGasMonitor: no API key provided – monitoring disabled.")
            self._running = False
            return
        logger.info("ETH Gas Monitor started")
        while self._running:
            try:
                item = await self.fetch_latest()
                if item:
                    self._items.extend(item)
                    if len(self._items) > 500:
                        self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"EthGas monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 1) -> List[DataItem]:
        """Fetch current ETH gas prices from Etherscan."""
        items: List[DataItem] = []
        if not self._api_key:
            return items

        try:
            params = {
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self._api_key,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.ETHERSCAN_API,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Etherscan API status {resp.status}")
                        return items
                    data = await resp.json()

            if data.get("status") != "1":
                logger.warning(f"Etherscan API error: {data.get('message')}")
                return items

            result = data.get("result", {})
            safe_gas = int(result.get("SafeGasPrice", 0))
            propose_gas = int(result.get("ProposeGasPrice", 0))
            fast_gas = int(result.get("FastGasPrice", 0))

            # Determine gas level and urgency
            if fast_gas > 200:
                gas_level = "extreme"
                urgency = 0.9
                signal = "DeFi liquidation cascade likely"
            elif fast_gas > 100:
                gas_level = "high"
                urgency = 0.7
                signal = "High DeFi activity"
            elif fast_gas > 50:
                gas_level = "medium"
                urgency = 0.5
                signal = "Normal activity"
            else:
                gas_level = "low"
                urgency = 0.3
                signal = "Low activity"

            # Create item if gas spiked >200 gwei or level changed
            if fast_gas > 200 or gas_level != self._prev_gas_level:
                content = (
                    f"ETH Gas: {gas_level} ({fast_gas} gwei fast, "
                    f"{propose_gas} gwei standard, {safe_gas} gwei safe) – {signal}"
                )

                item = DataItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    content=content,
                    timestamp=_utcnow(),
                    metadata={
                        "safe_gas_price": safe_gas,
                        "propose_gas_price": propose_gas,
                        "fast_gas_price": fast_gas,
                        "gas_level": gas_level,
                        "signal": signal,
                    },
                    relevance_score=min(1.0, fast_gas / 300),  # Scale with gas price
                    urgency_score=urgency,
                    mentioned_assets=["ETH"],
                )
                items.append(item)
                self._items_collected += 1
                self._prev_gas_level = gas_level
                logger.debug(f"ETH Gas: {gas_level}, {fast_gas} gwei")

            self._last_update = _utcnow()

        except Exception as exc:
            logger.warning(f"EthGas fetch error: {exc}")
            self._errors += 1

        return items
