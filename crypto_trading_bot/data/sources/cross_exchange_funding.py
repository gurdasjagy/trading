"""Cross-Exchange Funding Monitor — detects funding rate arbitrage opportunities."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CrossExchangeFundingMonitor(BaseSource):
    """Monitors funding rates across Binance, Bybit, OKX, and Gate.io.
    
    Detects arbitrage opportunities when funding rate spread exceeds 0.03%.
    """

    # Public funding rate endpoints
    ENDPOINTS = {
        "binance": "https://fapi.binance.com/fapi/v1/premiumIndex",
        "gateio": "https://api.gateio.ws/api/v4/futures/usdt/contracts",
        "okx": "https://www.okx.com/api/v5/public/funding-rate",
        "bybit": "https://api.bybit.com/v5/market/tickers?category=linear",
    }
    
    def __init__(
        self,
        polling_interval: int = 900,  # 15 minutes
        spread_threshold_pct: float = 0.03,  # 0.03% spread
    ) -> None:
        super().__init__("cross_exchange_funding", DataSourceType.REST_API, enabled=True)
        self._polling_interval = polling_interval
        self._spread_threshold = spread_threshold_pct / 100.0  # Convert to decimal
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Cross-Exchange Funding Monitor started (poll every {}s)", self._polling_interval)
        while self._running:
            try:
                await self._fetch_and_cache()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error("Cross-exchange funding fetch error: {}", exc)
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_and_cache()
        return self._items[-limit:]

    async def _fetch_and_cache(self) -> None:
        """Fetch funding rates from all exchanges and detect arbitrage opportunities."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Fetch from all exchanges concurrently
        tasks = [
            self._fetch_binance_funding(symbols),
            self._fetch_bybit_funding(symbols),
            self._fetch_okx_funding(symbols),
            self._fetch_gateio_funding(symbols),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_rates = {}
        for result in results:
            if isinstance(result, dict):
                all_rates.update(result)
        
        # Detect arbitrage opportunities
        new_items = []
        for symbol in symbols:
            rates = []
            exchanges = []
            
            for exchange in ["binance", "bybit", "okx", "gateio"]:
                key = f"{exchange}:{symbol}"
                if key in all_rates:
                    rates.append(all_rates[key])
                    exchanges.append(exchange)
            
            if len(rates) < 2:
                continue
            
            # Calculate spread
            max_rate = max(rates)
            min_rate = min(rates)
            spread = max_rate - min_rate
            
            if spread >= self._spread_threshold:
                max_exchange = exchanges[rates.index(max_rate)]
                min_exchange = exchanges[rates.index(min_rate)]
                
                content = (
                    f"Funding arbitrage: {symbol} spread {spread*100:.3f}% "
                    f"({max_exchange} {max_rate*100:.3f}% vs "
                    f"{min_exchange} {min_rate*100:.3f}%)"
                )
                
                # Relevance and urgency scale with spread size
                relevance = min(1.0, spread / 0.10)
                urgency = min(1.0, spread / 0.05)
                
                item = DataItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    content=content,
                    timestamp=_utcnow(),
                    metadata={
                        "symbol": symbol,
                        "spread_pct": spread * 100,
                        "max_exchange": max_exchange,
                        "max_rate": max_rate,
                        "min_exchange": min_exchange,
                        "min_rate": min_rate,
                        "all_rates": {ex: r for ex, r in zip(exchanges, rates)},
                    },
                    relevance_score=round(relevance, 3),
                    urgency_score=round(urgency, 3),
                    mentioned_assets=[symbol.replace("USDT", "")],
                )
                new_items.append(item)
                logger.info(
                    "Funding arbitrage: {} spread {:.3f}%",
                    symbol, spread * 100
                )
        
        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        self._last_update = _utcnow()
        logger.debug("Cross-exchange funding: cached {} items", len(new_items))

    async def _fetch_binance_funding(self, symbols: List[str]) -> dict:
        """Fetch funding rates from Binance."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.ENDPOINTS["binance"],
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
            
            rates = {}
            for item in data:
                symbol = item.get("symbol", "")
                if symbol in symbols:
                    rate = float(item.get("lastFundingRate", 0))
                    rates[f"binance:{symbol}"] = rate
            return rates
        except Exception as exc:
            logger.debug("Binance funding fetch error: {}", exc)
            return {}

    async def _fetch_bybit_funding(self, symbols: List[str]) -> dict:
        """Fetch funding rates from Bybit."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.ENDPOINTS["bybit"],
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
            
            rates = {}
            if "result" in data and "list" in data["result"]:
                for item in data["result"]["list"]:
                    symbol = item.get("symbol", "")
                    if symbol in symbols:
                        rate = float(item.get("fundingRate", 0))
                        rates[f"bybit:{symbol}"] = rate
            return rates
        except Exception as exc:
            logger.debug("Bybit funding fetch error: {}", exc)
            return {}

    async def _fetch_okx_funding(self, symbols: List[str]) -> dict:
        """Fetch funding rates from OKX."""
        try:
            rates = {}
            for symbol in symbols:
                # OKX uses different symbol format: BTC-USDT-SWAP
                okx_symbol = f"{symbol[:-4]}-USDT-SWAP"
                url = f"{self.ENDPOINTS['okx']}?instId={okx_symbol}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                
                if "data" in data and data["data"]:
                    rate = float(data["data"][0].get("fundingRate", 0))
                    rates[f"okx:{symbol}"] = rate
            return rates
        except Exception as exc:
            logger.debug("OKX funding fetch error: {}", exc)
            return {}

    async def _fetch_gateio_funding(self, symbols: List[str]) -> dict:
        """Fetch funding rates from Gate.io."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.ENDPOINTS["gateio"],
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
            
            rates = {}
            for item in data:
                name = item.get("name", "")
                # Gate.io format: BTC_USDT
                symbol = name.replace("_", "")
                if symbol in symbols:
                    rate = float(item.get("funding_rate", 0))
                    rates[f"gateio:{symbol}"] = rate
            return rates
        except Exception as exc:
            logger.debug("Gate.io funding fetch error: {}", exc)
            return {}
