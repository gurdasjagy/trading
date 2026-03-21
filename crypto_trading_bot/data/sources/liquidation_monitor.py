"""Liquidation monitor using Coinglass API."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LiquidationMonitor(BaseSource):
    """Monitors liquidation data from Coinglass API to detect cascade events."""

    COINGLASS_API = "https://open-api.coinglass.com/public/v2"
    COINGLASS_V3_API = "https://open-api-v3.coinglass.com/api"

    # Threshold in USD for a cascade-level liquidation event
    CASCADE_THRESHOLD_USD = 50_000_000

    def __init__(
        self,
        api_key: str = "",
        polling_interval: int = 30,
        symbols: Optional[List[str]] = None,
    ):
        super().__init__("liquidation", DataSourceType.REST_API)
        self._api_key = api_key
        self._polling_interval = polling_interval
        self._symbols = symbols or ["BTC", "ETH", "SOL", "BNB", "XRP"]
        self._items: List[DataItem] = []
        self._liq_history: List[dict] = []

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("Liquidation: no Coinglass API key – using public endpoints.")
        logger.info("Liquidation Monitor started")
        while self._running:
            try:
                liquidations = await self.monitor_liquidations()
                cascade = self.detect_cascade(liquidations)
                if cascade:
                    item = self._calculate_impact(cascade)
                    if item:
                        self._items.append(item)
                        self._items_collected += 1
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Liquidation monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            liqs = await self.monitor_liquidations()
            cascade = self.detect_cascade(liqs)
            if cascade:
                item = self._calculate_impact(cascade)
                if item:
                    self._items.append(item)
        return self._items[-limit:]

    async def monitor_liquidations(self) -> List[dict]:
        """Fetch liquidation data for tracked symbols."""
        liquidations: List[dict] = []
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["coinglassSecret"] = self._api_key

        try:
            async with aiohttp.ClientSession() as session:
                for symbol in self._symbols:
                    url = f"{self.COINGLASS_API}/liquidation_chart"
                    params = {"symbol": symbol, "interval": "1h", "limit": 24}
                    try:
                        async with session.get(
                            url,
                            params=params,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                liq_data = data.get("data") or []
                                for entry in liq_data:
                                    entry["symbol"] = symbol
                                liquidations.extend(liq_data)
                            else:
                                logger.debug(f"Coinglass API {resp.status} for {symbol}")
                    except Exception as exc:
                        logger.debug(f"Liquidation fetch for {symbol}: {exc}")
        except Exception as exc:
            logger.warning(f"Liquidation monitor_liquidations error: {exc}")
            self._errors += 1

        self._liq_history.extend(liquidations)
        if len(self._liq_history) > 10_000:
            self._liq_history = self._liq_history[-10_000:]
        self._last_update = _utcnow()
        return liquidations

    def detect_cascade(self, liquidations: List[dict]) -> Optional[dict]:
        """Identify if current liquidations constitute a cascade event."""
        if not liquidations:
            return None
        total_longs = 0.0
        total_shorts = 0.0
        per_symbol: Dict[str, float] = {}
        for entry in liquidations:
            symbol = entry.get("symbol", "UNKNOWN")
            longs = float(entry.get("buyUsdAmt") or entry.get("longAmount") or 0)
            shorts = float(entry.get("sellUsdAmt") or entry.get("shortAmount") or 0)
            total_longs += longs
            total_shorts += shorts
            per_symbol[symbol] = per_symbol.get(symbol, 0) + longs + shorts

        total_usd = total_longs + total_shorts
        if total_usd < self.CASCADE_THRESHOLD_USD:
            return None

        dominant_side = "longs" if total_longs > total_shorts else "shorts"
        return {
            "total_usd": total_usd,
            "total_longs": total_longs,
            "total_shorts": total_shorts,
            "dominant_side": dominant_side,
            "per_symbol": per_symbol,
            "timestamp": _utcnow(),
        }

    def _calculate_impact(self, cascade: dict) -> Optional[DataItem]:
        """Estimate market impact of a liquidation cascade and build a DataItem."""
        total_usd = cascade["total_usd"]
        dominant = cascade["dominant_side"]
        per_symbol = cascade["per_symbol"]
        top_symbol = max(per_symbol, key=lambda k: per_symbol[k]) if per_symbol else "CRYPTO"

        # Dominant side liquidated → price moved against them
        price_direction = "downward" if dominant == "longs" else "upward"
        content = (
            f"Liquidation Cascade: ${total_usd:,.0f} liquidated "
            f"({dominant} dominant), {price_direction} price pressure. "
            f"Top symbol: {top_symbol} (${per_symbol.get(top_symbol, 0):,.0f})"
        )
        assets = list(per_symbol.keys())
        magnitude = min(1.0, total_usd / 500_000_000)
        return DataItem(
            source_type=self.source_type,
            source_name="liquidation/coinglass",
            content=content,
            timestamp=cascade["timestamp"],
            metadata={
                "total_usd": total_usd,
                "total_longs": cascade["total_longs"],
                "total_shorts": cascade["total_shorts"],
                "dominant_side": dominant,
                "price_direction": price_direction,
                "per_symbol": per_symbol,
            },
            relevance_score=min(1.0, 0.5 + magnitude * 0.5),
            urgency_score=min(1.0, 0.6 + magnitude * 0.4),
            mentioned_assets=assets,
        )
