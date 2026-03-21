"""Open interest monitor across exchanges."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class OpenInterestMonitor(BaseSource):
    """Tracks open interest (OI) changes across exchanges via Coinglass and exchange APIs."""

    COINGLASS_API = "https://open-api.coinglass.com/public/v2"

    # Percentage change threshold to flag as significant
    SIGNIFICANT_CHANGE_PCT = 5.0

    def __init__(
        self,
        api_key: str = "",
        symbols: Optional[List[str]] = None,
        polling_interval: int = 300,
    ):
        super().__init__("open_interest", DataSourceType.REST_API)
        self._api_key = api_key
        self._symbols = symbols or ["BTC", "ETH", "SOL", "BNB", "XRP"]
        self._polling_interval = polling_interval
        self._prev_oi: Dict[str, float] = {}  # symbol -> previous total OI (USD)
        self._current_oi: Dict[str, dict] = {}  # symbol -> {exchange -> oi}
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("OpenInterest: no Coinglass API key – using public endpoints.")
        logger.info("Open Interest Monitor started")
        while self._running:
            try:
                await self.track_changes()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"OI monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self.track_changes()
        return self._items[-limit:]

    async def fetch_open_interest(self, symbol: str) -> Dict[str, float]:
        """Fetch open interest for a symbol across exchanges."""
        oi_by_exchange: Dict[str, float] = {}
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["coinglassSecret"] = self._api_key
        try:
            async with aiohttp.ClientSession() as session:
                params = {"symbol": symbol}
                async with session.get(
                    f"{self.COINGLASS_API}/open_interest",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for entry in data.get("data") or []:
                            exchange = entry.get("exchangeName", "unknown")
                            oi_usd = float(entry.get("openInterestAmount") or entry.get("oi") or 0)
                            oi_by_exchange[exchange] = oi_usd
                    else:
                        logger.debug(f"OI API status {resp.status} for {symbol}")
        except Exception as exc:
            logger.warning(f"fetch_open_interest({symbol}) error: {exc}")
            self._errors += 1
        self._current_oi[symbol] = oi_by_exchange
        self._last_update = _utcnow()
        return oi_by_exchange

    async def track_changes(self) -> List[DataItem]:
        """Poll OI for all symbols and generate items for significant changes."""
        new_items: List[DataItem] = []
        for symbol in self._symbols:
            oi_by_exchange = await self.fetch_open_interest(symbol)
            if not oi_by_exchange:
                continue
            total_oi = sum(oi_by_exchange.values())
            prev_oi = self._prev_oi.get(symbol)
            if prev_oi is not None and prev_oi > 0:
                change_pct = ((total_oi - prev_oi) / prev_oi) * 100
                items = self.detect_significant_changes(
                    symbol, total_oi, change_pct, oi_by_exchange
                )
                new_items.extend(items)
            self._prev_oi[symbol] = total_oi
        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        return new_items

    def detect_significant_changes(
        self,
        symbol: str,
        total_oi: float,
        change_pct: float,
        oi_by_exchange: Dict[str, float],
    ) -> List[DataItem]:
        """Generate DataItems when OI changes exceed the significance threshold."""
        if abs(change_pct) < self.SIGNIFICANT_CHANGE_PCT:
            return []
        direction = "increased" if change_pct > 0 else "decreased"
        signal = "bullish_build" if change_pct > 0 else "position_unwind"
        content = (
            f"Open Interest {direction} for {symbol}: {change_pct:+.1f}% "
            f"(total ${total_oi:,.0f}). Signal: {signal}"
        )
        magnitude = min(1.0, abs(change_pct) / 20.0)
        return [
            DataItem(
                source_type=self.source_type,
                source_name="open_interest/coinglass",
                content=content,
                timestamp=_utcnow(),
                metadata={
                    "symbol": symbol,
                    "total_oi_usd": total_oi,
                    "change_pct": change_pct,
                    "direction": direction,
                    "signal": signal,
                    "by_exchange": oi_by_exchange,
                },
                relevance_score=min(1.0, 0.5 + magnitude * 0.5),
                urgency_score=min(1.0, 0.4 + magnitude * 0.6),
                mentioned_assets=[symbol],
            )
        ]
