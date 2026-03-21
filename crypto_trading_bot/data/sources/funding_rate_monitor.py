"""Funding rate monitor across exchanges."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FundingRateMonitor(BaseSource):
    """Monitors funding rates across multiple exchanges."""

    def __init__(self, polling_interval: int = 300):
        super().__init__("funding_rate", DataSourceType.REST_API)
        self._polling_interval = polling_interval
        self._rates: Dict[str, Dict[str, float]] = {}  # symbol_key -> exchange -> rate
        self._extreme_threshold = 0.0005  # 0.05% per 8 h = extreme

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Funding Rate Monitor started")
        while self._running:
            try:
                await self._fetch_funding_rates()
                await asyncio.sleep(self._polling_interval)
            except Exception as e:
                logger.error(f"Funding rate fetch error: {e}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 20) -> List[DataItem]:
        if not self._rates:
            await self._fetch_funding_rates()
        extremes = self.detect_extremes()
        items: List[DataItem] = []
        for symbol, rate_info in list(extremes.items())[:limit]:
            content = (
                f"Extreme funding rate for {symbol}: "
                f"{rate_info['avg_rate']:.4%} (avg across exchanges)"
            )
            items.append(
                DataItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    content=content,
                    timestamp=_utcnow(),
                    metadata=rate_info,
                    relevance_score=0.9,
                    urgency_score=0.7,
                    mentioned_assets=[symbol.replace("USDT", "").replace("/USDT", "")],
                )
            )
        return items

    async def _fetch_funding_rates(self) -> None:
        """Fetch funding rates from exchange APIs via ccxt."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        try:
            import ccxt.async_support as ccxt  # type: ignore

            exchange = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})
            try:
                for symbol in symbols:
                    try:
                        rate_data = await exchange.fetch_funding_rate(symbol)
                        rate = float(rate_data.get("fundingRate", 0))
                        sym_key = symbol.replace("/", "")
                        if sym_key not in self._rates:
                            self._rates[sym_key] = {}
                        self._rates[sym_key]["mexc"] = rate
                    except Exception as inner:
                        logger.debug(f"Could not fetch funding rate for {symbol}: {inner}")
            finally:
                await exchange.close()
        except Exception as e:
            logger.warning(f"Failed to fetch funding rates via ccxt: {e}")
        self._last_update = _utcnow()
        self._items_collected += 1

    def detect_extremes(self) -> Dict[str, dict]:
        """Return symbols whose average funding rate exceeds the extreme threshold."""
        extremes: Dict[str, dict] = {}
        for symbol, exchange_rates in self._rates.items():
            if not exchange_rates:
                continue
            avg_rate = sum(exchange_rates.values()) / len(exchange_rates)
            if abs(avg_rate) >= self._extreme_threshold:
                direction = "long_heavy" if avg_rate > 0 else "short_heavy"
                signal = "short" if avg_rate > 0 else "long"
                extremes[symbol] = {
                    "avg_rate": avg_rate,
                    "direction": direction,
                    "signal": signal,
                    "rates_by_exchange": exchange_rates,
                }
        return extremes

    def get_rate(self, symbol: str, exchange: str = "mexc") -> Optional[float]:
        sym_key = symbol.replace("/", "")
        return self._rates.get(sym_key, {}).get(exchange)
