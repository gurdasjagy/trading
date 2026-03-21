"""CoinGecko market data source — no API key required for free endpoints."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Map CoinGecko coin IDs to canonical ticker symbols used by the trading bot
_COINGECKO_ID_TO_SYMBOL: Dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
    "dogecoin": "DOGE",
    "avalanche-2": "AVAX",
    "chainlink": "LINK",
    "polkadot": "DOT",
    "matic-network": "MATIC",
    "near": "NEAR",
    "arbitrum": "ARB",
}

_NOTABLE_MOVE_PCT = 5.0  # consider a coin "notable" when |24h change| > this value

# Reference market cap (in USD) used to scale relevance scores.
# At $1 trillion market cap the relevance contribution reaches 0.5.
_REFERENCE_MARKET_CAP_USD = 1e12


class CoinGeckoSource(BaseSource):
    """Fetches market data from CoinGecko's free public API.

    Retrieves trending coins, market-cap rankings, and flags coins that have
    moved more than :attr:`notable_move_pct` percent over the past 24 hours.
    No API key is required.
    """

    MARKETS_URL = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd"
        "&ids={coin_ids}"
        "&order=market_cap_desc"
        "&per_page=20"
        "&page=1"
        "&sparkline=false"
        "&price_change_percentage=24h"
    )
    TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"

    def __init__(
        self,
        polling_interval: int = 300,  # 5 minutes
        notable_move_pct: float = _NOTABLE_MOVE_PCT,
    ) -> None:
        super().__init__("coingecko", DataSourceType.REST_API, enabled=True)
        self._polling_interval = polling_interval
        self._notable_move_pct = notable_move_pct
        self._items: List[DataItem] = []

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("CoinGecko Monitor started (poll every {}s)", self._polling_interval)
        while self._running:
            try:
                await self._fetch_and_cache()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error("CoinGecko fetch error: {}", exc)
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_and_cache()
        return self._items[-limit:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_and_cache(self) -> None:
        market_items = await self._fetch_markets()
        trending_items = await self._fetch_trending()
        new_items = market_items + trending_items

        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        self._last_update = _utcnow()
        logger.debug("CoinGecko: fetched {} items", len(new_items))

    async def _get_json(self, url: str) -> Optional[Any]:
        """Perform a GET request and return parsed JSON, or None on failure."""
        headers = {"Accept": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 429:
                        logger.warning("CoinGecko: rate-limited (429)")
                        return None
                    if resp.status != 200:
                        logger.warning("CoinGecko: HTTP {} for {}", resp.status, url)
                        return None
                    return await resp.json(content_type=None)
        except Exception as exc:
            logger.error("CoinGecko HTTP error for {}: {}", url, exc)
            self._errors += 1
            return None

    async def _fetch_markets(self) -> List[DataItem]:
        """Fetch market data for the tracked coins and flag notable movers."""
        coin_ids = ",".join(_COINGECKO_ID_TO_SYMBOL.keys())
        url = self.MARKETS_URL.format(coin_ids=coin_ids)
        data = await self._get_json(url)
        if not isinstance(data, list):
            return []

        items: List[DataItem] = []
        for coin in data:
            item = self._market_coin_to_item(coin)
            if item is not None:
                items.append(item)
        return items

    async def _fetch_trending(self) -> List[DataItem]:
        """Fetch trending coins and create summary DataItems."""
        data = await self._get_json(self.TRENDING_URL)
        if not isinstance(data, dict):
            return []

        trending_coins = data.get("coins", [])
        if not trending_coins:
            return []

        names = [c.get("item", {}).get("name", "") for c in trending_coins[:7]]
        content = "Trending on CoinGecko: " + ", ".join(n for n in names if n)
        mentioned = self._extract_mentioned_assets(content)
        for c in trending_coins[:7]:
            sym = c.get("item", {}).get("symbol", "").upper()
            if sym and sym not in mentioned:
                mentioned.append(sym)

        return [
            DataItem(
                source_type=self.source_type,
                source_name=self.name,
                content=content,
                timestamp=_utcnow(),
                metadata={"trending_coins": trending_coins[:7]},
                relevance_score=0.6,
                urgency_score=0.3,
                mentioned_assets=mentioned,
            )
        ]

    def _market_coin_to_item(self, coin: dict) -> Optional[DataItem]:
        """Convert a CoinGecko markets entry into a :class:`DataItem`."""
        try:
            coin_id: str = coin.get("id", "")
            symbol_upper = _COINGECKO_ID_TO_SYMBOL.get(coin_id, coin.get("symbol", "").upper())
            name: str = coin.get("name", coin_id)
            price: float = float(coin.get("current_price") or 0)
            change_24h: float = float(coin.get("price_change_percentage_24h") or 0)
            market_cap: float = float(coin.get("market_cap") or 0)
            volume_24h: float = float(coin.get("total_volume") or 0)

            is_notable = abs(change_24h) >= self._notable_move_pct
            direction_word = "up" if change_24h >= 0 else "down"
            content = (
                f"{name} ({symbol_upper}) is {direction_word} "
                f"{abs(change_24h):.2f}% in 24h. "
                f"Price: ${price:,.2f}, "
                f"Market cap: ${market_cap:,.0f}, "
                f"24h volume: ${volume_24h:,.0f}."
            )
            if is_notable:
                content = f"[NOTABLE MOVE] {content}"

            # Relevance: higher for larger caps; urgency: proportional to move magnitude
            relevance = min(1.0, 0.4 + (market_cap / _REFERENCE_MARKET_CAP_USD) * 0.5)
            urgency = min(1.0, abs(change_24h) / 20.0 + (0.4 if is_notable else 0.1))

            return DataItem(
                source_type=self.source_type,
                source_name=self.name,
                content=content,
                timestamp=_utcnow(),
                metadata={
                    "coin_id": coin_id,
                    "symbol": symbol_upper,
                    "price": price,
                    "change_24h_pct": change_24h,
                    "market_cap": market_cap,
                    "volume_24h": volume_24h,
                    "is_notable": is_notable,
                },
                relevance_score=round(relevance, 3),
                urgency_score=round(urgency, 3),
                mentioned_assets=[symbol_upper] if symbol_upper else [],
            )
        except Exception as exc:
            logger.debug("CoinGecko: failed to parse coin {}: {}", coin.get("id"), exc)
            return None
