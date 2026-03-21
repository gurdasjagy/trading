"""Polymarket prediction market monitor for crypto events."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PolymarketMonitor(BaseSource):
    """Monitors Polymarket prediction markets for crypto-related events."""

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    CRYPTO_KEYWORDS = [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "crypto",
        "solana",
        "sol",
        "bnb",
        "xrp",
        "ripple",
        "defi",
        "nft",
        "blockchain",
        "altcoin",
        "coinbase",
        "binance",
        "sec",
        "etf",
    ]

    def __init__(self, polling_interval: int = 300):
        super().__init__("polymarket", DataSourceType.REST_API)
        self._polling_interval = polling_interval
        self._markets: Dict[str, dict] = {}  # condition_id -> market data
        self._prev_odds: Dict[str, float] = {}  # condition_id -> last recorded price
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Polymarket Monitor started")
        while self._running:
            try:
                await self.fetch_markets()
                await self.track_odds_changes()
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Polymarket monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._markets:
            await self.fetch_markets()
        return self._items[-limit:]

    async def fetch_markets(self) -> List[dict]:
        """Fetch active crypto-related markets from Polymarket Gamma API."""
        markets: List[dict] = []
        try:
            params = {"active": "true", "closed": "false", "_limit": 200}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.GAMMA_API}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for market in (data if isinstance(data, list) else data.get("markets", [])):
                            question = (market.get("question") or "").lower()
                            if any(kw in question for kw in self.CRYPTO_KEYWORDS):
                                cid = market.get("conditionId") or market.get("id", "")
                                if cid:
                                    self._markets[cid] = market
                                    markets.append(market)
                    else:
                        logger.warning(f"Polymarket API returned status {resp.status}")
            self._last_update = _utcnow()
        except Exception as exc:
            logger.warning(f"Polymarket fetch_markets error: {exc}")
            self._errors += 1
        return markets

    async def track_odds_changes(self) -> List[DataItem]:
        """Detect significant odds movements and generate signals."""
        new_items: List[DataItem] = []
        for cid, market in self._markets.items():
            try:
                outcome_prices = market.get("outcomePrices")
                if not outcome_prices:
                    continue
                # outcomePrices may be a JSON string or list
                if isinstance(outcome_prices, str):
                    import json

                    outcome_prices = json.loads(outcome_prices)
                yes_price = float(outcome_prices[0]) if outcome_prices else None
                if yes_price is None:
                    continue
                prev = self._prev_odds.get(cid)
                odds_change = (yes_price - prev) if prev is not None else 0.0
                self._prev_odds[cid] = yes_price
                if prev is not None and abs(odds_change) >= 0.05:  # 5% threshold
                    item = self._calculate_signal(market, odds_change)
                    if item:
                        new_items.append(item)
                        self._items_collected += 1
            except Exception as exc:
                logger.debug(f"Polymarket odds tracking error for {cid}: {exc}")
        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        return new_items

    def _calculate_signal(self, market: dict, odds_change: float) -> Optional[DataItem]:
        """Convert an odds movement into a DataItem signal."""
        question = market.get("question", "Unknown market")
        cid = market.get("conditionId") or market.get("id", "")
        yes_price = self._prev_odds.get(cid, 0.5)
        direction = "rising" if odds_change > 0 else "falling"
        content = (
            f"Polymarket: '{question}' YES price {direction} "
            f"({odds_change:+.1%}), now at {yes_price:.1%}"
        )
        assets = self._extract_mentioned_assets(question)
        magnitude = min(1.0, abs(odds_change) * 5)
        return DataItem(
            source_type=self.source_type,
            source_name="polymarket",
            content=content,
            url=f"https://polymarket.com/event/{cid}",
            timestamp=_utcnow(),
            raw_data=market,
            metadata={
                "condition_id": cid,
                "yes_price": yes_price,
                "odds_change": odds_change,
                "direction": direction,
            },
            relevance_score=min(1.0, 0.5 + magnitude),
            urgency_score=min(1.0, 0.3 + magnitude),
            mentioned_assets=assets,
        )
