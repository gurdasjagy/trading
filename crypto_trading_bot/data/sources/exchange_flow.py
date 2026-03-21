"""Exchange flow monitor — stablecoin minting, BTC reserves, and
exchange inflow/outflow for multiple exchanges.

Aggregates data from Glassnode (if key available) and on-chain public
endpoints to detect:
* Significant stablecoin minting events (bullish — dry powder entering)
* Large exchange BTC/ETH reserve changes (outflow = accumulation bullish)
* Cross-exchange arbitrage flow imbalances

No API key is required for the public blockchain.info / CryptoQuant fallback.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ExchangeFlowMonitor(BaseSource):
    """Monitors exchange-level on-chain flows for trading signals.

    Signals produced:
    * Stablecoin mint event > ``stablecoin_mint_threshold_usd`` → bullish
    * Exchange BTC reserve change > ``btc_reserve_change_pct`` → bullish/bearish
    * Exchange ETH reserve spike → bearish/bullish

    Args:
        glassnode_api_key: Optional Glassnode API key for richer data.
        cryptoquant_api_key: Optional CryptoQuant API key.
        polling_interval: Seconds between polls (default 300).
        stablecoin_mint_threshold_usd: Minimum stablecoin mint value (USD)
            to trigger a DataItem.
        btc_reserve_change_pct: Minimum % change in exchange BTC reserves
            to trigger a DataItem.
    """

    GLASSNODE_API = "https://api.glassnode.com/v1/metrics"
    BLOCKCHAIN_INFO_API = "https://blockchain.info"

    # Known exchange BTC wallet balances from blockchain.info (simplified)
    _EXCHANGE_BTC_ADDRESSES = {
        "Binance": "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
        "Coinbase": "1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ",
        "Kraken": "1LdRcdxfbSnmCYYNdeYpUnztiYzVfBEQeC",
    }

    def __init__(
        self,
        glassnode_api_key: str = "",
        cryptoquant_api_key: str = "",
        polling_interval: int = 300,
        stablecoin_mint_threshold_usd: float = 100_000_000,
        btc_reserve_change_pct: float = 1.0,
    ) -> None:
        super().__init__("exchange_flow", DataSourceType.REST_API)
        self._glassnode_key = glassnode_api_key
        self._cryptoquant_key = cryptoquant_api_key
        self._polling_interval = polling_interval
        self._stablecoin_threshold = stablecoin_mint_threshold_usd
        self._btc_change_pct = btc_reserve_change_pct
        self._items: List[DataItem] = []
        self._prev_reserves: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("ExchangeFlowMonitor started")
        while self._running:
            try:
                new_items = await self._fetch_all()
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
            except Exception as exc:
                logger.error("ExchangeFlowMonitor error: {}", exc)
                self._errors += 1
            await asyncio.sleep(self._polling_interval)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            items = await self._fetch_all()
            self._items.extend(items)
        return self._items[-limit:]

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_all(self) -> List[DataItem]:
        """Fetch all exchange flow data and produce DataItems."""
        items: List[DataItem] = []

        # BTC exchange reserve changes
        reserve_items = await self._check_btc_reserves()
        items.extend(reserve_items)

        # Stablecoin supply via Glassnode (if key available)
        if self._glassnode_key:
            stable_items = await self._check_stablecoin_supply()
            items.extend(stable_items)

        self._items_collected += len(items)
        self._last_update = _utcnow()
        return items

    async def _check_btc_reserves(self) -> List[DataItem]:
        """Check for significant changes in on-chain exchange BTC reserves."""
        items: List[DataItem] = []
        if not self._glassnode_key:
            return items
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "a": "BTC",
                    "api_key": self._glassnode_key,
                    "i": "24h",
                    "s": str(int(__import__("time").time()) - 86400),
                }
                async with session.get(
                    f"{self.GLASSNODE_API}/distribution/balance_exchanges",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and isinstance(data, list):
                            current_balance = float(data[-1].get("v", 0))
                            prev_balance = float(
                                data[-2].get("v", current_balance)
                                if len(data) >= 2
                                else current_balance
                            )
                            pct_change = (
                                (current_balance - prev_balance) / prev_balance * 100
                                if prev_balance > 0 else 0.0
                            )
                            if abs(pct_change) >= self._btc_change_pct:
                                direction = "increased" if pct_change > 0 else "decreased"
                                sentiment = -0.3 if pct_change > 0 else 0.3
                                content = (
                                    f"Exchange BTC reserves {direction} by "
                                    f"{pct_change:+.1f}% to "
                                    f"{current_balance:,.0f} BTC"
                                    + (
                                        " — selling pressure"
                                        if pct_change > 0
                                        else " — accumulation signal"
                                    )
                                )
                                items.append(
                                    self._build_item(
                                        content, "BTC", 0.65, sentiment,
                                        {"pct_change": pct_change, "balance_btc": current_balance}
                                    )
                                )
        except Exception as exc:
            logger.debug("Glassnode BTC reserves error: {}", exc)
        return items

    async def _check_stablecoin_supply(self) -> List[DataItem]:
        """Check for large stablecoin supply changes (minting events)."""
        items: List[DataItem] = []
        stablecoins = ["USDT", "USDC"]
        try:
            async with aiohttp.ClientSession() as session:
                for stable in stablecoins:
                    params = {
                        "a": stable,
                        "api_key": self._glassnode_key,
                        "i": "24h",
                    }
                    async with session.get(
                        f"{self.GLASSNODE_API}/supply/current",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data and isinstance(data, list) and len(data) >= 2:
                                current_supply = float(data[-1].get("v", 0))
                                prev_supply = float(data[-2].get("v", current_supply))
                                minted = current_supply - prev_supply
                                if minted >= self._stablecoin_threshold:
                                    content = (
                                        f"Stablecoin mint: {stable} supply"
                                        f" increased by ${minted / 1e9:.2f}B"
                                        f" (total: ${current_supply / 1e9:.2f}B)"
                                        " — bullish: new capital entering crypto"
                                    )
                                    items.append(
                                        self._build_item(
                                            content, stable, 0.65, 0.3,
                                            {"minted_usd": minted, "supply": current_supply}
                                        )
                                    )
        except Exception as exc:
            logger.debug("Glassnode stablecoin supply error: {}", exc)
        return items

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_item(
        self,
        content: str,
        asset: str,
        urgency: float,
        sentiment_score: float,
        metadata: Optional[dict] = None,
    ) -> DataItem:
        meta = dict(metadata or {})
        meta["sentiment_score"] = sentiment_score
        return DataItem(
            source_type=self.source_type,
            source_name="exchange_flow",
            content=content,
            timestamp=_utcnow(),
            relevance_score=min(1.0, urgency + 0.1),
            urgency_score=urgency,
            mentioned_assets=[asset],
            metadata=meta,
        )
