"""Binance futures data source — funding rates, open interest, and
long/short ratio from the Binance public API.

No API key is required for the endpoints used here.
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


class BinanceFuturesSource(BaseSource):
    """Monitors Binance perpetual futures for funding rate extremes,
    open interest trends, and long/short ratio extremes.

    Args:
        symbols: List of Binance futures symbols (e.g. ``["BTCUSDT"]``).
        polling_interval: Seconds between API polls.
        funding_extreme_threshold: Absolute funding rate above which an
            extreme-funding DataItem is emitted.
        ls_ratio_extreme: Long/short ratio outside
            ``[1/ls_ratio_extreme, ls_ratio_extreme]`` triggers a signal.
    """

    BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        polling_interval: int = 60,
        funding_extreme_threshold: float = 0.001,
        ls_ratio_extreme: float = 2.0,
    ) -> None:
        super().__init__("binance_futures", DataSourceType.REST_API)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._polling_interval = polling_interval
        self._funding_extreme_threshold = funding_extreme_threshold
        self._ls_ratio_extreme = ls_ratio_extreme
        self._items: List[DataItem] = []
        self._prev_oi: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("BinanceFuturesSource started: {}", self._symbols)
        while self._running:
            try:
                new_items = await self._fetch_all()
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
            except Exception as exc:
                logger.error("BinanceFuturesSource error: {}", exc)
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
        """Fetch data for all symbols and return DataItems for notable signals."""
        items: List[DataItem] = []
        async with aiohttp.ClientSession() as session:
            for symbol in self._symbols:
                try:
                    items.extend(await self._analyse_symbol(session, symbol))
                except Exception as exc:
                    logger.debug("BinanceFutures {} error: {}", symbol, exc)
                    self._errors += 1
        self._last_update = _utcnow()
        return items

    async def _analyse_symbol(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> List[DataItem]:
        """Fetch and analyse data for a single *symbol*."""
        items: List[DataItem] = []

        # --- Funding rate ---
        funding = await self._get_funding_rate(session, symbol)
        if funding is not None:
            if abs(funding) >= self._funding_extreme_threshold:
                direction = "positive" if funding > 0 else "negative"
                sentiment = -0.3 if funding > 0 else 0.3  # high funding → bearish
                content = (
                    f"Binance {symbol}: extreme {direction} funding rate "
                    f"{funding:.4%} — "
                    + (
                        "longs paying shorts (overcrowded long)"
                        if funding > 0
                        else "shorts paying longs (overcrowded short)"
                    )
                )
                items.append(
                    self._build_item(
                        content=content,
                        symbol=symbol,
                        urgency=0.65,
                        sentiment_score=sentiment,
                        metadata={"funding_rate": funding},
                    )
                )

        # --- Open interest trend ---
        oi = await self._get_open_interest(session, symbol)
        if oi is not None:
            prev_oi = self._prev_oi.get(symbol, oi)
            oi_change_pct = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0.0
            self._prev_oi[symbol] = oi
            if abs(oi_change_pct) >= 5.0:
                direction = "increased" if oi_change_pct > 0 else "decreased"
                content = (
                    f"Binance {symbol}: open interest {direction} by "
                    f"{oi_change_pct:+.1f}% to {oi:,.0f} contracts"
                )
                items.append(
                    self._build_item(
                        content=content,
                        symbol=symbol,
                        urgency=0.55,
                        sentiment_score=0.1 if oi_change_pct > 0 else -0.1,
                        metadata={"oi": oi, "oi_change_pct": oi_change_pct},
                    )
                )

        # --- Long/short ratio ---
        ls_ratio = await self._get_ls_ratio(session, symbol)
        if ls_ratio is not None:
            if ls_ratio > self._ls_ratio_extreme:
                content = (
                    f"Binance {symbol}: long/short ratio = {ls_ratio:.2f} "
                    f"(extreme long crowding — potential reversal signal)"
                )
                items.append(
                    self._build_item(
                        content=content,
                        symbol=symbol,
                        urgency=0.7,
                        sentiment_score=-0.3,
                        metadata={"ls_ratio": ls_ratio},
                    )
                )
            elif ls_ratio < (1.0 / self._ls_ratio_extreme):
                content = (
                    f"Binance {symbol}: long/short ratio = {ls_ratio:.2f} "
                    f"(extreme short crowding — potential short-squeeze signal)"
                )
                items.append(
                    self._build_item(
                        content=content,
                        symbol=symbol,
                        urgency=0.7,
                        sentiment_score=0.3,
                        metadata={"ls_ratio": ls_ratio},
                    )
                )

        self._items_collected += len(items)
        return items

    async def _get_funding_rate(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> Optional[float]:
        try:
            async with session.get(
                f"{self.BINANCE_FAPI}/premiumIndex",
                params={"symbol": symbol},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("lastFundingRate", 0))
        except Exception as exc:
            logger.debug("Binance premiumIndex {}: {}", symbol, exc)
        return None

    async def _get_open_interest(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> Optional[float]:
        try:
            async with session.get(
                f"{self.BINANCE_FAPI}/openInterest",
                params={"symbol": symbol},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("openInterest", 0))
        except Exception as exc:
            logger.debug("Binance openInterest {}: {}", symbol, exc)
        return None

    async def _get_ls_ratio(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> Optional[float]:
        """Fetch the global long/short ratio from Binance top-trader data."""
        try:
            async with session.get(
                f"{self.BINANCE_FAPI}/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "5m", "limit": 1},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and isinstance(data, list):
                        return float(data[0].get("longShortRatio", 1.0))
        except Exception as exc:
            logger.debug("Binance longShortRatio {}: {}", symbol, exc)
        return None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_item(
        self,
        content: str,
        symbol: str,
        urgency: float,
        sentiment_score: float,
        metadata: Optional[dict] = None,
    ) -> DataItem:
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        meta = dict(metadata or {})
        meta["sentiment_score"] = sentiment_score
        meta["symbol"] = symbol
        return DataItem(
            source_type=self.source_type,
            source_name="binance_futures",
            content=content,
            timestamp=_utcnow(),
            relevance_score=min(1.0, urgency),
            urgency_score=urgency,
            mentioned_assets=[base_asset] if base_asset else [],
            metadata=meta,
        )
