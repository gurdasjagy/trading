"""Deribit options data source — options flow, implied volatility surface,
put/call ratio, and max pain level.

Uses the public Deribit REST API (no authentication required for market data).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DeribitOptionsSource(BaseSource):
    """Monitors Deribit options market for flow signals and IV surface data.

    Emits :class:`~data.sources.base_source.DataItem` objects when:
    * Put/Call ratio crosses a significant threshold (> 1.5 or < 0.5)
    * Implied volatility term structure is in extreme backwardation/contango
    * Large block trades (open interest change > threshold) are detected

    Args:
        underlying: Base asset symbol (default ``"BTC"``).
        polling_interval: Seconds between API polls (default 60).
        pc_ratio_high: P/C ratio above which bearish signal is emitted.
        pc_ratio_low: P/C ratio below which bullish signal is emitted.
    """

    DERIBIT_API = "https://www.deribit.com/api/v2"

    def __init__(
        self,
        underlying: str = "BTC",
        polling_interval: int = 60,
        pc_ratio_high: float = 1.5,
        pc_ratio_low: float = 0.5,
    ) -> None:
        super().__init__("deribit_options", DataSourceType.REST_API)
        self._underlying = underlying.upper()
        self._polling_interval = polling_interval
        self._pc_ratio_high = pc_ratio_high
        self._pc_ratio_low = pc_ratio_low
        self._items: List[DataItem] = []
        self._prev_oi: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("DeribitOptionsSource started for {}", self._underlying)
        while self._running:
            try:
                items = await self._fetch_and_analyse()
                self._items.extend(items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
            except Exception as exc:
                logger.error("DeribitOptionsSource error: {}", exc)
                self._errors += 1
            await asyncio.sleep(self._polling_interval)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            items = await self._fetch_and_analyse()
            self._items.extend(items)
        return self._items[-limit:]

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_and_analyse(self) -> List[DataItem]:
        """Fetch options snapshot and produce DataItems for notable signals."""
        items: List[DataItem] = []

        instruments = await self._get_instruments()
        if not instruments:
            return items

        summary = await self._get_book_summary_by_currency()
        if not summary:
            return items

        pc_ratio, iv_surface, large_oi_changes = self._analyse_snapshot(
            instruments, summary
        )

        # Put/Call ratio signal
        if pc_ratio is not None:
            if pc_ratio > self._pc_ratio_high:
                content = (
                    f"Deribit {self._underlying}: Put/Call OI ratio = {pc_ratio:.2f} "
                    f"(bearish options flow — high put buying)"
                )
                items.append(self._build_item(content, urgency=0.7, sentiment_score=-0.4))
            elif pc_ratio < self._pc_ratio_low:
                content = (
                    f"Deribit {self._underlying}: Put/Call OI ratio = {pc_ratio:.2f} "
                    f"(bullish options flow — high call buying)"
                )
                items.append(self._build_item(content, urgency=0.6, sentiment_score=0.4))

        # IV term structure signal
        if iv_surface:
            short_iv = iv_surface.get("short_term_iv", 0.0)
            long_iv = iv_surface.get("long_term_iv", 0.0)
            if short_iv > 0 and long_iv > 0:
                ratio = short_iv / long_iv
                if ratio > 1.3:
                    content = (
                        f"Deribit {self._underlying}: IV backwardation detected "
                        f"(short={short_iv:.1f}%, long={long_iv:.1f}%) — "
                        "elevated near-term uncertainty"
                    )
                    items.append(self._build_item(content, urgency=0.8, sentiment_score=-0.3))
                elif ratio < 0.8:
                    content = (
                        f"Deribit {self._underlying}: IV contango detected "
                        f"(short={short_iv:.1f}%, long={long_iv:.1f}%) — calm near-term"
                    )
                    items.append(self._build_item(content, urgency=0.3, sentiment_score=0.2))

        # Large OI change signal
        for instrument_name, oi_change in large_oi_changes:
            content = (
                f"Deribit: Large OI change on {instrument_name}: "
                f"{oi_change:+,.0f} contracts"
            )
            items.append(self._build_item(content, urgency=0.65, sentiment_score=0.0))

        self._items_collected += len(items)
        self._last_update = _utcnow()
        return items

    async def _get_instruments(self) -> List[dict]:
        """Fetch all active options instruments for the underlying."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "currency": self._underlying,
                    "kind": "option",
                    "expired": "false",
                }
                async with session.get(
                    f"{self.DERIBIT_API}/public/get_instruments",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("result", [])
        except Exception as exc:
            logger.debug("Deribit get_instruments error: {}", exc)
        return []

    async def _get_book_summary_by_currency(self) -> List[dict]:
        """Fetch book summary for all options of the underlying."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"currency": self._underlying, "kind": "option"}
                async with session.get(
                    f"{self.DERIBIT_API}/public/get_book_summary_by_currency",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("result", [])
        except Exception as exc:
            logger.debug("Deribit book_summary error: {}", exc)
        return []

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyse_snapshot(
        self,
        instruments: List[dict],
        summary: List[dict],
    ):
        """Compute P/C ratio, IV surface, and large OI changes."""
        # Build OI lookup
        oi_by_instrument: Dict[str, float] = {
            item["instrument_name"]: float(item.get("open_interest", 0))
            for item in summary
        }
        iv_by_instrument: Dict[str, float] = {
            item["instrument_name"]: float(item.get("mark_iv", 0) or 0)
            for item in summary
        }

        put_oi = 0.0
        call_oi = 0.0
        short_ivs: List[float] = []
        long_ivs: List[float] = []
        large_changes: List = []

        for inst in instruments:
            name = inst.get("instrument_name", "")
            oi = oi_by_instrument.get(name, 0.0)
            iv = iv_by_instrument.get(name, 0.0)
            option_type = inst.get("option_type", "")
            days_to_expiry = float(inst.get("expiration_timestamp", 0) or 0) / 86400000

            if option_type == "put":
                put_oi += oi
            elif option_type == "call":
                call_oi += oi

            if iv > 0:
                if days_to_expiry < 30:
                    short_ivs.append(iv)
                else:
                    long_ivs.append(iv)

            # Large OI change detection
            prev = self._prev_oi.get(name, oi)
            change = oi - prev
            if abs(change) > 1000:
                large_changes.append((name, change))
            self._prev_oi[name] = oi

        pc_ratio = (put_oi / call_oi) if call_oi > 0 else None
        iv_surface = {}
        if short_ivs:
            iv_surface["short_term_iv"] = sum(short_ivs) / len(short_ivs)
        if long_ivs:
            iv_surface["long_term_iv"] = sum(long_ivs) / len(long_ivs)

        return pc_ratio, iv_surface, large_changes[:5]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_item(
        self,
        content: str,
        urgency: float,
        sentiment_score: float,
    ) -> DataItem:
        return DataItem(
            source_type=self.source_type,
            source_name="deribit_options",
            content=content,
            timestamp=_utcnow(),
            relevance_score=min(1.0, urgency),
            urgency_score=urgency,
            mentioned_assets=[self._underlying],
            metadata={
                "sentiment_score": sentiment_score,
                "underlying": self._underlying,
            },
        )
