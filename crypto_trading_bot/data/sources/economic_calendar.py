"""Macroeconomic calendar monitor for events affecting crypto markets."""

import asyncio
from datetime import datetime, timezone
from typing import List

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EconomicCalendarMonitor(BaseSource):
    """Monitors macroeconomic events via public economic calendar APIs."""

    # TradingEconomics public calendar (no key required for limited use)
    TRADINGECONOMICS_API = "https://api.tradingeconomics.com/calendar"
    # Forexfactory-style public calendar
    FOREXFACTORY_API = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    # Events with known high impact on crypto
    HIGH_IMPACT_EVENTS = [
        "fomc",
        "federal reserve",
        "interest rate decision",
        "cpi",
        "inflation",
        "nonfarm payroll",
        "gdp",
        "employment",
        "unemployment",
        "pce",
        "jackson hole",
        "powell",
        "yellen",
        "treasury",
        "etf approval",
        "sec decision",
        "bank failure",
        "debt ceiling",
    ]

    REDUCE_EXPOSURE_EVENTS = ["fomc", "interest rate decision", "cpi", "nonfarm payroll", "gdp"]

    def __init__(
        self,
        trading_economics_api_key: str = "",
        polling_interval: int = 3600,  # events don't change minute-to-minute
    ):
        super().__init__("economic_calendar", DataSourceType.REST_API)
        self._te_api_key = trading_economics_api_key
        self._polling_interval = polling_interval
        self._upcoming_events: List[dict] = []
        self._items: List[DataItem] = []
        self._seen_event_ids: set = set()

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("Economic Calendar Monitor started")
        while self._running:
            try:
                new_items = await self.fetch_upcoming_events()
                for item in new_items:
                    ev_id = item.metadata.get("event_id", "")
                    if ev_id and ev_id not in self._seen_event_ids:
                        self._seen_event_ids.add(ev_id)
                        self._items.append(item)
                        self._items_collected += 1
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Economic Calendar monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self.fetch_upcoming_events()
        return self._items[-limit:]

    async def fetch_upcoming_events(self) -> List[DataItem]:
        """Fetch this week's economic events from public calendar APIs."""
        events = await self._fetch_forexfactory()
        if not events and self._te_api_key:
            events = await self._fetch_tradingeconomics()

        self._upcoming_events = events
        self._last_update = _utcnow()
        items: List[DataItem] = []
        for event in events:
            impact = self.check_impact_level(event)
            if impact == "low":
                continue
            reduce = self._should_reduce_exposure([event])
            content = (
                f"Economic Event: {event.get('title','Unknown')} "
                f"[{event.get('country','?')}] – Impact: {impact}. "
                f"{'⚠️ Consider reducing exposure.' if reduce else ''}"
            )
            assets = self._extract_mentioned_assets(content)
            ts_str = event.get("date", "")
            try:
                ts = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=None)
            except Exception:
                ts = _utcnow()
            relevance = 0.9 if impact == "high" else 0.6
            urgency = 0.8 if reduce else (0.5 if impact == "high" else 0.3)
            event_id = event.get("id") or event.get("title", "") + ts_str
            items.append(
                DataItem(
                    source_type=self.source_type,
                    source_name="economic_calendar",
                    content=content,
                    timestamp=ts,
                    raw_data=event,
                    metadata={
                        "event_id": str(event_id),
                        "title": event.get("title", ""),
                        "country": event.get("country", ""),
                        "impact": impact,
                        "reduce_exposure": reduce,
                        "forecast": event.get("forecast"),
                        "previous": event.get("previous"),
                    },
                    relevance_score=relevance,
                    urgency_score=urgency,
                    mentioned_assets=assets,
                )
            )
        return items

    def check_impact_level(self, event: dict) -> str:
        """Classify event impact as high / medium / low."""
        title = (event.get("title") or event.get("event") or "").lower()
        # Some APIs provide an explicit impact/importance field
        raw_impact = str(event.get("impact") or event.get("importance") or "").lower()
        if raw_impact in ("high", "3", "red"):
            return "high"
        if raw_impact in ("medium", "2", "orange"):
            return "medium"
        if any(ev_kw in title for ev_kw in self.HIGH_IMPACT_EVENTS):
            return "high"
        if event.get("country", "").upper() in ("US", "EUR", "UK", "CN", "JP"):
            return "medium"
        return "low"

    def _should_reduce_exposure(self, events: List[dict]) -> bool:
        """Return True if any upcoming event warrants reducing market exposure."""
        for event in events:
            title = (event.get("title") or "").lower()
            if any(kw in title for kw in self.REDUCE_EXPOSURE_EVENTS):
                return True
        return False

    async def _fetch_forexfactory(self) -> List[dict]:
        """Fetch this week's calendar from ForexFactory public JSON."""
        events: List[dict] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.FOREXFACTORY_API,
                    headers={"User-Agent": "CryptoTradingBot/1.0"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        for entry in (data if isinstance(data, list) else []):
                            events.append(
                                {
                                    "id": entry.get("id", ""),
                                    "title": entry.get("title", ""),
                                    "country": entry.get("country", ""),
                                    "date": entry.get("date", ""),
                                    "impact": entry.get("impact", ""),
                                    "forecast": entry.get("forecast"),
                                    "previous": entry.get("previous"),
                                }
                            )
        except Exception as exc:
            logger.debug(f"ForexFactory fetch error: {exc}")
        return events

    async def _fetch_tradingeconomics(self) -> List[dict]:
        """Fetch calendar from TradingEconomics API (requires API key)."""
        events: List[dict] = []
        if not self._te_api_key:
            return events
        params = {"c": self._te_api_key, "importance": "2,3"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.TRADINGECONOMICS_API,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        for entry in (data if isinstance(data, list) else []):
                            events.append(
                                {
                                    "id": entry.get("CalendarId", ""),
                                    "title": entry.get("Event") or entry.get("Category", ""),
                                    "country": entry.get("Country", ""),
                                    "date": entry.get("Date", ""),
                                    "impact": str(entry.get("Importance", "")),
                                    "forecast": entry.get("Forecast"),
                                    "previous": entry.get("Previous"),
                                }
                            )
        except Exception as exc:
            logger.debug(f"TradingEconomics fetch error: {exc}")
        return events
