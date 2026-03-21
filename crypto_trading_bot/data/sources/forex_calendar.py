"""Forex economic calendar integration for news-driven trading decisions.

Fetches high-impact economic events from ForexFactory RSS feed and provides
an API for strategies to check for upcoming news events before trading.

Key features:
* Fetch and parse ForexFactory economic calendar events.
* Filter by impact level (low/medium/high).
* Track upcoming events within configurable time windows.
* Auto-pause trading before high-impact events.
* Feed events to news-driven strategies (NFP, FOMC, etc.).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import aiohttp
from loguru import logger


@dataclass
class EconomicEvent:
    """Represents a single economic calendar event."""

    timestamp: datetime
    currency: str  # USD, EUR, GBP, etc.
    title: str
    impact: str  # low, medium, high
    forecast: Optional[str] = None
    actual: Optional[str] = None
    previous: Optional[str] = None


class ForexEconomicCalendar:
    """Forex economic calendar with event tracking and trading pause logic.

    Fetches events from ForexFactory RSS feed and provides methods to:
    * Check if high-impact news is approaching.
    * Get upcoming events for specific currencies.
    * Auto-pause trading based on event proximity.

    Args:
        refresh_interval_minutes: How often to refresh the calendar (default: 60).
        pause_before_high_impact_minutes: Minutes to pause trading before high-impact events (default: 15).
    """

    # ForexFactory-like RSS feed URL (using a free alternative or mock)
    # In production, use: https://nfs.faireconomy.media/ff_calendar_thisweek.xml
    # Or Investing.com RSS: https://www.investing.com/rss/news_301.rss
    CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

    def __init__(
        self,
        refresh_interval_minutes: int = 60,
        pause_before_high_impact_minutes: int = 15,
    ) -> None:
        self.refresh_interval = refresh_interval_minutes
        self.pause_before_minutes = pause_before_high_impact_minutes
        self._events: List[EconomicEvent] = []
        self._last_refresh: Optional[datetime] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background calendar refresh task."""
        if self._running:
            logger.warning("ForexEconomicCalendar already running")
            return

        self._running = True
        await self.refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("ForexEconomicCalendar started (refresh every {} min)", self.refresh_interval)

    async def stop(self) -> None:
        """Stop background refresh task."""
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("ForexEconomicCalendar stopped")

    async def refresh(self) -> None:
        """Fetch and parse the latest economic calendar events."""
        try:
            logger.debug("Fetching economic calendar from {}", self.CALENDAR_URL)
            async with aiohttp.ClientSession() as session:
                async with session.get(self.CALENDAR_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Failed to fetch calendar: HTTP {}", resp.status)
                        return

                    xml_content = await resp.text()
                    self._events = self._parse_forexfactory_xml(xml_content)
                    self._last_refresh = datetime.now(tz=timezone.utc)
                    logger.info("Parsed {} economic events", len(self._events))

        except Exception as exc:
            logger.error("Error refreshing economic calendar: {}", exc)

    def should_pause_trading(self, currency: str = "USD") -> tuple[bool, Optional[EconomicEvent]]:
        """Check if trading should be paused due to upcoming high-impact news.

        Args:
            currency: Currency to check (e.g., "USD", "EUR").

        Returns:
            (should_pause, next_event): Tuple of bool and the upcoming event if any.
        """
        now = datetime.now(tz=timezone.utc)
        pause_threshold = now + timedelta(minutes=self.pause_before_minutes)

        for event in self._events:
            if event.impact == "high" and event.currency == currency:
                if now <= event.timestamp <= pause_threshold:
                    return (True, event)

        return (False, None)

    def get_upcoming_events(
        self,
        currency: Optional[str] = None,
        impact_filter: Optional[str] = None,
        hours_ahead: int = 24,
    ) -> List[EconomicEvent]:
        """Get upcoming events within the specified time window.

        Args:
            currency: Filter by currency (e.g., "USD"). If None, return all.
            impact_filter: Filter by impact level ("low", "medium", "high"). If None, return all.
            hours_ahead: How many hours ahead to look.

        Returns:
            List of upcoming EconomicEvent objects.
        """
        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        filtered = []
        for event in self._events:
            if event.timestamp < now or event.timestamp > cutoff:
                continue
            if currency and event.currency != currency:
                continue
            if impact_filter and event.impact != impact_filter:
                continue
            filtered.append(event)

        return sorted(filtered, key=lambda e: e.timestamp)

    def minutes_until_next_high_impact(self, currency: str = "USD") -> int:
        """Return minutes until the next high-impact event for the given currency.

        Args:
            currency: Currency to check.

        Returns:
            Minutes until next high-impact event, or 999999 if none within 24 hours.
        """
        events = self.get_upcoming_events(currency=currency, impact_filter="high", hours_ahead=24)
        if not events:
            return 999999

        now = datetime.now(tz=timezone.utc)
        delta = events[0].timestamp - now
        return int(delta.total_seconds() / 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        """Background task that refreshes the calendar periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.refresh_interval * 60)
                if self._running:
                    await self.refresh()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in calendar refresh loop: {}", exc)

    def _parse_forexfactory_xml(self, xml_content: str) -> List[EconomicEvent]:
        """Parse ForexFactory XML feed into EconomicEvent objects.

        Args:
            xml_content: Raw XML string from ForexFactory RSS.

        Returns:
            List of EconomicEvent objects.
        """
        # This is a simplified parser. In production, use xml.etree.ElementTree
        # or a proper XML parsing library.
        events = []

        # Mock parsing (replace with real XML parsing)
        # Example ForexFactory XML structure:
        # <event>
        #   <title>Non-Farm Payrolls</title>
        #   <country>USD</country>
        #   <date>2026-03-06 13:30:00</date>
        #   <impact>high</impact>
        #   <forecast>200K</forecast>
        #   <previous>180K</previous>
        # </event>

        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_content)
            for event_elem in root.findall(".//event"):
                title = event_elem.findtext("title", "")
                currency = event_elem.findtext("country", "USD")
                date_str = event_elem.findtext("date", "")
                impact = event_elem.findtext("impact", "medium")
                forecast = event_elem.findtext("forecast")
                actual = event_elem.findtext("actual")
                previous = event_elem.findtext("previous")

                # Parse date
                try:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                events.append(
                    EconomicEvent(
                        timestamp=timestamp,
                        currency=currency,
                        title=title,
                        impact=impact,
                        forecast=forecast,
                        actual=actual,
                        previous=previous,
                    )
                )

        except Exception as exc:
            logger.warning("Failed to parse ForexFactory XML: {}", exc)

        return events


# ------------------------------------------------------------------
# Mock calendar for testing (when ForexFactory is unavailable)
# ------------------------------------------------------------------


def create_mock_calendar() -> ForexEconomicCalendar:
    """Create a mock calendar with sample high-impact events for testing."""
    calendar = ForexEconomicCalendar()

    # Add mock NFP event (first Friday of month, 8:30 AM EST = 13:30 UTC)
    now = datetime.now(tz=timezone.utc)
    next_friday = now + timedelta(days=(4 - now.weekday()) % 7)
    nfp_time = next_friday.replace(hour=13, minute=30, second=0, microsecond=0)

    calendar._events = [
        EconomicEvent(
            timestamp=nfp_time,
            currency="USD",
            title="Non-Farm Payrolls",
            impact="high",
            forecast="200K",
            previous="180K",
        ),
        EconomicEvent(
            timestamp=now + timedelta(hours=2),
            currency="EUR",
            title="ECB Interest Rate Decision",
            impact="high",
        ),
        EconomicEvent(
            timestamp=now + timedelta(hours=8),
            currency="GBP",
            title="UK GDP Growth Rate",
            impact="medium",
        ),
    ]

    calendar._last_refresh = now
    logger.info("Mock calendar created with {} events", len(calendar._events))
    return calendar
