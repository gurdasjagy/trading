"""Economic calendar trade filter - blocks trading around high-impact events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from loguru import logger


class EconomicCalendarFilter:
    """Blocks trade execution during high-impact economic event windows.

    Pauses trading ``buffer_minutes_before`` minutes before and
    ``buffer_minutes_after`` minutes after events like FOMC, CPI, NFP.
    """

    KILL_SWITCH_EVENTS = [
        "fomc",
        "federal reserve",
        "interest rate decision",
        "cpi",
        "consumer price index",
        "inflation",
        "nonfarm payroll",
        "nfp",
        "employment situation",
        "gdp",
        "gross domestic product",
        "pce",
        "personal consumption",
        "jackson hole",
    ]

    def __init__(
        self,
        buffer_minutes_before: int = 15,
        buffer_minutes_after: int = 15,
    ) -> None:
        self._buffer_before = timedelta(minutes=buffer_minutes_before)
        self._buffer_after = timedelta(minutes=buffer_minutes_after)
        self._cached_events: List[dict] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

    async def refresh_events(self) -> None:
        """Fetch upcoming events from the economic calendar."""
        try:
            from data.sources.economic_calendar import EconomicCalendarMonitor

            monitor = EconomicCalendarMonitor()
            items = await monitor.fetch_upcoming_events()
            self._cached_events = [
                item.metadata
                for item in items
                if item.metadata.get("impact") == "high"
            ]
            self._last_fetch = datetime.now(tz=timezone.utc)
        except Exception as exc:
            logger.debug("Economic calendar refresh failed: {}", exc)

    async def is_trading_allowed(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is currently allowed based on economic calendar.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        now = datetime.now(tz=timezone.utc)

        # Refresh cache if stale
        if self._last_fetch is None or (now - self._last_fetch) > self._cache_ttl:
            await self.refresh_events()

        for event in self._cached_events:
            title = (event.get("title", "") or "").lower()
            is_kill_event = any(kw in title for kw in self.KILL_SWITCH_EVENTS)
            if not is_kill_event:
                continue

            try:
                event_time_str = event.get("date", "")
                event_time = datetime.fromisoformat(
                    event_time_str.rstrip("Z")
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue

            window_start = event_time - self._buffer_before
            window_end = event_time + self._buffer_after

            if window_start <= now <= window_end:
                reason = (
                    f"Trading paused: {event.get('title', 'Unknown event')} "
                    f"at {event_time.strftime('%H:%M UTC')} "
                    f"(window: {self._buffer_before.total_seconds() / 60:.0f}min before, "
                    f"{self._buffer_after.total_seconds() / 60:.0f}min after)"
                )
                logger.warning("🚫 {}", reason)
                return False, reason

        return True, None
