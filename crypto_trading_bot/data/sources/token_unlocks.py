"""Token unlock schedule monitor.

Tracks upcoming token vesting / unlock events using the Tokenomist.ai
public API.  Large unlocks can create significant sell pressure; the
monitor emits DataItems when a large unlock is detected within 7 days.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TokenUnlocksMonitor(BaseSource):
    """Monitors upcoming token unlock events.

    Emits a :class:`~data.sources.base_source.DataItem` for each unlock event
    that satisfies:
    * Unlock value ≥ ``min_unlock_usd`` (default 10 M USD)
    * Unlock date is within the next ``horizon_days`` days (default 7)

    Args:
        api_key: Tokenomist API key (leave empty to use public endpoint).
        symbols: List of token symbols to filter (e.g. ``["APT", "SUI"]``).
            Empty list = all symbols.
        polling_interval: Seconds between API polls (default 3600).
        min_unlock_usd: Minimum unlock value in USD to include.
        horizon_days: Number of days ahead to look for upcoming unlocks.
    """

    TOKENOMIST_API = "https://api.tokenomist.ai/v1/unlocks"
    CRYPTORANK_API = "https://api.cryptorank.io/v1/vesting"

    def __init__(
        self,
        api_key: str = "",
        symbols: Optional[List[str]] = None,
        polling_interval: int = 3600,
        min_unlock_usd: float = 10_000_000,
        horizon_days: int = 7,
    ) -> None:
        super().__init__("token_unlocks", DataSourceType.REST_API)
        self._api_key = api_key
        self._symbols = [s.upper() for s in (symbols or [])]
        self._polling_interval = polling_interval
        self._min_unlock_usd = min_unlock_usd
        self._horizon_days = horizon_days
        self._items: List[DataItem] = []
        self._seen_ids: set = set()

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info(
            "TokenUnlocksMonitor started: min_usd={:,.0f}, horizon={}d",
            self._min_unlock_usd,
            self._horizon_days,
        )
        while self._running:
            try:
                new_items = await self._fetch_unlocks()
                self._items.extend(new_items)
                if len(self._items) > 300:
                    self._items = self._items[-300:]
            except Exception as exc:
                logger.error("TokenUnlocksMonitor error: {}", exc)
                self._errors += 1
            await asyncio.sleep(self._polling_interval)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            items = await self._fetch_unlocks()
            self._items.extend(items)
        return self._items[-limit:]

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def _fetch_unlocks(self) -> List[DataItem]:
        """Fetch upcoming unlock events and produce DataItems."""
        items: List[DataItem] = []
        raw_events = await self._get_events()
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(days=self._horizon_days)

        for event in raw_events:
            try:
                unlock_dt = self._parse_datetime(event)
                if unlock_dt is None:
                    continue
                if not (now <= unlock_dt <= horizon):
                    continue

                symbol = str(event.get("token", event.get("symbol", "UNKNOWN"))).upper()
                if self._symbols and symbol not in self._symbols:
                    continue

                value_usd = float(event.get("amount_usd") or event.get("value_usd") or 0)
                if value_usd < self._min_unlock_usd:
                    continue

                event_id = str(event.get("id", f"{symbol}:{unlock_dt.isoformat()}"))
                if event_id in self._seen_ids:
                    continue
                self._seen_ids.add(event_id)

                days_ahead = (unlock_dt - now).days
                item = self._build_item(event, symbol, value_usd, unlock_dt, days_ahead)
                if item:
                    items.append(item)
                    self._items_collected += 1
            except Exception as exc:
                logger.debug("TokenUnlocks parse error: {}", exc)

        self._last_update = _utcnow()
        return items

    async def _get_events(self) -> List[dict]:
        """Fetch raw unlock events from Tokenomist API (or stub)."""
        params: Dict[str, str] = {}
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.TOKENOMIST_API,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("data", data) if isinstance(data, dict) else data
                    logger.debug("TokenUnlocks API status {}", resp.status)
        except Exception as exc:
            logger.debug("TokenUnlocks fetch error: {}", exc)

        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_datetime(self, event: dict) -> Optional[datetime]:
        """Parse the unlock datetime from various field formats."""
        for key in ("unlock_date", "date", "timestamp", "vesting_date"):
            raw = event.get(key)
            if raw is None:
                continue
            if isinstance(raw, (int, float)):
                ts = raw / 1000 if raw > 1e10 else raw
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            if isinstance(raw, str):
                for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = datetime.strptime(raw, fmt)
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
        return None

    def _build_item(
        self,
        event: dict,
        symbol: str,
        value_usd: float,
        unlock_dt: datetime,
        days_ahead: int,
    ) -> Optional[DataItem]:
        """Build a DataItem for a notable unlock event."""
        category = event.get("category", "vesting")
        pct_supply = float(event.get("percent_of_supply", 0) or 0)
        pct_str = f" ({pct_supply:.1f}% of circulating supply)" if pct_supply > 0 else ""

        urgency = min(1.0, 0.5 + (value_usd / 500_000_000))
        if days_ahead <= 1:
            urgency = min(1.0, urgency + 0.3)

        content = (
            f"Token Unlock: {symbol} — ${value_usd / 1e6:.1f}M unlocking "
            f"in {days_ahead} day{'s' if days_ahead != 1 else ''} "
            f"({unlock_dt.strftime('%Y-%m-%d')}){pct_str}. "
            f"Category: {category}."
        )

        return DataItem(
            source_type=self.source_type,
            source_name="token_unlocks",
            content=content,
            timestamp=_utcnow(),
            relevance_score=min(1.0, 0.6 + value_usd / 1_000_000_000),
            urgency_score=urgency,
            mentioned_assets=[symbol],
            metadata={
                "sentiment_score": -0.3 if value_usd > 50_000_000 else -0.1,
                "symbol": symbol,
                "value_usd": value_usd,
                "unlock_date": unlock_dt.isoformat(),
                "days_ahead": days_ahead,
                "category": category,
                "pct_supply": pct_supply,
            },
        )
