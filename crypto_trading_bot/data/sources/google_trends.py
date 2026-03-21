"""Google Trends monitor for crypto search interest."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GoogleTrendsMonitor(BaseSource):
    """Tracks Google Trends interest for crypto keywords using pytrends."""

    DEFAULT_KEYWORDS = [
        "bitcoin",
        "ethereum",
        "crypto",
        "buy bitcoin",
        "crypto crash",
        "crypto bull run",
        "altcoin",
    ]

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        geo: str = "",  # empty = worldwide
        polling_interval: int = 3600,  # Trends updates slowly; 1 hour is fine
    ):
        super().__init__("google_trends", DataSourceType.REST_API)
        self._keywords = keywords or self.DEFAULT_KEYWORDS
        self._geo = geo
        self._polling_interval = polling_interval
        self._prev_interest: Dict[str, float] = {}  # keyword -> last average interest
        self._items: List[DataItem] = []
        self._pytrends = None

    def _init_pytrends(self) -> bool:
        try:
            from pytrends.request import TrendReq  # type: ignore

            self._pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            return True
        except ImportError:
            logger.warning("Google Trends: pytrends not installed – monitoring disabled.")
            return False
        except Exception as exc:
            logger.warning(f"Google Trends: pytrends init failed: {exc}")
            return False

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._init_pytrends():
            self._running = False
            return
        logger.info(f"Google Trends Monitor started – {len(self._keywords)} keywords")
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                new_items = await loop.run_in_executor(None, self._poll_trends)
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Google Trends monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            if self._pytrends is None:
                self._init_pytrends()
            if self._pytrends:
                loop = asyncio.get_event_loop()
                new_items = await loop.run_in_executor(None, self._poll_trends)
                self._items.extend(new_items)
        return self._items[-limit:]

    def fetch_interest(self, keyword: str) -> Dict[str, float]:
        """Fetch interest-over-time for a single keyword (blocking, run in executor)."""
        if self._pytrends is None:
            return {}
        try:
            self._pytrends.build_payload([keyword], geo=self._geo, timeframe="now 7-d")
            df = self._pytrends.interest_over_time()
            if df is None or df.empty or keyword not in df.columns:
                return {}
            series = df[keyword].dropna()
            return {
                "current": float(series.iloc[-1]),
                "mean": float(series.mean()),
                "max": float(series.max()),
            }
        except Exception as exc:
            logger.debug(f"Google Trends fetch_interest({keyword}): {exc}")
            return {}

    def _poll_trends(self) -> List[DataItem]:
        """Synchronous polling of Google Trends for all keywords."""
        items: List[DataItem] = []
        # Batch keywords in groups of 5 (pytrends limit)
        batch_size = 5
        for i in range(0, len(self._keywords), batch_size):
            batch = self._keywords[i : i + batch_size]
            try:
                self._pytrends.build_payload(batch, geo=self._geo, timeframe="now 7-d")
                df = self._pytrends.interest_over_time()
                if df is None or df.empty:
                    continue
                for keyword in batch:
                    if keyword not in df.columns:
                        continue
                    series = df[keyword].dropna()
                    if series.empty:
                        continue
                    current = float(series.iloc[-1])
                    prev = self._prev_interest.get(keyword)
                    self._prev_interest[keyword] = current
                    if prev is not None:
                        item = self._calculate_trend_signal(
                            {
                                "keyword": keyword,
                                "current": current,
                                "prev": prev,
                                "mean": float(series.mean()),
                            }
                        )
                        if item:
                            items.append(item)
                            self._items_collected += 1
            except Exception as exc:
                logger.debug(f"Google Trends batch {batch}: {exc}")
        self._last_update = _utcnow()
        return items

    def track_trend_changes(self) -> Dict[str, float]:
        """Return percentage changes from previous snapshot."""
        changes: Dict[str, float] = {}
        for keyword, prev in self._prev_interest.items():
            if prev and prev > 0:
                current = self._prev_interest.get(keyword, prev)
                changes[keyword] = ((current - prev) / prev) * 100
        return changes

    def _calculate_trend_signal(self, data: dict) -> Optional[DataItem]:
        """Build a DataItem from a trend data dict."""
        keyword = data["keyword"]
        current = data["current"]
        prev = data["prev"]
        mean = data.get("mean", current)

        if prev == 0:
            return None
        change_pct = ((current - prev) / prev) * 100
        if abs(change_pct) < 20.0:  # only flag notable swings
            return None

        direction = "surging" if change_pct > 0 else "dropping"
        signal = (
            "fomo_bullish"
            if change_pct > 0 and current > mean
            else ("interest_fading" if change_pct < 0 else "neutral")
        )
        content = (
            f"Google Trends: '{keyword}' interest {direction} "
            f"({change_pct:+.1f}%), score={current:.0f}. Signal: {signal}"
        )
        assets = self._extract_mentioned_assets(keyword)
        magnitude = min(1.0, abs(change_pct) / 100.0)
        return DataItem(
            source_type=self.source_type,
            source_name="google_trends",
            content=content,
            url=f"https://trends.google.com/trends/explore?q={keyword}",
            timestamp=_utcnow(),
            metadata={
                "keyword": keyword,
                "current": current,
                "prev": prev,
                "change_pct": change_pct,
                "signal": signal,
            },
            relevance_score=min(1.0, 0.4 + magnitude * 0.6),
            urgency_score=min(1.0, 0.3 + magnitude * 0.7),
            mentioned_assets=assets,
        )
