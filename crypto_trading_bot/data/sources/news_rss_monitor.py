"""RSS feed monitor for crypto news sources."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Set

import aiohttp
import feedparser
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class NewsRSSMonitor(BaseSource):
    """Monitors multiple RSS feeds for crypto news."""

    DEFAULT_FEEDS = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
        "https://thedefiant.io/feed",
        "https://bitcoinmagazine.com/.rss/full/",
    ]

    def __init__(
        self,
        feed_urls: Optional[List[str]] = None,
        polling_interval: int = 120,
    ):
        super().__init__("news_rss", DataSourceType.RSS_FEED)
        self._feeds = feed_urls or self.DEFAULT_FEEDS
        self._polling_interval = polling_interval
        self._seen_urls: Set[str] = set()
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info(f"RSS Monitor started, tracking {len(self._feeds)} feeds")
        while self._running:
            try:
                new_items = await self._poll_all_feeds()
                self._items.extend(new_items)
                # Keep only last 500 items
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as e:
                logger.error(f"RSS polling error: {e}")
                self._errors += 1
                await asyncio.sleep(30)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._poll_all_feeds()
        return self._items[-limit:]

    async def _poll_all_feeds(self) -> List[DataItem]:
        tasks = [self._fetch_feed(url) for url in self._feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        new_items: List[DataItem] = []
        for result in results:
            if isinstance(result, list):
                new_items.extend(result)
        return new_items

    async def _fetch_feed(self, url: str) -> List[DataItem]:
        items: List[DataItem] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    content = await resp.read()
            feed = feedparser.parse(content)
            for entry in feed.entries[:20]:
                item_url = entry.get("link", "")
                if item_url in self._seen_urls:
                    continue
                self._seen_urls.add(item_url)

                title = entry.get("title", "")
                summary = entry.get("summary", "")
                content_text = f"{title}. {summary}"

                published = entry.get("published_parsed")
                if published:
                    # published_parsed is already UTC; build a naive UTC datetime directly
                    ts = datetime(*published[:6])
                else:
                    ts = _utcnow()

                assets = self._extract_mentioned_assets(content_text)
                items.append(
                    DataItem(
                        source_type=self.source_type,
                        source_name=f"rss:{feed.feed.get('title', url)[:30]}",
                        content=content_text,
                        url=item_url,
                        timestamp=ts,
                        metadata={"title": title, "feed_url": url},
                        relevance_score=0.8 if assets else 0.4,
                        urgency_score=self._calculate_urgency(content_text),
                        mentioned_assets=assets,
                    )
                )
                self._items_collected += 1
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feed {url}: {e}")
        return items
