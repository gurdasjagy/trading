"""YouTube crypto influencer content monitor."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class YouTubeMonitor(BaseSource):
    """Tracks YouTube crypto influencer content via the YouTube Data API v3."""

    YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

    # High-profile crypto YouTube channel IDs
    DEFAULT_CHANNELS: List[str] = [
        "UCRvqjQPSeaWn-uEx-w0XOIg",  # Coin Bureau
        "UCiRiQGCHGjDLT9FQWFWF5Vw",  # Benjamin Cowen
        "UC4nXW5zT3qmdTkZSRPhzN3A",  # DataDash
        "UCMQ57bHMUfqxW0GGxsggwHQ",  # Crypto Banter
        "UCEFJVYNiPp8xeIUyfaPCPQw",  # Altcoin Daily
    ]

    def __init__(
        self,
        api_key: str = "",
        channel_ids: Optional[List[str]] = None,
        polling_interval: int = 1800,  # 30 minutes
        max_results: int = 5,
    ):
        super().__init__("youtube", DataSourceType.REST_API)
        self._api_key = api_key
        self._channel_ids = channel_ids or self.DEFAULT_CHANNELS
        self._polling_interval = polling_interval
        self._max_results = max_results
        self._seen_video_ids: Set[str] = set()
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("YouTube: no API key – monitoring disabled.")
            self._running = False
            return
        logger.info(f"YouTube Monitor started – tracking {len(self._channel_ids)} channels")
        while self._running:
            try:
                for channel_id in self._channel_ids:
                    new_items = await self.fetch_videos(channel_id)
                    self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"YouTube monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    async def fetch_videos(self, channel_id: str) -> List[DataItem]:
        """Fetch the most recent videos from a YouTube channel."""
        if not self._api_key:
            return []
        items: List[DataItem] = []
        params = {
            "key": self._api_key,
            "channelId": channel_id,
            "part": "snippet",
            "order": "date",
            "maxResults": self._max_results,
            "type": "video",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.YOUTUBE_API}/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for result in data.get("items", []):
                            video_id = result.get("id", {}).get("videoId", "")
                            if not video_id or video_id in self._seen_video_ids:
                                continue
                            self._seen_video_ids.add(video_id)
                            snippet = result.get("snippet", {})
                            item = self._analyze_content(
                                video_id=video_id,
                                snippet=snippet,
                                channel_id=channel_id,
                            )
                            items.append(item)
                            self._items_collected += 1
                    elif resp.status == 403:
                        logger.warning("YouTube API quota exceeded or key invalid.")
                    else:
                        logger.debug(f"YouTube API {resp.status} for channel {channel_id}")
        except Exception as exc:
            logger.warning(f"YouTube fetch_videos({channel_id}) error: {exc}")
            self._errors += 1
        self._last_update = _utcnow()
        return items

    async def track_channel(self, channel_id: str) -> Optional[Dict]:
        """Fetch channel metadata (subscriber count, video count)."""
        if not self._api_key:
            return None
        params = {
            "key": self._api_key,
            "id": channel_id,
            "part": "statistics,snippet",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.YOUTUBE_API}/channels",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        if items:
                            ch = items[0]
                            stats = ch.get("statistics", {})
                            return {
                                "channel_id": channel_id,
                                "title": ch.get("snippet", {}).get("title", ""),
                                "subscribers": int(stats.get("subscriberCount", 0)),
                                "total_videos": int(stats.get("videoCount", 0)),
                                "view_count": int(stats.get("viewCount", 0)),
                            }
        except Exception as exc:
            logger.debug(f"track_channel({channel_id}): {exc}")
        return None

    def _analyze_content(
        self,
        video_id: str,
        snippet: dict,
        channel_id: str,
    ) -> DataItem:
        """Parse video snippet and derive relevance/urgency scores."""
        title = snippet.get("title", "")
        description = (snippet.get("description") or "")[:500]
        channel_title = snippet.get("channelTitle", channel_id)
        published_at = snippet.get("publishedAt", "")
        try:
            ts = datetime.fromisoformat(published_at.rstrip("Z")).replace(tzinfo=None)
        except Exception:
            ts = _utcnow()

        content = f"{title}. {description}"
        assets = self._extract_mentioned_assets(content)
        has_urgent = any(kw in title.lower() for kw in self._URGENT_KEYWORDS)
        relevance = 0.8 if assets else (0.6 if has_urgent else 0.4)
        return DataItem(
            source_type=self.source_type,
            source_name=f"youtube/{channel_title}",
            content=content,
            url=f"https://www.youtube.com/watch?v={video_id}",
            author=channel_title,
            timestamp=ts,
            raw_data={"video_id": video_id, "channel_id": channel_id},
            metadata={
                "video_id": video_id,
                "title": title,
                "channel_id": channel_id,
                "channel_title": channel_title,
                "has_urgent_keywords": has_urgent,
            },
            relevance_score=relevance,
            urgency_score=self._calculate_urgency(title, 0.5 if has_urgent else 0.3),
            mentioned_assets=assets,
        )
