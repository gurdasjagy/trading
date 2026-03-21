"""Reddit monitor for crypto subreddits."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Set

from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RedditMonitor(BaseSource):
    """Monitors crypto subreddits using PRAW."""

    DEFAULT_SUBREDDITS = [
        "cryptocurrency",
        "bitcoin",
        "ethtrader",
        "solana",
        "CryptoMarkets",
        "SatoshiStreetBets",
        "defi",
    ]

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        username: str = "",
        password: str = "",
        subreddits: Optional[List[str]] = None,
        polling_interval: int = 120,
    ):
        super().__init__("reddit", DataSourceType.REST_API)
        self._client_id = client_id
        self._client_secret = client_secret
        self._username = username
        self._password = password
        self._subreddits = subreddits or self.DEFAULT_SUBREDDITS
        self._polling_interval = polling_interval
        self._seen_ids: Set[str] = set()
        self._items: List[DataItem] = []
        self._reddit = None
        self._is_authenticated: bool = bool(client_id and client_secret)

    async def _init_reddit(self) -> None:
        try:
            import praw  # type: ignore

            self._reddit = praw.Reddit(
                client_id=self._client_id or "placeholder",
                client_secret=self._client_secret or "placeholder",
                username=self._username or None,
                password=self._password or None,
                user_agent="python:crypto_trading_bot:v1.0 (by /u/your_username)",
            )
            logger.info("Reddit PRAW client initialised")
        except Exception as e:
            logger.warning(f"Reddit init failed: {e}. Reddit monitoring disabled.")
            self._reddit = None

    async def start_monitoring(self) -> None:
        if not self._is_authenticated:
            logger.info("Reddit Monitor disabled — no API credentials provided")
            return
        self._running = True
        await self._init_reddit()
        logger.info(f"Reddit Monitor started, tracking: {', '.join(self._subreddits)}")
        loop = asyncio.get_event_loop()
        while self._running and self._is_authenticated:
            try:
                if self._reddit:
                    new_items = await loop.run_in_executor(None, self._fetch_hot_posts)
                    self._items.extend(new_items)
                    if len(self._items) > 500:
                        self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error("Reddit API Key Invalid - Disabling Source")
                    self._is_authenticated = False
                    break
                logger.error(f"Reddit monitoring error: {e}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    def _fetch_hot_posts(self) -> List[DataItem]:
        items: List[DataItem] = []
        try:
            for subreddit_name in self._subreddits:
                subreddit = self._reddit.subreddit(subreddit_name)
                for post in subreddit.hot(limit=10):
                    if post.id in self._seen_ids:
                        continue
                    self._seen_ids.add(post.id)
                    content = f"{post.title}. " f"{post.selftext[:500] if post.selftext else ''}"
                    assets = self._extract_mentioned_assets(content)
                    score_normalized = min(1.0, post.score / 10000)
                    items.append(
                        DataItem(
                            source_type=self.source_type,
                            source_name=f"reddit/r/{subreddit_name}",
                            content=content,
                            url=f"https://reddit.com{post.permalink}",
                            author=str(post.author) if post.author else "unknown",
                            timestamp=datetime.fromtimestamp(
                                post.created_utc, tz=timezone.utc
                            ).replace(tzinfo=None),
                            metadata={
                                "score": post.score,
                                "comments": post.num_comments,
                                "subreddit": subreddit_name,
                            },
                            relevance_score=score_normalized,
                            urgency_score=self._calculate_urgency(content, score_normalized),
                            mentioned_assets=assets,
                        )
                    )
                    self._items_collected += 1
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error("Reddit API Key Invalid - Disabling Source")
                self._is_authenticated = False
            else:
                logger.warning(f"Reddit fetch error: {e}")
        return items
