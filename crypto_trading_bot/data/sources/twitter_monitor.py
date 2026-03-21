"""Twitter/X monitor for crypto content."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Set

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TwitterMonitor(BaseSource):
    """Monitors Twitter/X for crypto content from influential accounts and hashtags."""

    DEFAULT_HASHTAGS = ["#crypto", "#bitcoin", "#ethereum", "#defi", "#altcoins"]

    def __init__(
        self,
        bearer_token: str = "",
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
        access_secret: str = "",
        accounts: Optional[List[str]] = None,
        hashtags: Optional[List[str]] = None,
        polling_interval: int = 60,
    ):
        super().__init__("twitter", DataSourceType.TWITTER)
        self._bearer_token = bearer_token
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._access_secret = access_secret
        self._accounts = accounts or []
        self._hashtags = hashtags or self.DEFAULT_HASHTAGS
        self._polling_interval = polling_interval
        self._seen_ids: Set[str] = set()
        self._items: List[DataItem] = []
        self._client = None  # tweepy.AsyncClient
        self._is_authenticated: bool = bool(bearer_token)

    async def _init_client(self) -> None:
        if not self._bearer_token:
            logger.warning("Twitter: no bearer token – monitoring disabled.")
            return
        try:
            import tweepy  # type: ignore

            self._client = tweepy.AsyncClient(
                bearer_token=self._bearer_token,
                consumer_key=self._api_key or None,
                consumer_secret=self._api_secret or None,
                access_token=self._access_token or None,
                access_token_secret=self._access_secret or None,
                wait_on_rate_limit=True,
            )
            logger.info("Twitter tweepy client initialised (OAuth 2.0)")
        except Exception as exc:
            logger.warning(f"Twitter tweepy init failed: {exc}. Falling back to basic HTTP.")

    async def start_monitoring(self) -> None:
        if not self._is_authenticated:
            logger.info("Twitter Monitor disabled — no bearer token provided")
            return
        self._running = True
        await self._init_client()
        logger.info(
            f"Twitter Monitor started – accounts={len(self._accounts)}, "
            f"hashtags={len(self._hashtags)}"
        )
        while self._running and self._is_authenticated:
            try:
                new_items: List[DataItem] = []
                for account in self._accounts:
                    new_items.extend(await self.fetch_tweets(account, count=10))
                for tag in self._hashtags:
                    new_items.extend(await self.search_tweets(tag))
                new_items = self._filter_relevant(new_items)
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Twitter monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    async def fetch_tweets(self, account: str, count: int = 10) -> List[DataItem]:
        """Fetch recent tweets from a specific account."""
        items: List[DataItem] = []
        if self._client is None:
            return await self._http_fallback(f"from:{account}", count)
        try:
            resp = await self._client.search_recent_tweets(
                query=f"from:{account} -is:retweet",
                max_results=min(count, 100),
                tweet_fields=["created_at", "author_id", "public_metrics"],
            )
            for tweet in resp.data or []:
                if str(tweet.id) in self._seen_ids:
                    continue
                self._seen_ids.add(str(tweet.id))
                items.append(
                    self._build_item(
                        text=tweet.text,
                        tweet_id=str(tweet.id),
                        author=account,
                        created_at=tweet.created_at,
                        metrics=tweet.public_metrics or {},
                    )
                )
                self._items_collected += 1
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.error("Twitter API Key Invalid - Disabling Source")
                self._is_authenticated = False
            else:
                logger.warning(f"Twitter fetch_tweets({account}) error: {exc}")
                self._errors += 1
        return items

    async def search_tweets(self, query: str) -> List[DataItem]:
        """Search recent tweets by query string."""
        items: List[DataItem] = []
        if self._client is None:
            return await self._http_fallback(query, 10)
        try:
            resp = await self._client.search_recent_tweets(
                query=f"{query} -is:retweet lang:en",
                max_results=10,
                tweet_fields=["created_at", "author_id", "public_metrics"],
            )
            for tweet in resp.data or []:
                if str(tweet.id) in self._seen_ids:
                    continue
                self._seen_ids.add(str(tweet.id))
                items.append(
                    self._build_item(
                        text=tweet.text,
                        tweet_id=str(tweet.id),
                        author=str(tweet.author_id),
                        created_at=tweet.created_at,
                        metrics=tweet.public_metrics or {},
                    )
                )
                self._items_collected += 1
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.error("Twitter API Key Invalid - Disabling Source")
                self._is_authenticated = False
            else:
                logger.warning(f"Twitter search_tweets({query}) error: {exc}")
                self._errors += 1
        return items

    async def _http_fallback(self, query: str, count: int) -> List[DataItem]:
        """Basic HTTP fallback using bearer token directly."""
        if not self._bearer_token:
            return []
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self._bearer_token}"}
        params = {
            "query": query,
            "max_results": min(count, 100),
            "tweet.fields": "created_at,author_id,public_metrics",
        }
        items: List[DataItem] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 401:
                        logger.error("Twitter API Key Invalid - Disabling Source")
                        self._is_authenticated = False
                        return []
                    if resp.status == 200:
                        data = await resp.json()
                        for tweet in data.get("data", []):
                            tid = tweet["id"]
                            if tid in self._seen_ids:
                                continue
                            self._seen_ids.add(tid)
                            items.append(
                                self._build_item(
                                    text=tweet.get("text", ""),
                                    tweet_id=tid,
                                    author=tweet.get("author_id", "unknown"),
                                    created_at=None,
                                    metrics=tweet.get("public_metrics", {}),
                                )
                            )
                            self._items_collected += 1
                    else:
                        logger.warning(f"Twitter HTTP fallback status {resp.status}")
        except Exception as exc:
            logger.warning(f"Twitter HTTP fallback error: {exc}")
        return items

    def _build_item(
        self,
        text: str,
        tweet_id: str,
        author: str,
        created_at: Optional[datetime],
        metrics: dict,
    ) -> DataItem:
        assets = self._extract_mentioned_assets(text)
        likes = metrics.get("like_count", 0)
        retweets = metrics.get("retweet_count", 0)
        influence = min(1.0, (likes + retweets * 3) / 5000)
        ts = created_at.replace(tzinfo=None) if created_at else _utcnow()
        return DataItem(
            source_type=self.source_type,
            source_name=f"twitter/@{author}",
            content=text,
            url=f"https://twitter.com/i/web/status/{tweet_id}",
            author=author,
            timestamp=ts,
            raw_data={"id": tweet_id, "metrics": metrics},
            metadata={"tweet_id": tweet_id, "likes": likes, "retweets": retweets},
            relevance_score=min(1.0, 0.4 + influence),
            urgency_score=self._calculate_urgency(text, influence),
            mentioned_assets=assets,
        )

    def _filter_relevant(self, tweets: List[DataItem]) -> List[DataItem]:
        """Keep only tweets that mention crypto assets or have high engagement."""
        return [t for t in tweets if t.mentioned_assets or t.relevance_score >= 0.6]

    def _calculate_urgency(self, text: str, author_influence: float = 0.5) -> float:
        return super()._calculate_urgency(text, author_influence)
