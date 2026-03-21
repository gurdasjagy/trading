"""CryptoPanic news data source."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CryptoPanicSource(BaseSource):
    """Fetches crypto news headlines and sentiment from the CryptoPanic free API.

    Polls every 5 minutes (configurable).  Requires a free CryptoPanic API key
    passed as *auth_token*.  When no key is provided the source is automatically
    disabled so the engine can still register it safely.
    """

    _BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(
        self,
        auth_token: Optional[str] = None,
        polling_interval: int = 300,  # 5 minutes
    ) -> None:
        enabled = bool(auth_token)
        super().__init__("cryptopanic", DataSourceType.REST_API, enabled=enabled)
        self._auth_token = auth_token
        self._polling_interval = polling_interval
        self._items: List[DataItem] = []

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        if not self._auth_token:
            logger.info("CryptoPanic source disabled — no API key provided")
            return
        self._running = True
        logger.info("CryptoPanic Monitor started (poll every {}s)", self._polling_interval)
        while self._running:
            try:
                await self._fetch_and_cache(currencies="BTC,ETH,SOL")
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error("CryptoPanic fetch error: {}", exc)
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_and_cache()
        return self._items[-limit:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_and_cache(self, currencies: Optional[str] = None) -> None:
        if not self._auth_token:
            return
        params: dict = {"auth_token": self._auth_token, "kind": "news"}
        if currencies:
            params["currencies"] = currencies
        # Build a loggable URL (without the API key) for diagnostics
        safe_params = {k: v for k, v in params.items() if k != "auth_token"}
        from urllib.parse import urlencode
        safe_url = self._BASE_URL + "?" + urlencode(safe_params)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 401:
                        logger.warning("CryptoPanic: invalid API key (401)")
                        return
                    if resp.status == 404:
                        logger.warning("CryptoPanic: 404 Not Found — URL: {}", safe_url)
                        return
                    if resp.status != 200:
                        logger.warning("CryptoPanic: HTTP {} response", resp.status)
                        return
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.error("CryptoPanic HTTP error: {}", exc)
            self._errors += 1
            return

        new_items: List[DataItem] = []
        for post in data.get("results", []):
            item = self._parse_post(post)
            if item is not None:
                new_items.append(item)

        self._items.extend(new_items)
        if len(self._items) > 500:
            self._items = self._items[-500:]
        self._items_collected += len(new_items)
        self._last_update = _utcnow()
        logger.debug("CryptoPanic: fetched {} new posts", len(new_items))

    def _parse_post(self, post: dict) -> Optional[DataItem]:
        """Convert a raw CryptoPanic post dict into a :class:`DataItem`."""
        try:
            title = post.get("title", "")
            if not title:
                return None

            # Currencies mentioned in the post
            currencies = post.get("currencies") or []
            mentioned_assets = [c["code"] for c in currencies if "code" in c]
            # Supplement with text-based extraction for assets not in currency list
            text_assets = self._extract_mentioned_assets(title)
            for a in text_assets:
                if a not in mentioned_assets:
                    mentioned_assets.append(a)

            # Sentiment from votes
            votes = post.get("votes") or {}
            positive = int(votes.get("positive", 0))
            negative = int(votes.get("negative", 0))
            important = int(votes.get("important", 0))
            total_votes = positive + negative + important
            if total_votes > 0:
                # Raw sentiment in [-1, 1]
                raw_sentiment = (positive - negative) / total_votes
                # Map to relevance: important votes boost relevance
                relevance = min(
                    1.0, max(0.0, 0.5 + raw_sentiment * 0.3 + (important / total_votes) * 0.2)
                )
            else:
                raw_sentiment = 0.0
                relevance = 0.5

            urgency = self._calculate_urgency(title)
            # Boost urgency when post is marked important
            if important > 0 and total_votes > 0:
                urgency = min(1.0, urgency + 0.15)

            # Parse timestamp
            ts_str = post.get("published_at") or post.get("created_at", "")
            try:
                ts = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
            except (ValueError, TypeError):
                ts = _utcnow()

            return DataItem(
                source_type=self.source_type,
                source_name=self.name,
                content=title,
                url=post.get("url"),
                timestamp=ts,
                raw_data=post,
                metadata={
                    "votes": votes,
                    "sentiment_score": raw_sentiment,
                    "kind": post.get("kind", "news"),
                },
                relevance_score=round(relevance, 3),
                urgency_score=round(urgency, 3),
                mentioned_assets=mentioned_assets,
            )
        except Exception as exc:
            logger.debug("CryptoPanic: failed to parse post: {}", exc)
            return None
