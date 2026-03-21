"""Discord server monitor for crypto signals."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Set

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DiscordMonitor(BaseSource):
    """Monitors Discord servers for crypto signals via discord.py or webhook fallback."""

    SIGNAL_KEYWORDS = [
        "buy",
        "sell",
        "long",
        "short",
        "entry",
        "target",
        "stop loss",
        "tp",
        "sl",
        "signal",
        "alert",
        "breakout",
        "pump",
        "dump",
    ]

    def __init__(
        self,
        bot_token: str = "",
        webhook_url: str = "",
        channel_ids: Optional[List[str]] = None,
        polling_interval: int = 60,
    ):
        super().__init__("discord", DataSourceType.DISCORD)
        self._bot_token = bot_token
        self._webhook_url = webhook_url
        self._channel_ids: List[str] = channel_ids or []
        self._polling_interval = polling_interval
        self._seen_ids: Set[str] = set()
        self._items: List[DataItem] = []
        self._discord_client = None

    async def _init_client(self) -> None:
        if not self._bot_token:
            logger.warning("Discord: no bot token – will poll via REST API if token provided.")
            return
        try:
            import discord  # type: ignore

            intents = discord.Intents.default()
            intents.message_content = True
            self._discord_client = discord.Client(intents=intents)

            @self._discord_client.event
            async def on_message(message: discord.Message) -> None:
                if message.author.bot:
                    return
                if str(message.channel.id) not in self._channel_ids and self._channel_ids:
                    return
                item = self._parse_message(
                    message.content,
                    str(message.id),
                    str(message.channel.id),
                    str(message.author),
                    message.created_at,
                )
                self._items.append(item)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                self._items_collected += 1

            asyncio.create_task(self._discord_client.start(self._bot_token))
            logger.info("Discord discord.py client initialised (event-driven)")
        except Exception as exc:
            logger.warning(f"Discord discord.py init failed: {exc}. Falling back to REST API.")
            self._discord_client = None

    async def start_monitoring(self) -> None:
        self._running = True
        await self._init_client()
        logger.info(f"Discord Monitor started – channels={len(self._channel_ids)}")
        while self._running:
            try:
                if self._discord_client is None and self._bot_token:
                    for channel_id in self._channel_ids:
                        new_items = await self.get_messages(channel_id, limit=50)
                        self._items.extend(new_items)
                    if len(self._items) > 500:
                        self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Discord monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False
        if self._discord_client is not None:
            try:
                await self._discord_client.close()
            except Exception:
                pass

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    async def get_messages(self, channel_id: str, limit: int = 50) -> List[DataItem]:
        """Fetch messages from a Discord channel via REST API."""
        if not self._bot_token:
            logger.debug("Discord: cannot fetch messages without bot token.")
            return []
        items: List[DataItem] = []
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {"Authorization": f"Bot {self._bot_token}"}
        params = {"limit": min(limit, 100)}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        messages = await resp.json()
                        for msg in messages:
                            msg_id = msg.get("id", "")
                            if msg_id in self._seen_ids:
                                continue
                            self._seen_ids.add(msg_id)
                            content = msg.get("content", "")
                            if not content:
                                continue
                            ts_str = msg.get("timestamp", "")
                            try:
                                ts = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=None)
                            except Exception:
                                ts = _utcnow()
                            author = msg.get("author", {}).get("username", "unknown")
                            item = self._parse_message(content, msg_id, channel_id, author, ts)
                            items.append(item)
                            self._items_collected += 1
                    else:
                        logger.warning(f"Discord API status {resp.status} for channel {channel_id}")
        except Exception as exc:
            logger.warning(f"Discord get_messages error: {exc}")
            self._errors += 1
        return items

    def _parse_message(
        self,
        content: str,
        msg_id: str,
        channel_id: str,
        author: str,
        timestamp: datetime,
    ) -> DataItem:
        assets = self._extract_mentioned_assets(content)
        has_signal = any(kw in content.lower() for kw in self.SIGNAL_KEYWORDS)
        return DataItem(
            source_type=self.source_type,
            source_name=f"discord/channel/{channel_id}",
            content=content,
            author=author,
            timestamp=timestamp if isinstance(timestamp, datetime) else _utcnow(),
            raw_data={"msg_id": msg_id, "channel_id": channel_id},
            metadata={"channel_id": channel_id, "has_signal": has_signal},
            relevance_score=0.9 if has_signal else (0.6 if assets else 0.3),
            urgency_score=self._calculate_urgency(content, 0.6 if has_signal else 0.3),
            mentioned_assets=assets,
        )
