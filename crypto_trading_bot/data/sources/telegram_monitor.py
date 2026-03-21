"""Telegram channel monitor for crypto signals."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Set

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TelegramMonitor(BaseSource):
    """Monitors Telegram channels for crypto signals via Telethon or HTTP/RSS fallback."""

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
        "pump",
        "dump",
        "breakout",
    ]

    def __init__(
        self,
        api_id: str = "",
        api_hash: str = "",
        phone: str = "",
        bot_token: str = "",
        channels: Optional[List[str]] = None,
        polling_interval: int = 60,
    ):
        super().__init__("telegram", DataSourceType.TELEGRAM)
        self._api_id = api_id
        self._api_hash = api_hash
        self._phone = phone
        self._bot_token = bot_token
        self._channels: List[str] = channels or []
        self._polling_interval = polling_interval
        self._seen_ids: Set[str] = set()
        self._items: List[DataItem] = []
        self._client = None  # Telethon TelegramClient

    async def _init_client(self) -> None:
        if not (self._api_id and self._api_hash):
            logger.warning(
                "Telegram: no API credentials – will use bot token fallback if available."
            )
            return
        try:
            from telethon import TelegramClient  # type: ignore
            from telethon.sessions import StringSession  # type: ignore

            self._client = TelegramClient(StringSession(), int(self._api_id), self._api_hash)
            await self._client.start(phone=self._phone or None)
            logger.info("Telegram Telethon client initialised")
        except Exception as exc:
            logger.warning(f"Telegram Telethon init failed: {exc}. Falling back to HTTP.")
            self._client = None

    async def start_monitoring(self) -> None:
        self._running = True
        await self._init_client()
        logger.info(f"Telegram Monitor started – channels={len(self._channels)}")
        while self._running:
            try:
                for channel in self._channels:
                    new_items = await self.get_messages(channel, limit=20)
                    self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Telegram monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    async def join_channel(self, channel: str) -> bool:
        """Join a Telegram channel (requires Telethon user session)."""
        if self._client is None:
            logger.warning("Cannot join channel: Telethon client not available.")
            return False
        try:
            from telethon.tl.functions.channels import JoinChannelRequest  # type: ignore

            await self._client(JoinChannelRequest(channel))
            logger.info(f"Joined Telegram channel: {channel}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to join channel {channel}: {exc}")
            return False

    async def get_messages(self, channel: str, limit: int = 20) -> List[DataItem]:
        """Fetch recent messages from a channel."""
        if self._client is not None:
            return await self._fetch_via_telethon(channel, limit)
        if self._bot_token:
            return await self._fetch_via_bot_api(channel, limit)
        logger.debug(f"No Telegram credentials available for channel {channel}")
        return []

    async def _fetch_via_telethon(self, channel: str, limit: int) -> List[DataItem]:
        items: List[DataItem] = []
        try:
            messages = await self._client.get_messages(channel, limit=limit)
            for msg in messages:
                msg_id = f"{channel}:{msg.id}"
                if msg_id in self._seen_ids:
                    continue
                self._seen_ids.add(msg_id)
                text = msg.text or msg.message or ""
                if not text:
                    continue
                item = self._parse_message(
                    text=text,
                    msg_id=str(msg.id),
                    channel=channel,
                    timestamp=msg.date.replace(tzinfo=None) if msg.date else _utcnow(),
                    sender=str(msg.sender_id) if msg.sender_id else "unknown",
                )
                items.append(item)
                self._items_collected += 1
        except Exception as exc:
            logger.warning(f"Telethon get_messages({channel}) error: {exc}")
            self._errors += 1
        return items

    async def _fetch_via_bot_api(self, channel: str, limit: int) -> List[DataItem]:
        """Fetch messages via Telegram Bot API (limited to chats the bot is in)."""
        items: List[DataItem] = []
        url = f"https://api.telegram.org/bot{self._bot_token}/getUpdates"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
            for update in data.get("result", [])[:limit]:
                message = update.get("message") or update.get("channel_post", {})
                if not message:
                    continue
                msg_id = str(update.get("update_id", ""))
                if msg_id in self._seen_ids:
                    continue
                self._seen_ids.add(msg_id)
                text = message.get("text", "")
                if not text:
                    continue
                ts_raw = message.get("date", 0)
                ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc).replace(tzinfo=None)
                item = self._parse_message(
                    text=text,
                    msg_id=msg_id,
                    channel=channel,
                    timestamp=ts,
                    sender=str(message.get("from", {}).get("username", "unknown")),
                )
                items.append(item)
                self._items_collected += 1
        except Exception as exc:
            logger.warning(f"Telegram bot API fetch error: {exc}")
            self._errors += 1
        return items

    def _parse_message(
        self,
        text: str,
        msg_id: str,
        channel: str,
        timestamp: datetime,
        sender: str = "unknown",
    ) -> DataItem:
        assets = self._extract_mentioned_assets(text)
        has_signal = self._detect_signal(text)
        return DataItem(
            source_type=self.source_type,
            source_name=f"telegram/{channel}",
            content=text,
            author=sender,
            timestamp=timestamp,
            raw_data={"msg_id": msg_id, "channel": channel},
            metadata={"channel": channel, "has_signal": has_signal},
            relevance_score=0.9 if has_signal else (0.6 if assets else 0.3),
            urgency_score=self._calculate_urgency(text, 0.6 if has_signal else 0.3),
            mentioned_assets=assets,
        )

    def _detect_signal(self, text: str) -> bool:
        """Return True if the message appears to contain a trading signal."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.SIGNAL_KEYWORDS)
