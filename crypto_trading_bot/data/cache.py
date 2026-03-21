"""Redis caching layer for the trading bot."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from loguru import logger


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CacheManager:
    """Redis-based caching with transparent in-memory fallback."""

    _instance: Optional["CacheManager"] = None

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._redis = None
        self._memory_cache: Dict[str, tuple] = {}  # key -> (value, expires_at | None)

    @classmethod
    def get_instance(cls) -> "CacheManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self) -> bool:
        """Attempt to connect to Redis; fall back silently to memory cache."""
        try:
            import redis.asyncio as aioredis  # type: ignore

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Redis connected successfully")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self._redis = None
            return False

    async def get(self, key: str) -> Optional[Any]:
        if self._redis:
            try:
                value = await self._redis.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.debug(f"Redis GET error: {e}")
        # Fallback: in-memory
        if key in self._memory_cache:
            value, expires_at = self._memory_cache[key]
            if expires_at is None or _utcnow() < expires_at:
                return value
            del self._memory_cache[key]
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        serialized = json.dumps(value, default=str)
        if self._redis:
            try:
                await self._redis.setex(key, ttl_seconds, serialized)
                return True
            except Exception as e:
                logger.debug(f"Redis SET error: {e}")
        # Fallback: in-memory
        expires_at = _utcnow() + timedelta(seconds=ttl_seconds)
        self._memory_cache[key] = (value, expires_at)
        return True

    async def delete(self, key: str) -> bool:
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception:
                pass
        self._memory_cache.pop(key, None)
        return True

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def get_or_set(self, key: str, factory: Callable, ttl_seconds: int = 300) -> Any:
        """Return a cached value, or compute it via *factory* and cache the result."""
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        await self.set(key, value, ttl_seconds)
        return value

    async def publish(self, channel: str, message: Any) -> None:
        if self._redis:
            try:
                await self._redis.publish(channel, json.dumps(message, default=str))
            except Exception as e:
                logger.debug(f"Redis PUBLISH error: {e}")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
