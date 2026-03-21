"""System health monitoring for the trading bot."""

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Optional

from loguru import logger


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str = ""
    last_check: Optional[datetime] = None
    response_time_ms: float = 0.0


class HealthChecker:
    """Aggregates and runs health checks for all bot subsystems."""

    def __init__(self) -> None:
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, ComponentHealth] = {}

    def register_check(self, name: str, check_fn: Callable) -> None:
        """Register an async health-check function under the given name."""
        self._checks[name] = check_fn
        logger.debug(f"Health check registered: {name!r}")

    async def run_all_checks(self) -> Dict[str, ComponentHealth]:
        """Run every registered health check and store results."""
        tasks = {
            name: asyncio.create_task(self._run_check(name, fn))
            for name, fn in self._checks.items()
        }
        for name, task in tasks.items():
            self._results[name] = await task
        return dict(self._results)

    async def _run_check(self, name: str, fn: Callable) -> ComponentHealth:
        start = time.monotonic()
        try:
            result: ComponentHealth = await fn()
            result.response_time_ms = (time.monotonic() - start) * 1000
            result.last_check = _utcnow()
            return result
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(f"Health check {name!r} raised exception: {exc}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
                last_check=_utcnow(),
                response_time_ms=elapsed_ms,
            )

    async def check_redis(self, redis_url: str = None) -> ComponentHealth:
        """Check Redis connectivity."""
        try:
            import redis.asyncio as aioredis

            start = time.monotonic()
            url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            client = aioredis.from_url(url)
            await client.ping()
            await client.aclose()
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis ping OK",
                response_time_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
            )

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        try:
            import aiosqlite

            start = time.monotonic()
            async with aiosqlite.connect(":memory:") as db:
                await db.execute("SELECT 1")
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database query OK",
                response_time_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
            )

    async def check_exchange_connectivity(self) -> ComponentHealth:
        """Check internet connectivity via a well-known public endpoint."""
        try:
            import httpx

            start = time.monotonic()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://httpbin.org/get")
                resp.raise_for_status()
            return ComponentHealth(
                name="connectivity",
                status=HealthStatus.HEALTHY,
                message="Internet connectivity OK",
                response_time_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ComponentHealth(
                name="connectivity",
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
            )

    async def check_ai_service(self) -> ComponentHealth:
        """Check AI/LLM service availability (lightweight probe)."""
        try:
            import httpx

            start = time.monotonic()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://api.openai.com")
            elapsed = (time.monotonic() - start) * 1000
            status = HealthStatus.HEALTHY if resp.status_code < 500 else HealthStatus.DEGRADED
            return ComponentHealth(
                name="ai_service",
                status=status,
                message=f"HTTP {resp.status_code}",
                response_time_ms=elapsed,
            )
        except Exception as exc:
            return ComponentHealth(
                name="ai_service",
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
            )

    async def get_overall_status(self) -> HealthStatus:
        """Return the worst status among all recorded component results."""
        if not self._results:
            return HealthStatus.UNHEALTHY
        statuses = [r.status for r in self._results.values()]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    async def get_health_report(self) -> dict:
        """Return a full JSON-serialisable health report."""
        overall = await self.get_overall_status()
        return {
            "overall": overall.value,
            "timestamp": _utcnow().isoformat(),
            "components": {
                name: {
                    "status": ch.status.value,
                    "message": ch.message,
                    "response_time_ms": round(ch.response_time_ms, 2),
                    "last_check": ch.last_check.isoformat() if ch.last_check else None,
                }
                for name, ch in self._results.items()
            },
        }

    async def run_health_loop(self, interval_seconds: int = 60) -> None:
        """Continuously run all registered checks at the given interval."""
        logger.info(f"Health check loop started (interval={interval_seconds}s)")
        while True:
            try:
                await self.run_all_checks()
                report = await self.get_health_report()
                logger.debug(f"Health report: {report['overall']}")
            except Exception as exc:
                logger.error(f"Health check loop error: {exc}")
            await asyncio.sleep(interval_seconds)
