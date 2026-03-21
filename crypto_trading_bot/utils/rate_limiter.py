"""API rate limiting utilities — token-bucket and per-exchange limiters."""

import asyncio
import time
from typing import Dict, Tuple

from loguru import logger


class RateLimiter:
    """Token-bucket rate limiter for API calls.

    Supports automatic detection and back-off for rate-limit errors
    (``ccxt.RateLimitExceeded``, ``ccxt.DDoSProtection``) via
    :meth:`handle_rate_limit_error`.  Each call to that method doubles the
    effective back-off interval (capped at 60 s) and increments the
    :attr:`rate_limit_hits` counter so the health-check endpoint can expose it.
    """

    def __init__(
        self,
        requests_per_second: float,
        burst_multiplier: float = 2.0,
    ) -> None:
        self.rate = requests_per_second
        self.max_tokens = requests_per_second * burst_multiplier
        self.tokens = self.max_tokens
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Rate-limit error tracking
        self._rate_limit_hits: int = 0
        self._backoff_until: float = 0.0     # monotonic timestamp; 0 = no active backoff
        self._backoff_interval: float = 0.0  # current extra backoff seconds (doubles on each hit)
        self._normal_interval: float = 1.0 / requests_per_second  # base inter-request interval

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rate_limit_hits(self) -> int:
        """Total number of rate-limit errors encountered since creation."""
        return self._rate_limit_hits

    # ------------------------------------------------------------------
    # Core acquire
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time (called while lock is held)."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until *tokens* tokens are available, then consume them.

        Also enforces any active back-off window set by
        :meth:`handle_rate_limit_error`.
        """
        # Honour active back-off period first (outside the lock for efficiency)
        now = time.monotonic()
        if self._backoff_until > now:
            wait = self._backoff_until - now
            logger.debug("Rate limiter: in back-off, waiting {:.2f}s", wait)
            await asyncio.sleep(wait)

        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate

            logger.debug("Rate limiter: waiting {:.3f}s for {} token(s)", wait_time, tokens)
            await asyncio.sleep(wait_time)

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        pass

    # ------------------------------------------------------------------
    # Rate-limit error handler
    # ------------------------------------------------------------------

    async def handle_rate_limit_error(self, exc: Exception | None = None) -> None:
        """Handle a ``ccxt.RateLimitExceeded`` or ``ccxt.DDoSProtection`` error.

        Backs off for ``2 × current_interval`` (doubling on each successive
        hit, capped at 60 s) and increments :attr:`rate_limit_hits`.

        Args:
            exc: The caught exception (logged at WARNING level).
        """
        self._rate_limit_hits += 1

        # Double the backoff on each successive rate-limit hit
        if self._backoff_interval <= 0:
            self._backoff_interval = self._normal_interval * 2
        else:
            self._backoff_interval = min(self._backoff_interval * 2, 60.0)

        self._backoff_until = time.monotonic() + self._backoff_interval

        logger.warning(
            "Rate limit hit #{}: backing off {:.1f}s (exc={!r})",
            self._rate_limit_hits,
            self._backoff_interval,
            exc,
        )
        await asyncio.sleep(self._backoff_interval)

    def reset_backoff(self) -> None:
        """Reset the backoff state after a successful request sequence."""
        self._backoff_interval = 0.0
        self._backoff_until = 0.0

    def get_stats(self) -> dict:
        """Return rate-limiter stats for health-check / metrics endpoints."""
        return {
            "rate_limit_hits": self._rate_limit_hits,
            "backoff_interval": self._backoff_interval,
            "in_backoff": time.monotonic() < self._backoff_until,
        }


class ExchangeRateLimiter:
    """Per-exchange singleton rate limiters with Rust telemetry aggregation.

    Trap 4 Fix: The Python rate limiter now aggregates both local Python API
    usage AND Rust engine's WS/REST throughput reported via the shared memory
    state recovery file. This provides a unified, accurate rate-limit view
    instead of Python being blind to Rust's actual exchange consumption.
    """

    _limiters: Dict[str, RateLimiter] = {}
    # Trap 4: Rust-reported rate-limit usage, keyed by exchange
    _rust_usage: Dict[str, dict] = {}

    @classmethod
    def get_limiter(cls, exchange: str, rps: float = 10.0) -> RateLimiter:
        """Return (and lazily create) the :class:`RateLimiter` for *exchange*."""
        if exchange not in cls._limiters:
            cls._limiters[exchange] = RateLimiter(requests_per_second=rps)
            logger.debug("Created rate limiter for {!r}: {} rps", exchange, rps)
        return cls._limiters[exchange]

    @classmethod
    async def limit(cls, exchange: str, rps: float = 10.0) -> None:
        """Convenience coroutine: acquire one token for *exchange*."""
        limiter = cls.get_limiter(exchange, rps)
        await limiter.acquire()

    @classmethod
    def update_rust_usage(cls, exchange: str, rust_metrics: dict) -> None:
        """Update the rate-limiter with Rust engine's reported API consumption.

        Trap 4 Fix: Called periodically by the cold-path orchestrator or
        health-check loop to ingest Rust's throughput metrics from shared
        memory / recovery file.

        Args:
            exchange: Exchange identifier (e.g. ``"gateio"``).
            rust_metrics: Dict with keys like ``ws_orders_per_sec``,
                ``rest_calls_per_sec``, ``total_orders_sent``.
        """
        cls._rust_usage[exchange] = {
            **rust_metrics,
            "updated_at": time.monotonic(),
        }

        # If Rust is being throttled, pre-emptively back off Python's limiter
        rust_rps = rust_metrics.get("ws_orders_per_sec", 0)
        if rust_rps > 30:  # Approaching typical exchange limits
            limiter = cls.get_limiter(exchange)
            if limiter._backoff_interval <= 0:
                limiter._backoff_interval = 0.5
                limiter._backoff_until = time.monotonic() + 0.5
                logger.warning(
                    "Rust engine high throughput ({} rps) — Python rate limiter backing off for {}",
                    exchange, limiter._backoff_interval,
                )

    @classmethod
    def get_all_stats(cls) -> Dict[str, dict]:
        """Return unified rate-limit stats for all exchanges.

        Trap 4 Fix: Includes both Python-local and Rust-reported usage.
        """
        result = {}
        for exchange, limiter in cls._limiters.items():
            stats = limiter.get_stats()
            # Merge Rust usage if available
            if exchange in cls._rust_usage:
                rust = cls._rust_usage[exchange]
                stats["rust_ws_orders_per_sec"] = rust.get("ws_orders_per_sec", 0)
                stats["rust_rest_calls_per_sec"] = rust.get("rest_calls_per_sec", 0)
                stats["rust_total_orders"] = rust.get("total_orders_sent", 0)
                stats["rust_usage_age_s"] = time.monotonic() - rust.get("updated_at", 0)
                stats["unified_rps"] = (
                    stats.get("rust_ws_orders_per_sec", 0)
                    + (limiter.rate - limiter.tokens)  # Python's consumption rate approximation
                )
            result[exchange] = stats
        return result

    @classmethod
    async def handle_rate_limit_error(
        cls, exchange: str, exc: Exception | None = None, rps: float = 10.0
    ) -> None:
        """Handle a rate-limit error for *exchange*.

        Delegates to the exchange-specific :class:`RateLimiter` instance.
        Creates the limiter if it does not yet exist.

        Args:
            exchange: Exchange identifier (e.g. ``"gateio"``).
            exc: The caught exception (logged at WARNING level).
            rps: RPS used when creating a new limiter (default 10).
        """
        limiter = cls.get_limiter(exchange, rps)
        await limiter.handle_rate_limit_error(exc)
