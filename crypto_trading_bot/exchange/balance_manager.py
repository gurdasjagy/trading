"""Balance and margin tracking with caching and reservation logic."""

import asyncio
import time
from typing import List, Optional, Tuple

from loguru import logger

from .base_exchange import Balance, BaseExchange


class BalanceManager:
    """Provides cached access to account balance with margin reservation support.

    The manager caches the balance for up to *cache_ttl_seconds* seconds and
    refreshes it automatically on reads that find a stale cache.  A simple
    in-process reservation system allows strategies to "lock" USDT margin
    before an order is placed so that concurrent strategies do not
    over-allocate.
    """

    # Default cache lifetime in seconds
    DEFAULT_CACHE_TTL = 30.0

    def __init__(
        self,
        exchange: BaseExchange,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL,
    ) -> None:
        self._exchange = exchange
        self._cache_ttl = cache_ttl_seconds
        self._balance: Optional[Balance] = None
        self._last_refresh: float = 0.0
        self._reserved_usdt: float = 0.0  # amount locked by pending orders
        self._balance_history: List[dict] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Core balance access
    # ------------------------------------------------------------------

    async def refresh(self) -> Balance:
        """Force-fetch the current balance from the exchange and update cache."""
        async with self._lock:
            balance = await self._exchange.get_balance()
            self._balance = balance
            self._last_refresh = time.monotonic()
            self._balance_history.append(
                {
                    "timestamp": time.time(),
                    "usdt_total": balance.usdt_total,
                    "usdt_free": balance.usdt_free,
                }
            )
            logger.debug(
                "Balance refreshed: total={:.2f} USDT, free={:.2f} USDT",
                balance.usdt_total,
                balance.usdt_free,
            )
            return balance

    async def get_balance(self) -> Balance:
        """Return a (possibly cached) :class:`Balance`, refreshing if stale.

        The staleness check and the refresh are performed inside the same lock
        acquisition to prevent a race condition where multiple concurrent
        callers all detect a stale cache and trigger redundant refreshes.
        """
        async with self._lock:
            age = time.monotonic() - self._last_refresh
            if self._balance is not None and age < self._cache_ttl:
                return self._balance
            # Cache is stale (or empty) — refresh inside the lock so only one
            # coroutine triggers the network call.
            balance = await self._exchange.get_balance()
            self._balance = balance
            self._last_refresh = time.monotonic()
            self._balance_history.append(
                {
                    "timestamp": time.time(),
                    "usdt_total": balance.usdt_total,
                    "usdt_free": balance.usdt_free,
                }
            )
            logger.debug(
                "Balance refreshed: total={:.2f} USDT, free={:.2f} USDT",
                balance.usdt_total,
                balance.usdt_free,
            )
            return balance

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    async def get_available_usdt(self) -> float:
        """Return free USDT minus any in-process reservations."""
        balance = await self.get_balance()
        available = balance.usdt_free - self._reserved_usdt
        return max(available, 0.0)

    async def get_total_usdt(self) -> float:
        """Return the total USDT account value (including positions)."""
        balance = await self.get_balance()
        return balance.usdt_total

    async def get_margin_ratio(self) -> float:
        """Return the fraction of balance currently in use (used / total).

        Returns 0.0 when total is zero.
        """
        balance = await self.get_balance()
        if balance.usdt_total == 0:
            return 0.0
        used = balance.usdt_total - balance.usdt_free
        return used / balance.usdt_total

    # ------------------------------------------------------------------
    # Affordability & reservations
    # ------------------------------------------------------------------

    async def can_afford(self, amount_usdt: float, leverage: int = 1) -> bool:
        """Return *True* if the account can open a position of *amount_usdt* notional.

        With leverage the required margin is ``amount_usdt / leverage``.
        """
        if leverage < 1:
            raise ValueError(f"Leverage must be >= 1, got {leverage}")
        required_margin = amount_usdt / leverage
        available = await self.get_available_usdt()
        result = available >= required_margin
        if not result:
            logger.warning(
                "can_afford: required={:.2f} USDT margin (leverage {}x), available={:.2f}",
                required_margin,
                leverage,
                available,
            )
        return result

    async def reserve_margin(self, amount_usdt: float) -> bool:
        """Reserve *amount_usdt* of free USDT so concurrent orders do not over-allocate.

        Returns *True* if the reservation succeeded (sufficient funds),
        *False* otherwise (reservation not made).
        """
        available = await self.get_available_usdt()
        if available < amount_usdt:
            logger.warning(
                "reserve_margin: cannot reserve {:.2f} USDT — only {:.2f} available",
                amount_usdt,
                available,
            )
            return False
        async with self._lock:
            self._reserved_usdt += amount_usdt
        logger.debug(
            "Reserved {:.2f} USDT (total reserved: {:.2f})", amount_usdt, self._reserved_usdt
        )
        return True

    async def release_margin(self, amount_usdt: float) -> None:
        """Release a previously reserved margin amount.

        The reserved total is floored at 0.0 to guard against double-releases.
        """
        async with self._lock:
            self._reserved_usdt = max(0.0, self._reserved_usdt - amount_usdt)
        logger.debug(
            "Released {:.2f} USDT reservation (total reserved: {:.2f})",
            amount_usdt,
            self._reserved_usdt,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    async def get_balance_history(self) -> list:
        """Return a list of historical balance snapshots captured during refreshes."""
        async with self._lock:
            return list(self._balance_history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_usdt_value(self, balance: Balance) -> Tuple[float, float]:
        """Extract total and free USDT values from a :class:`Balance`.

        Returns ``(usdt_total, usdt_free)``.  Subclasses may override this
        to incorporate real-time mark prices for non-USDT holdings.
        """
        return balance.usdt_total, balance.usdt_free
