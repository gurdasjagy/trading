"""Circuit Breaker pattern for per-symbol trading suspension.

After *failure_threshold* consecutive failures for a symbol, the circuit
opens and all further requests for that symbol are immediately rejected with
:class:`CircuitBreakerOpenError`, preventing infinite retry loops.  After
*recovery_timeout* seconds the circuit enters HALF_OPEN state and allows one
probe request; if it succeeds the circuit closes again.

Usage
-----
Apply :func:`with_circuit_breaker` as the **outermost** decorator (above
``@async_retry_decorator``) so it counts method-level failures rather than
individual retry attempts::

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3)
    async def get_ticker(self, symbol: str) -> Ticker: ...
"""

from __future__ import annotations

import functools
import inspect
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

from loguru import logger


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "CLOSED"       # Normal — requests pass through
    OPEN = "OPEN"           # Tripped — requests blocked immediately
    HALF_OPEN = "HALF_OPEN" # Recovery probe — one request allowed


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open for a symbol."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(
            f"Circuit breaker OPEN for {symbol!r} — trading suspended for this pair"
        )


class SymbolPermanentlyUnavailableError(CircuitBreakerOpenError):
    """Raised when a symbol is permanently unavailable on the exchange.

    This is a subclass of :class:`CircuitBreakerOpenError` so that:
    * The circuit breaker wrapper re-raises it WITHOUT recording a failure.
    * The retry decorator (once updated) skips retries immediately.

    Used when a precious-metals or other symbol is confirmed absent from the
    exchange's markets during :meth:`GateIOClient.connect` — the bot will
    never attempt that symbol again for the lifetime of the session.
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        # Override the message to be more descriptive for permanently unavailable symbols
        self.args = (
            f"Symbol {symbol!r} is permanently unavailable on this exchange — "
            "gold/precious-metals futures trading disabled for this session",
        )


class CircuitBreaker:
    """Per-symbol circuit breaker that suspends trading after repeated failures.

    Args:
        symbol: The trading symbol this breaker guards (e.g. ``"XAU/USDT"``).
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait in OPEN state before trying a probe.
    """

    def __init__(
        self,
        symbol: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
    ) -> None:
        self.symbol = symbol
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._opened_at: Optional[float] = None
        self._last_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow_request(self) -> bool:
        """Return *True* if a request may proceed, *False* to block it."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0.0)
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    "[CircuitBreaker] {} — entering HALF_OPEN after {:.0f}s (probe request allowed)",
                    self.symbol,
                    elapsed,
                )
                return True
            return False

        # HALF_OPEN: allow exactly one probe
        return True

    def record_success(self) -> None:
        """Mark the last request as successful — resets failure count."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(
                "[CircuitBreaker] {} — probe succeeded, circuit CLOSED",
                self.symbol,
            )
        self._consecutive_failures = 0
        self._last_error = None
        self.state = CircuitState.CLOSED

    def record_failure(self, exc: Optional[Exception] = None) -> None:
        """Mark the last request as failed — may open the circuit."""
        self._consecutive_failures += 1
        self._last_error = exc

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — reopen
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "[CircuitBreaker] {} — probe failed, circuit re-OPENED. Error: {}",
                self.symbol,
                exc,
            )
            self._send_alert(exc)
            return

        if (
            self._consecutive_failures >= self.failure_threshold
            and self.state == CircuitState.CLOSED
        ):
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "[CircuitBreaker] {} — circuit OPENED after {} consecutive failures. "
                "Trading suspended for this pair. Last error: {}",
                self.symbol,
                self._consecutive_failures,
                exc,
            )
            self._send_alert(exc)

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state (testing / admin override)."""
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._last_error = None
        logger.info("[CircuitBreaker] {} — manually reset to CLOSED", self.symbol)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_alert(self, exc: Optional[Exception]) -> None:
        """Emit a CRITICAL log alert.  Override this method to send Slack/email."""
        logger.critical(
            "[CircuitBreaker] ALERT ⚠️  Trading for {symbol!r} has been SUSPENDED "
            "after {n} consecutive failures.  The bot will continue trading other pairs.  "
            "Last error: {err}  (Auto-retry in {timeout:.0f}s)",
            symbol=self.symbol,
            n=self._consecutive_failures,
            err=exc,
            timeout=self.recovery_timeout,
        )


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_registry: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    symbol: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 300.0,
) -> CircuitBreaker:
    """Return (or lazily create) the :class:`CircuitBreaker` for *symbol*."""
    if symbol not in _registry:
        _registry[symbol] = CircuitBreaker(
            symbol,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _registry[symbol]


def reset_circuit_breaker(symbol: str) -> None:
    """Remove the circuit breaker for *symbol* from the registry.

    The next :func:`get_circuit_breaker` call for *symbol* will create a fresh
    breaker with whatever parameters are passed at that point.  Useful in tests
    and for manual admin overrides.
    """
    _registry.pop(symbol, None)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 300.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    symbol_arg: str = "symbol",
) -> Callable:
    """Decorator that wraps an async method with a per-symbol circuit breaker.

    Apply **above** ``@async_retry_decorator`` so that it counts method-level
    failures (after all internal retries are exhausted) rather than individual
    retry attempts::

        @with_circuit_breaker(failure_threshold=5)
        @async_retry_decorator(max_retries=3)
        async def get_ticker(self, symbol: str) -> Ticker: ...

    When the circuit is OPEN for a symbol, :class:`CircuitBreakerOpenError` is
    raised immediately without calling the underlying function, so the bot can
    continue trading other pairs unaffected.

    Args:
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout:  Seconds in OPEN state before a probe is allowed.
        exceptions:        Exception types that count as failures.
        symbol_arg:        Name of the parameter that carries the trading symbol.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        param_names = [p for p in sig.parameters if p != "self"]
        symbol_pos: Optional[int] = next(
            (i for i, p in enumerate(param_names) if p == symbol_arg), None
        )

        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Resolve symbol from positional or keyword arguments
            symbol: str = ""
            if symbol_arg in kwargs:
                symbol = str(kwargs[symbol_arg])
            elif symbol_pos is not None and symbol_pos < len(args):
                symbol = str(args[symbol_pos])

            cb: Optional[CircuitBreaker] = None
            if symbol:
                cb = get_circuit_breaker(
                    symbol,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
                if not cb.allow_request():
                    raise CircuitBreakerOpenError(symbol)

            try:
                result = await func(self, *args, **kwargs)
                if cb is not None:
                    cb.record_success()
                return result
            except CircuitBreakerOpenError:
                # Re-raise without recording another failure
                raise
            except exceptions as exc:
                if cb is not None:
                    cb.record_failure(exc)
                raise

        return wrapper

    return decorator
