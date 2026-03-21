"""Retry logic with exponential back-off for sync and async functions."""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Optional, Tuple, Type

from loguru import logger


# ---------------------------------------------------------------------------
# CCXT rate-limit exceptions — imported lazily so ccxt is not required at the
# module level (keeps the utils layer dependency-free).
# ---------------------------------------------------------------------------

def _ccxt_rate_limit_exceptions() -> Tuple[Type[Exception], ...]:
    """Return ccxt rate-limit exception types if ccxt is installed."""
    try:
        import ccxt  # type: ignore[import]
        return (ccxt.RateLimitExceeded, ccxt.DDoSProtection)
    except (ImportError, AttributeError):
        return ()


async def async_retry(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    *args,
    **kwargs,
) -> Any:
    """
    Call *func* with *args*/*kwargs*, retrying up to *max_retries* times on
    any exception listed in *exceptions*, using exponential back-off.
    """
    last_exc: Optional[Exception] = None
    _rate_limit_excs = _ccxt_rate_limit_exceptions()

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                logger.error(
                    f"async_retry: {func.__name__!r} failed after " f"{max_retries} retries — {exc}"
                )
                raise

            # Use a longer base delay for rate-limit errors
            if _rate_limit_excs and isinstance(exc, _rate_limit_excs):
                effective_base = max(base_delay * 2, 5.0)
                logger.warning(
                    f"async_retry: {func.__name__!r} rate-limited (attempt {attempt + 1}). "
                    f"Backing off harder."
                )
            else:
                effective_base = base_delay

            delay = min(effective_base * (exponential_base**attempt), max_delay)
            if jitter:
                delay *= random.uniform(0.5, 1.5)

            logger.warning(
                f"async_retry: {func.__name__!r} attempt {attempt + 1}/{max_retries} "
                f"failed ({exc}). Retrying in {delay:.2f}s…"
            )
            await asyncio.sleep(delay)

    raise (
        last_exc
        if last_exc is not None
        else RuntimeError(  # pragma: no cover
            f"async_retry: {func.__name__!r} failed for unknown reason."
        )
    )


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator that adds synchronous retry with exponential back-off."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            f"retry: {func.__name__!r} failed after "
                            f"{max_retries} retries — {exc}"
                        )
                        raise

                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    logger.warning(
                        f"retry: {func.__name__!r} attempt {attempt + 1}/{max_retries} "
                        f"failed ({exc}). Retrying in {delay:.2f}s…"
                    )
                    time.sleep(delay)
            raise (
                last_exc
                if last_exc is not None
                else RuntimeError(  # pragma: no cover
                    f"retry: {func.__name__!r} failed for unknown reason."
                )
            )

        return wrapper

    return decorator


def async_retry_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator that adds asynchronous retry with exponential back-off.

    Automatically applies a stronger back-off (2× base_delay, min 5 s) when
    the caught exception is a ``ccxt.RateLimitExceeded`` or
    ``ccxt.DDoSProtection`` error, while still respecting *max_delay*.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exc: Optional[Exception] = None
            _rate_limit_excs = _ccxt_rate_limit_exceptions()
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    # Never retry circuit-breaker exceptions — the circuit is open
                    # or the symbol is permanently unavailable; retrying is pointless.
                    try:
                        from utils.circuit_breaker import CircuitBreakerOpenError  # noqa: PLC0415
                        if isinstance(exc, CircuitBreakerOpenError):
                            raise
                    except ImportError:
                        pass
                    if attempt == max_retries:
                        logger.error(
                            f"async_retry_decorator: {func.__name__!r} failed after "
                            f"{max_retries} retries — {exc}"
                        )
                        raise

                    # Longer back-off for rate-limit errors
                    if _rate_limit_excs and isinstance(exc, _rate_limit_excs):
                        effective_base = max(base_delay * 2, 5.0)
                        logger.warning(
                            f"async_retry_decorator: {func.__name__!r} rate-limited "
                            f"(attempt {attempt + 1}/{max_retries}) — applying 2× back-off."
                        )
                        # Also trigger ExchangeRateLimiter back-off if we can infer exchange
                        try:
                            from utils.rate_limiter import ExchangeRateLimiter  # noqa: PLC0415
                            # Try to identify the exchange from the first arg (self._exchange_id)
                            self_arg = args[0] if args else None
                            exchange_id = (
                                getattr(self_arg, "_exchange_id", None)
                                or getattr(self_arg, "EXCHANGE_NAME", "unknown")
                            )
                            await ExchangeRateLimiter.handle_rate_limit_error(
                                str(exchange_id), exc=exc
                            )
                        except Exception:
                            pass
                    else:
                        effective_base = base_delay

                    delay = min(effective_base * (exponential_base**attempt), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    logger.warning(
                        f"async_retry_decorator: {func.__name__!r} attempt "
                        f"{attempt + 1}/{max_retries} failed ({exc}). "
                        f"Retrying in {delay:.2f}s…"
                    )
                    await asyncio.sleep(delay)
            raise (
                last_exc
                if last_exc is not None
                else RuntimeError(  # pragma: no cover
                    f"async_retry_decorator: {func.__name__!r} failed for unknown reason."
                )
            )

        return wrapper

    return decorator
