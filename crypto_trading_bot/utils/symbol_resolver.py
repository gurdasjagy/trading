"""Symbol resolution decorator for exchange API boundary safety.

The :func:`resolve_symbols` decorator ensures that any method decorated
with it will:

1. Look up the ``symbol`` parameter by name in the function signature.
2. Apply :meth:`_resolve_precious_metals_symbol` (e.g. XAU/USDT → XAUT/USDT).
3. Apply :meth:`_resolve_swap_symbol` (e.g. XAUT/USDT → XAUT/USDT:USDT).
4. Call the actual function with the **fully resolved** symbol.
5. Store the **original** symbol in :data:`original_symbol_ctx` (a
   :class:`~contextvars.ContextVar`) so that infinite WebSocket loops can
   still access it via ``original_symbol_ctx.get()``.
6. Post-process :class:`~exchange.base_exchange.Ticker` return values to
   restore the original symbol, so the rest of the bot always sees
   ``"XAU/USDT"`` rather than ``"XAUT/USDT:USDT"``.

This removes the need to remember to call ``_resolve_precious_metals_symbol``
and ``_resolve_swap_symbol`` in every single method, eliminating the class of
"forgot to map the symbol" bugs.

Usage
-----
::

    from utils.symbol_resolver import resolve_symbols, original_symbol_ctx

    class MyExchange(BaseExchange):

        @resolve_symbols
        async def get_ticker(self, symbol: str) -> Ticker:
            # symbol is already fully resolved (e.g. "XAUT/USDT:USDT")
            raw = await self._client.fetch_ticker(symbol)
            ...

        @resolve_symbols
        async def subscribe_ticker(self, symbol: str, callback) -> None:
            # symbol is resolved; use ContextVar for the original
            _ctx = original_symbol_ctx.get()
            original = _ctx if _ctx is not None else symbol
            while True:
                raw = await self._ws_client.watch_ticker(symbol)
                ticker = Ticker(symbol=original, ...)
                await callback(ticker)
"""

from __future__ import annotations

import functools
import inspect
from contextvars import ContextVar
from typing import Any, Callable, Optional

# ContextVar that holds the *original* (pre-resolution) symbol for the
# duration of the current async call.  Infinite WebSocket loops can read
# it via ``original_symbol_ctx.get()`` inside their body.
# None means no resolution is currently active.
original_symbol_ctx: ContextVar[Optional[str]] = ContextVar("original_symbol", default=None)


def resolve_symbols(func: Callable) -> Callable:
    """Decorator: auto-resolve precious-metals + swap symbol at API boundary.

    Finds the ``symbol`` parameter in *func*'s signature (by name, so it works
    regardless of position — e.g. ``cancel_order(self, order_id, symbol)``),
    applies both resolution steps, and passes the fully resolved string to the
    actual method.

    The original symbol is:
    * Stored in :data:`original_symbol_ctx` for the lifetime of the call.
    * Automatically restored in :class:`~exchange.base_exchange.Ticker`
      return values, so callers always see ``"XAU/USDT"`` not
      ``"XAUT/USDT:USDT"``.
    """
    sig = inspect.signature(func)
    param_names = [p for p in sig.parameters if p != "self"]
    symbol_pos: Optional[int] = next(
        (i for i, p in enumerate(param_names) if p == "symbol"), None
    )

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # --- Locate the symbol argument -----------------------------------
        if "symbol" in kwargs:
            original = str(kwargs["symbol"])
        elif symbol_pos is not None and symbol_pos < len(args):
            original = str(args[symbol_pos])
        else:
            # No symbol argument found — call unchanged
            return await func(self, *args, **kwargs)

        # --- Apply both resolution steps ---------------------------------
        mapped = self._resolve_precious_metals_symbol(original)
        resolved = self._resolve_swap_symbol(mapped)

        # --- Replace the symbol in the call ------------------------------
        if "symbol" in kwargs:
            kwargs = {**kwargs, "symbol": resolved}
        else:
            args_list = list(args)
            args_list[symbol_pos] = resolved  # type: ignore[index]
            args = tuple(args_list)

        # --- Store original in ContextVar for infinite WS loops ----------
        token = original_symbol_ctx.set(original)
        try:
            result = await func(self, *args, **kwargs)
        finally:
            original_symbol_ctx.reset(token)

        # --- Post-process: restore original symbol in Ticker responses ---
        try:
            from exchange.base_exchange import Ticker  # deferred — avoids circular import

            if isinstance(result, Ticker) and result.symbol != original:
                return result.model_copy(update={"symbol": original})
        except ImportError:
            pass

        return result

    return wrapper
