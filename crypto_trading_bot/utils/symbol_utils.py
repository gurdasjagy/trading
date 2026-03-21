"""Symbol normalization utilities.

Provides consistent helpers to convert between plain trading pair symbols
(e.g. ``SOL/USDT``) and the perpetual swap format used by CCXT and Gate.io
(e.g. ``SOL/USDT:USDT``).  Use these functions at all API boundaries to avoid
symbol-mismatch bugs such as conflicting-position detection failures.
"""

from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """Strip the swap suffix and normalize to a plain pair.

    Examples::

        normalize_symbol("SOL/USDT:USDT") -> "SOL/USDT"
        normalize_symbol("BTC/USDT")       -> "BTC/USDT"

    Args:
        symbol: Raw symbol string, possibly with a ``:QUOTE`` suffix.

    Returns:
        Plain ``BASE/QUOTE`` symbol without any colon suffix.
    """
    if ":" in symbol:
        return symbol.split(":")[0]
    return symbol


def to_swap_symbol(symbol: str) -> str:
    """Append the perpetual swap suffix if not already present.

    Examples::

        to_swap_symbol("SOL/USDT")      -> "SOL/USDT:USDT"
        to_swap_symbol("SOL/USDT:USDT") -> "SOL/USDT:USDT"

    Args:
        symbol: Plain ``BASE/QUOTE`` symbol, or one already containing a
            colon suffix.

    Returns:
        ``BASE/QUOTE:QUOTE`` swap symbol.
    """
    if ":" not in symbol and "/" in symbol:
        quote = symbol.split("/")[-1]
        return f"{symbol}:{quote}"
    return symbol
