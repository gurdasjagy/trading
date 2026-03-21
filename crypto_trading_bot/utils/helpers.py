"""General-purpose utility helpers for the trading bot."""

import uuid
from decimal import ROUND_DOWN, Decimal
from typing import Any, List, Tuple


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def round_price(price: float, precision: int = 8) -> float:
    """Round *price* to *precision* decimal places using ROUND_DOWN."""
    quantizer = Decimal(10) ** -precision
    return float(Decimal(str(price)).quantize(quantizer, rounding=ROUND_DOWN))


def round_amount(amount: float, step_size: float) -> float:
    """
    Round *amount* down to the nearest multiple of *step_size*.

    Used to satisfy exchange lot-size filters.
    """
    if step_size <= 0:
        return amount
    d_amount = Decimal(str(amount))
    d_step = Decimal(str(step_size))
    rounded = (d_amount // d_step) * d_step
    return float(rounded)


def calculate_pnl(
    entry: float,
    exit_price: float,
    amount: float,
    side: str,
) -> float:
    """
    Calculate realised P&L in quote currency.

    Args:
        entry:      Entry price.
        exit_price: Exit (close) price.
        amount:     Position size in base currency.
        side:       "buy" (long) or "sell" (short).

    Returns:
        P&L as a float (negative means a loss).
    """
    if side.lower() == "buy":
        return (exit_price - entry) * amount
    elif side.lower() == "sell":
        return (entry - exit_price) * amount
    else:
        raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")


def calculate_pnl_pct(entry: float, exit_price: float, side: str) -> float:
    """
    Calculate P&L as a percentage of the entry price.

    Returns:
        Percentage change (e.g. 2.5 means +2.5%).
    """
    if entry == 0:
        return 0.0
    if side.lower() == "buy":
        return ((exit_price - entry) / entry) * 100
    elif side.lower() == "sell":
        return ((entry - exit_price) / entry) * 100
    else:
        raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")


def format_pnl(pnl: float, pnl_pct: float) -> str:
    """
    Format P&L values into a human-readable string.

    Example:
        format_pnl(123.45, 1.23)  → "$+123.45 (+1.23%)"
        format_pnl(-50.0, -2.5)   → "$-50.00 (-2.50%)"
    """
    sign = "+" if pnl >= 0 else ""
    sign_pct = "+" if pnl_pct >= 0 else ""
    return f"${sign}{pnl:.2f} ({sign_pct}{pnl_pct:.2f}%)"


def chunk_list(lst: list, size: int) -> List[list]:
    """Split *lst* into consecutive sublists of at most *size* elements."""
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten_dict(d: dict, sep: str = ".", _prefix: str = "") -> dict:
    """
    Flatten a nested dict using *sep* as the key separator.

    Example:
        {"a": {"b": 1, "c": 2}} → {"a.b": 1, "a.c": 2}
    """
    result: dict = {}
    for key, value in d.items():
        full_key = f"{_prefix}{sep}{key}" if _prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, sep=sep, _prefix=full_key))
        else:
            result[full_key] = value
    return result


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into a copy of *base*.

    Nested dicts are merged; all other values are overwritten.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def generate_order_id(prefix: str = "BOT") -> str:
    """Generate a unique order ID with the given prefix."""
    unique = uuid.uuid4().hex[:12].upper()
    return f"{prefix}-{unique}"


def truncate_string(s: str, max_len: int = 100) -> str:
    """Truncate *s* to *max_len* characters, appending '…' if needed."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def is_valid_symbol(symbol: str) -> bool:
    """
    Return True if *symbol* matches the "BASE/QUOTE" format.

    Examples:
        "BTC/USDT" → True
        "btcusdt"  → False
        "BTC-USDT" → False
    """
    parts = symbol.split("/")
    if len(parts) != 2:
        return False
    base, quote = parts
    return bool(base) and bool(quote) and base.isalpha() and quote.isalpha()


def symbol_to_pair(symbol: str) -> Tuple[str, str]:
    """
    Split "BASE/QUOTE" into a (base, quote) tuple.

    Raises:
        ValueError: if *symbol* is not in the expected format.
    """
    if not is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {symbol!r}. Expected 'BASE/QUOTE'.")
    base, quote = symbol.split("/")
    return base.upper(), quote.upper()
