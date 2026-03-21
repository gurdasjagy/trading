"""Crypto-specific utility functions for the trading bot."""

TRADING_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "NEAR/USDT",
    "ARB/USDT",
]

# Pairs that are considered highly correlated with BTC
_BTC_CORRELATED = {
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "NEAR/USDT",
    "ARB/USDT",
}


def calculate_liquidation_price(
    entry: float,
    leverage: int,
    side: str,
    margin_rate: float = 0.005,
) -> float:
    """
    Estimate the liquidation price for a leveraged position.

    Uses the simplified formula:
        Long:  liq_price = entry * (1 - 1/leverage + margin_rate)
        Short: liq_price = entry * (1 + 1/leverage - margin_rate)

    Args:
        entry:       Entry price.
        leverage:    Leverage multiplier (e.g. 10 for 10×).
        side:        "buy" (long) or "sell" (short).
        margin_rate: Maintenance margin rate (default 0.5%).

    Returns:
        Estimated liquidation price.
    """
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    if side.lower() == "buy":
        return entry * (1 - 1 / leverage + margin_rate)
    elif side.lower() == "sell":
        return entry * (1 + 1 / leverage - margin_rate)
    else:
        raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")


def calculate_margin_required(
    position_size: float,
    price: float,
    leverage: int,
) -> float:
    """
    Calculate the initial margin required to open a position.

    Args:
        position_size: Amount of base currency (e.g. 0.1 BTC).
        price:         Current price of the asset.
        leverage:      Leverage multiplier.

    Returns:
        Required margin in quote currency.
    """
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    notional = position_size * price
    return notional / leverage


def estimate_funding_cost(
    position_size: float,
    price: float,
    funding_rate: float,
    hours: float = 8,
) -> float:
    """
    Estimate the funding cost for holding a perpetual futures position.

    Args:
        position_size: Amount of base currency.
        price:         Current asset price.
        funding_rate:  Funding rate per interval (e.g. 0.0001 = 0.01%).
        hours:         Hours until the next funding payment (default 8).

    Returns:
        Estimated funding cost in quote currency (positive = cost to longs).
    """
    notional = position_size * price
    intervals = hours / 8  # standard 8-hour funding interval
    return notional * funding_rate * intervals


def calculate_kelly_position(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    half_kelly: bool = True,
) -> float:
    """
    Calculate the optimal position size using the Kelly Criterion.

    Args:
        win_rate:  Probability of a winning trade (0–1).
        avg_win:   Average profit per winning trade (in quote currency).
        avg_loss:  Average loss per losing trade (positive value, quote currency).
        capital:   Total available capital.
        half_kelly: If True, use half the Kelly fraction (more conservative).

    Returns:
        Recommended position size in quote currency.
    """
    if avg_loss == 0:
        return 0.0
    loss_rate = 1 - win_rate
    win_loss_ratio = avg_win / avg_loss
    kelly_fraction = win_rate - (loss_rate / win_loss_ratio)
    kelly_fraction = max(0.0, kelly_fraction)
    if half_kelly:
        kelly_fraction /= 2
    return capital * kelly_fraction


def normalize_symbol(symbol: str) -> str:
    """
    Normalise a raw symbol string to the "BASE/QUOTE" format.

    Examples:
        "btcusdt"  → "BTC/USDT"
        "BTC-USDT" → "BTC/USDT"
        "BTC/USDT" → "BTC/USDT"
    """
    symbol = symbol.upper().strip()
    # Already normalised
    if "/" in symbol:
        return symbol
    # Hyphen separator
    if "-" in symbol:
        return symbol.replace("-", "/")
    # Attempt to split common quote currencies
    for quote in ("USDT", "BUSD", "BTC", "ETH", "BNB", "USD"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            return f"{base}/{quote}"
    return symbol  # Return as-is if we cannot determine the split


def get_base_currency(symbol: str) -> str:
    """Return the base currency from "BASE/QUOTE" (e.g. "BTC/USDT" → "BTC")."""
    parts = symbol.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid symbol format: {symbol!r}")
    return parts[0].upper()


def get_quote_currency(symbol: str) -> str:
    """Return the quote currency from "BASE/QUOTE" (e.g. "BTC/USDT" → "USDT")."""
    parts = symbol.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid symbol format: {symbol!r}")
    return parts[1].upper()


def calculate_position_value(amount: float, price: float) -> float:
    """Return the notional value of a position (amount × price)."""
    return amount * price


def calculate_unrealized_pnl(
    entry: float,
    current: float,
    amount: float,
    side: str,
) -> float:
    """
    Calculate the unrealised P&L for an open position.

    Args:
        entry:   Entry price.
        current: Current market price.
        amount:  Position size in base currency.
        side:    "buy" (long) or "sell" (short).

    Returns:
        Unrealised P&L in quote currency.
    """
    if side.lower() == "buy":
        return (current - entry) * amount
    elif side.lower() == "sell":
        return (entry - current) * amount
    else:
        raise ValueError(f"Invalid side: {side!r}. Must be 'buy' or 'sell'.")


def calculate_roe(pnl: float, margin: float) -> float:
    """
    Calculate Return on Equity as a percentage.

    Args:
        pnl:    Realised or unrealised P&L.
        margin: Initial margin used to open the position.

    Returns:
        ROE percentage (e.g. 10.0 means 10%).
    """
    if margin == 0:
        return 0.0
    return (pnl / margin) * 100


def is_correlated(symbol1: str, symbol2: str) -> bool:
    """
    Return True when the two symbols are known to be highly correlated.

    BTC/USDT is considered correlated with most major altcoins.
    Meme coins (DOGE) are treated as uncorrelated with everything.
    """
    s1 = normalize_symbol(symbol1)
    s2 = normalize_symbol(symbol2)
    if s1 == s2:
        return True

    correlated_pairs = {
        frozenset({"BTC/USDT", "ETH/USDT"}),
        frozenset({"BTC/USDT", "BNB/USDT"}),
        frozenset({"BTC/USDT", "SOL/USDT"}),
        frozenset({"BTC/USDT", "AVAX/USDT"}),
        frozenset({"BTC/USDT", "LINK/USDT"}),
        frozenset({"BTC/USDT", "DOT/USDT"}),
        frozenset({"BTC/USDT", "MATIC/USDT"}),
        frozenset({"BTC/USDT", "NEAR/USDT"}),
        frozenset({"BTC/USDT", "ARB/USDT"}),
        frozenset({"ETH/USDT", "BNB/USDT"}),
        frozenset({"ETH/USDT", "SOL/USDT"}),
        frozenset({"ETH/USDT", "AVAX/USDT"}),
        frozenset({"ETH/USDT", "LINK/USDT"}),
        frozenset({"ETH/USDT", "MATIC/USDT"}),
        frozenset({"ETH/USDT", "ARB/USDT"}),
    }
    return frozenset({s1, s2}) in correlated_pairs
