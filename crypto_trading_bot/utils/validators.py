"""Input validation utilities for the trading bot."""

import re
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, field_validator, model_validator

_VALID_SIDES = {"buy", "sell"}
_VALID_ORDER_TYPES = {"market", "limit", "stop_limit", "stop_market"}

# Rough length heuristics for API keys per exchange
_API_KEY_MIN_LEN: dict = {
    "binance": 60,
    "okx": 30,
    "coinbase": 20,
    "default": 10,
}


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalise a trading symbol to "BASE/QUOTE" format.

    Raises:
        ValueError: if the symbol is not in the expected format.
    """
    if not isinstance(symbol, str) or "/" not in symbol:
        raise ValueError(
            f"Invalid symbol {symbol!r}. Expected format: 'BASE/QUOTE' (e.g. 'BTC/USDT')."
        )
    parts = symbol.split("/")
    if len(parts) != 2 or not all(p.isalpha() and p for p in parts):
        raise ValueError(f"Invalid symbol {symbol!r}. Both BASE and QUOTE must be alphabetic.")
    return symbol.upper()


def validate_leverage(leverage: int, max_leverage: int = 20) -> int:
    """
    Ensure leverage is within the allowed range [1, max_leverage].

    Raises:
        ValueError: if leverage is out of range.
    """
    if not isinstance(leverage, int) or leverage < 1:
        raise ValueError(f"Leverage must be a positive integer, got {leverage!r}.")
    if leverage > max_leverage:
        raise ValueError(
            f"Leverage {leverage}× exceeds the maximum allowed value of {max_leverage}×."
        )
    return leverage


def validate_amount(amount: float, min_amount: float = 0.0) -> float:
    """
    Ensure *amount* is a positive finite number ≥ *min_amount*.

    Raises:
        ValueError: if validation fails.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError(f"Amount must be a number, got {amount!r}.")
    if amount <= 0:
        raise ValueError(f"Amount must be greater than zero, got {amount}.")
    if amount < min_amount:
        raise ValueError(f"Amount {amount} is below the minimum required {min_amount}.")
    return amount


def validate_price(price: float) -> float:
    """
    Ensure *price* is a positive finite number.

    Raises:
        ValueError: if validation fails.
    """
    try:
        price = float(price)
    except (TypeError, ValueError):
        raise ValueError(f"Price must be a number, got {price!r}.")
    if price <= 0:
        raise ValueError(f"Price must be greater than zero, got {price}.")
    return price


def validate_side(side: str) -> str:
    """
    Validate that *side* is "buy" or "sell".

    Raises:
        ValueError: for any other value.
    """
    normalised = side.lower().strip()
    if normalised not in _VALID_SIDES:
        raise ValueError(f"Invalid side {side!r}. Must be one of {sorted(_VALID_SIDES)}.")
    return normalised


def validate_order_type(order_type: str) -> str:
    """
    Validate that *order_type* is a supported order type.

    Raises:
        ValueError: for unsupported types.
    """
    normalised = order_type.lower().strip()
    if normalised not in _VALID_ORDER_TYPES:
        raise ValueError(
            f"Invalid order type {order_type!r}. " f"Must be one of {sorted(_VALID_ORDER_TYPES)}."
        )
    return normalised


def validate_api_key_format(api_key: str, exchange: str) -> bool:
    """
    Perform a lightweight sanity check on an API key's length and character set.

    Returns:
        True if the key looks plausible, False otherwise.
    """
    if not isinstance(api_key, str) or not api_key:
        return False
    min_len = _API_KEY_MIN_LEN.get(exchange.lower(), _API_KEY_MIN_LEN["default"])
    if len(api_key) < min_len:
        logger.warning(
            f"API key for {exchange!r} looks too short " f"(len={len(api_key)}, min={min_len})."
        )
        return False
    # Keys are typically alphanumeric with hyphens/underscores
    if not re.fullmatch(r"[A-Za-z0-9\-_]+", api_key):
        logger.warning(f"API key for {exchange!r} contains unexpected characters.")
        return False
    return True


def validate_trading_pair(pair: str, supported_pairs: List[str]) -> bool:
    """
    Return True if *pair* is in *supported_pairs* (case-insensitive).
    """
    upper_pair = pair.upper()
    return upper_pair in [p.upper() for p in supported_pairs]


def validate_percentage(
    value: float,
    min_val: float = 0.0,
    max_val: float = 100.0,
) -> float:
    """
    Validate that *value* is a percentage within [min_val, max_val].

    Raises:
        ValueError: if out of range.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Percentage must be a number, got {value!r}.")
    if not (min_val <= value <= max_val):
        raise ValueError(f"Percentage {value} is out of range [{min_val}, {max_val}].")
    return value


class TradeSignalValidator(BaseModel):
    """Pydantic model for validating a trade signal before execution."""

    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    price: Optional[float] = None
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return validate_symbol(v)

    @field_validator("side")
    @classmethod
    def _validate_side(cls, v: str) -> str:
        return validate_side(v)

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, v: float) -> float:
        return validate_amount(v)

    @field_validator("price")
    @classmethod
    def _validate_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return validate_price(v)
        return v

    @field_validator("leverage")
    @classmethod
    def _validate_leverage(cls, v: int) -> int:
        return validate_leverage(v)

    @field_validator("stop_loss")
    @classmethod
    def _validate_stop_loss(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return validate_price(v)
        return v

    @field_validator("take_profit")
    @classmethod
    def _validate_take_profit(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return validate_price(v)
        return v

    @model_validator(mode="after")
    def _validate_stop_take_consistency(self) -> "TradeSignalValidator":
        """Ensure stop_loss and take_profit are on the correct side of the entry."""
        if self.price is None:
            return self
        if self.side == "buy":
            if self.stop_loss is not None and self.stop_loss >= self.price:
                raise ValueError(
                    f"stop_loss ({self.stop_loss}) must be below entry price ({self.price}) for a long."
                )
            if self.take_profit is not None and self.take_profit <= self.price:
                raise ValueError(
                    f"take_profit ({self.take_profit}) must be above entry price ({self.price}) for a long."
                )
        elif self.side == "sell":
            if self.stop_loss is not None and self.stop_loss <= self.price:
                raise ValueError(
                    f"stop_loss ({self.stop_loss}) must be above entry price ({self.price}) for a short."
                )
            if self.take_profit is not None and self.take_profit >= self.price:
                raise ValueError(
                    f"take_profit ({self.take_profit}) must be below entry price ({self.price}) for a short."
                )
        return self
