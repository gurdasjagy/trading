"""Custom exception hierarchy for the crypto trading bot."""

from typing import Optional


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.__class__.__name__}: {self.message} [{detail_str}]"
        return f"{self.__class__.__name__}: {self.message}"


class ExchangeError(TradingBotError):
    """Raised when an exchange operation fails."""


class InsufficientFundsError(ExchangeError):
    """Raised when account has insufficient funds for an operation."""

    def __init__(
        self,
        message: str = "Insufficient funds",
        required: Optional[float] = None,
        available: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if required is not None:
            d["required"] = required
        if available is not None:
            d["available"] = available
        super().__init__(message, d)


class OrderError(ExchangeError):
    """Raised when order placement or management fails."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if order_id is not None:
            d["order_id"] = order_id
        super().__init__(message, d)


class ConnectionError(TradingBotError):
    """Raised when a network or connection issue occurs."""

    def __init__(
        self,
        message: str = "Connection failed",
        host: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if host is not None:
            d["host"] = host
        super().__init__(message, d)


class RiskLimitError(TradingBotError):
    """Raised when a trade or action violates risk management rules."""

    def __init__(
        self,
        message: str,
        rule: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if rule is not None:
            d["rule"] = rule
        super().__init__(message, d)


class CircuitBreakerError(TradingBotError):
    """Raised when the circuit breaker is active and trading is halted."""

    def __init__(
        self,
        message: str = "Circuit breaker is active — trading halted",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message, details)


class DataSourceError(TradingBotError):
    """Raised when a data source fails to return valid data."""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if source is not None:
            d["source"] = source
        super().__init__(message, d)


class AIError(TradingBotError):
    """Raised when an AI/LLM operation fails."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if model is not None:
            d["model"] = model
        super().__init__(message, d)


class ConfigurationError(TradingBotError):
    """Raised when configuration is missing or invalid."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if field is not None:
            d["field"] = field
        super().__init__(message, d)


class StrategyError(TradingBotError):
    """Raised when a trading strategy encounters an error."""

    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if strategy is not None:
            d["strategy"] = strategy
        super().__init__(message, d)


class BacktestError(TradingBotError):
    """Raised when a backtesting operation fails."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if symbol is not None:
            d["symbol"] = symbol
        super().__init__(message, d)


class AuthenticationError(ExchangeError):
    """Raised when API authentication fails."""

    def __init__(
        self,
        message: str = "API authentication failed",
        exchange: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if exchange is not None:
            d["exchange"] = exchange
        super().__init__(message, d)


class RateLimitError(ExchangeError):
    """Raised when the exchange rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if retry_after is not None:
            d["retry_after"] = retry_after
        super().__init__(message, d)


class ExchangeNetworkError(ExchangeError):
    """Transient network error — retry with exponential backoff."""


class ExchangeAuthError(ExchangeError):
    """Authentication/permission error — stop trading and alert operator."""

    def __init__(
        self,
        message: str = "Exchange authentication failed",
        exchange: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if exchange is not None:
            d["exchange"] = exchange
        super().__init__(message, d)


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance to place the order — skip trade and alert."""

    def __init__(
        self,
        message: str = "Insufficient balance",
        required: Optional[float] = None,
        available: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if required is not None:
            d["required"] = required
        if available is not None:
            d["available"] = available
        super().__init__(message, d)


class ExchangeMaintenanceError(ExchangeError):
    """Exchange is under maintenance — pause for 5 minutes then retry."""


class OrderRejectedError(ExchangeError):
    """Order was rejected by the exchange (invalid params, limits, etc.)."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        d = details or {}
        if order_id is not None:
            d["order_id"] = order_id
        super().__init__(message, d)


def classify_exchange_error(exc: Exception) -> str:
    """Classify an exception into a human-readable error category string.

    Returns one of: "network", "auth", "insufficient_balance",
    "maintenance", "order_rejected", "rate_limit", "unknown".
    """
    try:
        import ccxt
        if isinstance(exc, (ccxt.NetworkError, ccxt.RequestTimeout)):
            return "network"
        if isinstance(exc, ccxt.AuthenticationError):
            return "auth"
        if isinstance(exc, ccxt.InsufficientFunds):
            return "insufficient_balance"
        if isinstance(exc, ccxt.ExchangeNotAvailable):
            return "maintenance"
        if isinstance(exc, ccxt.InvalidOrder):
            return "order_rejected"
        if isinstance(exc, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)):
            return "rate_limit"
    except ImportError:
        pass

    if isinstance(exc, ExchangeNetworkError):
        return "network"
    if isinstance(exc, ExchangeAuthError):
        return "auth"
    if isinstance(exc, InsufficientBalanceError):
        return "insufficient_balance"
    if isinstance(exc, ExchangeMaintenanceError):
        return "maintenance"
    if isinstance(exc, OrderRejectedError):
        return "order_rejected"
    if isinstance(exc, RateLimitError):
        return "rate_limit"

    exc_msg = str(exc).lower()
    if "network" in exc_msg or "timeout" in exc_msg or "connection" in exc_msg:
        return "network"
    if "auth" in exc_msg or "key" in exc_msg or "permission" in exc_msg:
        return "auth"
    if "insufficient" in exc_msg or "balance" in exc_msg:
        return "insufficient_balance"
    if "maintenance" in exc_msg or "not available" in exc_msg:
        return "maintenance"
    if "rate limit" in exc_msg or "too many" in exc_msg:
        return "rate_limit"
    if "invalid order" in exc_msg or "rejected" in exc_msg:
        return "order_rejected"

    return "unknown"
