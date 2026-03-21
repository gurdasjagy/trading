"""Abstract base class defining the common interface for all exchange clients."""

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class MarginType(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


class Order(BaseModel):
    """Normalized order representation."""

    id: str
    symbol: str
    type: OrderType
    side: OrderSide
    amount: float
    price: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    status: OrderStatus
    timestamp: int  # milliseconds since epoch
    fee: float = 0.0
    info: Dict[str, Any] = {}


class Position(BaseModel):
    """Normalized open position representation."""

    symbol: str
    side: PositionSide
    amount: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    margin: float = 0.0
    liquidation_price: float = 0.0
    timestamp: int  # milliseconds since epoch
    # Exchange-grade extended fields
    mark_price: float = 0.0
    margin_ratio: float = 0.0
    roe_pct: float = 0.0
    position_value: float = 0.0
    funding_rate: Optional[float] = None


class Balance(BaseModel):
    """Normalized account balance."""

    total: Dict[str, float] = {}  # currency -> total amount
    free: Dict[str, float] = {}  # currency -> available amount
    used: Dict[str, float] = {}  # currency -> locked amount
    usdt_total: float = 0.0
    usdt_free: float = 0.0


class Ticker(BaseModel):
    """Normalized ticker snapshot."""

    symbol: str
    bid: float
    ask: float
    last: float
    high: float = 0.0
    low: float = 0.0
    volume: float
    timestamp: int  # milliseconds since epoch
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None


class BaseExchange(ABC):
    """Abstract base class for all exchange implementations.

    All concrete exchange clients must implement every abstract method.
    Concrete helper methods such as :meth:`get_mid_price` and
    :meth:`get_multiple_tickers` are provided here.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str = "",
        testnet: bool = False,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.testnet = testnet
        self._client: Any = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Initialise the underlying exchange client and load markets."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close all open connections gracefully."""

    # ------------------------------------------------------------------
    # Market data — REST
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_balance(self) -> Balance:
        """Return the current account balance."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Return the latest ticker for *symbol*."""

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Return the order-book for *symbol* with *limit* levels per side."""

    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Return OHLCV candles as a :class:`pandas.DataFrame`."""

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a market order and return the resulting :class:`Order`."""

    @abstractmethod
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a limit order at *price* and return the resulting :class:`Order`."""

    @abstractmethod
    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a stop-loss order triggered at *stop_price*."""

    @abstractmethod
    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a take-profit order triggered at *tp_price*."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a single order by *order_id*."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for *symbol*."""

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch the current state of a single order."""

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all open orders, optionally filtered by *symbol*."""

    # ------------------------------------------------------------------
    # Position & leverage management
    # ------------------------------------------------------------------

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set the leverage multiplier for *symbol*."""

    @abstractmethod
    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Switch between cross and isolated margin for *symbol*."""

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Return all currently open positions."""

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for *symbol*, or *None* if flat."""

    @abstractmethod
    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close (or partially close) the open position for *symbol*."""

    # ------------------------------------------------------------------
    # Derivatives-specific data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Return the current funding rate for *symbol*."""

    @abstractmethod
    async def get_open_interest(self, symbol: str) -> float:
        """Return the current open interest for *symbol*."""

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time ticker updates for *symbol*."""

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time order-book updates for *symbol*."""

    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time public trade feed for *symbol*."""

    @abstractmethod
    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to private user-data events (fills, balance changes, etc.)."""

    async def watch_order_book(self, symbol: str) -> Dict[str, Any]:
        """Return a single order-book snapshot via WebSocket.

        Subclasses backed by ``ccxt.pro`` should override this to call
        ``watch_order_book`` on the underlying WebSocket client.  The
        default implementation raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support watch_order_book"
        )

    async def close(self) -> None:
        """Reset all underlying connections (REST + WebSocket).

        Calls :meth:`disconnect` by default.  Exchange implementations that
        maintain a separate WebSocket client should override this to close
        that client as well, forcing a fresh reconnect on the next call.
        """
        await self.disconnect()

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    async def get_mid_price(self, symbol: str) -> float:
        """Return the mid-price ``(bid + ask) / 2`` for *symbol*."""
        ticker = await self.get_ticker(symbol)
        return (ticker.bid + ticker.ask) / 2.0

    async def get_multiple_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Return a mapping of symbol → :class:`Ticker` for every *symbol*.

        Requests are issued concurrently via :func:`asyncio.gather`.
        """
        results = await asyncio.gather(
            *[self.get_ticker(s) for s in symbols], return_exceptions=True
        )
        tickers: Dict[str, Ticker] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch ticker for {symbol}: {result}")
            else:
                tickers[symbol] = result
        return tickers

    def format_symbol(self, symbol: str) -> str:
        """Return *symbol* in the exchange-native format.

        Subclasses may override this for exchange-specific quoting conventions
        (e.g. ``BTC/USDT:USDT`` for perpetual swaps).  The default
        implementation returns the symbol unchanged.
        """
        return symbol

    @property
    def name(self) -> str:
        """Human-readable exchange name.  Subclasses should override."""
        return self.__class__.__name__
