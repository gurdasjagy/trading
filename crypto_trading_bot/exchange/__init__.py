"""Exchange module — unified interface to multiple crypto exchanges."""

from .balance_manager import BalanceManager
from .base_exchange import (
    Balance,
    BaseExchange,
    MarginType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Ticker,
)


from .ccxt_exchange import SUPPORTED_EXCHANGES, CcxtExchange
from .exchange_factory import create_exchange
from .gateio_client import GateIOClient
from .local_orderbook import LocalOrderBookManager

from .order_manager import OrderManager, OrderTracker
from .paper_exchange import PaperExchange
from .position_manager import PositionManager, PositionTracker
from .websocket_feeds import MarketDataFeed

__all__ = [
    # Abstract base & data models
    "BaseExchange",
    "Balance",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "PositionSide",
    "MarginType",
    "Ticker",
    # Exchange clients
    
    "GateIOClient",
    
    
    "CcxtExchange",
    "PaperExchange",
    "SUPPORTED_EXCHANGES",
    # Factory
    "create_exchange",
    # Managers
    "OrderManager",
    "OrderTracker",
    "PositionManager",
    "PositionTracker",
    "BalanceManager",
    "LocalOrderBookManager",
    # Feed
    "MarketDataFeed",
]
