"""Gate.io TradFi MT5 integration for Forex and Precious Metals trading.

Gate.io's TradFi platform is powered by MetaTrader 5 (MT5) infrastructure, providing:

* Access to Forex pairs (EURUSD, GBPUSD, etc.) and Precious Metals (XAUUSD, XAGUSD)
* Up to 500x leverage for TradFi contracts
* USDx margin unit (pegged 1:1 to USDT)
* Fixed leverage (cannot be adjusted dynamically per-symbol)
* Market hours awareness (Forex markets close on weekends)
* Native MT5 Python integration for Windows deployments
* rpyc bridge support for Linux/Docker deployments

This client extends BaseExchange and maps all operations to MT5 API calls:

* ``connect()`` → ``mt5.initialize()`` with Gate.io TradFi server
* ``get_ticker()`` → ``mt5.symbol_info_tick()``
* ``get_ohlcv()`` → ``mt5.copy_rates_from_pos()``
* ``create_market_order()`` → ``mt5.order_send()`` with TRADE_ACTION_DEAL
* ``get_positions()`` → ``mt5.positions_get()``
* ``get_balance()`` → ``mt5.account_info()`` (returns USDx balance)

**Usage**:

.. code-block:: python

    # Live trading
    client = GateIOTradFiMT5Client(
        login="12345678",
        password="your_password",
        server="GateIO-TradFi",
        testnet=False
    )
    await client.connect()
    ticker = await client.get_ticker("XAUUSD")

Args:
    login: Gate.io TradFi MT5 account login number.
    password: Gate.io TradFi MT5 account password.
    server: Gate.io TradFi MT5 server (e.g., "GateIO-TradFi", "GateIO-TradFi-Demo").
    account_type: Account type ("Standard" or "Pro").
    testnet: When ``True``, connects to demo account.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

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

# Conditionally import MT5 (only available on Windows or via rpyc bridge)
_HAS_MT5 = False
_MT5_IMPORT_ERROR = None
try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError as e:
    mt5 = None
    _MT5_IMPORT_ERROR = e
    # Don't log warning on import - only log when client is actually instantiated


class GateIOTradFiMT5Client(BaseExchange):
    """Gate.io TradFi MT5 client for Forex and Precious Metals trading.

    This client connects to Gate.io's TradFi platform via MetaTrader 5 protocol.

    **Key Features**:
    - Support for Forex pairs and Precious Metals via MT5
    - Up to 500x leverage (fixed at account level)
    - USDx margin (1:1 with USDT)
    - Market hours awareness
    - Session-based trading (London, New York, Tokyo, Sydney)
    - Native async/await support

    **Symbol Format**:
    - Input: XAUUSD, XAU/USD, XAU_USDT → MT5: XAUUSD
    - Input: EURUSD, EUR/USD, EUR_USDT → MT5: EURUSD
    """

    EXCHANGE_NAME = "gateio_tradfi_mt5"

    # ------------------------------------------------------------------
    # TradFi pair configurations
    # ------------------------------------------------------------------

    TRADFI_PAIRS: Dict[str, Dict[str, Any]] = {
        "XAUUSD": {
            "mt5_symbol": "XAUUSD",
            "display_symbol": "XAU/USDT",
            "contract_size": 1.0,      # 1 oz per contract (Gate.io specific)
            "pip_size": 0.01,          # $0.01 for gold
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,          # Gate.io TradFi limits
            "typical_spread_pips": 0.3,
            "max_acceptable_spread_pips": 1.5,
            "max_leverage": 500,       # Gate.io TradFi max
        },
        "XAGUSD": {
            "mt5_symbol": "XAGUSD",
            "display_symbol": "XAG/USDT",
            "contract_size": 1.0,      # 1 oz per contract
            "pip_size": 0.001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.5,
            "max_acceptable_spread_pips": 2.0,
            "max_leverage": 500,
        },
        "EURUSD": {
            "mt5_symbol": "EURUSD",
            "display_symbol": "EUR/USDT",
            "contract_size": 10000,    # Gate.io uses smaller contracts than standard
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.1,
            "max_acceptable_spread_pips": 1.0,
            "max_leverage": 500,
        },
        "GBPUSD": {
            "mt5_symbol": "GBPUSD",
            "display_symbol": "GBP/USDT",
            "contract_size": 10000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.2,
            "max_acceptable_spread_pips": 1.2,
            "max_leverage": 500,
        },
        "USDJPY": {
            "mt5_symbol": "USDJPY",
            "display_symbol": "JPY/USDT",
            "contract_size": 10000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.1,
            "max_acceptable_spread_pips": 1.0,
            "max_leverage": 500,
        },
        "AUDUSD": {
            "mt5_symbol": "AUDUSD",
            "display_symbol": "AUD/USDT",
            "contract_size": 10000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.15,
            "max_acceptable_spread_pips": 1.0,
            "max_leverage": 500,
        },
        "USDCAD": {
            "mt5_symbol": "USDCAD",
            "display_symbol": "CAD/USDT",
            "contract_size": 10000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.15,
            "max_acceptable_spread_pips": 1.0,
            "max_leverage": 500,
        },
        "USDCHF": {
            "mt5_symbol": "USDCHF",
            "display_symbol": "CHF/USDT",
            "contract_size": 10000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.15,
            "max_acceptable_spread_pips": 1.0,
            "max_leverage": 500,
        },
        "NZDUSD": {
            "mt5_symbol": "NZDUSD",
            "display_symbol": "NZD/USDT",
            "contract_size": 10000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.2,
            "max_acceptable_spread_pips": 1.5,
            "max_leverage": 500,
        },
        "GBPJPY": {
            "mt5_symbol": "GBPJPY",
            "display_symbol": "GBPJPY/USDT",
            "contract_size": 10000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.3,
            "max_acceptable_spread_pips": 2.0,
            "max_leverage": 500,
        },
        "EURJPY": {
            "mt5_symbol": "EURJPY",
            "display_symbol": "EURJPY/USDT",
            "contract_size": 10000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.2,
            "max_acceptable_spread_pips": 1.5,
            "max_leverage": 500,
        },
        "EURGBP": {
            "mt5_symbol": "EURGBP",
            "display_symbol": "EURGBP/USDT",
            "contract_size": 10000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 0.15,
            "max_acceptable_spread_pips": 1.0,
            "max_leverage": 500,
        },
    }

    # Trading session time ranges (UTC hours)
    SESSIONS = {
        "sydney": (22, 7),      # 22:00 - 07:00 UTC
        "tokyo": (0, 9),        # 00:00 - 09:00 UTC
        "london": (8, 16),      # 08:00 - 16:00 UTC
        "new_york": (13, 21),   # 13:00 - 21:00 UTC
    }

    def __init__(
        self,
        login: str,
        password: str,
        server: str = "GateIO-TradFi",
        account_type: str = "Standard",
        testnet: bool = False,
    ) -> None:
        """Initialize Gate.io TradFi MT5 client.

        Args:
            login: MT5 account login number
            password: MT5 account password
            server: MT5 server address (e.g., "GateIO-TradFi", "GateIO-TradFi-Demo")
            account_type: Account type ("Standard" or "Pro")
            testnet: If True, use demo credentials/server
        """
        super().__init__(api_key=login, secret_key=password)
        self.login = int(login)
        self.password = password
        self.server = server
        self.account_type = account_type
        self.testnet = testnet

        if not _HAS_MT5:
            raise RuntimeError(
                "MetaTrader5 package is required for Gate.io TradFi integration. "
                "Install with: pip install MetaTrader5 (Windows only) "
                "or set up rpyc bridge for Linux/Docker deployments."
            )

        self._connected = False
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 300  # 5 minutes
        self._max_retries = 5
        self._retry_delay = 2.0  # seconds

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize MT5 connection to Gate.io TradFi server."""
        if self._connected:
            logger.debug("Already connected to Gate.io TradFi MT5")
            return

        logger.info(
            "Connecting to Gate.io TradFi MT5: server={}, login={}, testnet={}",
            self.server, self.login, self.testnet
        )

        # Retry connection with exponential backoff
        for attempt in range(1, self._max_retries + 1):
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                mt5.initialize,
                None,  # path (None = default)
                self.login,
                self.password,
                self.server,
                30000,  # timeout ms
            )

            if success:
                # Verify connection by fetching account info
                account_info = await asyncio.get_event_loop().run_in_executor(
                    None, mt5.account_info
                )
                if account_info:
                    self._connected = True
                    self._last_heartbeat = time.time()
                    logger.info(
                        "Connected to Gate.io TradFi MT5 - Account: {}, Balance: ${:.2f} USDx, Server: {}",
                        account_info.login,
                        account_info.balance,
                        account_info.server
                    )
                    return
                else:
                    error = mt5.last_error()
                    logger.warning(
                        "MT5 initialized but account_info failed: {} (attempt {}/{})",
                        error, attempt, self._max_retries
                    )
            else:
                error = mt5.last_error()
                logger.warning(
                    "Gate.io TradFi MT5 connection failed: {} (attempt {}/{})",
                    error, attempt, self._max_retries
                )

            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying in {:.1f}s...", delay)
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Failed to connect to Gate.io TradFi MT5 after {self._max_retries} attempts. "
            f"Server: {self.server}, Login: {self.login}"
        )

    async def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if not self._connected:
            return

        await asyncio.get_event_loop().run_in_executor(None, mt5.shutdown)
        self._connected = False
        logger.info("Disconnected from Gate.io TradFi MT5")

    async def _ensure_connected(self) -> None:
        """Ensure connection is alive, reconnect if necessary."""
        if not self._connected:
            await self.connect()
            return

        # Check heartbeat timeout
        if time.time() - self._last_heartbeat > self._heartbeat_interval:
            logger.debug("Heartbeat timeout, checking Gate.io TradFi MT5 connection...")
            account_info = await asyncio.get_event_loop().run_in_executor(
                None, mt5.account_info
            )
            if not account_info:
                logger.warning("MT5 connection lost, reconnecting...")
                self._connected = False
                await self.connect()
            else:
                self._last_heartbeat = time.time()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Fetch account balance (USDx)."""
        await self._ensure_connected()

        account_info = await asyncio.get_event_loop().run_in_executor(
            None, mt5.account_info
        )

        if not account_info:
            logger.error("Failed to fetch Gate.io TradFi MT5 account info")
            return Balance(
                total={"USDx": 0.0},
                free={"USDx": 0.0},
                used={"USDx": 0.0},
                usdt_total=0.0,
                usdt_free=0.0,
            )

        # USDx is 1:1 with USDT
        return Balance(
            total={"USDx": account_info.balance, "USDT": account_info.balance},
            free={"USDx": account_info.margin_free, "USDT": account_info.margin_free},
            used={"USDx": account_info.margin, "USDT": account_info.margin},
            usdt_total=account_info.balance,
            usdt_free=account_info.margin_free,
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch latest ticker for symbol."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol)
        if not mt5_symbol:
            raise ValueError(f"Unsupported symbol: {symbol}")

        tick = await asyncio.get_event_loop().run_in_executor(
            None, mt5.symbol_info_tick, mt5_symbol
        )

        if not tick:
            raise RuntimeError(f"Failed to fetch ticker for {mt5_symbol}")

        return Ticker(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            high=tick.last,  # MT5 tick doesn't have 24h high/low
            low=tick.last,
            volume=tick.volume_real,
            timestamp=int(tick.time * 1000),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch order book (not available via MT5)."""
        logger.warning("Order book not available via MT5 for {}", symbol)
        return {"bids": [], "asks": [], "timestamp": int(time.time() * 1000)}

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol)
        if not mt5_symbol:
            raise ValueError(f"Unsupported symbol: {symbol}")

        mt5_timeframe = self._map_timeframe(timeframe)

        rates = await asyncio.get_event_loop().run_in_executor(
            None,
            mt5.copy_rates_from_pos,
            mt5_symbol,
            mt5_timeframe,
            0,  # start from current bar
            limit
        )

        if rates is None or len(rates) == 0:
            logger.warning("No OHLCV data returned for {} {}", mt5_symbol, timeframe)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict = {}
    ) -> Order:
        """Place a market order."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol)
        if not mt5_symbol:
            raise ValueError(f"Unsupported symbol: {symbol}")

        config = self.TRADFI_PAIRS.get(self._normalize_symbol(symbol))
        if not config:
            raise ValueError(f"No configuration for symbol: {symbol}")

        # Convert amount to lot size
        lot_size = amount / config["contract_size"]
        lot_size = self._round_lot_size(lot_size, config)

        # Get current price for market order
        tick = await asyncio.get_event_loop().run_in_executor(
            None, mt5.symbol_info_tick, mt5_symbol
        )
        if not tick:
            raise RuntimeError(f"Failed to fetch price for {mt5_symbol}")

        price = tick.ask if side == OrderSide.BUY else tick.bid

        # Determine order type
        order_type = mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": params.get("slippage_points", 50),
            "magic": params.get("magic", 0),
            "comment": params.get("comment", "GateIOTradFiMT5"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Add SL/TP if provided
        if "sl" in params:
            request["sl"] = params["sl"]
        if "tp" in params:
            request["tp"] = params["tp"]

        # Send order
        result = await asyncio.get_event_loop().run_in_executor(
            None, mt5.order_send, request
        )

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error()
            error_msg = self._get_error_message(result.retcode if result else 0)
            raise RuntimeError(
                f"Order failed: {error_msg} (retcode={result.retcode if result else 'N/A'})"
            )

        logger.info(
            "Market order placed: {} {} {} lots @ {} (order={})",
            side.value, mt5_symbol, lot_size, price, result.order
        )

        return Order(
            id=str(result.order),
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            price=price,
            amount=amount,
            filled=amount,
            remaining=0.0,
            status=OrderStatus.FILLED,
            timestamp=int(time.time() * 1000),
        )

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict = {}
    ) -> Order:
        """Place a limit order."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol)
        if not mt5_symbol:
            raise ValueError(f"Unsupported symbol: {symbol}")

        config = self.TRADFI_PAIRS.get(self._normalize_symbol(symbol))
        if not config:
            raise ValueError(f"No configuration for symbol: {symbol}")

        lot_size = amount / config["contract_size"]
        lot_size = self._round_lot_size(lot_size, config)

        order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": mt5_symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": params.get("slippage_points", 50),
            "magic": params.get("magic", 0),
            "comment": params.get("comment", "GateIOTradFiMT5"),
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if "sl" in params:
            request["sl"] = params["sl"]
        if "tp" in params:
            request["tp"] = params["tp"]

        result = await asyncio.get_event_loop().run_in_executor(
            None, mt5.order_send, request
        )

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = self._get_error_message(result.retcode if result else 0)
            raise RuntimeError(f"Limit order failed: {error_msg}")

        logger.info(
            "Limit order placed: {} {} {} lots @ {} (order={})",
            side.value, mt5_symbol, lot_size, price, result.order
        )

        return Order(
            id=str(result.order),
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            price=price,
            amount=amount,
            filled=0.0,
            remaining=amount,
            status=OrderStatus.OPEN,
            timestamp=int(time.time() * 1000),
        )

    async def cancel_order(self, order_id: str, symbol: str = "") -> bool:
        """Cancel a pending order."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol) if symbol else ""

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
            "comment": "Cancel via GateIOTradFiMT5",
        }

        result = await asyncio.get_event_loop().run_in_executor(
            None, mt5.order_send, request
        )

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Order {} cancelled", order_id)
            return True
        else:
            error_msg = self._get_error_message(result.retcode if result else 0)
            logger.error("Failed to cancel order {}: {}", order_id, error_msg)
            return False

    async def get_open_orders(self, symbol: str = "") -> List[Order]:
        """Get all open orders."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol) if symbol else None

        if mt5_symbol:
            orders = await asyncio.get_event_loop().run_in_executor(
                None, mt5.orders_get, symbol=mt5_symbol
            )
        else:
            orders = await asyncio.get_event_loop().run_in_executor(
                None, mt5.orders_get
            )

        if not orders:
            return []

        return [self._parse_order(order) for order in orders]

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def get_positions(self, symbol: str = "") -> List[Position]:
        """Get all open positions."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_mt5_symbol(symbol) if symbol else None

        if mt5_symbol:
            positions = await asyncio.get_event_loop().run_in_executor(
                None, mt5.positions_get, symbol=mt5_symbol
            )
        else:
            positions = await asyncio.get_event_loop().run_in_executor(
                None, mt5.positions_get
            )

        if not positions:
            return []

        return [self._parse_position(pos) for pos in positions]

    async def close_position(self, symbol: str, params: Dict = {}) -> bool:
        """Close an open position."""
        await self._ensure_connected()

        positions = await self.get_positions(symbol)
        if not positions:
            logger.warning("No open position to close for {}", symbol)
            return False

        position = positions[0]

        # Close by opening opposite order
        opposite_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        try:
            await self.create_market_order(
                symbol=symbol,
                side=opposite_side,
                amount=position.amount,
                params=params
            )
            logger.info("Position closed: {} {}", symbol, position.side.value)
            return True
        except Exception as e:
            logger.error("Failed to close position for {}: {}", symbol, e)
            return False

    # ------------------------------------------------------------------
    # Leverage and margin
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage (warning: Gate.io TradFi leverage is fixed at account level)."""
        logger.warning(
            "Gate.io TradFi leverage is fixed at account level (up to 500x). "
            "Cannot adjust per-symbol leverage for {}. Requested: {}x",
            symbol, leverage
        )
        return True

    async def set_margin_mode(self, symbol: str, margin_type: MarginType) -> bool:
        """Set margin mode (always cross margin in MT5)."""
        logger.info("Gate.io TradFi MT5 uses cross margin mode by default")
        return True

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _resolve_mt5_symbol(self, symbol: str) -> Optional[str]:
        """Convert user symbol to MT5 symbol format.

        Examples:
            XAUUSD -> XAUUSD
            XAU/USD -> XAUUSD
            XAU_USDT -> XAUUSD
            XAU/USDT -> XAUUSD
        """
        # Normalize symbol
        normalized = self._normalize_symbol(symbol)
        config = self.TRADFI_PAIRS.get(normalized)

        if config:
            return config["mt5_symbol"]

        # Try direct lookup
        for key, cfg in self.TRADFI_PAIRS.items():
            if key == symbol.upper() or cfg["mt5_symbol"] == symbol.upper():
                return cfg["mt5_symbol"]

        return None

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to base format (e.g., XAUUSD, EURUSD)."""
        symbol = symbol.upper().replace("/", "").replace("_", "")

        # Handle USDT suffix
        if symbol.endswith("USDT"):
            symbol = symbol[:-4]

        # Handle USD suffix
        if symbol.endswith("USD"):
            # Check if it's already a complete pair
            if symbol in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]:
                return symbol
            # Otherwise, it's a metal: XAUUSD, XAGUSD
            return symbol

        # Try to match against known pairs
        for key in self.TRADFI_PAIRS.keys():
            if key.replace("/", "").replace("_", "") == symbol:
                return key

        return symbol

    def _map_timeframe(self, timeframe: str) -> int:
        """Map timeframe string to MT5 constant."""
        mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
        }
        return mapping.get(timeframe, mt5.TIMEFRAME_H1)

    def _round_lot_size(self, lot_size: float, config: Dict) -> float:
        """Round lot size to valid increment."""
        lot_step = config["lot_step"]
        min_lot = config["min_lot"]
        max_lot = config["max_lot"]

        rounded = round(lot_size / lot_step) * lot_step
        rounded = max(min_lot, min(rounded, max_lot))

        return round(rounded, 2)

    def _get_error_message(self, retcode: int) -> str:
        """Get human-readable error message for MT5 return code."""
        error_messages = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote — price changed",
            mt5.TRADE_RETCODE_REJECT: "Order rejected",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid SL/TP",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid lot size",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Insufficient margin",
            mt5.TRADE_RETCODE_PRICE_OFF: "Price off quotes",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
        }
        return error_messages.get(retcode, f"Unknown error (code={retcode})")

    def _parse_order(self, mt5_order) -> Order:
        """Parse MT5 order to Order object."""
        # Resolve symbol back to user format
        symbol = mt5_order.symbol
        for key, cfg in self.TRADFI_PAIRS.items():
            if cfg["mt5_symbol"] == symbol:
                symbol = cfg["display_symbol"]
                break

        side = OrderSide.BUY if mt5_order.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT) else OrderSide.SELL
        order_type = OrderType.LIMIT if mt5_order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT) else OrderType.MARKET

        return Order(
            id=str(mt5_order.ticket),
            symbol=symbol,
            side=side,
            type=order_type,
            price=mt5_order.price_open,
            amount=mt5_order.volume_initial,
            filled=mt5_order.volume_current,
            remaining=mt5_order.volume_initial - mt5_order.volume_current,
            status=OrderStatus.OPEN,
            timestamp=int(mt5_order.time_setup * 1000),
        )

    def _parse_position(self, mt5_pos) -> Position:
        """Parse MT5 position to Position object."""
        # Resolve symbol
        symbol = mt5_pos.symbol
        for key, cfg in self.TRADFI_PAIRS.items():
            if cfg["mt5_symbol"] == symbol:
                symbol = cfg["display_symbol"]
                config = cfg
                break
        else:
            config = None

        side = PositionSide.LONG if mt5_pos.type == mt5.POSITION_TYPE_BUY else PositionSide.SHORT

        amount = mt5_pos.volume
        if config:
            amount *= config["contract_size"]

        # Calculate ROE
        roe_pct = 0.0
        if amount > 0 and mt5_pos.price_open > 0:
            roe_pct = (mt5_pos.profit / (amount * mt5_pos.price_open)) * 100

        return Position(
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=mt5_pos.price_open,
            current_price=mt5_pos.price_current,
            unrealized_pnl=mt5_pos.profit,
            leverage=1,  # MT5 doesn't expose per-position leverage
            margin=0.0,
            roe_pct=roe_pct,
        )

    # ------------------------------------------------------------------
    # Market info methods
    # ------------------------------------------------------------------

    def get_current_session(self) -> str:
        """Get currently active trading sessions."""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour

        active_sessions = []
        for session, (start, end) in self.SESSIONS.items():
            if start <= end:
                if start <= hour < end:
                    active_sessions.append(session)
            else:  # Crosses midnight (Sydney)
                if hour >= start or hour < end:
                    active_sessions.append(session)

        return ",".join(active_sessions) if active_sessions else "closed"

    async def get_spread(self, symbol: str) -> Dict[str, Any]:
        """Get current spread for symbol."""
        ticker = await self.get_ticker(symbol)

        config = self.TRADFI_PAIRS.get(self._normalize_symbol(symbol))
        pip_size = config["pip_size"] if config else 0.0001

        spread_pips = (ticker.ask - ticker.bid) / pip_size if pip_size > 0 else 0.0

        return {
            "spread_pips": spread_pips,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "typical_spread": config["typical_spread_pips"] if config else 0.0,
        }
