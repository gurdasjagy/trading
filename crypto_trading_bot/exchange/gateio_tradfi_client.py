"""Gate.io TradFi API client for Forex and Precious Metals trading.

Gate.io launched their dedicated "TradFi Trading API" and "Precious Metals Zone"
in early 2026, supporting USDT-margined perpetual contracts for traditional Forex,
Gold, Silver, and Indices with up to 500x leverage.

This client implements direct REST API calls to the TradFi endpoints, as CCXT
does not yet fully support the GATEIOTRADFI namespace.

Supported symbols:
- Precious Metals: XAU_USDT (Gold), XAG_USDT (Silver)
- Forex: EUR_USDT, GBP_USDT, JPY_USDT, AUD_USDT, CAD_USDT, CHF_USDT, etc.
- Indices: SPX_USDT, DXY_USDT, etc.

API Base URL: https://api.gateio.ws/api/v4/
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
import pandas as pd
from loguru import logger

from utils.rate_limiter import ExchangeRateLimiter
from utils.retry import async_retry_decorator

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


class GateIOTradFiClient(BaseExchange):
    """Gate.io TradFi API client for Forex and Precious Metals.

    This client uses direct aiohttp REST requests and WebSocket connections
    to interact with Gate.io's TradFi trading endpoints, which are separate
    from their standard futures API.

    Args:
        api_key: Gate.io API key
        secret_key: Gate.io secret key
        passphrase: Not used for Gate.io (kept for interface compatibility)
        testnet: Whether to use testnet (if available)
    """

    EXCHANGE_NAME = "gateio_tradfi"
    BASE_URL = "https://api.gateio.ws/api/v4"
    WS_URL = "wss://api.gateio.ws/ws/v4/"

    # Forex symbol mapping: MT5/Traditional → Gate.io TradFi
    FOREX_SYMBOL_MAPPING = {
        "EURUSD": "EUR_USDT",
        "GBPUSD": "GBP_USDT",
        "USDJPY": "JPY_USDT",
        "AUDUSD": "AUD_USDT",
        "USDCAD": "CAD_USDT",
        "USDCHF": "CHF_USDT",
        "NZDUSD": "NZD_USDT",
        "EURGBP": "EURGBP_USDT",
        "EURJPY": "EURJPY_USDT",
        "GBPJPY": "GBPJPY_USDT",
        # Precious metals
        "XAUUSD": "XAU_USDT",
        "XAU/USD": "XAU_USDT",
        "XAU/USDT": "XAU_USDT",
        "XAGUSD": "XAG_USDT",
        "XAG/USD": "XAG_USDT",
        "XAG/USDT": "XAG_USDT",
    }

    # Reverse mapping for display
    TRADFI_TO_DISPLAY = {v: k for k, v in FOREX_SYMBOL_MAPPING.items()}

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str = "",
        testnet: bool = False,
    ) -> None:
        super().__init__(api_key, secret_key, passphrase, testnet)
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize aiohttp session and verify connectivity."""
        self._session = aiohttp.ClientSession()
        # Test connectivity
        try:
            await self._public_request("GET", "/futures/usdt/contracts")
            logger.info("GateIOTradFiClient connected (testnet={})", self.testnet)
        except Exception as e:
            logger.error("Failed to connect to Gate.io TradFi API: {}", e)
            raise

    async def disconnect(self) -> None:
        """Close aiohttp session and WebSocket connections."""
        if self._session:
            await self._session.close()
            self._session = None
        for ws in self._ws_connections.values():
            await ws.close()
        self._ws_connections.clear()
        logger.info("GateIOTradFiClient disconnected")

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def _resolve_tradfi_symbol(self, symbol: str) -> str:
        """Convert user-friendly symbol to Gate.io TradFi format.

        Examples:
            "XAUUSD" → "XAU_USDT"
            "XAU/USDT" → "XAU_USDT"
            "EURUSD" → "EUR_USDT"

        Args:
            symbol: User-friendly symbol

        Returns:
            Gate.io TradFi format symbol (e.g., "XAU_USDT")
        """
        # Check direct mapping first
        if symbol in self.FOREX_SYMBOL_MAPPING:
            return self.FOREX_SYMBOL_MAPPING[symbol]

        # Check if already in TradFi format (contains underscore)
        if "_" in symbol:
            return symbol

        # Try to convert slash format to underscore
        if "/" in symbol:
            base, quote = symbol.split("/")
            # For TradFi, everything is margined in USDT
            if quote == "USDT":
                return f"{base}_USDT"
            elif quote == "USD":
                # Traditional forex: USD quote → USDT margin
                return f"{base}_USDT"

        logger.warning("Could not resolve TradFi symbol for: {}", symbol)
        return symbol

    def _resolve_display_symbol(self, tradfi_symbol: str) -> str:
        """Convert Gate.io TradFi symbol to display format.

        Examples:
            "XAU_USDT" → "XAU/USDT"
            "EUR_USDT" → "EURUSD"

        Args:
            tradfi_symbol: Gate.io TradFi format (e.g., "XAU_USDT")

        Returns:
            User-friendly display format
        """
        # Check if we have a specific display mapping
        if tradfi_symbol in self.TRADFI_TO_DISPLAY:
            return self.TRADFI_TO_DISPLAY[tradfi_symbol]

        # Default: convert underscore to slash
        if "_" in tradfi_symbol:
            return tradfi_symbol.replace("_", "/")

        return tradfi_symbol

    # ------------------------------------------------------------------
    # HTTP request helpers
    # ------------------------------------------------------------------

    def _generate_signature(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body_string: str = "",
    ) -> Dict[str, str]:
        """Generate Gate.io API signature headers.

        Gate.io uses HMAC-SHA512 for authentication.  The sign string must
        include the **full URL path** (i.e. including the ``/api/v4`` prefix),
        not just the endpoint path.  Using only the endpoint causes a 401
        Unauthorized response from the server.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path (e.g., "/futures/usdt/orders").
                  The ``/api/v4`` prefix is prepended automatically.
            query_string: URL query string
            body_string: JSON body string

        Returns:
            Dict with "KEY", "SIGN", and "Timestamp" headers
        """
        timestamp = str(int(time.time()))
        hashed_payload = hashlib.sha512(body_string.encode()).hexdigest()

        # Gate.io requires the full URL path including the /api/v4 prefix in
        # the signature.  The caller passes only the endpoint (e.g.
        # "/futures/usdt/accounts"), so we prepend the API version prefix here.
        full_path = f"/api/v4{path}"
        sign_string = f"{method}\n{full_path}\n{query_string}\n{hashed_payload}\n{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            sign_string.encode(),
            hashlib.sha512
        ).hexdigest()

        return {
            "KEY": self.api_key,
            "SIGN": signature,
            "Timestamp": timestamp,
        }

    async def _public_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a public (unauthenticated) API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On HTTP errors
        """
        if not self._session:
            raise RuntimeError("Client not connected. Call connect() first.")

        await ExchangeRateLimiter.limit(self.EXCHANGE_NAME, rps=10.0)
        url = f"{self.BASE_URL}{endpoint}"
        query_string = ""
        if params:
            query_string = urlencode(sorted(params.items()))
            url = f"{url}?{query_string}"

        async with self._session.request(method, url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _private_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a private (authenticated) API request.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            params: Query parameters
            body: JSON body

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On HTTP errors
        """
        if not self._session:
            raise RuntimeError("Client not connected. Call connect() first.")

        await ExchangeRateLimiter.limit(self.EXCHANGE_NAME, rps=10.0)
        url = f"{self.BASE_URL}{endpoint}"
        query_string = ""
        if params:
            query_string = urlencode(sorted(params.items()))
            url = f"{url}?{query_string}"

        body_string = ""
        json_body = None
        if body:
            json_body = body
            body_string = json.dumps(body)

        # Generate signature
        headers = self._generate_signature(method, endpoint, query_string, body_string)
        headers["Content-Type"] = "application/json"

        async with self._session.request(method, url, headers=headers, json=json_body) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Market data — REST
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_balance(self) -> Balance:
        """Fetch USDT futures account balance."""
        try:
            result = await self._private_request("GET", "/futures/usdt/accounts")
            total_usdt = float(result.get("total", "0") or "0")
            available_usdt = float(result.get("available", "0") or "0")
            used_usdt = total_usdt - available_usdt

            return Balance(
                total={"USDT": total_usdt},
                free={"USDT": available_usdt},
                used={"USDT": used_usdt},
                usdt_total=total_usdt,
                usdt_free=available_usdt,
            )
        except Exception as e:
            logger.error("Failed to fetch balance: {}", e)
            return Balance(
                total={"USDT": 0.0},
                free={"USDT": 0.0},
                used={"USDT": 0.0},
                usdt_total=0.0,
                usdt_free=0.0,
            )

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch latest ticker for symbol."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            # Validate the contract exists before fetching ticker data.
            # The response is not used directly — we only need the side-effect
            # of raising an error when the symbol is not found.
            await self._public_request(
                "GET",
                f"/futures/usdt/contracts/{tradfi_symbol}",
            )
            # Get ticker data
            ticker_data = await self._public_request(
                "GET",
                "/futures/usdt/tickers",
                params={"contract": tradfi_symbol},
            )

            if ticker_data and len(ticker_data) > 0:
                tick = ticker_data[0]
                return Ticker(
                    symbol=symbol,
                    bid=float(tick.get("last", 0)),
                    ask=float(tick.get("last", 0)),
                    last=float(tick.get("last", 0)),
                    high=float(tick.get("high_24h", 0)),
                    low=float(tick.get("low_24h", 0)),
                    volume=float(tick.get("volume_24h", 0)),
                    timestamp=int(time.time() * 1000),
                )
        except Exception as e:
            logger.error("Failed to fetch ticker for {}: {}", symbol, e)

        return Ticker(
            symbol=symbol,
            bid=0.0,
            ask=0.0,
            last=0.0,
            high=0.0,
            low=0.0,
            volume=0.0,
            timestamp=int(time.time() * 1000),
        )

    async def get_multiple_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Fetch latest tickers for multiple symbols in one request."""
        result: Dict[str, Ticker] = {}
        for symbol in symbols:
            try:
                ticker = await self.get_ticker(symbol)
                result[symbol] = ticker
            except Exception as e:
                logger.warning("Failed to fetch ticker for {}: {}", symbol, e)
        return result

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch order book for symbol."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            result = await self._public_request(
                "GET",
                "/futures/usdt/order_book",
                params={"contract": tradfi_symbol, "limit": limit},
            )
            return {
                "bids": [[float(p), float(a)] for p, a in result.get("bids", [])],
                "asks": [[float(p), float(a)] for p, a in result.get("asks", [])],
                "timestamp": int(time.time() * 1000),
            }
        except Exception as e:
            logger.error("Failed to fetch orderbook for {}: {}", symbol, e)
            return {"bids": [], "asks": [], "timestamp": int(time.time() * 1000)}

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)

        # Convert timeframe to seconds
        timeframe_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        interval = timeframe_map.get(timeframe, 3600)

        try:
            result = await self._public_request(
                "GET",
                "/futures/usdt/candlesticks",
                params={
                    "contract": tradfi_symbol,
                    "interval": f"{interval}s",
                    "limit": limit,
                },
            )

            if result:
                data = []
                for candle in result:
                    data.append([
                        int(candle["t"]) * 1000,  # timestamp in ms
                        float(candle["o"]),  # open
                        float(candle["h"]),  # high
                        float(candle["l"]),  # low
                        float(candle["c"]),  # close
                        float(candle.get("v", 0)),  # volume
                    ])

                df = pd.DataFrame(
                    data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                return df
        except Exception as e:
            logger.error("Failed to fetch OHLCV for {}: {}", symbol, e)

        # Return empty DataFrame on error
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a market order."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)

        # Determine order size (contracts); Gate.io uses negative for SELL
        size = int(amount) if amount >= 1 else 1
        if side == OrderSide.SELL:
            size = -size

        # Set leverage if provided
        leverage = params.get("leverage", 1)
        if leverage:
            try:
                await self.set_leverage(symbol, int(leverage))
            except Exception as e:
                logger.warning("Failed to set leverage for {}: {}", symbol, e)

        body = {
            "contract": tradfi_symbol,
            "size": size,
            "price": "0",  # Market order
            "tif": "ioc",  # Immediate or cancel
            "reduce_only": params.get("reduceOnly", False),
        }

        try:
            result = await self._private_request("POST", "/futures/usdt/orders", body=body)

            return Order(
                id=str(result.get("id", "")),
                symbol=symbol,
                type=OrderType.MARKET,
                side=side,
                amount=float(size),
                price=None,
                filled=float(result.get("fill_price", 0)),
                remaining=0.0,
                status=OrderStatus.CLOSED if result.get("status") == "finished" else OrderStatus.OPEN,
                timestamp=int(result.get("create_time", time.time()) * 1000),
                fee=0.0,
                info=result,
            )
        except Exception as e:
            logger.error("Failed to create market order for {}: {}", symbol, e)
            raise

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a limit order."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)

        size = int(amount) if amount >= 1 else 1

        # Set leverage if provided
        leverage = params.get("leverage", 1)
        if leverage:
            try:
                await self.set_leverage(symbol, int(leverage))
            except Exception as e:
                logger.warning("Failed to set leverage for {}: {}", symbol, e)

        body = {
            "contract": tradfi_symbol,
            "size": size,
            "price": str(price),
            "tif": "gtc",  # Good till cancelled
            "reduce_only": params.get("reduceOnly", False),
        }

        try:
            result = await self._private_request("POST", "/futures/usdt/orders", body=body)

            return Order(
                id=str(result.get("id", "")),
                symbol=symbol,
                type=OrderType.LIMIT,
                side=side,
                amount=float(size),
                price=price,
                filled=float(result.get("fill_price", 0)),
                remaining=float(size) - float(result.get("fill_price", 0)),
                status=OrderStatus.CLOSED if result.get("status") == "finished" else OrderStatus.OPEN,
                timestamp=int(result.get("create_time", time.time()) * 1000),
                fee=0.0,
                info=result,
            )
        except Exception as e:
            logger.error("Failed to create limit order for {}: {}", symbol, e)
            raise

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a stop-loss order."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        size = int(amount) if amount >= 1 else 1

        body = {
            "contract": tradfi_symbol,
            "size": size,
            "trigger": {
                "strategy_type": 0,  # Stop loss
                "price_type": 0,  # Latest price
                "price": str(stop_price),
            },
            "order_type": "market",
            "reduce_only": True,
        }

        try:
            result = await self._private_request("POST", "/futures/usdt/price_orders", body=body)

            return Order(
                id=str(result.get("id", "")),
                symbol=symbol,
                type=OrderType.STOP_LOSS,
                side=side,
                amount=float(size),
                price=stop_price,
                filled=0.0,
                remaining=float(size),
                status=OrderStatus.OPEN,
                timestamp=int(time.time() * 1000),
                fee=0.0,
                info=result,
            )
        except Exception as e:
            logger.error("Failed to create stop-loss order for {}: {}", symbol, e)
            raise

    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a take-profit order."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        size = int(amount) if amount >= 1 else 1

        body = {
            "contract": tradfi_symbol,
            "size": size,
            "trigger": {
                "strategy_type": 1,  # Take profit
                "price_type": 0,  # Latest price
                "price": str(tp_price),
            },
            "order_type": "market",
            "reduce_only": True,
        }

        try:
            result = await self._private_request("POST", "/futures/usdt/price_orders", body=body)

            return Order(
                id=str(result.get("id", "")),
                symbol=symbol,
                type=OrderType.TAKE_PROFIT,
                side=side,
                amount=float(size),
                price=tp_price,
                filled=0.0,
                remaining=float(size),
                status=OrderStatus.OPEN,
                timestamp=int(time.time() * 1000),
                fee=0.0,
                info=result,
            )
        except Exception as e:
            logger.error("Failed to create take-profit order for {}: {}", symbol, e)
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            result = await self._private_request(
                "DELETE",
                f"/futures/usdt/orders/{order_id}",
            )
            logger.info("Order {} cancelled for {}", order_id, symbol)
            return result
        except Exception as e:
            logger.error("Failed to cancel order {}: {}", order_id, e)
            raise

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all orders for symbol."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            result = await self._private_request(
                "DELETE",
                "/futures/usdt/orders",
                params={"contract": tradfi_symbol},
            )
            logger.info("All orders cancelled for {}", symbol)
            return result if isinstance(result, list) else [result]
        except Exception as e:
            logger.error("Failed to cancel all orders for {}: {}", symbol, e)
            return []

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch order status."""
        try:
            result = await self._private_request(
                "GET",
                f"/futures/usdt/orders/{order_id}",
            )

            return Order(
                id=str(result.get("id", "")),
                symbol=symbol,
                type=OrderType.MARKET,
                side=OrderSide.BUY if result.get("size", 0) > 0 else OrderSide.SELL,
                amount=abs(float(result.get("size", 0))),
                price=float(result.get("price", 0)) if result.get("price") else None,
                filled=float(result.get("fill_price", 0)),
                remaining=float(result.get("left", 0)),
                status=OrderStatus.CLOSED if result.get("status") == "finished" else OrderStatus.OPEN,
                timestamp=int(result.get("create_time", 0) * 1000),
                fee=0.0,
                info=result,
            )
        except Exception as e:
            logger.error("Failed to fetch order {}: {}", order_id, e)
            raise

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Fetch all open orders."""
        params = {}
        if symbol:
            params["contract"] = self._resolve_tradfi_symbol(symbol)

        try:
            result = await self._private_request("GET", "/futures/usdt/orders", params=params)

            orders = []
            for order_data in result:
                orders.append(Order(
                    id=str(order_data.get("id", "")),
                    symbol=self._resolve_display_symbol(order_data.get("contract", "")),
                    type=OrderType.MARKET,
                    side=OrderSide.BUY if order_data.get("size", 0) > 0 else OrderSide.SELL,
                    amount=abs(float(order_data.get("size", 0))),
                    price=float(order_data.get("price", 0)) if order_data.get("price") else None,
                    filled=float(order_data.get("fill_price", 0)),
                    remaining=float(order_data.get("left", 0)),
                    status=OrderStatus.OPEN,
                    timestamp=int(order_data.get("create_time", 0) * 1000),
                    fee=0.0,
                    info=order_data,
                ))
            return orders
        except Exception as e:
            logger.error("Failed to fetch open orders: {}", e)
            return []

    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch personal trade history via GET /futures/usdt/my_trades."""
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["contract"] = self._resolve_tradfi_symbol(symbol)
        try:
            result = await self._private_request("GET", "/futures/usdt/my_trades", params=params)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error("Failed to fetch trade history: {}", e)
            return []

    # ------------------------------------------------------------------
    # Position & leverage management
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for symbol."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            body = {
                "contract": tradfi_symbol,
                "leverage": str(leverage),
            }
            result = await self._private_request(
                "POST",
                "/futures/usdt/positions/{}/leverage".format(tradfi_symbol),
                body=body,
            )
            logger.info("Leverage set to {}x for {}", leverage, symbol)
            return result
        except Exception as e:
            logger.warning("Failed to set leverage for {}: {}", symbol, e)
            return {}

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Set margin type (cross/isolated) for symbol."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            body = {
                "contract": tradfi_symbol,
                "mode": "dual_long_short" if margin_type == MarginType.CROSS else "single",
            }
            result = await self._private_request(
                "POST",
                "/futures/usdt/positions/{}/margin_mode".format(tradfi_symbol),
                body=body,
            )
            logger.info("Margin type set to {} for {}", margin_type.value, symbol)
            return result
        except Exception as e:
            logger.warning("Failed to set margin type for {}: {}", symbol, e)
            return {}

    async def get_positions(self) -> List[Position]:
        """Fetch all open positions."""
        try:
            result = await self._private_request("GET", "/futures/usdt/positions")

            positions = []
            for pos_data in result:
                size = float(pos_data.get("size", 0))
                if size == 0:
                    continue

                positions.append(Position(
                    symbol=self._resolve_display_symbol(pos_data.get("contract", "")),
                    side=PositionSide.LONG if size > 0 else PositionSide.SHORT,
                    amount=abs(size),
                    entry_price=float(pos_data.get("entry_price", 0)),
                    current_price=float(pos_data.get("mark_price", 0)),
                    unrealized_pnl=float(pos_data.get("unrealised_pnl", 0)),
                    leverage=int(pos_data.get("leverage", 1)),
                    margin=float(pos_data.get("margin", 0)),
                    liquidation_price=float(pos_data.get("liq_price", 0)),
                    timestamp=int(time.time() * 1000),
                ))
            return positions
        except Exception as e:
            logger.error("Failed to fetch positions: {}", e)
            return []

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Fetch position for specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close a position."""
        position = await self.get_position(symbol)
        if not position:
            raise ValueError(f"No open position for {symbol}")

        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_amount = amount if amount is not None else position.amount

        logger.info("Closing position {} amount={}", symbol, close_amount)
        return await self.create_market_order(
            symbol,
            close_side,
            close_amount,
            {"reduceOnly": True}
        )

    # ------------------------------------------------------------------
    # Derivatives data
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            result = await self._public_request(
                "GET",
                f"/futures/usdt/contracts/{tradfi_symbol}",
            )
            return float(result.get("funding_rate", 0))
        except Exception as e:
            logger.warning("Failed to fetch funding rate for {}: {}", symbol, e)
            return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        """Fetch open interest."""
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        try:
            result = await self._public_request(
                "GET",
                f"/futures/usdt/contracts/{tradfi_symbol}",
            )
            return float(result.get("open_interest", 0))
        except Exception as e:
            logger.warning("Failed to fetch open interest for {}: {}", symbol, e)
            return 0.0

    # ------------------------------------------------------------------
    # WebSocket subscriptions — real implementation
    # ------------------------------------------------------------------

    async def _ws_send(self, ws: Any, channel: str, event: str, payload: Any) -> None:
        """Send a WebSocket message with Gate.io v4 format."""
        msg = {
            "time": int(time.time()),
            "channel": channel,
            "event": event,
            "payload": payload,
        }
        await ws.send_str(json.dumps(msg))

    async def _ws_auth(self, ws: Any) -> None:
        """Authenticate a WebSocket connection using HMAC-SHA512."""
        ts = int(time.time())
        sign_str = f"channel=futures.login\nevent=api\ntimestamp={ts}"
        signature = hmac.new(
            self.secret_key.encode(),
            sign_str.encode(),
            hashlib.sha512,
        ).hexdigest()
        auth_msg = {
            "time": ts,
            "channel": "futures.login",
            "event": "api",
            "payload": {
                "api_key": self.api_key,
                "signature": signature,
                "timestamp": str(ts),
            },
        }
        await ws.send_str(json.dumps(auth_msg))

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time ticker updates via WebSocket.

        Uses the ``futures.tickers`` channel on wss://api.gateio.ws/ws/v4/.
        """
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        key = f"ticker_{tradfi_symbol}"
        if key in self._ws_connections:
            logger.debug("Ticker WS already subscribed for {}", tradfi_symbol)
            return


        async def _run() -> None:
            url = self.WS_URL
            while True:
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.ws_connect(url) as ws:
                            self._ws_connections[key] = ws
                            await self._ws_send(ws, "futures.tickers", "subscribe", [tradfi_symbol])
                            logger.info("Subscribed to ticker WS for {}", tradfi_symbol)
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    if data.get("channel") == "futures.tickers" and data.get("event") == "update":
                                        result = data.get("result", {})
                                        ticker = Ticker(
                                            symbol=symbol,
                                            bid=float(result.get("last", 0)),
                                            ask=float(result.get("last", 0)),
                                            last=float(result.get("last", 0)),
                                            high=float(result.get("high_24h", 0)),
                                            low=float(result.get("low_24h", 0)),
                                            volume=float(result.get("volume_24h", 0)),
                                            timestamp=int(time.time() * 1000),
                                        )
                                        try:
                                            await callback(ticker)
                                        except Exception as cb_err:
                                            logger.warning("Ticker callback error: {}", cb_err)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    logger.warning("Ticker WS closed for {}, reconnecting", tradfi_symbol)
                                    break
                except Exception as e:
                    logger.warning("Ticker WS error for {}: {}, retrying in 5s", tradfi_symbol, e)
                    await asyncio.sleep(5)

        asyncio.ensure_future(_run())

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time orderbook updates via WebSocket.

        Uses the ``futures.order_book`` channel.
        """
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        key = f"orderbook_{tradfi_symbol}"
        if key in self._ws_connections:
            return


        async def _run() -> None:
            url = self.WS_URL
            while True:
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.ws_connect(url) as ws:
                            self._ws_connections[key] = ws
                            await self._ws_send(
                                ws, "futures.order_book", "subscribe",
                                [tradfi_symbol, "20", "0"]
                            )
                            logger.info("Subscribed to orderbook WS for {}", tradfi_symbol)
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    if data.get("channel") == "futures.order_book" and data.get("event") == "update":
                                        result = data.get("result", {})
                                        book = {
                                            "bids": [[float(b["p"]), float(b["s"])] for b in result.get("b", [])],
                                            "asks": [[float(a["p"]), float(a["s"])] for a in result.get("a", [])],
                                            "timestamp": int(time.time() * 1000),
                                        }
                                        try:
                                            await callback(book)
                                        except Exception as cb_err:
                                            logger.warning("Orderbook callback error: {}", cb_err)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    logger.warning("Orderbook WS closed for {}, reconnecting", tradfi_symbol)
                                    break
                except Exception as e:
                    logger.warning("Orderbook WS error for {}: {}, retrying in 5s", tradfi_symbol, e)
                    await asyncio.sleep(5)

        asyncio.ensure_future(_run())

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time trade feed via WebSocket.

        Uses the ``futures.trades`` channel.
        """
        tradfi_symbol = self._resolve_tradfi_symbol(symbol)
        key = f"trades_{tradfi_symbol}"
        if key in self._ws_connections:
            return


        async def _run() -> None:
            url = self.WS_URL
            while True:
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.ws_connect(url) as ws:
                            self._ws_connections[key] = ws
                            await self._ws_send(ws, "futures.trades", "subscribe", [tradfi_symbol])
                            logger.info("Subscribed to trades WS for {}", tradfi_symbol)
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    if data.get("channel") == "futures.trades" and data.get("event") == "update":
                                        try:
                                            await callback(data.get("result", {}))
                                        except Exception as cb_err:
                                            logger.warning("Trades callback error: {}", cb_err)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                except Exception as e:
                    logger.warning("Trades WS error for {}: {}, retrying in 5s", tradfi_symbol, e)
                    await asyncio.sleep(5)

        asyncio.ensure_future(_run())

    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to private user data: orders, positions, balances.

        Uses futures.orders, futures.positions, futures.balances channels.
        Requires authentication.
        """
        key = "user_data"
        if key in self._ws_connections:
            return


        async def _run() -> None:
            url = self.WS_URL
            while True:
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.ws_connect(url) as ws:
                            self._ws_connections[key] = ws
                            await self._ws_auth(ws)
                            for channel in ("futures.orders", "futures.positions", "futures.balances"):
                                await self._ws_send(ws, channel, "subscribe", ["!all"])
                            logger.info("Subscribed to user data WS channels")
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    channel = data.get("channel", "")
                                    if data.get("event") == "update" and channel in (
                                        "futures.orders", "futures.positions", "futures.balances"
                                    ):
                                        try:
                                            await callback(data)
                                        except Exception as cb_err:
                                            logger.warning("User data callback error: {}", cb_err)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    logger.warning("User data WS closed, reconnecting")
                                    break
                except Exception as e:
                    logger.warning("User data WS error: {}, retrying in 5s", e)
                    await asyncio.sleep(5)

        asyncio.ensure_future(_run())

    async def start_user_data_stream(self, callback: Callable) -> None:
        """Start real-time position/order tracking via WebSocket.

        Alias for :meth:`subscribe_user_data` that matches the naming
        convention used by the Binance SDK and other exchanges.
        """
        await self.subscribe_user_data(callback)

    @property
    def name(self) -> str:
        return "Gate.io TradFi"
