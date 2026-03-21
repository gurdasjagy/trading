"""BingX exchange client built on top of ccxt async support."""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import ccxt.async_support as ccxt
import pandas as pd
import websockets
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

_WS_URL = "wss://open-api-ws.bingx.com/market"
_WS_RECONNECT_DELAY = 5  # seconds


class BingXClient(BaseExchange):
    """BingX exchange client using ``ccxt.async_support.bingx``.

    Supports perpetual futures via the ``defaultType='swap'`` option.
    """

    EXCHANGE_NAME = "bingx"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str = "",
        testnet: bool = False,
    ) -> None:
        super().__init__(api_key, secret_key, passphrase, testnet)
        self._ws_connections: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._ws_lock = asyncio.Lock()
        self._rate_limiter = ExchangeRateLimiter.get_limiter(self.EXCHANGE_NAME, rps=10.0)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the ccxt BingX client and pre-load market data."""
        self._client = ccxt.bingx(
            {
                "apiKey": self.api_key,
                "secret": self.secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        if self.testnet:
            self._client.set_sandbox_mode(True)
        await self._client.load_markets()
        logger.info("BingXClient connected (testnet={})", self.testnet)

    async def disconnect(self) -> None:
        """Close the ccxt HTTP session."""
        if self._client:
            await self._client.close()
            logger.info("BingXClient disconnected")

    # ------------------------------------------------------------------
    # Market data — REST
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_balance(self) -> Balance:
        """Fetch account balance and return a normalised :class:`Balance`."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_balance()
        total = {k: float(v) for k, v in raw.get("total", {}).items() if v}
        free = {k: float(v) for k, v in raw.get("free", {}).items() if v}
        used = {k: float(v) for k, v in raw.get("used", {}).items() if v}
        return Balance(
            total=total,
            free=free,
            used=used,
            usdt_total=total.get("USDT", 0.0),
            usdt_free=free.get("USDT", 0.0),
        )

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch the latest ticker snapshot for *symbol*."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,
            bid=float(raw.get("bid") or 0),
            ask=float(raw.get("ask") or 0),
            last=float(raw.get("last") or 0),
            high=float(raw.get("high") or 0),
            low=float(raw.get("low") or 0),
            volume=float(raw.get("baseVolume") or 0),
            timestamp=int(raw.get("timestamp") or 0),
        )

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Return the level-2 order book for *symbol*."""
        await self._rate_limiter.acquire()
        return await self._client.fetch_order_book(symbol, limit)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Return OHLCV candles as a :class:`pandas.DataFrame` indexed by time."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

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
        """Place a market order for *amount* contracts on *symbol*."""
        await self._rate_limiter.acquire()
        raw = await self._client.create_market_order(symbol, side.value, amount, params=params)
        logger.info("Market order placed: {} {} {} @ market", side.value, amount, symbol)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a limit order for *amount* contracts at *price*."""
        await self._rate_limiter.acquire()
        raw = await self._client.create_limit_order(
            symbol, side.value, amount, price, params=params
        )
        logger.info("Limit order placed: {} {} {} @ {}", side.value, amount, symbol, price)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a stop-market order triggered at *stop_price*."""
        await self._rate_limiter.acquire()
        p = {**params, "stopPrice": stop_price, "workingType": "MARK_PRICE"}
        raw = await self._client.create_order(
            symbol, "STOP_MARKET", side.value, amount, stop_price, p
        )
        logger.info("Stop-loss placed: {} {} {} trigger={}", side.value, amount, symbol, stop_price)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a take-profit market order triggered at *tp_price*."""
        await self._rate_limiter.acquire()
        p = {**params, "stopPrice": tp_price, "workingType": "MARK_PRICE"}
        raw = await self._client.create_order(
            symbol, "TAKE_PROFIT_MARKET", side.value, amount, tp_price, p
        )
        logger.info("Take-profit placed: {} {} {} trigger={}", side.value, amount, symbol, tp_price)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a single open order."""
        await self._rate_limiter.acquire()
        result = await self._client.cancel_order(order_id, symbol)
        logger.info("Order {} cancelled on {}", order_id, symbol)
        return result

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for *symbol*."""
        await self._rate_limiter.acquire()
        result = await self._client.cancel_all_orders(symbol)
        logger.info("All orders cancelled on {}", symbol)
        return result if isinstance(result, list) else []

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch the current state of an order by its ID."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_order(order_id, symbol)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all open orders, optionally filtered by *symbol*."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_open_orders(symbol)
        return [self._parse_order(o) for o in raw]

    # ------------------------------------------------------------------
    # Position & leverage management
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for *symbol*."""
        await self._rate_limiter.acquire()
        result = await self._client.set_leverage(leverage, symbol)
        logger.info("Leverage set to {}x on {}", leverage, symbol)
        return result

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Switch between cross / isolated margin for *symbol*."""
        await self._rate_limiter.acquire()
        result = await self._client.set_margin_mode(margin_type.value, symbol)
        logger.info("Margin type set to {} on {}", margin_type.value, symbol)
        return result

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_positions(self) -> List[Position]:
        """Return all non-zero open positions."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_positions()
        return [self._parse_position(p) for p in raw if float(p.get("contracts") or 0) != 0]

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for *symbol*, or *None* if flat."""
        positions = await self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close (or partially close) the open position for *symbol*."""
        position = await self.get_position(symbol)
        if not position:
            raise ValueError(f"No open position for {symbol}")
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_amount = amount if amount is not None else position.amount
        logger.info("Closing position {} amount={}", symbol, close_amount)
        return await self.create_market_order(
            symbol, close_side, close_amount, {"reduceOnly": True}
        )

    # ------------------------------------------------------------------
    # Derivatives-specific data
    # ------------------------------------------------------------------

    def _resolve_swap_symbol(self, symbol: str) -> str:
        """Return the swap market symbol for *symbol*, appending ``:USDT`` if needed."""
        if ":" in symbol:
            return symbol
        markets = getattr(self._client, "markets", None) or {}
        for market_symbol, market_info in markets.items():
            if isinstance(market_info, dict):
                if (
                    market_info.get("type") in ("swap", "future")
                    and market_info.get("spot") is False
                    and market_info.get("base") == symbol.split("/")[0]
                    and market_info.get("quote") == symbol.split("/")[-1]
                ):
                    return market_symbol
        if "/" in symbol and ":" not in symbol:
            quote = symbol.split("/")[-1]
            return f"{symbol}:{quote}"
        return symbol

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_funding_rate(self, symbol: str) -> float:
        """Return the current funding rate for *symbol*, or 0.0 on error."""
        try:
            await self._rate_limiter.acquire()
            swap_symbol = self._resolve_swap_symbol(symbol)
            result = await self._client.fetch_funding_rate(swap_symbol)
            return float(result.get("fundingRate") or 0)
        except Exception as exc:
            logger.warning("get_funding_rate failed for {}: {}", symbol, exc)
            return 0.0

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_open_interest(self, symbol: str) -> float:
        """Return the current open interest for *symbol*, or 0.0 on error."""
        try:
            await self._rate_limiter.acquire()
            swap_symbol = self._resolve_swap_symbol(symbol)
            result = await self._client.fetch_open_interest(swap_symbol)
            return float(result.get("openInterest") or 0)
        except Exception as exc:
            logger.warning("get_open_interest failed for {}: {}", symbol, exc)
            return 0.0

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time ticker updates for *symbol*."""
        asyncio.create_task(self._ws_ticker_loop(symbol, callback))

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time order-book snapshots for *symbol*."""
        asyncio.create_task(self._ws_orderbook_loop(symbol, callback))

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to the real-time public trade feed for *symbol*."""
        asyncio.create_task(self._ws_trades_loop(symbol, callback))

    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to private account-update events."""
        asyncio.create_task(self._ws_user_data_loop(callback))

    # ------------------------------------------------------------------
    # WebSocket loops (private)
    # ------------------------------------------------------------------

    def _bingx_symbol(self, symbol: str) -> str:
        """Convert ccxt symbol to BingX format, e.g. ``BTC/USDT`` → ``BTC-USDT``."""
        return symbol.split(":")[0].replace("/", "-")

    async def _ws_ticker_loop(self, symbol: str, callback: Callable) -> None:
        bx_sym = self._bingx_symbol(symbol)
        subscribe_msg = {
            "id": f"ticker_{bx_sym}",
            "reqType": "sub",
            "dataType": f"{bx_sym}@ticker",
        }
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    async for raw_msg in ws:
                        # BingX may send gzip-compressed data
                        try:
                            text = raw_msg if isinstance(raw_msg, str) else raw_msg.decode()
                        except Exception:
                            import gzip

                            text = gzip.decompress(raw_msg).decode()
                        data = json.loads(text)
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
            except Exception as exc:
                logger.warning(
                    "WS ticker error for {}: {} — reconnecting in {}s",
                    symbol,
                    exc,
                    _WS_RECONNECT_DELAY,
                )
                await asyncio.sleep(_WS_RECONNECT_DELAY)

    async def _ws_orderbook_loop(self, symbol: str, callback: Callable) -> None:
        bx_sym = self._bingx_symbol(symbol)
        subscribe_msg = {
            "id": f"depth_{bx_sym}",
            "reqType": "sub",
            "dataType": f"{bx_sym}@depth20",
        }
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    async for raw_msg in ws:
                        try:
                            text = raw_msg if isinstance(raw_msg, str) else raw_msg.decode()
                        except Exception:
                            import gzip

                            text = gzip.decompress(raw_msg).decode()
                        data = json.loads(text)
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
            except Exception as exc:
                logger.warning(
                    "WS orderbook error for {}: {} — reconnecting in {}s",
                    symbol,
                    exc,
                    _WS_RECONNECT_DELAY,
                )
                await asyncio.sleep(_WS_RECONNECT_DELAY)

    async def _ws_trades_loop(self, symbol: str, callback: Callable) -> None:
        bx_sym = self._bingx_symbol(symbol)
        subscribe_msg = {
            "id": f"trade_{bx_sym}",
            "reqType": "sub",
            "dataType": f"{bx_sym}@trade",
        }
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    async for raw_msg in ws:
                        try:
                            text = raw_msg if isinstance(raw_msg, str) else raw_msg.decode()
                        except Exception:
                            import gzip

                            text = gzip.decompress(raw_msg).decode()
                        data = json.loads(text)
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
            except Exception as exc:
                logger.warning(
                    "WS trades error for {}: {} — reconnecting in {}s",
                    symbol,
                    exc,
                    _WS_RECONNECT_DELAY,
                )
                await asyncio.sleep(_WS_RECONNECT_DELAY)

    async def _ws_user_data_loop(self, callback: Callable) -> None:
        """Private user-data stream; BingX requires a listenKey."""
        while True:
            try:
                # Fetch a listen key via REST before opening WebSocket
                listen_key_data = await self._client.fetch_private_listen_key()  # type: ignore[attr-defined]
                listen_key = listen_key_data.get("listenKey", "")
                ws_url = f"{_WS_URL}?listenKey={listen_key}"
                async with websockets.connect(ws_url) as ws:
                    async for raw_msg in ws:
                        try:
                            text = raw_msg if isinstance(raw_msg, str) else raw_msg.decode()
                        except Exception:
                            import gzip

                            text = gzip.decompress(raw_msg).decode()
                        data = json.loads(text)
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
            except Exception as exc:
                logger.warning(
                    "WS user-data error: {} — reconnecting in {}s",
                    exc,
                    _WS_RECONNECT_DELAY,
                )
                await asyncio.sleep(_WS_RECONNECT_DELAY)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_order(self, raw: Dict[str, Any]) -> Order:
        """Convert a ccxt order dict into a normalised :class:`Order`."""
        raw_type = (raw.get("type") or "market").lower()
        type_map = {
            "stop_market": "stop_loss",
            "take_profit_market": "take_profit",
        }
        order_type_str = type_map.get(raw_type, raw_type)
        try:
            order_type = OrderType(order_type_str)
        except ValueError:
            order_type = OrderType.MARKET

        raw_status = (raw.get("status") or "open").lower()
        status_map = {"filled": "closed", "cancelled": "canceled"}
        order_status_str = status_map.get(raw_status, raw_status)
        try:
            order_status = OrderStatus(order_status_str)
        except ValueError:
            order_status = OrderStatus.OPEN

        fee_cost = 0.0
        if raw.get("fee") and isinstance(raw["fee"], dict):
            fee_cost = float(raw["fee"].get("cost") or 0)

        return Order(
            id=str(raw.get("id") or ""),
            symbol=raw.get("symbol") or "",
            type=order_type,
            side=OrderSide(raw.get("side") or "buy"),
            amount=float(raw.get("amount") or 0),
            price=float(raw["price"]) if raw.get("price") else None,
            filled=float(raw.get("filled") or 0),
            remaining=float(raw.get("remaining") or 0),
            status=order_status,
            timestamp=int(raw.get("timestamp") or 0),
            fee=fee_cost,
            info=raw.get("info") or {},
        )

    def _parse_position(self, raw: Dict[str, Any]) -> Position:
        """Convert a ccxt position dict into a normalised :class:`Position`."""
        raw_side = (raw.get("side") or "long").lower()
        try:
            side = PositionSide(raw_side)
        except ValueError:
            side = PositionSide.LONG

        return Position(
            symbol=raw.get("symbol") or "",
            side=side,
            amount=abs(float(raw.get("contracts") or 0)),
            entry_price=float(raw.get("entryPrice") or 0),
            current_price=float(raw.get("markPrice") or 0),
            unrealized_pnl=float(raw.get("unrealizedPnl") or 0),
            leverage=int(raw.get("leverage") or 1),
            margin=float(raw.get("initialMargin") or 0),
            liquidation_price=float(raw.get("liquidationPrice") or 0),
            timestamp=int(raw.get("timestamp") or 0),
        )

    @property
    def name(self) -> str:
        return "BingX"
