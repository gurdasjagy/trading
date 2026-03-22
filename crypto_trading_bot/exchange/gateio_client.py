"""Gate.io exchange client built on top of ccxt async support.

.. deprecated::
    Direct order-placement methods (``create_market_order``,
    ``create_limit_order``, ``create_stop_loss_order``,
    ``create_take_profit_order``) are deprecated for live trading.
    Use the Rust ``trading_engine`` binary (see ``rust_engine/src/gateio_gateway.rs``)
    which provides HMAC-SHA512 signing, connection pooling, adaptive rate
    limiting, and exponential backoff in a sub-2 ms hot path.

    For reading positions, prefer :meth:`get_positions_from_rust` which
    queries the Rust engine's shared state (zero latency) rather than
    issuing a REST call.
"""

import asyncio
import hashlib
import hmac
import json
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

import ccxt.async_support as ccxt
import pandas as pd
import websockets
from loguru import logger

from utils.circuit_breaker import SymbolPermanentlyUnavailableError, with_circuit_breaker
from utils.rate_limiter import ExchangeRateLimiter
from utils.retry import async_retry_decorator
from utils.symbol_resolver import resolve_symbols

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

try:
    from rust_trading_engine.ws_parser import parse_ws_message as _rust_parse_ws_message
    _USE_RUST_WS_PARSER = True
except ImportError:
    _USE_RUST_WS_PARSER = False

_WS_URL = "wss://api.gateio.ws/ws/v4/"
_WS_RECONNECT_DELAY = 5  # seconds


class GateIOClient(BaseExchange):
    """Gate.io exchange client using ``ccxt.async_support.gateio``.

    Supports futures (perpetual contracts) via the ``defaultType='swap'`` option.
    Gate.io WebSocket channels use the ``futures.*`` namespace for derivatives.
    """

    EXCHANGE_NAME = "gateio"

    # Gate.io does NOT have native XAU/USDT on their standard futures/swap API.
    # The XAU/USDT perpetual visible in the Gate.io app is on their TradFi (MT5)
    # platform, not the standard futures API.
    # XAUT/USDT (Tether Gold, backed 1:1 by physical gold) IS available as a
    # perpetual contract on Gate.io's standard futures API and tracks the gold price.
    PRECIOUS_METALS_MAPPING: Dict[str, str] = {
        "XAU/USDT": "XAUT/USDT",  # Tether Gold (1 XAUT = 1 troy oz gold)
        "XAG/USDT": "XAUT/USDT",  # No silver token; fallback to gold token
    }

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
        # Use lower RPS on testnet to avoid stricter rate limits
        _rps = 5.0 if testnet else 10.0
        self._rate_limiter = ExchangeRateLimiter.get_limiter(self.EXCHANGE_NAME, rps=_rps)
        # WebSocket user-data caches (updated in real-time via start_user_data_stream)
        self._ws_positions: Dict[str, Any] = {}   # symbol -> position dict
        self._ws_orders: Dict[str, Any] = {}       # order_id -> order dict
        self._ws_last_update: float = 0.0          # epoch seconds of last WS update
        self._event_bus: Optional[Any] = None      # injected by engine after init
        # Pending order fill futures: order_id -> asyncio.Future
        self._pending_fills: Dict[str, asyncio.Future] = {}
        # Symbols that are permanently unavailable on this exchange — tried once,
        # failed, never retried.  Populated by _validate_precious_metals_availability()
        # after markets are loaded.
        self._permanently_unavailable_symbols: set = set()
        self._refresh_markets_task: Optional[asyncio.Task] = None
        # Shared ccxt.pro WebSocket client for market-data streams (lazy-created)
        self._pro_ws_client: Any = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the ccxt Gate.io client and pre-load market data."""
        self._client = ccxt.gateio(
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
        # Check which precious-metals symbols are actually available.
        # This runs exactly once so the bot never wastes cycles retrying
        # symbols that do not exist on the exchange (e.g. XAUT/USDT:USDT).
        self._validate_precious_metals_availability()
        self._refresh_markets_task = asyncio.create_task(self._refresh_markets_loop())
        logger.info("GateIOClient connected (testnet={})", self.testnet)

    async def disconnect(self) -> None:
        """Close the ccxt HTTP session."""
        if self._refresh_markets_task and not self._refresh_markets_task.done():
            self._refresh_markets_task.cancel()
        if self._client:
            await self._client.close()
            logger.info("GateIOClient disconnected")

    def _get_ws_client(self) -> Any:
        """Return a shared ccxt.pro Gate.io WebSocket client, creating it if necessary.

        The client is lazily initialised on the first call and cached in
        ``self._pro_ws_client`` for all subsequent calls.  Because this runs
        inside an asyncio event loop (single-threaded), no lock is needed.

        Used by :class:`~.ws_data_manager.WebSocketDataManager` to obtain a
        proper ``ccxt.pro.gateio`` client whose ``has`` dict advertises genuine
        ``watchTicker``/``watchTickers`` support, unlike the ``ccxt.async_support``
        REST client whose stub methods raise ``NotSupported`` at runtime.

        Returns:
            A ``ccxt.pro.gateio`` exchange instance configured for perpetual
            swap markets.

        Raises:
            ImportError: If ``ccxt.pro`` is not installed.
            RuntimeError: If Gate.io is not supported by the installed ``ccxt.pro``.
        """
        if self._pro_ws_client is not None:
            return self._pro_ws_client

        import ccxt.pro as ccxtpro  # type: ignore[import]

        ws_class = getattr(ccxtpro, "gateio", None)
        if ws_class is None:
            raise RuntimeError("ccxt.pro has no class for 'gateio'")

        cfg: Dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        self._pro_ws_client = ws_class(cfg)
        if self.testnet:
            self._pro_ws_client.set_sandbox_mode(True)
        return self._pro_ws_client

    async def _refresh_markets_loop(self, interval_hours: float = 4.0) -> None:
        """Refresh market metadata every N hours in the background."""
        while True:
            await asyncio.sleep(interval_hours * 3600)
            try:
                await self._client.load_markets(reload=True)
                logger.info("Markets refreshed ({} markets loaded)", len(self._client.markets))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Market refresh failed: {}", exc)

    # ------------------------------------------------------------------
    # Market data — REST
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_balance(self) -> Balance:
        """Fetch account balance and return a normalised :class:`Balance`.

        Explicitly requests the USDT-settled swap (futures) balance so the
        correct endpoint is called regardless of CCXT version behaviour with
        the ``defaultType`` option.  The ``settle`` parameter pins the
        settlement currency and avoids ambiguous routing on Gate.io testnet.
        """
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_balance({"type": "swap", "settle": "usdt"})
        total = {k: float(v) for k, v in raw.get("total", {}).items() if v is not None}
        free = {k: float(v) for k, v in raw.get("free", {}).items() if v is not None}
        used = {k: float(v) for k, v in raw.get("used", {}).items() if v is not None}
        balance = Balance(
            total=total,
            free=free,
            used=used,
            usdt_total=total.get("USDT", 0.0),
            usdt_free=free.get("USDT", 0.0),
        )
        logger.debug(
            "GateIOClient balance: usdt_total={:.2f} usdt_free={:.2f} (testnet={})",
            balance.usdt_total,
            balance.usdt_free,
            self.testnet,
        )
        return balance

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_trade_history(self, symbol: str, limit: int = 50) -> List[Any]:
        """Return recent user trade history for *symbol*."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_my_trades(symbol, limit=limit)
        class _TradeDict:
            def __init__(self, d):
                self.__dict__.update(d)
                self.price = float(d.get("price", 0))
                self.amount = float(d.get("amount", 0))
                self.side = d.get("side", "")
                
        return [_TradeDict(t) for t in raw]

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch the latest ticker snapshot for *symbol*."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols.
        # @resolve_symbols post-processes Ticker to restore original symbol.
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,  # original restored by @resolve_symbols decorator
            bid=float(raw.get("bid") or 0),
            ask=float(raw.get("ask") or 0),
            last=float(raw.get("last") or 0),
            high=float(raw.get("high") or 0),
            low=float(raw.get("low") or 0),
            volume=float(raw.get("baseVolume") or 0),
            timestamp=int(raw.get("timestamp") or 0),
        )

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Return the level-2 order book for *symbol*."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        await self._rate_limiter.acquire()
        return await self._client.fetch_order_book(symbol, limit)

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Return OHLCV candles as a :class:`pandas.DataFrame` indexed by time."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent public trades for *symbol*."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_trades(symbol, limit=limit)
        return [
            {
                "id": str(t.get("id", "")),
                "price": float(t.get("price", 0)),
                "amount": float(t.get("amount", 0)),
                "side": t.get("side", ""),
                "timestamp": t.get("timestamp", 0),
            }
            for t in raw
        ]

    async def get_multiple_tickers(self, symbols: List[str]) -> Dict[str, "Ticker"]:
        """Fetch tickers for multiple symbols concurrently.

        Permanently unavailable symbols (e.g. XAU/USDT when XAUT not listed)
        are silently skipped so they do not block tickers for other symbols.
        """
        tasks = [self._fetch_single_ticker(sym) for sym in symbols]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: Dict[str, "Ticker"] = {}
        for sym, result in zip(symbols, gathered):
            if isinstance(result, Exception):
                logger.debug("get_multiple_tickers: error for {}: {}", sym, result)
                continue
            results[sym] = result
        return results

    async def _fetch_single_ticker(self, symbol: str) -> "Ticker":
        """Helper to fetch a single ticker; used by :meth:`get_multiple_tickers`."""
        return await self.get_ticker(symbol)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def _apply_leverage_from_params(self, swap_symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract leverage from *params*, call set_leverage on the exchange, and return
        a copy of *params* with the leverage value normalised to an int.

        This ensures Gate.io sets the correct leverage before every contract order,
        preventing "Balance not enough" errors caused by the exchange defaulting to 1x.
        """
        leverage = params.get("leverage")
        if not leverage:
            return {**params}
        try:
            if hasattr(self._client, "set_leverage"):
                await self._client.set_leverage(int(leverage), swap_symbol)
            elif hasattr(self._client, "set_margin_mode"):
                await self._client.set_margin_mode("cross", swap_symbol, {"leverage": int(leverage)})
        except Exception as e:
            logger.warning("Could not set leverage for {}: {}", swap_symbol, e)
        return {**params, "leverage": int(leverage)}

    def _validate_contract_minimum(self, symbol: str, swap_symbol: str, amount: float) -> float:
        """Ensure *amount* meets the minimum of 1 contract for contract-based markets.

        Returns the adjusted amount (rounded up to 1 if needed, and rounded to a
        whole integer for contract markets).  Previously this raised a *ValueError*;
        now it corrects the amount in-place so the order can still be placed.
        """
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        if market_info.get("contract"):
            if amount < 1.0:
                logger.warning(
                    "Order size ({}) is less than 1 contract for {}. "
                    "Rounding up to minimum 1 contract.",
                    amount, swap_symbol,
                )
                return 1.0
            return float(int(round(amount)))  # whole integer, kept as float for type consistency
        return amount

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a market order for *amount* contracts on *symbol*.

        .. deprecated::
            Use the Rust ``GateIoGateway`` (``trading_engine`` binary) for
            live trading. This Python path is retained for back-testing only.
        """
        warnings.warn(
            "GateIOClient.create_market_order is deprecated for live trading. "
            "Use the Rust GateIoGateway (gateio_gateway.rs) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map precious metals symbols first
        symbol = self._resolve_precious_metals_symbol(symbol)
        await self._rate_limiter.acquire()
        # FIX: Ensure we hit the Futures API by mapping the symbol
        swap_symbol = self._resolve_swap_symbol(symbol)
        # Markets are pre-loaded in connect() and refreshed by _refresh_markets_loop()
        # For contract markets, Gate.io requires WHOLE INTEGER contracts.
        # amount_to_precision can produce fractional values (e.g. 0.4) which
        # the exchange rejects, so we skip it for contract markets.
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        is_contract = bool(market_info.get("contract", False))
        if is_contract:
            amount = max(1, int(round(amount)))
            logger.debug("Contract market {}: using {} whole contracts", swap_symbol, amount)
        else:
            try:
                amount = float(self._client.amount_to_precision(swap_symbol, amount))
            except Exception as exc:
                logger.debug("amount_to_precision failed for {} ({}); using raw amount", swap_symbol, exc)
        if amount <= 0:
            raise ValueError(
                f"Order amount for {symbol} is 0 after applying exchange precision. "
                "Position size is too small to purchase a single contract."
            )
        amount = self._validate_contract_minimum(symbol, swap_symbol, amount)
        order_params = await self._apply_leverage_from_params(swap_symbol, params)
        raw = await self._client.create_market_order(swap_symbol, side.value, amount, params=order_params)
        logger.info("Market order placed: {} {} {} @ market", side.value, amount, swap_symbol)
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
        """Place a limit order for *amount* contracts at *price*.

        .. deprecated::
            Use the Rust ``GateIoGateway`` for live trading.
        """
        warnings.warn(
            "GateIOClient.create_limit_order is deprecated for live trading. "
            "Use the Rust GateIoGateway (gateio_gateway.rs) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map precious metals symbols first
        symbol = self._resolve_precious_metals_symbol(symbol)
        await self._rate_limiter.acquire()
        # FIX: Ensure we hit the Futures API by mapping the symbol
        swap_symbol = self._resolve_swap_symbol(symbol)
        # Markets are pre-loaded in connect() and refreshed by _refresh_markets_loop()
        # For contract markets, skip amount_to_precision to avoid fractional values.
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        is_contract = bool(market_info.get("contract", False))
        if is_contract:
            amount = max(1, int(round(amount)))
            logger.debug("Contract market {}: using {} whole contracts", swap_symbol, amount)
        else:
            try:
                amount = float(self._client.amount_to_precision(swap_symbol, amount))
            except Exception as exc:
                logger.debug("amount_to_precision failed for {} ({}); using raw amount", swap_symbol, exc)
        if amount <= 0:
            raise ValueError(
                f"Order amount for {symbol} is 0 after applying exchange precision. "
                "Position size is too small to purchase a single contract."
            )
        amount = self._validate_contract_minimum(symbol, swap_symbol, amount)
        order_params = await self._apply_leverage_from_params(swap_symbol, params)
        raw = await self._client.create_limit_order(
            swap_symbol, side.value, amount, price, params=order_params
        )
        logger.info("Limit order placed: {} {} {} @ {}", side.value, amount, swap_symbol, price)
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
        """Place a stop-market order triggered at *stop_price*.

        .. deprecated::
            Use the Rust ``GateIoGateway`` for live trading.

        Uses Gate.io price-triggered orders with ``reduceOnly=True`` so the SL
        order can only close an existing position (never open a new one).
        Falls back to the plain ``"stop_market"`` order type if the first attempt fails.

        Gate.io trigger rules:
          rule=1 → trigger when price <= stop_price (SELL SL closing a LONG)
          rule=2 → trigger when price >= stop_price (BUY SL closing a SHORT)
        """
        warnings.warn(
            "GateIOClient.create_stop_loss_order is deprecated for live trading. "
            "Use the Rust GateIoGateway (gateio_gateway.rs) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        original_symbol = symbol
        symbol = self._resolve_precious_metals_symbol(symbol)
        await self._rate_limiter.acquire()
        swap_symbol = self._resolve_swap_symbol(symbol)
        # Ensure whole integer contracts for contract markets
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        if market_info.get("contract"):
            amount = max(1, int(round(amount)))

        # Fetch current price to validate trigger direction and set initial_price.
        current_price: float = 0.0
        try:
            ticker = await self.get_ticker(symbol)
            current_price = ticker.last
        except Exception as _ticker_exc:
            logger.debug("create_stop_loss_order: ticker fetch failed {}, skipping price validation", _ticker_exc)

        # Validate stop_price direction and set correct Gate.io rule.
        if side == OrderSide.SELL:
            # Closing a LONG: SL must be BELOW current price.
            # rule=1 → Gate.io triggers when price drops to/below stop_price.
            if current_price > 0 and stop_price >= current_price:
                logger.warning(
                    "SL price {:.4f} >= current {:.4f} for SELL — adjusting to 0.3% below",
                    stop_price, current_price,
                )
                stop_price = current_price * 0.997
            rule = 1
        else:
            # OrderSide.BUY — Closing a SHORT: SL must be ABOVE current price.
            # rule=2 → Gate.io triggers when price rises to/above stop_price.
            if current_price > 0 and stop_price <= current_price:
                logger.warning(
                    "SL price {:.4f} <= current {:.4f} for BUY — adjusting to 0.3% above",
                    stop_price, current_price,
                )
                stop_price = current_price * 1.003
            rule = 2

        effective_initial_price = str(current_price) if current_price > 0 else str(stop_price)
        p = {
            **params,
            "stopPrice": stop_price,
            "reduceOnly": True,
            "triggerPrice": stop_price,
            "rule": rule,
            "initial_price": effective_initial_price,
            "expiration": 0,
        }
        try:
            # Use type="market" + triggerPrice to bypass CCXT's broken type mapping
            raw = await self._client.create_order(
                swap_symbol, "market", side.value, amount, None, p
            )
        except Exception as e:
            logger.debug("First SL attempt failed: {} — retrying with stop_market type", e)
            raw = await self._client.create_order(
                swap_symbol, "stop_market", side.value, amount, stop_price, p
            )
        logger.info(
            "Stop-loss placed: {} {} {} trigger={:.4f} rule={}",
            side.value, amount, swap_symbol, stop_price, rule,
        )
        order = self._parse_order(raw)
        return order.model_copy(update={"symbol": original_symbol})

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a take-profit market order triggered at *tp_price*.

        .. deprecated::
            Use the Rust ``GateIoGateway`` for live trading.

        Gate.io price-triggered order direction rules:
          rule=1 → trigger when price <= trigger price
                   (used for BUY TP orders that close a SHORT position,
                    i.e. trigger when price FALLS to TP)
          rule=2 → trigger when price >= trigger price
                   (used for SELL TP orders that close a LONG position,
                    i.e. trigger when price RISES to TP)

        Uses ``type="market"`` with ``triggerPrice`` to bypass CCXT's broken
        ``"take_profit"`` type mapping that overrides the Gate.io ``rule``
        parameter and causes ``AUTO_TRIGGER_PRICE_GREATE_LAST`` rejections.
        """
        warnings.warn(
            "GateIOClient.create_take_profit_order is deprecated for live trading. "
            "Use the Rust GateIoGateway (gateio_gateway.rs) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        original_symbol = symbol
        # Map precious metals symbols first
        symbol = self._resolve_precious_metals_symbol(symbol)
        await self._rate_limiter.acquire()
        swap_symbol = self._resolve_swap_symbol(symbol)
        # Ensure whole integer contracts for contract markets
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        if market_info.get("contract"):
            amount = max(1, int(round(amount)))

        # Fetch current price to validate trigger direction and set initial_price.
        current_price: float = 0.0
        try:
            ticker = await self.get_ticker(symbol)
            current_price = ticker.last
        except Exception as _ticker_exc:
            logger.debug("create_take_profit_order: ticker fetch failed {}, skipping price validation", _ticker_exc)

        # Validate tp_price direction and set correct Gate.io rule.
        if side == OrderSide.SELL:
            # Closing a LONG: TP must be ABOVE current price.
            # rule=2 → trigger when price rises to/above tp_price.
            if current_price > 0 and tp_price <= current_price:
                logger.warning(
                    "TP price {:.4f} <= current {:.4f} for SELL — price has moved past TP level. Skipping.",
                    tp_price, current_price,
                )
                raise ValueError(
                    f"TP price {tp_price:.4f} is no longer valid: current price {current_price:.4f} "
                    f"has moved past the take-profit level for a LONG position."
                )
            rule = 2
        else:
            # OrderSide.BUY — Closing a SHORT: TP must be BELOW current price.
            # rule=1 → trigger when price drops to/below tp_price.
            if current_price > 0 and tp_price >= current_price:
                logger.warning(
                    "TP price {:.4f} >= current {:.4f} for BUY — price has moved past TP level. Skipping.",
                    tp_price, current_price,
                )
                raise ValueError(
                    f"TP price {tp_price:.4f} is no longer valid: current price {current_price:.4f} "
                    f"has moved past the take-profit level for a SHORT position."
                )
            rule = 1

        effective_initial_price = str(current_price) if current_price > 0 else str(tp_price)
        p = {
            **params,
            "triggerPrice": tp_price,
            "stopPrice": tp_price,
            "rule": rule,
            "initial_price": effective_initial_price,
            "reduceOnly": True,
            "expiration": 0,
        }
        try:
            # Use type="market" + triggerPrice to bypass CCXT's broken "take_profit"
            # type mapping which overrides the rule and causes Gate.io rejections.
            raw = await self._client.create_order(
                swap_symbol, "market", side.value, amount, None, p
            )
        except Exception as e:
            err_str = str(e)
            # Gate.io error 1029/1026: trigger price violates direction rule.
            # The TP price is fundamentally invalid — don't retry with stale price.
            if "1029" in err_str or "AUTO_TRIGGER_PRICE" in err_str or "1026" in err_str:
                raise ValueError(
                    f"Gate.io rejected TP order: {e}. "
                    f"TP price {tp_price} is invalid relative to current market price {current_price}."
                ) from e
            logger.debug("First TP attempt failed: {} — retrying with stop_market type", e)
            raw = await self._client.create_order(
                swap_symbol, "stop_market", side.value, amount, tp_price, p
            )
        logger.info(
            "Take-profit placed: {} {} {} trigger={:.4f} rule={}",
            side.value, amount, swap_symbol, tp_price, rule,
        )
        order = self._parse_order(raw)
        return order.model_copy(update={"symbol": original_symbol})

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a single open order.

        Returns a ``{"id": order_id, "status": "not_found"}`` dict (without
        raising) when Gate.io responds with ORDER_NOT_FOUND so that callers
        do not trigger unnecessary retries for already-filled or never-placed
        orders.
        """
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        await self._rate_limiter.acquire()
        try:
            result = await self._client.cancel_order(order_id, symbol)
            logger.info("Order {} cancelled on {}", order_id, symbol)
            return result
        except Exception as exc:
            err_str = str(exc)
            if "ORDER_NOT_FOUND" in err_str or "order not found" in err_str.lower():
                logger.warning(
                    "Order {} not found on {} (already filled or cancelled) — skipping cancel",
                    order_id, symbol,
                )
                return {"id": order_id, "status": "not_found"}
            raise

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for *symbol*."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        await self._rate_limiter.acquire()
        result = await self._client.cancel_all_orders(symbol)
        logger.info("All orders cancelled on {}", symbol)
        return result if isinstance(result, list) else []

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch the current state of an order by its ID."""
        # Map precious metals symbols first
        symbol = self._resolve_precious_metals_symbol(symbol)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_order(order_id, symbol)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all open orders, optionally filtered by *symbol*.

        Gate.io separates regular orders from price-triggered orders
        (stop-loss / take-profit conditional orders).  CCXT's
        ``fetch_open_orders`` only returns regular orders by default; passing
        ``params={'stop': True}`` fetches the price-triggered orders
        (``/futures/usdt/price_orders``).  Both sets are merged so that callers
        — in particular the watchdog — can reliably detect whether a position
        already has a stop-loss attached.
        """
        # Map precious metals symbols first if symbol provided
        if symbol:
            symbol = self._resolve_precious_metals_symbol(symbol)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_open_orders(symbol)
        # Also fetch price-triggered orders (SL/TP conditional orders on Gate.io).
        # These live under /futures/usdt/price_orders and are NOT included in the
        # default fetch_open_orders response.
        try:
            await self._rate_limiter.acquire()
            trigger_raw = await self._client.fetch_open_orders(symbol, params={"stop": True})
            raw = raw + trigger_raw
        except Exception as exc:
            logger.debug("Could not fetch trigger orders (non-critical): {}", exc)
        return [self._parse_order(o) for o in raw]

    # ------------------------------------------------------------------
    # Position & leverage management
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for *symbol* using CCXT or native Gate.io API as fallback."""
        # Map precious metals symbols first
        symbol = self._resolve_precious_metals_symbol(symbol)
        swap_symbol = self._resolve_swap_symbol(symbol)
        await self._rate_limiter.acquire()
        try:
            result = await self._client.set_leverage(leverage, swap_symbol)
            logger.info("Leverage set to {}x on {}", leverage, swap_symbol)
            return result
        except Exception as e:
            exc_str = str(e).lower()
            if "not support" in exc_str or "not implemented" in exc_str:
                # Fallback: use Gate.io's native REST API
                try:
                    settle = "usdt"
                    contract = self._gateio_contract(swap_symbol)
                    result = await self._client.privateFuturesPostSettlePositionsContractLeverage(
                        {"settle": settle, "contract": contract, "leverage": str(leverage)}
                    )
                    logger.info("Leverage set to {}x on {} via native API", leverage, swap_symbol)
                    return result
                except Exception as native_exc:
                    logger.warning(
                        "Native leverage API also failed for {}: {}", swap_symbol, native_exc
                    )
                    raise
            raise

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Switch between cross / isolated margin for *symbol*.

        Gate.io does NOT support CCXT's ``setMarginMode()``.  The margin mode is
        configured at the account level in Gate.io settings.  We attempt the call
        once (no retries) and skip gracefully if the exchange returns a
        "not supported" error, avoiding the previous 12-second retry storm.
        """
        symbol = self._resolve_precious_metals_symbol(symbol)
        swap_symbol = self._resolve_swap_symbol(symbol)
        await self._rate_limiter.acquire()
        try:
            result = await self._client.set_margin_mode(margin_type.value, swap_symbol)
            logger.info("Margin type set to {} on {}", margin_type.value, swap_symbol)
            return result
        except Exception as e:
            exc_str = str(e).lower()
            if any(kw in exc_str for kw in ("not support", "not implemented", "not need", "already")):
                logger.debug(
                    "Gate.io does not support setMarginMode via CCXT — margin mode must be "
                    "configured in Gate.io account settings. Skipping for {}.", swap_symbol
                )
                return {"symbol": swap_symbol, "marginType": margin_type.value, "skipped": True}
            raise

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_positions(self) -> List[Position]:
        """Return all non-zero open positions."""
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_positions()
        return [self._parse_position(p) for p in raw if float(p.get("contracts") or 0) != 0]

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for *symbol*, or *None* if flat."""
        # Map precious metals symbols first
        symbol = self._resolve_precious_metals_symbol(symbol)
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

    async def add_margin(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Add margin to an existing isolated position for *symbol*.

        Uses Gate.io's native futures REST API
        ``POST /api/v4/futures/{settle}/positions/{contract}/margin``
        to top-up the isolated margin.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            amount: USDT margin to add (must be > 0).

        Returns:
            dict with ``symbol`` and ``added_margin`` keys.
        """
        symbol = self._resolve_precious_metals_symbol(symbol)
        swap_symbol = self._resolve_swap_symbol(symbol)
        await self._rate_limiter.acquire()
        settle = "usdt"
        contract = self._gateio_contract(swap_symbol)
        try:
            result = await self._client.privateFuturesPostSettlePositionsContractMargin(
                {"settle": settle, "contract": contract, "change": str(amount)}
            )
            logger.info("Added {} USDT margin to {}", amount, swap_symbol)
            return {"symbol": swap_symbol, "added_margin": amount, "raw": result}
        except Exception as exc:
            logger.error("Failed to add margin to {}: {}", swap_symbol, exc)
            raise

    # ------------------------------------------------------------------
    # Derivatives-specific data
    # ------------------------------------------------------------------

    def _resolve_precious_metals_symbol(self, symbol: str) -> str:
        """Map XAU/USDT and XAG/USDT to their CCXT-accessible equivalents.

        Gate.io does NOT have native XAU/USDT or XAG/USDT on their standard
        futures/swap API.  The XAU/USDT perpetual visible in the Gate.io app
        is on their TradFi (MT5) platform, not the standard futures API.

        We map to XAUT/USDT (Tether Gold), which IS available as a perpetual
        contract on Gate.io's standard futures API (XAUT/USDT:USDT).  Tether
        Gold is backed 1:1 by physical gold and closely tracks the spot price.

        This mapping must be applied BEFORE _resolve_swap_symbol() is called.
        The original symbol is preserved in all response objects so the rest
        of the bot logic still sees "XAU/USDT".

        Args:
            symbol: User-provided symbol (e.g., "XAU/USDT", "BTC/USDT")

        Returns:
            Mapped symbol (e.g. "XAUT/USDT" for "XAU/USDT") or the original
            symbol if no mapping exists.
        """
        if symbol in self._permanently_unavailable_symbols:
            raise SymbolPermanentlyUnavailableError(symbol)
        mapped = self.PRECIOUS_METALS_MAPPING.get(symbol)
        if mapped:
            logger.debug("Precious metals mapping: {} -> {}", symbol, mapped)
            return mapped
        return symbol

    def _validate_precious_metals_availability(self) -> None:
        """Check each precious-metals mapping against loaded markets.

        Called once after :meth:`connect` loads markets.  Logs availability
        status but does NOT permanently disable symbols — the symbol may still
        work with a slightly different key format or become available later.
        """
        markets = getattr(self._client, "markets", None)
        if not markets:
            # Markets not loaded yet — skip validation to avoid false positives
            logger.debug("_validate_precious_metals_availability: markets not loaded, skipping.")
            return
        for original, mapped in self.PRECIOUS_METALS_MAPPING.items():
            quote = mapped.split("/")[-1] if "/" in mapped else ""
            swap_candidate = f"{mapped}:{quote}" if quote else mapped
            # Check multiple possible key formats Gate.io / CCXT may use
            found = (
                swap_candidate in markets
                or mapped in markets
                or mapped.replace("/", "_") in markets  # Gate.io native format
            )
            if not found:
                # Don't permanently disable — just warn. The symbol might become
                # available later or work with a different format on first use.
                logger.warning(
                    "Symbol {} (mapped → {}) not found in initial market load. "
                    "Gold trading may not be available. Will retry on first use.",
                    original,
                    swap_candidate,
                )
                # DO NOT add to _permanently_unavailable_symbols here.
            else:
                logger.info(
                    "Precious metals symbol {} → {} confirmed in exchange markets.",
                    original,
                    swap_candidate,
                )

    def _resolve_swap_symbol(self, symbol: str) -> str:
        """Return the fully-resolved Gate.io swap market symbol for *symbol*.

        Applies precious-metals mapping first (e.g. XAU/USDT → XAUT/USDT) so
        that a single call to this method is sufficient at every API boundary.
        Then aggressively converts to the CCXT perpetual standard format
        (e.g. XAUT/USDT → XAUT/USDT:USDT).

        Gate.io ``fetchFundingRate`` only works with perpetual swap symbols
        (e.g. ``BTC/USDT:USDT``), not spot symbols (``BTC/USDT``).  This
        helper maps a spot symbol to the corresponding swap symbol.

        Args:
            symbol: Market symbol, e.g. ``"XAU/USDT"``, ``"BTC/USDT"``, or
                ``"BTC/USDT:USDT"``.

        Returns:
            Fully resolved swap symbol, e.g. ``"XAUT/USDT:USDT"`` or
            ``"BTC/USDT:USDT"``.
        """
        # Step 1 — precious metals mapping (XAU/USDT → XAUT/USDT)
        symbol = self._resolve_precious_metals_symbol(symbol)
        # Step 2 — aggressively force CCXT perpetual standard format
        if ":" not in symbol and "/" in symbol:
            quote = symbol.split("/")[-1]
            candidate = f"{symbol}:{quote}"
            # Validate against loaded markets — fall back gracefully if not found
            markets = getattr(self._client, "markets", None)
            if markets is not None and candidate in markets:
                return candidate
            if markets is not None and symbol in markets:
                return symbol
            # Return the candidate and let the caller handle the error
            return candidate
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

    @resolve_symbols
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time ticker updates for *symbol*."""
        # symbol is fully resolved by @resolve_symbols — pass to the WS loop
        asyncio.create_task(self._ws_ticker_loop(symbol, callback))

    @resolve_symbols
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time order-book snapshots for *symbol*."""
        # symbol is fully resolved by @resolve_symbols — pass to the WS loop
        asyncio.create_task(self._ws_orderbook_loop(symbol, callback))

    @resolve_symbols
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to the real-time public trade feed for *symbol*."""
        # symbol is fully resolved by @resolve_symbols — pass to the WS loop
        asyncio.create_task(self._ws_trades_loop(symbol, callback))

    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to private account-update events."""
        asyncio.create_task(self._ws_user_data_loop(callback))

    async def start_user_data_stream(self) -> None:
        """Start the WebSocket user-data stream.

        Subscribes to futures.orders, futures.balances, and futures.positions
        channels.  Parses incoming messages and updates the internal
        ``_ws_positions`` and ``_ws_orders`` caches in real-time.
        Emits events via the EventBus when positions or orders change.
        Reconnects automatically with exponential backoff on failure.
        """
        asyncio.create_task(self._ws_full_user_data_loop())

    def get_positions_from_ws(self) -> Dict[str, Any]:
        """Return the WebSocket-cached positions dict (zero latency).

        Returns:
            Dict mapping symbol -> position dict, or empty dict if no WS data.
        """
        return dict(self._ws_positions)

    def get_positions_from_rust(self) -> Dict[str, Any]:
        """Return positions from the Rust engine's ZeroMQ telemetry state cache.

        Queries the last ``microstructure`` and ``fill`` telemetry messages
        published by the Rust ``trading_engine`` binary via ZeroMQ and
        constructs a position-like snapshot.  This is a **zero-latency** read
        from an in-process cache — no REST call is issued.

        Requires ``pyzmq`` and the Rust binary to be running.  Falls back
        transparently to :meth:`get_positions_from_ws` if Rust telemetry is
        not available.

        Returns:
            Dict mapping symbol → position dict, compatible with the
            ``_ws_positions`` format returned by :meth:`get_positions_from_ws`.
        """
        try:
            import zmq

            ctx = zmq.Context.instance()
            # Use a DEALER socket to do a single non-blocking REQ to the telemetry feed
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"fill")
            sock.setsockopt(zmq.RCVTIMEO, 200)  # 200 ms timeout
            sock.connect("tcp://127.0.0.1:5555")

            positions: Dict[str, Any] = {}
            deadline = time.monotonic() + 0.2
            try:
                while time.monotonic() < deadline:
                    try:
                        raw = sock.recv_string(flags=zmq.NOBLOCK)
                        space_idx = raw.find(" ")
                        if space_idx == -1:
                            continue
                        payload_str = raw[space_idx + 1:]
                        payload = json.loads(payload_str)
                        symbol = payload.get("symbol", "")
                        if symbol:
                            positions[symbol] = payload
                    except zmq.Again:
                        break
            finally:
                sock.close()

            if positions:
                return positions

        except Exception as exc:
            logger.debug("get_positions_from_rust: ZMQ unavailable ({}), falling back to WS cache.", exc)

        # Fallback to WebSocket cache
        return self.get_positions_from_ws()

    def is_ws_data_fresh(self, max_stale_seconds: float = 30.0) -> bool:
        """Return True if WebSocket data has been updated within max_stale_seconds."""
        import time as _time
        return (self._ws_last_update > 0
                and (_time.time() - self._ws_last_update) < max_stale_seconds)

    def register_fill_waiter(self, order_id: str) -> "asyncio.Future[Any]":
        """Register a Future that will be resolved when order_id is filled via WebSocket.

        Args:
            order_id: The exchange order ID to watch for a fill event.

        Returns:
            An :class:`asyncio.Future` that is resolved with the raw order data
            dict when a ``finished`` or ``cancelled`` WS event arrives for this
            order, or raises :class:`asyncio.TimeoutError` if wrapped with
            :func:`asyncio.wait_for`.
        """
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Any]" = loop.create_future()
        self._pending_fills[order_id] = future
        return future

    # ------------------------------------------------------------------
    # WebSocket loops (private)
    # ------------------------------------------------------------------

    def _gateio_contract(self, symbol: str) -> str:
        """Convert a ccxt symbol such as ``BTC/USDT:USDT`` to Gate.io contract format ``BTC_USDT``."""
        base = symbol.split(":")[0]
        return base.replace("/", "_")

    def _ws_auth_payload(self, channel: str, event: str) -> Dict[str, Any]:
        """Build a Gate.io WebSocket authentication payload."""
        ts = int(time.time())
        message = f"channel={channel}&event={event}&time={ts}"
        sign = hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha512).hexdigest()
        return {
            "time": ts,
            "channel": channel,
            "event": event,
            "auth": {"method": "api_key", "KEY": self.api_key, "SIGN": sign},
        }

    async def _ws_ticker_loop(self, symbol: str, callback: Callable) -> None:
        # symbol should already be fully resolved when called via subscribe_ticker
        # (which is decorated with @resolve_symbols).  The precious-metals step
        # below is a belt-and-suspenders guard for any direct callers.
        mapped_symbol = self._resolve_precious_metals_symbol(symbol)
        contract = self._gateio_contract(mapped_symbol)
        subscribe_msg = {
            "time": int(time.time()),
            "channel": "futures.tickers",
            "event": "subscribe",
            "payload": [contract],
        }
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    async for raw_msg in ws:
                        if _USE_RUST_WS_PARSER:
                            raw_bytes = raw_msg if isinstance(raw_msg, (bytes, bytearray)) else raw_msg.encode()
                            try:
                                data = _rust_parse_ws_message(raw_bytes)
                            except Exception:
                                data = json.loads(raw_msg)
                        else:
                            data = json.loads(raw_msg)
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
        # Belt-and-suspenders: ensure precious metals are mapped even for direct callers.
        mapped_symbol = self._resolve_precious_metals_symbol(symbol)
        contract = self._gateio_contract(mapped_symbol)
        subscribe_msg = {
            "time": int(time.time()),
            "channel": "futures.order_book",
            "event": "subscribe",
            "payload": [contract, "20", "0"],
        }
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    async for raw_msg in ws:
                        if _USE_RUST_WS_PARSER:
                            raw_bytes = raw_msg if isinstance(raw_msg, (bytes, bytearray)) else raw_msg.encode()
                            try:
                                data = _rust_parse_ws_message(raw_bytes)
                            except Exception:
                                data = json.loads(raw_msg)
                        else:
                            data = json.loads(raw_msg)
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
        # Belt-and-suspenders: ensure precious metals are mapped even for direct callers.
        mapped_symbol = self._resolve_precious_metals_symbol(symbol)
        contract = self._gateio_contract(mapped_symbol)
        subscribe_msg = {
            "time": int(time.time()),
            "channel": "futures.trades",
            "event": "subscribe",
            "payload": [contract],
        }
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    async for raw_msg in ws:
                        if _USE_RUST_WS_PARSER:
                            raw_bytes = raw_msg if isinstance(raw_msg, (bytes, bytearray)) else raw_msg.encode()
                            try:
                                data = _rust_parse_ws_message(raw_bytes)
                            except Exception:
                                data = json.loads(raw_msg)
                        else:
                            data = json.loads(raw_msg)
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
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    # Subscribe to orders and balance updates with auth
                    for channel in ("futures.orders", "futures.balances"):
                        auth_payload = self._ws_auth_payload(channel, "subscribe")
                        await ws.send(json.dumps(auth_payload))
                    async for raw_msg in ws:
                        data = json.loads(raw_msg)
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

    async def _ws_full_user_data_loop(self) -> None:
        """Full user-data WebSocket loop: positions + orders + balances with caching."""
        backoff = _WS_RECONNECT_DELAY
        while True:
            try:
                async with websockets.connect(_WS_URL) as ws:
                    backoff = _WS_RECONNECT_DELAY  # reset on successful connect
                    channels = ("futures.orders", "futures.balances", "futures.positions")
                    for channel in channels:
                        auth_payload = self._ws_auth_payload(channel, "subscribe")
                        await ws.send(json.dumps(auth_payload))
                    logger.info("GateIOClient: user-data WS stream connected ({})", channels)
                    async for raw_msg in ws:
                        try:
                            data = json.loads(raw_msg)
                            await self._handle_ws_user_data(data)
                        except Exception as parse_exc:
                            logger.debug("WS user-data parse error: {}", parse_exc)
            except Exception as exc:
                logger.warning(
                    "WS full user-data error: {} — reconnecting in {}s",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # exponential backoff up to 60s

    async def _handle_ws_user_data(self, data: Dict[str, Any]) -> None:
        """Parse a WebSocket user-data message and update caches."""
        channel = data.get("channel", "")
        event = data.get("event", "")
        result = data.get("result")

        if event not in ("update", "subscribe") or result is None:
            return

        self._ws_last_update = time.time()

        if channel == "futures.positions" and isinstance(result, list):
            for pos_data in result:
                symbol = pos_data.get("contract", "").replace("_", "/")
                if symbol:
                    # Normalize to ccxt swap format
                    if "/" in symbol and ":" not in symbol:
                        symbol = f"{symbol}:{symbol.split('/')[-1]}"
                    contracts = float(pos_data.get("size", 0))
                    if contracts == 0:
                        self._ws_positions.pop(symbol, None)
                    else:
                        self._ws_positions[symbol] = pos_data
                    # Emit event if event bus is available
                    if self._event_bus is not None:
                        try:
                            await self._event_bus.emit(
                                "position_update",
                                {"symbol": symbol, "data": pos_data, "source": "ws"},
                            )
                        except Exception:
                            pass

        elif channel == "futures.orders" and isinstance(result, list):
            for order_data in result:
                order_id = str(order_data.get("id", ""))
                status = order_data.get("status", "")
                if order_id:
                    if status in ("finished", "cancelled"):
                        self._ws_orders.pop(order_id, None)
                        # Resolve pending fill future (if any waiter registered)
                        fill_future = self._pending_fills.pop(order_id, None)
                        if fill_future is not None and not fill_future.done():
                            fill_future.set_result(order_data)
                        # SL/TP fill detected: emit event
                        if self._event_bus is not None:
                            try:
                                await self._event_bus.emit(
                                    "order_fill",
                                    {"order_id": order_id, "data": order_data, "source": "ws"},
                                )
                            except Exception:
                                pass
                    else:
                        self._ws_orders[order_id] = order_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_order(self, raw: Dict[str, Any]) -> Order:
        """Convert a ccxt order dict into a normalised :class:`Order`."""
        raw_type = (raw.get("type") or "market").lower()
        type_map = {
            "stop": "stop_loss",
            "take_profit": "take_profit",
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

        entry_price = float(raw.get("entryPrice") or 0)
        amount = abs(float(raw.get("contracts") or 0))
        leverage = int(raw.get("leverage") or 1)
        # Use initialMargin from exchange; fall back to computed value when missing/zero.
        margin = float(raw.get("initialMargin") or 0)
        if margin == 0 and entry_price > 0 and amount > 0 and leverage > 0:
            contract_size = float(raw.get("contractSize") or 1.0)
            margin = (entry_price * amount * contract_size) / leverage

        unrealized_pnl = float(raw.get("unrealizedPnl") or 0)
        mark_price = float(raw.get("markPrice") or 0)
        # Compute roe_pct and position_value when not supplied by the exchange.
        roe_pct = (unrealized_pnl / margin * 100.0) if margin > 0 else 0.0
        contract_size_val = float(raw.get("contractSize") or 1.0)
        position_value = entry_price * amount * contract_size_val

        return Position(
            symbol=raw.get("symbol") or "",
            side=side,
            amount=amount,
            entry_price=entry_price,
            current_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            leverage=leverage,
            margin=margin,
            liquidation_price=float(raw.get("liquidationPrice") or 0),
            timestamp=int(raw.get("timestamp") or 0),
            mark_price=mark_price,
            roe_pct=roe_pct,
            position_value=position_value,
        )

    # ------------------------------------------------------------------
    # Gate.io-specific advanced order types
    # ------------------------------------------------------------------

    async def create_iceberg_order(
        self,
        symbol: str,
        side: OrderSide,
        total_amount: float,
        visible_amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> List[Order]:
        """Split a large order into iceberg (hidden-size) chunks.

        Places *ceil(total_amount / visible_amount)* limit orders of
        *visible_amount* each at *price*, hiding the true order size
        from the public order book.

        Args:
            symbol          : Trading pair, e.g. "BTC/USDT".
            side            : BUY or SELL.
            total_amount    : Full order size in contracts.
            visible_amount  : Visible chunk size per slice (≥ 1 contract).
            price           : Limit price for all slices.
            params          : Extra CCXT params forwarded to each slice.

        Returns:
            List of :class:`Order` objects, one per placed slice.
        """
        symbol = self._resolve_precious_metals_symbol(symbol)
        swap_symbol = self._resolve_swap_symbol(symbol)
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        is_contract = bool(market_info.get("contract", False))

        # Normalise to whole contracts when applicable
        if is_contract:
            total_amount = max(1, int(round(total_amount)))
            visible_amount = max(1, int(round(visible_amount)))
        else:
            try:
                total_amount = float(self._client.amount_to_precision(swap_symbol, total_amount))
                visible_amount = float(
                    self._client.amount_to_precision(swap_symbol, visible_amount)
                )
            except Exception:
                pass

        placed_orders: List[Order] = []
        remaining = total_amount

        while remaining > 0:
            chunk = min(visible_amount, remaining)
            await self._rate_limiter.acquire()
            order_params = await self._apply_leverage_from_params(swap_symbol, dict(params))
            try:
                raw = await self._client.create_limit_order(
                    swap_symbol, side.value, chunk, price, params=order_params
                )
                placed_orders.append(self._parse_order(raw))
                logger.info(
                    "Iceberg slice: {} {} {} @ {} ({}/{} placed)",
                    side.value, chunk, swap_symbol, price,
                    sum(o.amount for o in placed_orders), total_amount,
                )
            except Exception as exc:
                logger.error("Iceberg slice failed for {}: {}", swap_symbol, exc)
                break
            remaining -= chunk

        return placed_orders

    async def create_post_only_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a post-only limit order (rejected by exchange if it would cross).

        Post-only orders guarantee maker execution and capture the Gate.io
        maker rebate.  If the order would immediately cross the spread the
        exchange cancels it rather than filling at taker rates.

        Args:
            symbol  : Trading pair.
            side    : BUY or SELL.
            amount  : Order size in contracts.
            price   : Limit price.
            params  : Extra CCXT params.

        Returns:
            Normalised :class:`Order`.
        """
        symbol = self._resolve_precious_metals_symbol(symbol)
        swap_symbol = self._resolve_swap_symbol(symbol)
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(swap_symbol, {}) or markets.get(symbol, {})
        is_contract = bool(market_info.get("contract", False))

        if is_contract:
            amount = max(1, int(round(amount)))
        else:
            try:
                amount = float(self._client.amount_to_precision(swap_symbol, amount))
            except Exception:
                pass

        if amount <= 0:
            raise ValueError(
                f"Post-only order amount for {symbol} is 0 after precision rounding."
            )

        await self._rate_limiter.acquire()
        order_params = await self._apply_leverage_from_params(swap_symbol, dict(params))
        # Gate.io post-only flag: timeInForce='POK' or ccxt normalised postOnly=True
        order_params.setdefault("postOnly", True)
        raw = await self._client.create_limit_order(
            swap_symbol, side.value, amount, price, params=order_params
        )
        logger.info("Post-only order placed: {} {} {} @ {}", side.value, amount, swap_symbol, price)
        return self._parse_order(raw)

    async def get_fee_tier(self) -> Dict[str, Any]:
        """Return current Gate.io fee tier and maker/taker rates.

        Queries the account's trading fee schedule.  On failure returns the
        default VIP 0 rates so callers always get a usable result.

        Returns:
            dict with keys: maker_fee, taker_fee, vip_level, raw
        """
        _default = {
            "maker_fee": -0.00025,
            "taker_fee": 0.00075,
            "vip_level": 0,
            "raw": {},
        }
        try:
            await self._rate_limiter.acquire()
            fees = await self._client.fetch_trading_fees()
            # ccxt returns a dict keyed by symbol or a top-level dict
            first = next(iter(fees.values())) if isinstance(fees, dict) and fees else fees
            if isinstance(first, dict):
                maker = float(first.get("maker", _default["maker_fee"]))
                taker = float(first.get("taker", _default["taker_fee"]))
            else:
                maker = _default["maker_fee"]
                taker = _default["taker_fee"]
            return {
                "maker_fee": maker,
                "taker_fee": taker,
                "vip_level": 0,
                "raw": first if isinstance(first, dict) else {},
            }
        except Exception as exc:
            logger.debug("get_fee_tier: could not fetch fees ({}); using defaults", exc)
            return _default

    @property
    def name(self) -> str:
        return "Gate.io"
