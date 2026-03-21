"""Generic CCXT-based exchange client.

Supports Gate.io and other exchanges via ``ccxt.async_support``.
Credentials are read from settings/environment variables.

.. deprecated::
    Direct order-placement methods (``create_market_order``,
    ``create_limit_order``, ``create_stop_loss_order``,
    ``create_take_profit_order``) are deprecated for live trading.
    Use the Rust ``trading_engine`` binary (``rust_engine/src/execution_gateway.rs``)
    which achieves <2 ms order submission latency with connection pooling,
    adaptive rate limiting, and exponential backoff.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Callable, Dict, List, Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from utils.circuit_breaker import CircuitBreakerOpenError, with_circuit_breaker
from utils.rate_limiter import ExchangeRateLimiter
from utils.retry import async_retry_decorator
from utils.symbol_resolver import original_symbol_ctx, resolve_symbols

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

# ---------------------------------------------------------------------------
# Supported exchange registry
# ---------------------------------------------------------------------------

#: Mapping of lowercase exchange name → (ccxt class, recommended rps, default options)
_EXCHANGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mexc": {
        "class": ccxt.mexc,
        "rps": 10.0,
        "options": {"defaultType": "swap"},
    },
    "gateio": {
        "class": ccxt.gateio,
        "rps": 10.0,
        "options": {
            "defaultType": "swap",
            "defaultMarginMode": "cross",
            "createMarketBuyOrderRequiresPrice": False,
        },
    },
    "bingx": {
        "class": ccxt.bingx,
        "rps": 5.0,
        "options": {"defaultType": "swap"},
    },
    "bitget": {
        "class": ccxt.bitget,
        "rps": 10.0,
        "options": {"defaultType": "swap"},
    },
}

SUPPORTED_EXCHANGES: List[str] = list(_EXCHANGE_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExchangeNotSupportedError(ValueError):
    """Raised when an unsupported exchange identifier is provided."""


class ExchangeAPIError(RuntimeError):
    """Raised when an exchange API call returns a non-retryable error."""


# ---------------------------------------------------------------------------
# CcxtExchange
# ---------------------------------------------------------------------------


class CcxtExchange(BaseExchange):
    """Concrete :class:`~.base_exchange.BaseExchange` backed by ``ccxt.async_support``.

    All abstract methods are implemented.  Every exchange interaction is logged
    via loguru and retried with exponential back-off on transient failures.

    Args:
        exchange_id: Lowercase exchange name, e.g. ``"mexc"``, ``"gateio"``.
        api_key: Exchange API key.
        secret_key: Exchange API secret.
        passphrase: Exchange passphrase (required by Bitget).
        testnet: When *True*, switches to sandbox/testnet mode.
    """

    # Gate.io does NOT have native XAU/USDT on their standard futures/swap API.
    # We map to XAUT/USDT (Tether Gold) which IS available as a perpetual contract.
    # Applied before passing symbol to CCXT; original symbol preserved in responses.
    PRECIOUS_METALS_MAPPING: Dict[str, str] = {
        "XAU/USDT": "XAUT/USDT",  # Tether Gold (1 XAUT = 1 troy oz gold)
        "XAG/USDT": "XAUT/USDT",  # No silver token; fallback to gold token
    }

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        secret_key: str,
        passphrase: str = "",
        testnet: bool = False,
    ) -> None:
        super().__init__(api_key, secret_key, passphrase, testnet)
        self._exchange_id = exchange_id.lower()
        if self._exchange_id not in _EXCHANGE_REGISTRY:
            raise ExchangeNotSupportedError(
                f"Unsupported exchange: {exchange_id!r}. "
                f"Supported exchanges: {SUPPORTED_EXCHANGES}"
            )
        cfg = _EXCHANGE_REGISTRY[self._exchange_id]
        self._exchange_class = cfg["class"]
        self._default_options: Dict[str, Any] = dict(cfg.get("options", {}))
        self._rate_limiter = ExchangeRateLimiter.get_limiter(self._exchange_id, rps=cfg["rps"])
        # Single ccxt.pro WebSocket client shared across all subscriptions (lazy-created)
        self._ws_client: Any = None

    # ------------------------------------------------------------------
    # BaseExchange properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable exchange name."""
        return self._exchange_id

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the underlying ccxt client and pre-load market metadata."""
        config: Dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
            "enableRateLimit": False,
            "options": self._default_options,
        }
        if self.passphrase:
            config["password"] = self.passphrase

        self._client = self._exchange_class(config)
        if self.testnet:
            self._client.set_sandbox_mode(True)

        await self._client.load_markets()
        logger.info(
            "CcxtExchange connected: exchange={} testnet={}",
            self._exchange_id,
            self.testnet,
        )

    async def disconnect(self) -> None:
        """Close all open HTTP sessions gracefully."""
        if self._client is not None:
            await self._client.close()
            logger.info("CcxtExchange disconnected: {}", self._exchange_id)

    # ------------------------------------------------------------------
    # Market data — REST
    # ------------------------------------------------------------------

    async def get_markets(self) -> Dict[str, Any]:
        """Return the loaded markets dict (symbol → market info).

        Returns the already-loaded markets from the CCXT client; re-loads them
        if they have not been fetched yet.
        """
        if self._client is None:
            return {}
        if not getattr(self._client, "markets", None):
            await self._client.load_markets()
        return dict(self._client.markets or {})

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_balance(self) -> Balance:
        """Fetch current account balance."""
        logger.debug("[{}] Fetching balance", self._exchange_id)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_balance()
        total = {k: float(v) for k, v in raw.get("total", {}).items() if v}
        free = {k: float(v) for k, v in raw.get("free", {}).items() if v}
        used = {k: float(v) for k, v in raw.get("used", {}).items() if v}
        balance = Balance(
            total=total,
            free=free,
            used=used,
            usdt_total=total.get("USDT", 0.0),
            usdt_free=free.get("USDT", 0.0),
        )
        logger.info(
            "[{}] Balance: usdt_total={:.2f} usdt_free={:.2f}",
            self._exchange_id,
            balance.usdt_total,
            balance.usdt_free,
        )
        return balance

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch the latest ticker for *symbol*."""
        # symbol is fully resolved by @resolve_symbols; original preserved in Ticker
        # by the decorator's post-processing step.
        logger.debug("[{}] Fetching ticker: {}", self._exchange_id, symbol)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_ticker(symbol)
        info = raw.get("info") or {}
        funding_rate_raw = info.get("fundingRate")
        return Ticker(
            symbol=symbol,
            bid=float(raw.get("bid") or 0),
            ask=float(raw.get("ask") or 0),
            last=float(raw.get("last") or 0),
            high=float(raw.get("high") or 0),
            low=float(raw.get("low") or 0),
            volume=float(raw.get("baseVolume") or 0),
            timestamp=int(raw.get("timestamp") or 0),
            funding_rate=float(funding_rate_raw) if funding_rate_raw is not None else None,
        )

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch the level-2 order book for *symbol*."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        logger.debug("[{}] Fetching orderbook: {} limit={}", self._exchange_id, symbol, limit)
        await self._rate_limiter.acquire()
        return await self._client.fetch_order_book(symbol, limit)

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Return OHLCV candles as a :class:`~pandas.DataFrame` indexed by time."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        logger.debug(
            "[{}] Fetching OHLCV: {} {} limit={}",
            self._exchange_id,
            symbol,
            timeframe,
            limit,
        )
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def _apply_leverage_from_params(self, symbol: str, params: Dict[str, Any]) -> None:
        """Call set_leverage on the CCXT client for *symbol* if *params* contains a
        ``"leverage"`` key and the market is a contract market.

        This ensures the exchange uses the correct multiplier before every order,
        preventing "Balance not enough" errors caused by Gate.io defaulting to 1x.
        Does nothing (and warns) if the call fails.
        """
        leverage = params.get("leverage")
        if not leverage:
            return
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(symbol, {})
        if not market_info.get("contract"):
            return
        swap_symbol = self._resolve_swap_symbol(symbol)
        try:
            if hasattr(self._client, "set_leverage"):
                await self._client.set_leverage(int(leverage), swap_symbol)
            elif hasattr(self._client, "set_margin_mode"):
                await self._client.set_margin_mode("cross", swap_symbol, {"leverage": int(leverage)})
        except Exception as e:
            logger.warning("[{}] Could not set leverage for {}: {}", self._exchange_id, swap_symbol, e)

    def _validate_contract_minimum(self, symbol: str, amount: float) -> None:
        """Raise *ValueError* if *amount* is below the minimum of 1 contract for
        contract-based markets (e.g. Gate.io perpetuals).
        """
        markets = getattr(self._client, "markets", None) or {}
        market_info = markets.get(symbol, {})
        if amount < 1.0 and market_info.get("contract"):
            raise ValueError(
                f"Order size ({amount}) is less than 1 contract. Gate.io requires whole integer contracts. "
                "Increase your USDT margin or leverage."
            )

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a market order.

        .. deprecated::
            Use the Rust execution gateway (``trading_engine`` binary) for
            live trading.  This Python path remains available for back-testing
            and paper trading only.

        The *amount* is automatically formatted to the exchange's precision for
        *symbol* (e.g. integer contracts for perpetual swaps) before being sent
        to the exchange API.

        If *params* contains ``"slippage_pct"`` (e.g. ``0.01`` for 1%), the
        order is converted to a marketable limit order placed ``slippage_pct``
        beyond the current mid-price.  This caps worst-case fill price while
        still guaranteeing execution in normal market conditions.
        """
        warnings.warn(
            "CcxtExchange.create_market_order is deprecated for live trading. "
            "Use the Rust execution_gateway (trading_engine binary) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        original_symbol = symbol
        symbol = self._resolve_precious_metals_symbol(symbol)
        # Apply exchange amount precision to prevent "order too small" errors
        try:
            amount = float(self._client.amount_to_precision(symbol, amount))
        except Exception as exc:
            logger.debug("[{}] amount_to_precision failed for {} ({}); using raw amount", self._exchange_id, symbol, exc)

        # Block zero-sized orders before they reach the exchange API
        if amount <= 0:
            raise ValueError(
                f"Order amount for {original_symbol} rounded to 0 after applying exchange precision. "
                "Position size is too small to purchase a single contract."
            )

        self._validate_contract_minimum(symbol, amount)
        await self._apply_leverage_from_params(symbol, params)

        # Slippage protection: convert to a marketable limit order when caller
        # specifies a maximum acceptable slippage percentage.
        slippage_pct = params.pop("slippage_pct", None) if isinstance(params, dict) else None
        if slippage_pct is not None:
            try:
                slippage_pct = float(slippage_pct)
                ticker_raw = await self._client.fetch_ticker(symbol)
                mid_price = float(ticker_raw.get("last") or ticker_raw.get("close") or 0)
                if mid_price > 0:
                    if side == OrderSide.BUY:
                        limit_price = mid_price * (1 + slippage_pct)
                    else:
                        limit_price = mid_price * (1 - slippage_pct)
                    limit_price = float(self._client.price_to_precision(symbol, limit_price))
                    logger.info(
                        "[{}] Slippage-limited limit order (slippage={:.2%}): {} {} {} @ {}",
                        self._exchange_id, slippage_pct, side.value, amount, symbol, limit_price,
                    )
                    await self._rate_limiter.acquire()
                    raw = await self._client.create_limit_order(
                        symbol, side.value, amount, limit_price, params=params
                    )
                    order = self._parse_order(raw)
                    order = order.model_copy(update={"symbol": original_symbol})
                    logger.info(
                        "[{}] Slippage-limited order placed: id={} status={}",
                        self._exchange_id, order.id, order.status.value,
                    )
                    return order
            except Exception as slip_exc:
                logger.warning(
                    "[{}] Slippage-limit order failed ({}); falling back to market order",
                    self._exchange_id, slip_exc,
                )

        logger.info(
            "[{}] Placing market order: {} {} {}",
            self._exchange_id,
            side.value,
            amount,
            symbol,
        )
        await self._rate_limiter.acquire()
        raw = await self._client.create_market_order(symbol, side.value, amount, params=params)
        order = self._parse_order(raw)
        order = order.model_copy(update={"symbol": original_symbol})
        logger.info(
            "[{}] Market order placed: id={} status={}",
            self._exchange_id,
            order.id,
            order.status.value,
        )
        return order

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a limit order at *price*.

        .. deprecated::
            Use the Rust execution gateway for live trading.

        The *amount* is automatically formatted to the exchange's precision for
        *symbol* before being sent to the exchange API.
        """
        warnings.warn(
            "CcxtExchange.create_limit_order is deprecated for live trading. "
            "Use the Rust execution_gateway (trading_engine binary) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        original_symbol = symbol
        symbol = self._resolve_precious_metals_symbol(symbol)
        # Apply exchange amount precision
        try:
            amount = float(self._client.amount_to_precision(symbol, amount))
        except Exception as exc:
            logger.debug("[{}] amount_to_precision failed for {} ({}); using raw amount", self._exchange_id, symbol, exc)

        # Block zero-sized orders before they reach the exchange API
        if amount <= 0:
            raise ValueError(
                f"Order amount for {original_symbol} rounded to 0 after applying exchange precision. "
                "Position size is too small to purchase a single contract."
            )

        self._validate_contract_minimum(symbol, amount)
        await self._apply_leverage_from_params(symbol, params)

        logger.info(
            "[{}] Placing limit order: {} {} {} @ {}",
            self._exchange_id,
            side.value,
            amount,
            symbol,
            price,
        )
        await self._rate_limiter.acquire()
        raw = await self._client.create_limit_order(
            symbol, side.value, amount, price, params=params
        )
        order = self._parse_order(raw)
        order = order.model_copy(update={"symbol": original_symbol})
        logger.info(
            "[{}] Limit order placed: id={} status={}",
            self._exchange_id,
            order.id,
            order.status.value,
        )
        return order

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
            Use the Rust execution gateway for live trading.
        """
        warnings.warn(
            "CcxtExchange.create_stop_loss_order is deprecated for live trading. "
            "Use the Rust execution_gateway (trading_engine binary) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        original_symbol = symbol
        symbol = self._resolve_precious_metals_symbol(symbol)
        logger.info(
            "[{}] Placing stop-loss: {} {} {} trigger={}",
            self._exchange_id,
            side.value,
            amount,
            symbol,
            stop_price,
        )
        await self._rate_limiter.acquire()
        # CCXT standard for stop-market orders: use type="market" with triggerPrice param
        p = {**params, "triggerPrice": stop_price, "reduceOnly": True}
        raw = await self._client.create_order(
            symbol, "market", side.value, amount, None, p
        )
        order = self._parse_order(raw)
        order = order.model_copy(update={"symbol": original_symbol})
        logger.info("[{}] Stop-loss placed: id={}", self._exchange_id, order.id)
        return order

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
            Use the Rust execution gateway for live trading.
        """
        warnings.warn(
            "CcxtExchange.create_take_profit_order is deprecated for live trading. "
            "Use the Rust execution_gateway (trading_engine binary) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        original_symbol = symbol
        symbol = self._resolve_precious_metals_symbol(symbol)
        logger.info(
            "[{}] Placing take-profit: {} {} {} trigger={}",
            self._exchange_id,
            side.value,
            amount,
            symbol,
            tp_price,
        )
        await self._rate_limiter.acquire()
        # CCXT standard for TP-market orders: use type="market" with triggerPrice param
        p = {**params, "triggerPrice": tp_price, "reduceOnly": True}
        raw = await self._client.create_order(
            symbol, "market", side.value, amount, None, p
        )
        order = self._parse_order(raw)
        order = order.model_copy(update={"symbol": original_symbol})
        logger.info("[{}] Take-profit placed: id={}", self._exchange_id, order.id)
        return order

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a single order by *order_id*."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        logger.info("[{}] Cancelling order: {} on {}", self._exchange_id, order_id, symbol)
        await self._rate_limiter.acquire()
        result = await self._client.cancel_order(order_id, symbol)
        logger.info("[{}] Order cancelled: {}", self._exchange_id, order_id)
        return result

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for *symbol*."""
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        logger.info("[{}] Cancelling all orders on {}", self._exchange_id, symbol)
        await self._rate_limiter.acquire()
        result = await self._client.cancel_all_orders(symbol)
        return result if isinstance(result, list) else []

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch the current state of a single order."""
        logger.debug("[{}] Fetching order: {} on {}", self._exchange_id, order_id, symbol)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_order(order_id, symbol)
        return self._parse_order(raw)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all open orders, optionally filtered by *symbol*."""
        logger.debug("[{}] Fetching open orders (symbol={})", self._exchange_id, symbol)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_open_orders(symbol)
        return [self._parse_order(o) for o in raw]

    async def get_trade_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return recent trade history for *symbol* (or all symbols)."""
        logger.debug(
            "[{}] Fetching trade history (symbol={} limit={})",
            self._exchange_id,
            symbol,
            limit,
        )
        await self._rate_limiter.acquire()
        return await self._client.fetch_my_trades(symbol, limit=limit)

    @with_circuit_breaker(failure_threshold=5)
    @async_retry_decorator(max_retries=3, base_delay=1.0)
    @resolve_symbols
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent public market trades for *symbol*.

        Unlike :meth:`get_trade_history` (which returns *your* trades via
        ``fetch_my_trades``), this fetches publicly visible market trades via
        ``fetch_trades`` and normalises each entry to a plain dict with keys
        ``id``, ``price``, ``amount``, ``side``, and ``timestamp``.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            limit: Maximum number of trades to return (default 50).

        Returns:
            List of trade dicts, newest first.
        """
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        logger.debug(
            "[{}] Fetching recent trades (symbol={} limit={})",
            self._exchange_id,
            symbol,
            limit,
        )
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_trades(symbol, limit=limit)
        trades = []
        for t in raw:
            trades.append({
                "id": t.get("id"),
                "price": float(t.get("price") or 0),
                "amount": float(t.get("amount") or 0),
                "side": t.get("side"),
                "timestamp": t.get("timestamp"),
            })
        return trades

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return a raw ticker dict for *symbol* (alias for ccxt-style callers).

        Internally calls :meth:`get_ticker` and converts the resulting
        :class:`~.base_exchange.Ticker` dataclass to a plain dict so callers
        that expect a raw ccxt-style dict (e.g. dashboard liquidation heatmap)
        work without modification.
        """
        ticker = await self.get_ticker(symbol)
        return {
            "last": ticker.last,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "high": ticker.high,
            "low": ticker.low,
            "baseVolume": ticker.volume,
            "timestamp": ticker.timestamp,
        }

    # ------------------------------------------------------------------
    # Position & leverage management
    # ------------------------------------------------------------------

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set the leverage multiplier for *symbol*.

        Tries the swap/perpetual symbol first (e.g. ``BTC/USDT:USDT``) since
        some exchanges (Gate.io) only support leverage on contract markets.
        If leverage setting is not supported for the market type the error is
        caught and logged at DEBUG level so callers are not interrupted.
        """
        logger.info("[{}] Setting leverage to {}x on {}", self._exchange_id, leverage, symbol)
        await self._rate_limiter.acquire()
        swap_symbol = self._resolve_swap_symbol(symbol)
        try:
            result = await self._client.set_leverage(leverage, swap_symbol)
            return result
        except Exception as exc:
            exc_str = str(exc)
            # Some exchanges don't support leverage on spot/certain market types
            if "not support" in exc_str.lower() or "market type" in exc_str.lower():
                logger.debug(
                    "[{}] set_leverage not supported for {}: {}",
                    self._exchange_id,
                    symbol,
                    exc,
                )
                return {"symbol": symbol, "leverage": leverage, "skipped": True}
            raise

    async def modify_leverage(self, symbol: str, new_leverage: int) -> Dict[str, Any]:
        """Set leverage for *symbol* and return a normalised result dict.

        This wraps :meth:`set_leverage` with a consistent return shape matching
        the paper exchange implementation.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            new_leverage: New leverage multiplier (must be >= 1).

        Returns:
            dict with ``symbol``, ``leverage`` keys.
        """
        result = await self.set_leverage(symbol, new_leverage)
        return {"symbol": symbol, "leverage": new_leverage, "raw": result}

    async def add_margin(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Add margin to an isolated position for *symbol*.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            amount: USDT amount to add as margin (must be > 0).

        Returns:
            dict with ``symbol``, ``added_margin`` keys.
        """
        logger.info("[{}] Adding {:.4f} USDT margin to {}", self._exchange_id, amount, symbol)
        await self._rate_limiter.acquire()
        result = await self._client.add_margin(symbol, amount)
        return {"symbol": symbol, "added_margin": amount, "raw": result}

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Switch between cross and isolated margin for *symbol*."""
        logger.info(
            "[{}] Setting margin type to {} on {}",
            self._exchange_id,
            margin_type.value,
            symbol,
        )
        await self._rate_limiter.acquire()
        result = await self._client.set_margin_mode(margin_type.value, symbol)
        return result

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def get_positions(self) -> List[Position]:
        """Return all non-zero open positions."""
        logger.debug("[{}] Fetching positions", self._exchange_id)
        await self._rate_limiter.acquire()
        raw = await self._client.fetch_positions()
        markets = getattr(self._client, "markets", None)
        if not isinstance(markets, dict):
            markets = {}
        result = []
        for p in raw:
            if float(p.get("contracts") or 0) == 0:
                continue
            market = markets.get(p.get("symbol", ""), {})
            contract_size = float(market.get("contractSize") or 1.0)
            result.append(self._parse_position(p, contract_size))
        return result

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for *symbol*, or *None* if flat."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def update_stop_loss(self, symbol: str, new_price: float) -> Order:
        """Cancel existing stop-loss orders for *symbol* and place a new one at *new_price*."""
        logger.info(
            "[{}] Updating stop-loss for {} → new_price={}", self._exchange_id, symbol, new_price
        )
        # Cancel all existing open stop-loss orders for this symbol
        try:
            open_orders = await self.get_open_orders(symbol)
            for order in open_orders:
                if order.type == OrderType.STOP_LOSS:
                    try:
                        await self.cancel_order(order.id, symbol)
                    except Exception as exc:
                        logger.debug(
                            "[{}] Could not cancel SL order {}: {}", self._exchange_id, order.id, exc
                        )
        except Exception as exc:
            logger.warning(
                "[{}] Could not fetch open orders to cancel SL: {}", self._exchange_id, exc
            )
        # Fetch current position to determine close side and amount
        position = await self.get_position(symbol)
        if not position:
            raise ValueError(f"No open position for {symbol}")
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        return await self.create_stop_loss_order(symbol, close_side, position.amount, new_price)

    async def update_take_profit(self, symbol: str, new_price: float) -> Order:
        """Cancel existing take-profit orders for *symbol* and place a new one at *new_price*."""
        logger.info(
            "[{}] Updating take-profit for {} → new_price={}", self._exchange_id, symbol, new_price
        )
        # Cancel all existing open take-profit orders for this symbol
        try:
            open_orders = await self.get_open_orders(symbol)
            for order in open_orders:
                if order.type == OrderType.TAKE_PROFIT:
                    try:
                        await self.cancel_order(order.id, symbol)
                    except Exception as exc:
                        logger.debug(
                            "[{}] Could not cancel TP order {}: {}", self._exchange_id, order.id, exc
                        )
        except Exception as exc:
            logger.warning(
                "[{}] Could not fetch open orders to cancel TP: {}", self._exchange_id, exc
            )
        # Fetch current position to determine close side and amount
        position = await self.get_position(symbol)
        if not position:
            raise ValueError(f"No open position for {symbol}")
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        return await self.create_take_profit_order(symbol, close_side, position.amount, new_price)

    @async_retry_decorator(max_retries=3, base_delay=1.0)
    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close (or partially close) the open position for *symbol*."""
        position = await self.get_position(symbol)
        if not position:
            raise ValueError(f"No open position for {symbol}")
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_amount = amount if amount is not None else position.amount
        logger.info("[{}] Closing position: {} amount={}", self._exchange_id, symbol, close_amount)
        return await self.create_market_order(
            symbol, close_side, close_amount, {"reduceOnly": True}
        )

    # ------------------------------------------------------------------
    # Derivatives-specific data
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        """Return the current funding rate for *symbol*, or 0.0 on error.

        Automatically resolves spot symbols to their swap equivalent (e.g.
        ``BTC/USDT`` → ``BTC/USDT:USDT``) so the call works on exchanges that
        only support ``fetchFundingRate`` for swap/perpetual markets.
        """
        try:
            await self._rate_limiter.acquire()
            swap_symbol = self._resolve_swap_symbol(symbol)
            result = await self._client.fetch_funding_rate(swap_symbol)
            rate = float(result.get("fundingRate") or 0)
            logger.debug("[{}] Funding rate for {}: {}", self._exchange_id, symbol, rate)
            return rate
        except Exception as exc:
            logger.warning(
                "[{}] get_funding_rate failed for {}: {}", self._exchange_id, symbol, exc
            )
            return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        """Return the current open interest for *symbol*, or 0.0 on error.

        Automatically resolves spot symbols to their swap equivalent so the
        call works on exchanges that only support ``fetchOpenInterest`` for
        swap/perpetual markets.
        """
        try:
            await self._rate_limiter.acquire()
            swap_symbol = self._resolve_swap_symbol(symbol)
            result = await self._client.fetch_open_interest(swap_symbol)
            oi = float(result.get("openInterest") or 0)
            logger.debug("[{}] Open interest for {}: {}", self._exchange_id, symbol, oi)
            return oi
        except Exception as exc:
            logger.warning(
                "[{}] get_open_interest failed for {}: {}", self._exchange_id, symbol, exc
            )
            return 0.0

    def _resolve_precious_metals_symbol(self, symbol: str) -> str:
        """Map precious metals symbols to their CCXT-accessible equivalents.

        Gate.io does NOT have native XAU/USDT on their standard futures/swap API.
        We map to XAUT/USDT (Tether Gold) which IS available as a perpetual contract.
        The original symbol is preserved in all responses so the rest of the bot
        sees "XAU/USDT" consistently.

        This mapping is applied before _resolve_swap_symbol().

        Args:
            symbol: User-provided symbol (e.g., "XAU/USDT", "BTC/USDT")

        Returns:
            Mapped symbol or the original if no mapping exists.
        """
        mapped = self.PRECIOUS_METALS_MAPPING.get(symbol)
        if mapped:
            logger.debug(
                "[{}] Precious metals mapping: {} -> {}", self._exchange_id, symbol, mapped
            )
            return mapped
        return symbol

    def _resolve_swap_symbol(self, symbol: str) -> str:
        """Return the fully-resolved swap market symbol for *symbol*.

        Applies precious-metals mapping first (e.g. XAU/USDT → XAUT/USDT) so
        that a single call to this method is sufficient at every API boundary.
        Then converts spot symbols to the perpetual/swap format expected by
        CCXT (e.g. XAUT/USDT → XAUT/USDT:USDT).

        Many exchanges (e.g. Gate.io) only support ``fetchFundingRate`` for
        perpetual/swap symbols such as ``BTC/USDT:USDT`` rather than the spot
        form ``BTC/USDT``.  If *symbol* is already a swap symbol (contains
        ``:``) it is returned unchanged after the precious-metals step.
        Otherwise the method looks up the loaded markets to find the correct
        swap symbol; if not found it appends ``:USDT`` as a best-effort
        fallback.

        Args:
            symbol: Market symbol, e.g. ``"XAU/USDT"`` or ``"BTC/USDT:USDT"``.

        Returns:
            Fully resolved swap/perpetual market symbol.
        """
        # Step 1 — precious metals mapping (XAU/USDT → XAUT/USDT)
        symbol = self._resolve_precious_metals_symbol(symbol)
        # Step 2 — spot → swap format
        if ":" in symbol:
            return symbol
        markets = getattr(self._client, "markets", None) or {}
        # Look for a swap market that matches the base symbol
        for market_symbol, market_info in markets.items():
            if isinstance(market_info, dict):
                if (
                    market_info.get("type") in ("swap", "future")
                    and market_info.get("spot") is False
                    and market_info.get("base") == symbol.split("/")[0]
                    and market_info.get("quote") == symbol.split("/")[-1]
                ):
                    return market_symbol
        # Fallback: append :<quote> for USDT-margined perpetuals
        if "/" in symbol and ":" not in symbol:
            quote = symbol.split("/")[-1]
            return f"{symbol}:{quote}"
        return symbol

    # ------------------------------------------------------------------
    # WebSocket subscriptions (ccxt.pro when available, REST polling fallback)
    # ------------------------------------------------------------------

    def _get_ws_client(self) -> Any:
        """Return the shared ccxt.pro WebSocket client, creating it if necessary.

        Raises:
            ImportError: If ``ccxt.pro`` is not installed.
            RuntimeError: If the exchange is not supported by ``ccxt.pro``.
        """
        if self._ws_client is not None:
            return self._ws_client

        import ccxt.pro as ccxtpro  # type: ignore[import]

        ws_class = getattr(ccxtpro, self._exchange_id, None)
        if ws_class is None:
            raise RuntimeError(f"ccxt.pro has no class for {self._exchange_id!r}")

        cfg: Dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
            "options": dict(self._default_options),
        }
        if self.passphrase:
            cfg["password"] = self.passphrase

        self._ws_client = ws_class(cfg)
        if self.testnet:
            self._ws_client.set_sandbox_mode(True)
        return self._ws_client

    @resolve_symbols
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to live ticker updates for *symbol* via WebSocket.

        Uses the shared ``ccxt.pro`` WebSocket client if available and falls
        back to REST polling every 5 seconds when ``ccxt.pro`` is not installed.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            callback: Async callable invoked with a :class:`Ticker` on each update.
        """
        # symbol is fully resolved (e.g. "XAUT/USDT:USDT") by @resolve_symbols.
        # original_symbol_ctx holds the original "XAU/USDT" for Ticker construction.
        _ctx = original_symbol_ctx.get()
        original = _ctx if _ctx is not None else symbol
        try:
            ws_client = self._get_ws_client()
            logger.info("[{}] Starting WS ticker subscription: {}", self._exchange_id, symbol)
            while True:
                raw = await ws_client.watch_ticker(symbol)
                # Safely extract values with proper defaults
                last_val = raw.get("last")
                bid_val = raw.get("bid")
                ask_val = raw.get("ask")
                high_val = raw.get("high")
                low_val = raw.get("low")
                volume_val = raw.get("baseVolume")
                timestamp_val = raw.get("timestamp")

                ticker = Ticker(
                    symbol=original,
                    last=float(last_val) if last_val is not None else 0.0,
                    bid=float(bid_val) if bid_val is not None else 0.0,
                    ask=float(ask_val) if ask_val is not None else 0.0,
                    high=float(high_val) if high_val is not None else 0.0,
                    low=float(low_val) if low_val is not None else 0.0,
                    volume=float(volume_val) if volume_val is not None else 0.0,
                    timestamp=int(timestamp_val) if timestamp_val is not None else 0,
                )
                await callback(ticker)
        except (ImportError, RuntimeError):
            logger.info(
                "[{}] ccxt.pro not available — falling back to REST polling for {}",
                self._exchange_id,
                original,
            )
            while True:
                try:
                    ticker = await self.get_ticker(original)
                    await callback(ticker)
                except Exception as exc:
                    logger.debug("[{}] REST ticker poll error for {}: {}", self._exchange_id, original, exc)
                await asyncio.sleep(5)

    @resolve_symbols
    async def watch_order_book(self, symbol: str) -> Dict[str, Any]:
        """Return one order-book snapshot via the ccxt.pro WebSocket client.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT"``.

        Returns:
            Raw order-book dict with ``bids`` / ``asks`` lists.

        Raises:
            NotImplementedError: When ``ccxt.pro`` is not installed or the
                exchange is not supported.
        """
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols
        ws_client = self._get_ws_client()
        return await ws_client.watch_order_book(symbol)

    async def close(self) -> None:
        """Close both the REST and WebSocket connections.

        Closing the WebSocket client forces ccxt.pro to establish a fresh
        connection on the next ``watch_order_book`` call, which is the
        recommended recovery strategy after transport-level errors (e.g.
        aiohttp ``parse_frame`` exceptions).
        """
        if self._ws_client is not None:
            try:
                await self._ws_client.close()
            except Exception as exc:
                logger.debug("[{}] WS client close error (ignored): {}", self._exchange_id, exc)
            self._ws_client = None
        await self.disconnect()

    @resolve_symbols
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to live order-book updates for *symbol* via WebSocket.

        Uses the shared ``ccxt.pro`` WebSocket client if available and falls
        back to REST polling every 5 seconds when ``ccxt.pro`` is not installed.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            callback: Async callable invoked with a raw order-book dict on each update.
        """
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols.
        _ctx = original_symbol_ctx.get()
        original = _ctx if _ctx is not None else symbol
        try:
            ws_client = self._get_ws_client()
            logger.info("[{}] Starting WS order-book subscription: {}", self._exchange_id, symbol)
            while True:
                raw = await ws_client.watch_order_book(symbol)
                await callback(raw)
        except (ImportError, RuntimeError):
            logger.info(
                "[{}] ccxt.pro not available — falling back to REST polling for order-book {}",
                self._exchange_id,
                original,
            )
            while True:
                try:
                    orderbook = await self.get_orderbook(original)
                    await callback(orderbook)
                except Exception as exc:
                    logger.debug(
                        "[{}] REST order-book poll error for {}: {}", self._exchange_id, original, exc
                    )
                await asyncio.sleep(5)

    @resolve_symbols
    async def subscribe_ohlcv(
        self, symbol: str, timeframe: str, callback: Callable
    ) -> None:
        """Subscribe to OHLCV (candlestick) updates for *symbol* via WebSocket.

        Uses the shared ``ccxt.pro`` WebSocket client if available and falls
        back to REST polling every 30 seconds when ``ccxt.pro`` is not installed.

        The *callback* is an async callable invoked as
        ``await callback(symbol, timeframe, candles)`` where *candles* is a
        list of ``[timestamp_ms, open, high, low, close, volume]`` lists.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            timeframe: Candle timeframe, e.g. ``"1m"`` or ``"5m"``.
            callback: Async callable invoked with ``(symbol, timeframe, candles)``
                on each update.
        """
        # symbol is fully resolved (precious metals + swap) by @resolve_symbols.
        # Pass original symbol back to callback so the rest of the bot sees the
        # user-visible symbol (e.g. "XAU/USDT") rather than "XAUT/USDT:USDT".
        _ctx = original_symbol_ctx.get()
        original = _ctx if _ctx is not None else symbol
        try:
            ws_client = self._get_ws_client()
            logger.info(
                "[{}] Starting WS OHLCV subscription: {} {}",
                self._exchange_id,
                symbol,
                timeframe,
            )
            while True:
                raw = await ws_client.watch_ohlcv(symbol, timeframe)
                await callback(original, timeframe, raw)
        except (ImportError, RuntimeError):
            logger.info(
                "[{}] ccxt.pro not available — falling back to REST polling for OHLCV {} {}",
                self._exchange_id,
                original,
                timeframe,
            )
            poll_interval = 30 if timeframe == "1m" else 60
            while True:
                try:
                    df = await self.get_ohlcv(original, timeframe=timeframe, limit=200)
                    candles = []
                    for ts, row in df.iterrows():
                        candles.append(
                            [
                                int(ts.timestamp() * 1000),
                                float(row["open"]),
                                float(row["high"]),
                                float(row["low"]),
                                float(row["close"]),
                                float(row["volume"]),
                            ]
                        )
                    await callback(original, timeframe, candles)
                except Exception as exc:
                    logger.debug(
                        "[{}] REST OHLCV poll error for {} {}: {}",
                        self._exchange_id,
                        original,
                        timeframe,
                        exc,
                    )
                await asyncio.sleep(poll_interval)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Not implemented — use REST polling via :meth:`get_trade_history`."""
        logger.warning("[{}] subscribe_trades not implemented; use REST polling", self._exchange_id)

    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to private user-data events via ccxt.pro WebSocket.

        Streams order fills, position updates, and balance changes in real-time.
        Falls back to REST polling every 3 seconds if ccxt.pro is unavailable.
        """
        try:
            ws_client = self._get_ws_client()
            logger.info("[{}] Starting user data WebSocket stream", self._exchange_id)

            async def _watch_orders() -> None:
                while True:
                    try:
                        orders = await ws_client.watch_orders()
                        for order_data in orders:
                            parsed = self._parse_order(order_data)
                            await callback({
                                "type": "order_update",
                                "order": parsed,
                                "raw": order_data,
                            })
                    except Exception as exc:
                        logger.debug("[{}] watch_orders error: {}", self._exchange_id, exc)
                        await asyncio.sleep(1)

            async def _watch_positions() -> None:
                while True:
                    try:
                        if hasattr(ws_client, "watch_positions"):
                            positions = await ws_client.watch_positions()
                            markets = getattr(ws_client, "markets", {})
                            parsed = []
                            for p in positions:
                                if abs(float(p.get("contracts", 0))) > 1e-8:
                                    cs = float(
                                        markets.get(p.get("symbol", ""), {}).get("contractSize", 1.0)
                                    )
                                    parsed.append(self._parse_position(p, cs))
                            await callback({
                                "type": "position_update",
                                "positions": parsed,
                            })
                        else:
                            await asyncio.sleep(60)  # No position streaming, rely on order events
                    except Exception as exc:
                        logger.debug("[{}] watch_positions error: {}", self._exchange_id, exc)
                        await asyncio.sleep(1)

            await asyncio.gather(_watch_orders(), _watch_positions())

        except (ImportError, RuntimeError):
            logger.info(
                "[{}] ccxt.pro not available - falling back to REST polling for user data",
                self._exchange_id,
            )
            while True:
                try:
                    orders = await self.get_open_orders()
                    positions = await self.get_positions()
                    await callback({
                        "type": "poll_update",
                        "orders": orders,
                        "positions": positions,
                    })
                except Exception as exc:
                    logger.debug("[{}] User data poll error: {}", self._exchange_id, exc)
                await asyncio.sleep(3)  # 3 seconds instead of 10

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_order_status(raw_status: Optional[str]) -> OrderStatus:
        _map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.CLOSED,
            "canceled": OrderStatus.CANCELED,
            "cancelled": OrderStatus.CANCELED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "rejected": OrderStatus.REJECTED,
        }
        return _map.get((raw_status or "").lower(), OrderStatus.OPEN)

    @staticmethod
    def _parse_order_type(raw_type: Optional[str]) -> OrderType:
        _map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_loss": OrderType.STOP_LOSS,
            "stop_market": OrderType.STOP_LOSS,
            "take_profit": OrderType.TAKE_PROFIT,
            "take_profit_market": OrderType.TAKE_PROFIT,
        }
        return _map.get((raw_type or "").lower(), OrderType.MARKET)

    def _parse_order(self, raw: Dict[str, Any]) -> Order:
        fee_info = raw.get("fee") or {}
        price_raw = raw.get("price")
        return Order(
            id=str(raw.get("id", "")),
            symbol=str(raw.get("symbol", "")),
            type=self._parse_order_type(raw.get("type")),
            side=OrderSide(str(raw.get("side", "buy")).lower()),
            amount=float(raw.get("amount") or 0),
            price=float(price_raw) if price_raw is not None else None,
            filled=float(raw.get("filled") or 0),
            remaining=float(raw.get("remaining") or 0),
            status=self._parse_order_status(raw.get("status")),
            timestamp=int(raw.get("timestamp") or 0),
            fee=float(fee_info.get("cost") or 0),
            info=dict(raw.get("info") or {}),
        )

    @staticmethod
    def _parse_position(raw: Dict[str, Any], contract_size: float = 1.0) -> Position:
        side_str = str(raw.get("side") or "long").lower()
        side = PositionSide.LONG if side_str == "long" else PositionSide.SHORT
        entry_price = float(raw.get("entryPrice") or 0)
        mark_price = float(raw.get("markPrice") or 0)
        contracts = float(raw.get("contracts") or 0)
        leverage = int(raw.get("leverage") or 1)

        # Use the exchange-reported notional when available; fall back to calculating
        # it from first principles using contractSize so Gate.io displays correctly.
        notional = float(raw.get("notional") or 0)
        price_for_notional = mark_price if mark_price else entry_price
        if notional <= 0 and price_for_notional > 0:
            notional = contracts * contract_size * price_for_notional

        margin = float(raw.get("initialMargin") or 0)
        if margin == 0 and leverage > 0 and notional > 0:
            # Some exchanges (e.g. OKX in cross-margin mode) omit
            # initialMargin.  Calculate it from first principles:
            #   margin = notional / leverage
            margin = notional / leverage

        unrealized_pnl = float(raw.get("unrealizedPnl") or 0)
        position_value = round(notional, 4)
        roe_pct = (unrealized_pnl / margin * 100.0) if margin > 0 else 0.0
        return Position(
            symbol=str(raw.get("symbol", "")),
            side=side,
            amount=contracts,
            entry_price=entry_price,
            current_price=mark_price,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            leverage=leverage,
            margin=margin,
            liquidation_price=float(raw.get("liquidationPrice") or 0),
            timestamp=int(raw.get("timestamp") or 0),
            roe_pct=roe_pct,
            position_value=position_value,
        )
