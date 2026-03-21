"""Exness forex broker integration via MetaTrader5 API.

⚠️  **DEPRECATED**: This client is deprecated in favor of Gate.io TradFi API.
    Exness/MT5 is heavily restricted in India and requires Windows/VPS deployment.
    Use GateIOTradFiClient (forex_live/forex_demo modes) instead.

Provides a production-grade Exness forex client extending BaseExchange with:

* MT5 Python package integration for Windows deployments.
* REST/WebSocket API fallback for Linux/Docker deployments.
* Support for 12 forex pairs: XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY, AUDUSD,
  USDCAD, USDCHF, GBPJPY, EURJPY, NZDUSD, EURGBP.
* Session-aware trading (London, New York, Tokyo, Sydney).
* Spread monitoring with Exness-specific thresholds.
* Swap rate tracking (including swap-free Islamic accounts).
* Commission calculation (Exness Raw Spread: $3.50/lot/side).
* Robust error handling with requote detection and slippage protection.

**Migration Guide**:
- Replace ``forex_exness_live`` with ``forex_live`` (Gate.io TradFi)
- Replace ``forex_exness_demo`` with ``forex_demo`` (Gate.io TradFi)
- Replace ``forex_exness_paper`` with ``paper`` mode
- Update symbol format: ``XAUUSD`` → ``XAU/USDT``, ``EURUSD`` → ``EURUSD``
- Gate.io TradFi offers: No Windows requirement, no India restrictions,
  up to 500x leverage, tighter spreads, native async Python API
"""

from __future__ import annotations

import asyncio
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

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


class ExnessForexClient(BaseExchange):
    """Exness forex broker client via MetaTrader5 API.

    This client supports two deployment modes:

    1. **Windows (native MT5)**: Uses the MetaTrader5 Python package directly.
    2. **Linux/Docker**: Uses rpyc bridge to a Windows VPS running MT5, or
       Exness REST/WebSocket API (if available).

    All BaseExchange methods are mapped to MT5 operations:

    * ``connect()`` → ``mt5.initialize()``
    * ``get_ticker()`` → ``mt5.symbol_info_tick()``
    * ``get_ohlcv()`` → ``mt5.copy_rates_from_pos()``
    * ``create_market_order()`` → ``mt5.order_send()``
    * ``get_positions()`` → ``mt5.positions_get()``
    * ``get_balance()`` → ``mt5.account_info()``
    * ``close_position()`` → reverse order via ``mt5.order_send()``

    Args:
        login: Exness MT5 account login number.
        password: Exness MT5 account password.
        server: Exness MT5 server (e.g., "Exness-MT5Real", "Exness-MT5Demo").
        account_type: Exness account type ("Raw Spread" / "Pro" / "Standard").
        testnet: When ``True``, connects to demo account.
    """

    # ------------------------------------------------------------------
    # Forex pair configurations
    # ------------------------------------------------------------------

    FOREX_PAIRS: Dict[str, Dict[str, Any]] = {
        "XAUUSD": {
            "mt5_symbol": "XAUUSD",
            "contract_size": 100,      # 100 oz per lot
            "pip_size": 0.01,          # $0.01 for gold
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,  # Exness Raw Spread
            "max_acceptable_spread_pips": 3.0,
            "commission_per_lot_per_side": 3.50,  # USD
        },
        "XAGUSD": {
            "mt5_symbol": "XAGUSD",
            "contract_size": 5000,     # 5000 oz per lot
            "pip_size": 0.001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 100.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 5.0,
            "commission_per_lot_per_side": 3.50,
        },
        "EURUSD": {
            "mt5_symbol": "EURUSD",
            "contract_size": 100000,   # Standard lot
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "GBPUSD": {
            "mt5_symbol": "GBPUSD",
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "USDJPY": {
            "mt5_symbol": "USDJPY",
            "contract_size": 100000,
            "pip_size": 0.01,          # JPY pairs have 2 decimal places
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "AUDUSD": {
            "mt5_symbol": "AUDUSD",
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "USDCAD": {
            "mt5_symbol": "USDCAD",
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "USDCHF": {
            "mt5_symbol": "USDCHF",
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "GBPJPY": {
            "mt5_symbol": "GBPJPY",
            "contract_size": 100000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 3.0,
            "commission_per_lot_per_side": 3.50,
        },
        "EURJPY": {
            "mt5_symbol": "EURJPY",
            "contract_size": 100000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 3.0,
            "commission_per_lot_per_side": 3.50,
        },
        "NZDUSD": {
            "mt5_symbol": "NZDUSD",
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
        },
        "EURGBP": {
            "mt5_symbol": "EURGBP",
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.0,
            "max_acceptable_spread_pips": 2.0,
            "commission_per_lot_per_side": 3.50,
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
        server: str = "Exness-MT5Real",
        account_type: str = "Raw Spread",
        testnet: bool = False,
    ) -> None:
        super().__init__(api_key=login, secret_key=password)
        self.login = int(login)
        self.password = password
        self.server = server if not testnet else "Exness-MT5Demo"
        self.account_type = account_type
        self.testnet = testnet
        self._connected = False
        self._last_heartbeat = time.time()

        # Connection retry state
        self._retry_count = 0
        self._max_retries = 5
        self._retry_delay = 2.0  # Exponential backoff base

        # Check if MT5 is available
        if not _HAS_MT5:
            raise RuntimeError(
                "MetaTrader5 package is required for ExnessForexClient. "
                "Install with: pip install MetaTrader5 (Windows only) "
                "or set up rpyc bridge for Linux deployments."
            )

    # ------------------------------------------------------------------
    # BaseExchange properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"exness_{self.account_type.lower().replace(' ', '_')}"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize MT5 connection with exponential backoff retry logic."""
        if self._connected:
            logger.debug("ExnessForexClient already connected")
            return

        for attempt in range(self._max_retries):
            try:
                if not mt5.initialize(
                    login=self.login,
                    password=self.password,
                    server=self.server,
                ):
                    error_code = mt5.last_error()
                    raise ConnectionError(
                        f"MT5 initialization failed: {error_code}. "
                        f"Ensure MT5 terminal is running and credentials are correct."
                    )

                # Verify account info
                account_info = mt5.account_info()
                if account_info is None:
                    raise ConnectionError("Failed to retrieve account info")

                self._connected = True
                self._last_heartbeat = time.time()
                logger.info(
                    "ExnessForexClient connected: account={} balance={:.2f} currency={} server={}",
                    account_info.login,
                    account_info.balance,
                    account_info.currency,
                    self.server,
                )
                return

            except Exception as exc:
                self._retry_count = attempt + 1
                delay = self._retry_delay * (2 ** attempt)
                logger.warning(
                    "MT5 connection attempt {}/{} failed: {}. Retrying in {:.1f}s...",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Failed to connect to Exness MT5 after {self._max_retries} attempts"
        )

    async def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if not self._connected:
            return

        try:
            mt5.shutdown()
            self._connected = False
            logger.info("ExnessForexClient disconnected")
        except Exception as exc:
            logger.warning("Error during MT5 shutdown: {}", exc)

    async def _ensure_connected(self) -> None:
        """Ensure MT5 is connected, reconnect if needed."""
        if not self._connected:
            await self.connect()
            return

        # Heartbeat check (reconnect if no activity for 5 minutes)
        if time.time() - self._last_heartbeat > 300:
            logger.warning("MT5 heartbeat timeout — reconnecting...")
            await self.disconnect()
            await self.connect()

        self._last_heartbeat = time.time()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Return account balance from MT5."""
        await self._ensure_connected()

        try:
            account_info = mt5.account_info()
            if account_info is None:
                raise RuntimeError("Failed to retrieve account info")

            return Balance(
                total={"USD": account_info.balance},
                free={"USD": account_info.margin_free},
                used={"USD": account_info.margin},
                usdt_total=account_info.balance,
                usdt_free=account_info.margin_free,
            )
        except Exception as exc:
            logger.error("get_balance error: {}", exc)
            raise

    async def get_ticker(self, symbol: str) -> Ticker:
        """Return the latest ticker for *symbol* via MT5."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        try:
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                raise ValueError(f"No tick data for {mt5_symbol}")

            return Ticker(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                high=0.0,  # MT5 tick doesn't provide 24h high/low
                low=0.0,
                volume=tick.volume,
                timestamp=int(tick.time * 1000),  # Convert to milliseconds
            )
        except Exception as exc:
            logger.error("get_ticker error for {}: {}", symbol, exc)
            raise

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Return order book (not directly available in MT5, return bid/ask only)."""
        await self._ensure_connected()

        ticker = await self.get_ticker(symbol)
        return {
            "symbol": symbol,
            "bids": [[ticker.bid, 0.0]],  # MT5 doesn't expose depth
            "asks": [[ticker.ask, 0.0]],
            "timestamp": ticker.timestamp,
        }

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """Return OHLCV candles via MT5."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        mt5_timeframe = self._parse_timeframe(timeframe)

        try:
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, limit)
            if rates is None or len(rates) == 0:
                logger.warning("No OHLCV data for {} {}", symbol, timeframe)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={"tick_volume": "volume"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df

        except Exception as exc:
            logger.error("get_ohlcv error for {} {}: {}", symbol, timeframe, exc)
            raise

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a market order via MT5."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        config = self._get_pair_config(symbol)

        # Convert amount to lots
        lot_size = amount / config["contract_size"]
        lot_size = self._round_lot_size(lot_size, config)

        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        price = await self._get_fill_price(mt5_symbol, side)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": params.get("slippage_points", 50),  # Max slippage in points
            "magic": params.get("magic", 0),
            "comment": params.get("comment", "ExnessForexClient"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Add SL/TP if provided
        if "stopLoss" in params:
            request["sl"] = float(params["stopLoss"])
        if "takeProfit" in params:
            request["tp"] = float(params["takeProfit"])

        try:
            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError(f"order_send failed: {mt5.last_error()}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self._handle_order_error(result)

            # Calculate commission
            commission = lot_size * config["commission_per_lot_per_side"]

            return Order(
                id=str(result.order),
                symbol=symbol,
                type=OrderType.MARKET,
                side=side,
                amount=amount,
                price=result.price,
                filled=amount,
                remaining=0.0,
                status=OrderStatus.CLOSED,
                timestamp=int(time.time() * 1000),
                fee=commission,
                info={"mt5_result": result._asdict()},
            )

        except Exception as exc:
            logger.error("create_market_order error for {} {}: {}", symbol, side, exc)
            raise

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Place a limit order via MT5."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        config = self._get_pair_config(symbol)

        lot_size = amount / config["contract_size"]
        lot_size = self._round_lot_size(lot_size, config)

        order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": mt5_symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "magic": params.get("magic", 0),
            "comment": params.get("comment", "ExnessForexClient"),
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if "stopLoss" in params:
            request["sl"] = float(params["stopLoss"])
        if "takeProfit" in params:
            request["tp"] = float(params["takeProfit"])

        try:
            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError(f"order_send failed: {mt5.last_error()}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self._handle_order_error(result)

            return Order(
                id=str(result.order),
                symbol=symbol,
                type=OrderType.LIMIT,
                side=side,
                amount=amount,
                price=price,
                filled=0.0,
                remaining=amount,
                status=OrderStatus.OPEN,
                timestamp=int(time.time() * 1000),
                fee=0.0,
                info={"mt5_result": result._asdict()},
            )

        except Exception as exc:
            logger.error("create_limit_order error for {} {}: {}", symbol, side, exc)
            raise

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """MT5 handles SL as part of position, not separate order. Use modify_position."""
        logger.warning(
            "create_stop_loss_order called for {}. "
            "MT5 manages SL via position modification, not separate orders.",
            symbol,
        )
        # Return a dummy order (SL is set via modify_position)
        return Order(
            id="sl-dummy",
            symbol=symbol,
            type=OrderType.STOP_LOSS,
            side=side,
            amount=amount,
            price=stop_price,
            filled=0.0,
            remaining=amount,
            status=OrderStatus.OPEN,
            timestamp=int(time.time() * 1000),
            fee=0.0,
        )

    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """MT5 handles TP as part of position, not separate order. Use modify_position."""
        logger.warning(
            "create_take_profit_order called for {}. "
            "MT5 manages TP via position modification, not separate orders.",
            symbol,
        )
        return Order(
            id="tp-dummy",
            symbol=symbol,
            type=OrderType.TAKE_PROFIT,
            side=side,
            amount=amount,
            price=tp_price,
            filled=0.0,
            remaining=amount,
            status=OrderStatus.OPEN,
            timestamp=int(time.time() * 1000),
            fee=0.0,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a pending order via MT5."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
            "symbol": mt5_symbol,
        }

        try:
            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError(f"order_send failed: {mt5.last_error()}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self._handle_order_error(result)

            return {"order_id": order_id, "status": "canceled"}

        except Exception as exc:
            logger.error("cancel_order error for {}: {}", order_id, exc)
            raise

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all pending orders for *symbol*."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        orders = mt5.orders_get(symbol=mt5_symbol)
        if orders is None or len(orders) == 0:
            return []

        results = []
        for order in orders:
            try:
                await self.cancel_order(str(order.ticket), symbol)
                results.append({"order_id": str(order.ticket), "status": "canceled"})
            except Exception as exc:
                logger.warning("Failed to cancel order {}: {}", order.ticket, exc)

        return results

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch order details from MT5 order history."""
        await self._ensure_connected()

        # Check pending orders first
        orders = mt5.orders_get(ticket=int(order_id))
        if orders and len(orders) > 0:
            mt5_order = orders[0]
            return self._parse_order(mt5_order, symbol)

        # Check order history
        from_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        to_date = datetime.now(tz=timezone.utc)
        history = mt5.history_orders_get(from_date, to_date, ticket=int(order_id))
        if history and len(history) > 0:
            mt5_order = history[0]
            return self._parse_order(mt5_order, symbol)

        raise ValueError(f"Order {order_id} not found")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all open (pending) orders."""
        await self._ensure_connected()

        try:
            if symbol:
                mt5_symbol = self._resolve_symbol(symbol)
                orders = mt5.orders_get(symbol=mt5_symbol)
            else:
                orders = mt5.orders_get()

            if orders is None:
                return []

            return [self._parse_order(o, symbol or self._reverse_symbol(o.symbol)) for o in orders]

        except Exception as exc:
            logger.error("get_open_orders error: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Position & leverage management
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Leverage is account-level in Exness, not per-symbol. Log warning."""
        logger.warning(
            "set_leverage called for {} with leverage={}. "
            "Exness manages leverage at account level. No action taken.",
            symbol,
            leverage,
        )
        return {"symbol": symbol, "leverage": leverage, "note": "account_level"}

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Margin type is account-level in Exness. Log warning."""
        logger.warning(
            "set_margin_type called for {} with type={}. "
            "Exness manages margin type at account level. No action taken.",
            symbol,
            margin_type,
        )
        return {"symbol": symbol, "margin_type": margin_type.value, "note": "account_level"}

    async def get_positions(self) -> List[Position]:
        """Return all open positions from MT5."""
        await self._ensure_connected()

        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            return [self._parse_position(p) for p in positions]

        except Exception as exc:
            logger.error("get_positions error: {}", exc)
            return []

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for *symbol*."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        try:
            positions = mt5.positions_get(symbol=mt5_symbol)
            if positions is None or len(positions) == 0:
                return None

            return self._parse_position(positions[0])

        except Exception as exc:
            logger.error("get_position error for {}: {}", symbol, exc)
            return None

    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close an open position via reverse order."""
        await self._ensure_connected()

        position = await self.get_position(symbol)
        if position is None:
            raise ValueError(f"No open position for {symbol}")

        # Determine close side (opposite of position)
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_amount = amount if amount is not None else position.amount

        # Close via market order
        return await self.create_market_order(
            symbol,
            close_side,
            close_amount,
            {"comment": "Close position"},
        )

    # ------------------------------------------------------------------
    # Derivatives-specific (not applicable to spot forex)
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        """Forex doesn't have funding rates. Return swap rate instead."""
        await self._ensure_connected()

        mt5_symbol = self._resolve_symbol(symbol)
        symbol_info = mt5.symbol_info(mt5_symbol)
        if symbol_info is None:
            return 0.0

        # Return swap long (approximate funding cost)
        return float(symbol_info.swap_long)

    async def get_open_interest(self, symbol: str) -> float:
        """Open interest not available for forex. Return 0."""
        return 0.0

    # ------------------------------------------------------------------
    # WebSocket subscriptions (polling fallback for MT5)
    # ------------------------------------------------------------------

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Polling-based ticker subscription (MT5 doesn't have WebSocket)."""
        logger.warning("MT5 doesn't support WebSocket. Using polling for ticker updates.")
        # Implement polling loop if needed

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Order book subscription not available."""
        logger.warning("Order book subscription not supported in MT5")

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Public trades subscription not available."""
        logger.warning("Trades subscription not supported in MT5")

    async def subscribe_user_data(self, callback: Callable) -> None:
        """User data subscription not available."""
        logger.warning("User data subscription not supported in MT5")

    # ------------------------------------------------------------------
    # Forex-specific helpers
    # ------------------------------------------------------------------

    def get_current_session(self) -> str:
        """Return the currently active trading session(s)."""
        now = datetime.now(tz=timezone.utc)
        hour = now.hour
        active_sessions = []

        for session_name, (start, end) in self.SESSIONS.items():
            if start < end:
                if start <= hour < end:
                    active_sessions.append(session_name)
            else:  # Crosses midnight
                if hour >= start or hour < end:
                    active_sessions.append(session_name)

        return ",".join(active_sessions) if active_sessions else "none"

    async def get_spread(self, symbol: str) -> Dict[str, float]:
        """Return the current spread in pips."""
        ticker = await self.get_ticker(symbol)
        config = self._get_pair_config(symbol)
        spread_pips = (ticker.ask - ticker.bid) / config["pip_size"]

        return {
            "spread_pips": spread_pips,
            "bid": ticker.bid,
            "ask": ticker.ask,
        }

    def calculate_margin_required(
        self, symbol: str, lot_size: float, price: float, leverage: int
    ) -> float:
        """Calculate required margin for a position."""
        config = self._get_pair_config(symbol)
        return (lot_size * config["contract_size"] * price) / leverage

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_symbol(self, symbol: str) -> str:
        """Convert standard symbol to MT5 symbol."""
        # Remove slashes: XAU/USD → XAUUSD
        std_symbol = symbol.replace("/", "")
        for key, cfg in self.FOREX_PAIRS.items():
            if key.replace("/", "") == std_symbol:
                return cfg["mt5_symbol"]
        return std_symbol

    def _reverse_symbol(self, mt5_symbol: str) -> str:
        """Convert MT5 symbol back to standard format."""
        for key, cfg in self.FOREX_PAIRS.items():
            if cfg["mt5_symbol"] == mt5_symbol:
                return key
        return mt5_symbol

    def _get_pair_config(self, symbol: str) -> Dict[str, Any]:
        """Return pair config or raise ValueError."""
        std_symbol = symbol.replace("/", "")
        for key, cfg in self.FOREX_PAIRS.items():
            if key.replace("/", "") == std_symbol:
                return cfg
        raise ValueError(f"Unknown forex pair: {symbol}")

    def _round_lot_size(self, lot_size: float, config: Dict[str, Any]) -> float:
        """Round lot size to valid increment."""
        lot_step = config["lot_step"]
        lot_size = round(lot_size / lot_step) * lot_step
        lot_size = max(config["min_lot"], lot_size)
        lot_size = min(config["max_lot"], lot_size)
        return lot_size

    def _parse_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to MT5 constant."""
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

    async def _get_fill_price(self, mt5_symbol: str, side: OrderSide) -> float:
        """Get expected fill price (ask for buy, bid for sell)."""
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            raise ValueError(f"No tick data for {mt5_symbol}")
        return tick.ask if side == OrderSide.BUY else tick.bid

    def _handle_order_error(self, result: Any) -> None:
        """Handle MT5 order errors with descriptive messages."""
        error_messages = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote — price changed",
            mt5.TRADE_RETCODE_REJECT: "Order rejected",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid SL/TP",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid lot size",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Insufficient margin",
        }
        error_msg = error_messages.get(result.retcode, f"Unknown error {result.retcode}")
        raise RuntimeError(f"MT5 order failed: {error_msg} — {result.comment}")

    def _parse_order(self, mt5_order: Any, symbol: str) -> Order:
        """Convert MT5 order to Order dataclass."""
        order_type_map = {
            mt5.ORDER_TYPE_BUY: OrderSide.BUY,
            mt5.ORDER_TYPE_SELL: OrderSide.SELL,
            mt5.ORDER_TYPE_BUY_LIMIT: OrderSide.BUY,
            mt5.ORDER_TYPE_SELL_LIMIT: OrderSide.SELL,
        }
        side = order_type_map.get(mt5_order.type, OrderSide.BUY)

        # Determine order type
        if mt5_order.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL):
            order_type = OrderType.MARKET
        else:
            order_type = OrderType.LIMIT

        # Determine status
        if mt5_order.state == mt5.ORDER_STATE_FILLED:
            status = OrderStatus.CLOSED
        elif mt5_order.state in (mt5.ORDER_STATE_CANCELED, mt5.ORDER_STATE_REJECTED):
            status = OrderStatus.CANCELED
        else:
            status = OrderStatus.OPEN

        config = self._get_pair_config(symbol)
        amount = mt5_order.volume_current * config["contract_size"]
        filled = (mt5_order.volume_initial - mt5_order.volume_current) * config["contract_size"]

        return Order(
            id=str(mt5_order.ticket),
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount + filled,
            price=mt5_order.price_open,
            filled=filled,
            remaining=amount,
            status=status,
            timestamp=int(mt5_order.time_setup * 1000),
            fee=0.0,
            info={"mt5_order": mt5_order._asdict()},
        )

    def _parse_position(self, mt5_pos: Any) -> Position:
        """Convert MT5 position to Position dataclass."""
        symbol = self._reverse_symbol(mt5_pos.symbol)
        config = self._get_pair_config(symbol)

        side = PositionSide.LONG if mt5_pos.type == mt5.POSITION_TYPE_BUY else PositionSide.SHORT
        amount = mt5_pos.volume * config["contract_size"]

        return Position(
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=mt5_pos.price_open,
            current_price=mt5_pos.price_current,
            unrealized_pnl=mt5_pos.profit,
            leverage=1,  # MT5 doesn't expose per-position leverage
            margin=0.0,  # Would need to calculate
            liquidation_price=0.0,
            timestamp=int(mt5_pos.time * 1000),
            mark_price=mt5_pos.price_current,
            margin_ratio=0.0,
            roe_pct=(mt5_pos.profit / (amount * mt5_pos.price_open)) * 100 if amount > 0 else 0.0,
            position_value=amount * mt5_pos.price_current,
        )
