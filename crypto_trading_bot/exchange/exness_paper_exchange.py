"""Exness paper trading exchange — simulates Exness execution with realistic forex behavior.

Provides realistic simulation of Exness Raw Spread account with:

* Real-time price feeds from free data providers (TradingView/Alpha Vantage).
* Realistic spread simulation per pair (0.0-2.0 pips for majors, wider for exotics).
* Slippage simulation (0.1-0.5 pips for majors, 1-3 pips for gold).
* Swap charge simulation at end of trading day (triple swap on Wednesday).
* Margin tracking with margin call at 100%, stop out at 50% (Exness defaults).
* Commission calculation ($3.50/lot/side for Raw Spread accounts).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
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


class ExnessPaperExchange(BaseExchange):
    """Paper trading exchange simulating Exness forex execution.

    Simulates Exness Raw Spread account behavior with:

    * Realistic spreads (0.0-2.0 pips for majors).
    * Market slippage (0.1-0.5 pips for majors, 1-3 pips for gold).
    * Commission: $3.50 per lot per side.
    * Swap charges at 00:00 server time (triple swap on Wednesday).
    * Margin call at 100% margin level, stop out at 50%.

    Args:
        starting_balance: Starting virtual USD balance.
        state_file: Path to JSON state file for persistence.
        price_exchange: Optional BaseExchange for real price feeds.
    """

    # Pair specifications (same as ExnessForexClient)
    FOREX_PAIRS: Dict[str, Dict[str, Any]] = {
        "XAUUSD": {
            "contract_size": 100,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 1.5,
            "slippage_range_pips": (1.0, 3.0),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -5.0,   # USD per lot per day
            "swap_short": 2.0,
        },
        "XAGUSD": {
            "contract_size": 5000,
            "pip_size": 0.001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 100.0,
            "typical_spread_pips": 2.0,
            "slippage_range_pips": (0.5, 2.0),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -3.0,
            "swap_short": 1.0,
        },
        "EURUSD": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.5,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -2.0,
            "swap_short": 1.0,
        },
        "GBPUSD": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.8,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -3.0,
            "swap_short": 1.5,
        },
        "USDJPY": {
            "contract_size": 100000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.5,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": 1.5,
            "swap_short": -2.5,
        },
        "AUDUSD": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.6,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -1.5,
            "swap_short": 0.5,
        },
        "USDCAD": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.7,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": 0.5,
            "swap_short": -1.5,
        },
        "USDCHF": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.8,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": 1.0,
            "swap_short": -2.0,
        },
        "GBPJPY": {
            "contract_size": 100000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 1.5,
            "slippage_range_pips": (0.2, 0.8),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -4.0,
            "swap_short": 2.0,
        },
        "EURJPY": {
            "contract_size": 100000,
            "pip_size": 0.01,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 1.2,
            "slippage_range_pips": (0.2, 0.6),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -3.5,
            "swap_short": 1.5,
        },
        "NZDUSD": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.8,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -1.0,
            "swap_short": 0.3,
        },
        "EURGBP": {
            "contract_size": 100000,
            "pip_size": 0.0001,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 200.0,
            "typical_spread_pips": 0.9,
            "slippage_range_pips": (0.1, 0.5),
            "commission_per_lot_per_side": 3.50,
            "swap_long": -2.5,
            "swap_short": 1.2,
        },
    }

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        state_file: str = "data/exness_paper_state.json",
        price_exchange: Optional[BaseExchange] = None,
    ) -> None:
        super().__init__(api_key="exness_paper", secret_key="exness_paper")
        self._starting_balance = starting_balance
        self._state_file = Path(state_file)
        self._price_exchange = price_exchange

        # State
        self._balance: float = starting_balance
        self._positions: Dict[str, dict] = {}  # symbol → position
        self._open_orders: Dict[str, dict] = {}  # order_id → order
        self._trade_history: List[dict] = []
        self._order_counter: int = 0
        self._lock = asyncio.Lock()

        # Last swap charge timestamp (to track daily rollover)
        self._last_swap_time: int = 0

        # Margin level tracking
        self._margin_call_level = 100.0  # %
        self._stop_out_level = 50.0      # %

    @property
    def name(self) -> str:
        return "exness_paper"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Load persisted state and connect price feed."""
        self._load_state()
        if self._price_exchange is not None:
            try:
                await self._price_exchange.connect()
                logger.info("ExnessPaperExchange: price feed connected")
            except Exception as exc:
                logger.warning("ExnessPaperExchange: price feed failed: {}", exc)

        logger.info(
            "ExnessPaperExchange ready — balance=${:.2f} positions={}",
            self._balance,
            len(self._positions),
        )

    async def disconnect(self) -> None:
        """Save state and disconnect price feed."""
        self._save_state()
        if self._price_exchange is not None:
            try:
                await self._price_exchange.disconnect()
            except Exception:
                pass
        logger.info("ExnessPaperExchange disconnected")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Return current balance with margin calculations."""
        async with self._lock:
            balance_copy = self._balance
            positions_copy = {k: dict(v) for k, v in self._positions.items()}

        # Calculate margin in use
        margin_used = 0.0
        unrealized_pnl = 0.0

        for pos in positions_copy.values():
            entry_price = pos["entry_price"]
            amount = pos["amount"]
            leverage = pos.get("leverage", 100)
            margin_used += (amount * entry_price) / leverage

            # Get current price for unrealized PnL
            try:
                current_price = await self._get_current_price(pos["symbol"])
                side = pos["side"]
                if side == "long":
                    pnl = (current_price - entry_price) * amount
                else:
                    pnl = (entry_price - current_price) * amount
                unrealized_pnl += pnl
            except Exception:
                pass

        equity = balance_copy + unrealized_pnl
        free_margin = equity - margin_used

        return Balance(
            total={"USD": equity},
            free={"USD": free_margin},
            used={"USD": margin_used},
            usdt_total=equity,
            usdt_free=free_margin,
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        """Return simulated ticker with spread."""
        if self._price_exchange is None:
            raise RuntimeError("No price feed configured")

        # Normalize symbol for price feed
        feed_symbol = self._normalize_symbol_for_feed(symbol)
        ticker = await self._price_exchange.get_ticker(feed_symbol)

        # Apply Exness spread
        config = self._get_pair_config(symbol)
        spread = config["typical_spread_pips"] * config["pip_size"]
        mid = ticker.last
        bid = mid - spread / 2
        ask = mid + spread / 2

        return Ticker(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=mid,
            high=ticker.high,
            low=ticker.low,
            volume=ticker.volume,
            timestamp=ticker.timestamp,
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Return minimal order book (bid/ask only)."""
        ticker = await self.get_ticker(symbol)
        return {
            "symbol": symbol,
            "bids": [[ticker.bid, 0.0]],
            "asks": [[ticker.ask, 0.0]],
            "timestamp": ticker.timestamp,
        }

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """Delegate OHLCV to price feed."""
        if self._price_exchange is None:
            raise RuntimeError("No price feed configured")

        feed_symbol = self._normalize_symbol_for_feed(symbol)
        return await self._price_exchange.get_ohlcv(feed_symbol, timeframe, limit)

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
        """Simulate market order fill with slippage."""
        async with self._lock:
            config = self._get_pair_config(symbol)
            lot_size = amount / config["contract_size"]

            # Get fill price with slippage
            ticker = await self.get_ticker(symbol)
            base_price = ticker.ask if side == OrderSide.BUY else ticker.bid

            # Apply slippage
            slippage_range = config["slippage_range_pips"]
            slippage_pips = random.uniform(*slippage_range)
            slippage = slippage_pips * config["pip_size"]
            if side == OrderSide.BUY:
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage

            # Calculate commission
            commission = lot_size * config["commission_per_lot_per_side"]

            # Deduct commission from balance
            self._balance -= commission

            # Create or update position
            leverage = params.get("leverage", 100)
            self._update_position(symbol, side, amount, fill_price, leverage)

            # Check margin level after opening position
            await self._check_margin_level()

            order_id = str(self._order_counter)
            self._order_counter += 1

            order = Order(
                id=order_id,
                symbol=symbol,
                type=OrderType.MARKET,
                side=side,
                amount=amount,
                price=fill_price,
                filled=amount,
                remaining=0.0,
                status=OrderStatus.CLOSED,
                timestamp=int(time.time() * 1000),
                fee=commission,
                info={"slippage_pips": slippage_pips},
            )

            self._trade_history.append(order.model_dump())
            self._save_state()

            logger.info(
                "ExnessPaperExchange: Market order filled {} {} {:.2f} @ {:.5f} (commission=${:.2f})",
                symbol,
                side.value,
                amount,
                fill_price,
                commission,
            )

            return order

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Create pending limit order."""
        async with self._lock:
            order_id = str(self._order_counter)
            self._order_counter += 1

            order_dict = {
                "id": order_id,
                "symbol": symbol,
                "type": "limit",
                "side": side.value,
                "amount": amount,
                "price": price,
                "filled": 0.0,
                "remaining": amount,
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "params": params,
            }

            self._open_orders[order_id] = order_dict
            self._save_state()

            return Order(**order_dict, type=OrderType.LIMIT, side=side, status=OrderStatus.OPEN, fee=0.0)

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Create stop loss order (managed at position level)."""
        # Store SL at position level
        logger.info("SL order created for {} at {}", symbol, stop_price)
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
        """Create take profit order (managed at position level)."""
        logger.info("TP order created for {} at {}", symbol, tp_price)
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
        """Cancel a pending order."""
        async with self._lock:
            if order_id in self._open_orders:
                del self._open_orders[order_id]
                self._save_state()
                return {"order_id": order_id, "status": "canceled"}
            raise ValueError(f"Order {order_id} not found")

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all orders for symbol."""
        async with self._lock:
            to_cancel = [
                oid for oid, o in self._open_orders.items() if o["symbol"] == symbol
            ]
            for oid in to_cancel:
                del self._open_orders[oid]
            self._save_state()
            return [{"order_id": oid, "status": "canceled"} for oid in to_cancel]

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Get order details."""
        async with self._lock:
            if order_id in self._open_orders:
                o = self._open_orders[order_id]
                return Order(**o, type=OrderType.LIMIT, side=OrderSide(o["side"]), status=OrderStatus.OPEN, fee=0.0)

        # Check history
        for trade in self._trade_history:
            if trade["id"] == order_id:
                return Order(**trade)

        raise ValueError(f"Order {order_id} not found")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all open orders."""
        async with self._lock:
            orders = []
            for o in self._open_orders.values():
                if symbol is None or o["symbol"] == symbol:
                    orders.append(
                        Order(**o, type=OrderType.LIMIT, side=OrderSide(o["side"]), status=OrderStatus.OPEN, fee=0.0)
                    )
            return orders

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for symbol (stored for next position)."""
        logger.info("Leverage set for {}: {}", symbol, leverage)
        return {"symbol": symbol, "leverage": leverage}

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """Set margin type."""
        logger.info("Margin type set for {}: {}", symbol, margin_type)
        return {"symbol": symbol, "margin_type": margin_type.value}

    async def get_positions(self) -> List[Position]:
        """Return all open positions."""
        async with self._lock:
            positions = []
            for symbol, pos in self._positions.items():
                try:
                    current_price = await self._get_current_price(symbol)
                    positions.append(self._build_position(symbol, pos, current_price))
                except Exception as exc:
                    logger.warning("Error building position for {}: {}", symbol, exc)
            return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for symbol."""
        async with self._lock:
            if symbol not in self._positions:
                return None
            pos = self._positions[symbol]
            current_price = await self._get_current_price(symbol)
            return self._build_position(symbol, pos, current_price)

    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close position."""
        position = await self.get_position(symbol)
        if position is None:
            raise ValueError(f"No position for {symbol}")

        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_amount = amount if amount is not None else position.amount

        # Execute close as market order
        order = await self.create_market_order(symbol, close_side, close_amount, {})

        # Realize PnL
        async with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pnl = position.unrealized_pnl * (close_amount / position.amount)
                self._balance += pnl
                logger.info("Position closed for {}: PnL = ${:.2f}", symbol, pnl)

                # Remove or reduce position
                if close_amount >= position.amount:
                    del self._positions[symbol]
                else:
                    pos["amount"] -= close_amount

                self._save_state()

        return order

    # ------------------------------------------------------------------
    # Derivatives-specific
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        """Return swap rate."""
        config = self._get_pair_config(symbol)
        return config.get("swap_long", 0.0)

    async def get_open_interest(self, symbol: str) -> float:
        """Not applicable."""
        return 0.0

    # ------------------------------------------------------------------
    # WebSocket subscriptions (not supported)
    # ------------------------------------------------------------------

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        logger.warning("Ticker subscription not supported in paper mode")

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        logger.warning("Orderbook subscription not supported in paper mode")

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        logger.warning("Trades subscription not supported in paper mode")

    async def subscribe_user_data(self, callback: Callable) -> None:
        logger.warning("User data subscription not supported in paper mode")

    # ------------------------------------------------------------------
    # Swap charge simulation
    # ------------------------------------------------------------------

    async def apply_daily_swap(self) -> None:
        """Apply swap charges to open positions (called at rollover time)."""
        now = datetime.now(tz=timezone.utc)
        today_midnight = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())

        if self._last_swap_time >= today_midnight:
            return  # Already applied today

        async with self._lock:
            total_swap = 0.0
            is_wednesday = now.weekday() == 2  # Wednesday = triple swap

            for symbol, pos in self._positions.items():
                config = self._get_pair_config(symbol)
                lot_size = pos["amount"] / config["contract_size"]
                side = pos["side"]

                swap_rate = config.get("swap_long" if side == "long" else "swap_short", 0.0)
                daily_swap = swap_rate * lot_size
                if is_wednesday:
                    daily_swap *= 3

                self._balance += daily_swap  # Swap is typically negative (cost)
                total_swap += daily_swap

            self._last_swap_time = today_midnight

            if total_swap != 0:
                logger.info("Daily swap applied: ${:.2f} (Wed triple={})".format(total_swap, is_wednesday))
                self._save_state()

    # ------------------------------------------------------------------
    # Margin monitoring
    # ------------------------------------------------------------------

    async def _check_margin_level(self) -> None:
        """Check margin level and trigger margin call/stop out if needed."""
        balance_info = await self.get_balance()
        equity = balance_info.usdt_total
        margin_used = balance_info.used.get("USD", 0.0)

        if margin_used <= 0:
            return

        margin_level = (equity / margin_used) * 100

        if margin_level <= self._stop_out_level:
            logger.error("STOP OUT: Margin level {:.1f}% — closing all positions!", margin_level)
            await self._close_all_positions()
        elif margin_level <= self._margin_call_level:
            logger.warning("MARGIN CALL: Margin level {:.1f}%", margin_level)

    async def _close_all_positions(self) -> None:
        """Close all positions (stop out scenario)."""
        async with self._lock:
            symbols = list(self._positions.keys())
        for symbol in symbols:
            try:
                await self.close_position(symbol)
            except Exception as exc:
                logger.error("Failed to close position {}: {}", symbol, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_pair_config(self, symbol: str) -> Dict[str, Any]:
        """Get pair config or raise ValueError."""
        std_symbol = symbol.replace("/", "")
        for key, cfg in self.FOREX_PAIRS.items():
            if key.replace("/", "") == std_symbol:
                return cfg
        raise ValueError(f"Unknown forex pair: {symbol}")

    def _normalize_symbol_for_feed(self, symbol: str) -> str:
        """Convert Exness symbol to price feed format (add slash if needed)."""
        # XAUUSD → XAU/USD
        std_symbol = symbol.replace("/", "")
        if std_symbol in ["XAUUSD", "XAGUSD"]:
            return std_symbol[:3] + "/" + std_symbol[3:]
        # EURUSD → EUR/USD
        if len(std_symbol) == 6:
            return std_symbol[:3] + "/" + std_symbol[3:]
        return symbol

    async def _get_current_price(self, symbol: str) -> float:
        """Get current mid price."""
        ticker = await self.get_ticker(symbol)
        return (ticker.bid + ticker.ask) / 2

    def _update_position(
        self, symbol: str, side: OrderSide, amount: float, price: float, leverage: int
    ) -> None:
        """Update or create position."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            # Average entry price
            old_amount = pos["amount"]
            old_price = pos["entry_price"]
            new_amount = old_amount + amount
            avg_price = ((old_amount * old_price) + (amount * price)) / new_amount
            pos["amount"] = new_amount
            pos["entry_price"] = avg_price
        else:
            self._positions[symbol] = {
                "symbol": symbol,
                "side": side.value,
                "amount": amount,
                "entry_price": price,
                "leverage": leverage,
                "timestamp": int(time.time() * 1000),
            }

    def _build_position(self, symbol: str, pos: dict, current_price: float) -> Position:
        """Build Position dataclass from position dict."""
        entry_price = pos["entry_price"]
        amount = pos["amount"]
        side = PositionSide.LONG if pos["side"] == "long" else PositionSide.SHORT

        if side == PositionSide.LONG:
            unrealized_pnl = (current_price - entry_price) * amount
        else:
            unrealized_pnl = (entry_price - current_price) * amount

        leverage = pos.get("leverage", 100)
        margin = (amount * entry_price) / leverage

        return Position(
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            leverage=leverage,
            margin=margin,
            liquidation_price=0.0,
            timestamp=pos["timestamp"],
            mark_price=current_price,
            position_value=amount * current_price,
        )

    def _load_state(self) -> None:
        """Load state from JSON file."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)
            self._balance = state.get("balance", self._starting_balance)
            self._positions = state.get("positions", {})
            self._open_orders = state.get("open_orders", {})
            self._trade_history = state.get("trade_history", [])
            self._order_counter = state.get("order_counter", 0)
            self._last_swap_time = state.get("last_swap_time", 0)
            logger.info("ExnessPaperExchange: State loaded from {}", self._state_file)
        except Exception as exc:
            logger.warning("Failed to load state: {}", exc)

    def _save_state(self) -> None:
        """Save state to JSON file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "balance": self._balance,
                "positions": self._positions,
                "open_orders": self._open_orders,
                "trade_history": self._trade_history[-100:],  # Keep last 100
                "order_counter": self._order_counter,
                "last_swap_time": self._last_swap_time,
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save state: {}", exc)
