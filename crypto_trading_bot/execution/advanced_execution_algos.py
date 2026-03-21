"""Advanced execution algorithms for institutional-grade order execution.

Implements TWAP (Time-Weighted Average Price), VWAP (Volume-Weighted Average Price),
and other sophisticated execution strategies to minimize market impact and slippage.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from loguru import logger
import numpy as np
import pandas as pd

from exchange.base_exchange import BaseExchange, OrderSide, Order


class ExecutionAlgorithm:
    """Base class for execution algorithms."""

    def __init__(self, exchange: BaseExchange):
        self.exchange = exchange
        self._active = False
        self._filled_amount = 0.0
        self._total_cost = 0.0
        self._orders: List[Order] = []

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        total_amount: float,
        **kwargs,
    ) -> Dict[str, float]:
        """Execute the order using the specific algorithm.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            total_amount: Total amount to execute
            **kwargs: Algorithm-specific parameters

        Returns:
            Dict with execution stats: average_price, filled_amount, slippage, etc.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the execution algorithm."""
        self._active = False
        logger.info("Execution algorithm stopped")

    @property
    def filled_amount(self) -> float:
        """Get total filled amount."""
        return self._filled_amount

    @property
    def average_price(self) -> float:
        """Get average fill price."""
        if self._filled_amount == 0:
            return 0.0
        return self._total_cost / self._filled_amount


class TWAPExecutor(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm.

    Splits a large order into smaller child orders executed at regular intervals
    to minimize market impact and achieve an average price close to the time-weighted
    average over the execution period.

    Args:
        exchange: Exchange client instance
        duration_minutes: Total duration for execution in minutes
        num_slices: Number of child orders to split into
        randomize: Add random jitter to order times to avoid detection
    """

    def __init__(
        self,
        exchange: BaseExchange,
        duration_minutes: int = 60,
        num_slices: int = 10,
        randomize: bool = True,
    ):
        super().__init__(exchange)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.randomize = randomize

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        total_amount: float,
        **kwargs,
    ) -> Dict[str, float]:
        """Execute order using TWAP strategy."""
        logger.info(
            f"Starting TWAP execution: {side.value} {total_amount} {symbol} "
            f"over {self.duration_minutes} minutes in {self.num_slices} slices"
        )

        self._active = True
        self._filled_amount = 0.0
        self._total_cost = 0.0
        self._orders = []

        # Calculate slice size and interval
        slice_amount = total_amount / self.num_slices
        interval_seconds = (self.duration_minutes * 60) / self.num_slices

        start_time = time.time()
        execution_times = []

        # Calculate execution times with optional randomization
        for i in range(self.num_slices):
            base_time = start_time + (i * interval_seconds)

            if self.randomize and i > 0:  # Don't randomize first order
                # Add ±20% random jitter
                jitter = interval_seconds * 0.2 * (np.random.random() - 0.5)
                execution_time = base_time + jitter
            else:
                execution_time = base_time

            execution_times.append(execution_time)

        # Execute child orders
        for i, exec_time in enumerate(execution_times):
            if not self._active:
                logger.info("TWAP execution stopped early")
                break

            # Wait until execution time
            now = time.time()
            if exec_time > now:
                await asyncio.sleep(exec_time - now)

            # Execute child order
            try:
                logger.debug(
                    f"TWAP slice {i+1}/{self.num_slices}: "
                    f"{side.value} {slice_amount:.6f} {symbol}"
                )

                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=slice_amount,
                )

                self._orders.append(order)
                self._filled_amount += order.filled
                fill_price = order.price if order.price else 0.0
                self._total_cost += order.filled * fill_price

                logger.debug(
                    f"TWAP slice {i+1} filled: {order.filled:.6f} @ {fill_price:.2f}"
                )

            except Exception as exc:
                logger.error(f"TWAP slice {i+1} failed: {exc}")
                # Continue with remaining slices

        # Calculate execution statistics
        avg_price = self.average_price
        completion_pct = (self._filled_amount / total_amount) * 100

        logger.info(
            f"TWAP execution complete: filled {self._filled_amount:.6f} / {total_amount:.6f} "
            f"({completion_pct:.1f}%) @ avg price {avg_price:.2f}"
        )

        return {
            "algorithm": "TWAP",
            "filled_amount": self._filled_amount,
            "total_amount": total_amount,
            "average_price": avg_price,
            "completion_pct": completion_pct,
            "num_orders": len(self._orders),
        }


class VWAPExecutor(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm.

    Executes orders following the market's natural volume profile to minimize
    market impact. Child order sizes are weighted by historical volume patterns.

    Args:
        exchange: Exchange client instance
        duration_minutes: Total duration for execution in minutes
        lookback_periods: Number of historical periods to analyze for volume profile
    """

    def __init__(
        self,
        exchange: BaseExchange,
        duration_minutes: int = 60,
        lookback_periods: int = 20,
    ):
        super().__init__(exchange)
        self.duration_minutes = duration_minutes
        self.lookback_periods = lookback_periods

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        total_amount: float,
        **kwargs,
    ) -> Dict[str, float]:
        """Execute order using VWAP strategy."""
        logger.info(
            f"Starting VWAP execution: {side.value} {total_amount} {symbol} "
            f"over {self.duration_minutes} minutes"
        )

        self._active = True
        self._filled_amount = 0.0
        self._total_cost = 0.0
        self._orders = []

        # Get historical volume profile
        try:
            # Fetch OHLCV data to analyze volume patterns
            df = await self.exchange.get_ohlcv(
                symbol=symbol,
                timeframe="5m",
                limit=self.lookback_periods,
            )

            if df.empty:
                logger.warning("No volume data available, falling back to TWAP")
                return await self._fallback_twap(symbol, side, total_amount)

            # Calculate volume profile (percentage of total volume per period)
            total_volume = df["volume"].sum()
            volume_profile = (df["volume"] / total_volume).tolist()

        except Exception as exc:
            logger.error(f"Failed to fetch volume profile: {exc}, falling back to TWAP")
            return await self._fallback_twap(symbol, side, total_amount)

        # Determine number of slices based on duration
        num_slices = max(1, self.duration_minutes // 5)  # One slice per 5 minutes
        interval_seconds = (self.duration_minutes * 60) / num_slices

        # Distribute total amount according to volume profile
        # Use most recent volume profile or repeat pattern if needed
        slice_amounts = []
        for i in range(num_slices):
            profile_idx = i % len(volume_profile)
            weight = volume_profile[profile_idx]
            slice_amount = total_amount * weight
            slice_amounts.append(slice_amount)

        # Normalize to ensure sum equals total_amount
        actual_sum = sum(slice_amounts)
        if actual_sum > 0:
            slice_amounts = [amt * (total_amount / actual_sum) for amt in slice_amounts]

        start_time = time.time()

        # Execute child orders
        for i, slice_amount in enumerate(slice_amounts):
            if not self._active:
                logger.info("VWAP execution stopped early")
                break

            # Wait for next interval
            exec_time = start_time + (i * interval_seconds)
            now = time.time()
            if exec_time > now:
                await asyncio.sleep(exec_time - now)

            # Skip tiny slices
            if slice_amount < 0.0001:
                continue

            # Execute child order
            try:
                logger.debug(
                    f"VWAP slice {i+1}/{num_slices}: "
                    f"{side.value} {slice_amount:.6f} {symbol}"
                )

                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=slice_amount,
                )

                self._orders.append(order)
                self._filled_amount += order.filled
                fill_price = order.price if order.price else 0.0
                self._total_cost += order.filled * fill_price

                logger.debug(
                    f"VWAP slice {i+1} filled: {order.filled:.6f} @ {fill_price:.2f}"
                )

            except Exception as exc:
                logger.error(f"VWAP slice {i+1} failed: {exc}")
                # Continue with remaining slices

        # Calculate execution statistics
        avg_price = self.average_price
        completion_pct = (self._filled_amount / total_amount) * 100

        logger.info(
            f"VWAP execution complete: filled {self._filled_amount:.6f} / {total_amount:.6f} "
            f"({completion_pct:.1f}%) @ avg price {avg_price:.2f}"
        )

        return {
            "algorithm": "VWAP",
            "filled_amount": self._filled_amount,
            "total_amount": total_amount,
            "average_price": avg_price,
            "completion_pct": completion_pct,
            "num_orders": len(self._orders),
        }

    async def _fallback_twap(
        self, symbol: str, side: OrderSide, total_amount: float
    ) -> Dict[str, float]:
        """Fallback to TWAP if VWAP cannot execute."""
        twap = TWAPExecutor(self.exchange, self.duration_minutes, num_slices=10)
        return await twap.execute(symbol, side, total_amount)


class IcebergOrderExecutor(ExecutionAlgorithm):
    """Iceberg order execution - hide large order size from order book.

    Displays only a small portion of the total order size, automatically
    replenishing as fills occur to avoid revealing full order size.

    Args:
        exchange: Exchange client instance
        display_quantity: Maximum quantity to display in order book
        price_offset_pct: Limit price offset from mid as percentage (0.0 = mid price)
    """

    def __init__(
        self,
        exchange: BaseExchange,
        display_quantity: float = 0.1,
        price_offset_pct: float = 0.05,
    ):
        super().__init__(exchange)
        self.display_quantity = display_quantity
        self.price_offset_pct = price_offset_pct

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        total_amount: float,
        **kwargs,
    ) -> Dict[str, float]:
        """Execute order using iceberg strategy."""
        logger.info(
            f"Starting Iceberg execution: {side.value} {total_amount} {symbol} "
            f"with display quantity {self.display_quantity}"
        )

        self._active = True
        self._filled_amount = 0.0
        self._total_cost = 0.0
        self._orders = []

        remaining_amount = total_amount

        while remaining_amount > 0.0001 and self._active:
            # Determine visible order size
            visible_amount = min(self.display_quantity, remaining_amount)

            # Get current mid price for limit order placement
            try:
                ticker = await self.exchange.get_ticker(symbol)
                mid_price = (ticker.bid + ticker.ask) / 2.0

                # Calculate limit price with offset
                if side == OrderSide.BUY:
                    # For buy, offset below mid (more aggressive = closer to mid)
                    limit_price = mid_price * (1 - self.price_offset_pct)
                else:
                    # For sell, offset above mid
                    limit_price = mid_price * (1 + self.price_offset_pct)

                logger.debug(
                    f"Iceberg child order: {side.value} {visible_amount:.6f} {symbol} "
                    f"@ {limit_price:.2f} (mid: {mid_price:.2f})"
                )

                # Place limit order
                order = await self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=visible_amount,
                    price=limit_price,
                )

                self._orders.append(order)

                # Wait for fill with timeout
                timeout = 30.0  # 30 second timeout per child order
                start_wait = time.time()

                while time.time() - start_wait < timeout and self._active:
                    # Check order status
                    current_order = await self.exchange.get_order(order.id, symbol)

                    if current_order.filled > 0:
                        # Update stats
                        fill_delta = current_order.filled - order.filled
                        self._filled_amount += fill_delta
                        fill_price = current_order.price if current_order.price else limit_price
                        self._total_cost += fill_delta * fill_price

                        remaining_amount -= fill_delta
                        order = current_order

                        logger.debug(
                            f"Iceberg partial fill: {fill_delta:.6f} @ {fill_price:.2f}, "
                            f"remaining: {remaining_amount:.6f}"
                        )

                    if current_order.status.value in ["closed", "canceled"]:
                        break

                    await asyncio.sleep(1.0)  # Check every second

                # Cancel any unfilled portion
                if order.status.value == "open":
                    try:
                        await self.exchange.cancel_order(order.id, symbol)
                        logger.debug(f"Cancelled unfilled iceberg order {order.id}")
                    except Exception:
                        pass  # Already filled or cancelled

            except Exception as exc:
                logger.error(f"Iceberg child order failed: {exc}")
                await asyncio.sleep(2.0)  # Brief pause before retry

        # Calculate execution statistics
        avg_price = self.average_price
        completion_pct = (self._filled_amount / total_amount) * 100

        logger.info(
            f"Iceberg execution complete: filled {self._filled_amount:.6f} / {total_amount:.6f} "
            f"({completion_pct:.1f}%) @ avg price {avg_price:.2f}"
        )

        return {
            "algorithm": "Iceberg",
            "filled_amount": self._filled_amount,
            "total_amount": total_amount,
            "average_price": avg_price,
            "completion_pct": completion_pct,
            "num_orders": len(self._orders),
        }


class AdaptiveExecutor(ExecutionAlgorithm):
    """Adaptive execution algorithm that dynamically adjusts to market conditions.

    Monitors market volatility, spread, and volume to adjust execution pace
    and aggressiveness. Slows down in illiquid conditions and speeds up
    when opportunities arise.

    Args:
        exchange: Exchange client instance
        target_duration_minutes: Target duration (can be adjusted)
        aggressiveness: Execution aggressiveness (0.0-1.0)
    """

    def __init__(
        self,
        exchange: BaseExchange,
        target_duration_minutes: int = 60,
        aggressiveness: float = 0.5,
    ):
        super().__init__(exchange)
        self.target_duration_minutes = target_duration_minutes
        self.aggressiveness = aggressiveness

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        total_amount: float,
        **kwargs,
    ) -> Dict[str, float]:
        """Execute order with adaptive strategy."""
        logger.info(
            f"Starting Adaptive execution: {side.value} {total_amount} {symbol} "
            f"target duration {self.target_duration_minutes}m, "
            f"aggressiveness {self.aggressiveness:.2f}"
        )

        self._active = True
        self._filled_amount = 0.0
        self._total_cost = 0.0
        self._orders = []

        remaining_amount = total_amount
        start_time = time.time()
        target_end_time = start_time + (self.target_duration_minutes * 60)

        while remaining_amount > 0.0001 and self._active and time.time() < target_end_time:
            # Assess market conditions
            try:
                ticker = await self.exchange.get_ticker(symbol)
                orderbook = await self.exchange.get_orderbook(symbol, limit=20)

                # Calculate market metrics
                mid_price = (ticker.bid + ticker.ask) / 2.0
                spread_pct = (ticker.ask - ticker.bid) / mid_price if mid_price > 0 else 0.01

                # Calculate order book imbalance
                bid_volume = sum(bid[1] for bid in orderbook.get("bids", []))
                ask_volume = sum(ask[1] for ask in orderbook.get("asks", []))
                total_ob_volume = bid_volume + ask_volume
                imbalance = (
                    (bid_volume - ask_volume) / total_ob_volume
                    if total_ob_volume > 0
                    else 0.0
                )

                # Determine execution urgency based on market conditions
                time_remaining = target_end_time - time.time()
                time_pct_remaining = time_remaining / (self.target_duration_minutes * 60)
                amount_pct_remaining = remaining_amount / total_amount

                # If we're behind schedule, increase urgency
                urgency = amount_pct_remaining / max(time_pct_remaining, 0.01)

                # Adjust slice size based on market conditions and urgency
                # Start with base slice of 5% of remaining
                base_slice = remaining_amount * 0.05

                # Adjust for spread (wider spread = smaller slices)
                spread_factor = 1.0 - min(spread_pct * 10, 0.5)  # Max 50% reduction

                # Adjust for urgency
                urgency_factor = 0.5 + (urgency * 0.5)  # 0.5x to 1.5x multiplier

                # Adjust for order book imbalance (favorable imbalance = larger slices)
                if side == OrderSide.BUY:
                    imbalance_factor = 1.0 + max(imbalance, 0) * 0.5  # More asks = good for buying
                else:
                    imbalance_factor = 1.0 + max(-imbalance, 0) * 0.5  # More bids = good for selling

                # Calculate final slice size
                slice_amount = (
                    base_slice
                    * spread_factor
                    * urgency_factor
                    * imbalance_factor
                    * (1.0 + self.aggressiveness)
                )
                slice_amount = min(slice_amount, remaining_amount)

                logger.debug(
                    f"Adaptive slice: {slice_amount:.6f} (spread={spread_pct*100:.3f}%, "
                    f"imbalance={imbalance:.3f}, urgency={urgency:.2f})"
                )

                # Execute child order
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=slice_amount,
                )

                self._orders.append(order)
                self._filled_amount += order.filled
                remaining_amount -= order.filled
                fill_price = order.price if order.price else mid_price
                self._total_cost += order.filled * fill_price

                # Dynamic wait time based on market conditions
                # Tighter spread and favorable imbalance = shorter wait
                base_wait = 5.0  # Base 5 seconds
                wait_time = base_wait * (1.0 + spread_pct * 10)  # Longer wait if wide spread
                wait_time = max(1.0, wait_time)  # Minimum 1 second

                logger.debug(f"Waiting {wait_time:.1f}s before next slice")
                await asyncio.sleep(wait_time)

            except Exception as exc:
                logger.error(f"Adaptive execution error: {exc}")
                await asyncio.sleep(5.0)  # Brief pause on error

        # Calculate execution statistics
        avg_price = self.average_price
        completion_pct = (self._filled_amount / total_amount) * 100
        actual_duration = (time.time() - start_time) / 60

        logger.info(
            f"Adaptive execution complete: filled {self._filled_amount:.6f} / {total_amount:.6f} "
            f"({completion_pct:.1f}%) @ avg price {avg_price:.2f} in {actual_duration:.1f}m"
        )

        return {
            "algorithm": "Adaptive",
            "filled_amount": self._filled_amount,
            "total_amount": total_amount,
            "average_price": avg_price,
            "completion_pct": completion_pct,
            "num_orders": len(self._orders),
            "duration_minutes": actual_duration,
        }
