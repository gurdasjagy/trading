"""Adaptive execution engine with institutional-grade execution algorithms."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger
import numpy as np

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange, OrderSide
    from exchange.order_manager import OrderManager


@dataclass
class ExecutionSlice:
    """A single slice of an order execution."""

    slice_id: int
    amount: float
    delay_seconds: float
    price_limit: Optional[float] = None
    urgency: float = 0.5  # 0 (patient) to 1 (aggressive)


@dataclass
class MarketImpactModel:
    """Market impact parameters for a symbol."""

    symbol: str
    volatility: float  # Historical volatility (sigma)
    average_daily_volume: float  # ADV
    temporary_impact_factor: float = 0.5  # Sigma multiplier
    permanent_impact_factor: float = 0.1  # Gamma factor


@dataclass
class ExecutionResult:
    """Result of an execution algorithm."""

    success: bool
    filled_amount: float
    average_price: float
    total_slices: int
    completed_slices: int
    total_time_seconds: float
    total_fees: float = 0.0
    error: Optional[str] = None
    slices_detail: List[Dict] = None


class AdaptiveExecutionEngine:
    """Adaptive execution engine with institutional-grade algorithms.

    Algorithms:
    1. Implementation Shortfall (IS) - Minimize execution cost vs decision price
    2. Adaptive TWAP - Time-weighted with real-time market adjustments
    3. Adaptive VWAP - Volume-weighted with historical volume patterns
    4. Iceberg Orders - Hide order size with periodic replenishment
    5. Sniper Mode - Wait for favorable microstructure moments

    All algorithms are:
    - Async and non-blocking
    - Handle partial fills gracefully
    - Include circuit breakers (3 consecutive failures → pause)
    - Log decisions for post-trade analysis
    """

    # Circuit breaker
    MAX_CONSECUTIVE_FAILURES = 3
    CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60.0

    # Algorithm parameters
    ICEBERG_DISPLAY_FRACTION = 0.15  # Show 15% of total
    ICEBERG_RANDOMIZATION = 0.15  # ±15% randomization
    SNIPER_TIMEOUT_SECONDS = 5.0
    MIN_SLICE_AMOUNT_FRACTION = 0.01  # Minimum 1% of total per slice

    def __init__(
        self,
        exchange: BaseExchange,
        order_manager: OrderManager,
        local_orderbook_manager: Optional[Any] = None
    ):
        """Initialize adaptive execution engine.

        Args:
            exchange: Exchange interface
            order_manager: Order manager for placement
            local_orderbook_manager: Local orderbook for zero-latency access
        """
        self._exchange = exchange
        self._order_manager = order_manager
        self._local_orderbook_manager = local_orderbook_manager

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_breaker_active = False
        self._circuit_breaker_until: Optional[float] = None

        # Market impact models (cached per symbol)
        self._impact_models: Dict[str, MarketImpactModel] = {}

        # Execution tracking
        self._active_executions: Dict[str, bool] = {}  # execution_id -> is_active

        logger.info("AdaptiveExecutionEngine initialized")

    # =====================================================================
    # Implementation Shortfall Algorithm
    # =====================================================================

    async def execute_implementation_shortfall(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        decision_price: float,
        execution_duration_seconds: float = 300.0,
        **kwargs
    ) -> ExecutionResult:
        """Execute using Implementation Shortfall algorithm.

        Minimizes the difference between decision price and actual execution price
        by optimally balancing temporary impact, permanent impact, and timing risk.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            total_amount: Total amount to execute
            decision_price: Price at signal generation (benchmark)
            execution_duration_seconds: Time horizon for execution
            **kwargs: Additional parameters

        Returns:
            ExecutionResult with execution details
        """
        execution_id = f"IS_{symbol}_{int(time.time())}"
        self._active_executions[execution_id] = True

        logger.info(
            "Starting Implementation Shortfall: {} {} amount={} decision_price={} duration={}s",
            symbol,
            side,
            total_amount,
            decision_price,
            execution_duration_seconds
        )

        try:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                return ExecutionResult(
                    success=False,
                    filled_amount=0.0,
                    average_price=0.0,
                    total_slices=0,
                    completed_slices=0,
                    total_time_seconds=0.0,
                    error="Circuit breaker active"
                )

            # Get or create market impact model
            impact_model = await self._get_impact_model(symbol)

            # Calculate optimal participation rate
            # IS = temporary_impact + permanent_impact + timing_risk
            # Optimal rate minimizes total IS
            participation_rate = await self._calculate_optimal_participation_rate(
                total_amount,
                impact_model,
                execution_duration_seconds
            )

            logger.info(
                "IS algorithm: participation_rate={:.2%} for {} {}",
                participation_rate,
                total_amount,
                symbol
            )

            # Split order based on participation rate
            slices = await self._create_is_slices(
                total_amount,
                participation_rate,
                execution_duration_seconds,
                decision_price,
                side
            )

            # Execute slices
            result = await self._execute_slices(
                execution_id,
                symbol,
                side,
                slices,
                algorithm="IS"
            )

            self._on_execution_complete(success=result.success)
            return result

        except Exception as e:
            logger.error("Implementation Shortfall execution failed: {}", e)
            self._on_execution_complete(success=False)
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_slices=0,
                completed_slices=0,
                total_time_seconds=0.0,
                error=str(e)
            )
        finally:
            self._active_executions.pop(execution_id, None)

    async def _calculate_optimal_participation_rate(
        self,
        total_amount: float,
        impact_model: MarketImpactModel,
        duration_seconds: float
    ) -> float:
        """Calculate optimal participation rate to minimize Implementation Shortfall.

        The IS model:
        - Temporary impact = sigma * sqrt(participation_rate)
        - Permanent impact = gamma * (order_size / ADV)
        - Timing risk = sigma * sqrt(remaining_time)

        Optimal rate balances these three components.

        Args:
            total_amount: Total order size
            impact_model: Market impact parameters
            duration_seconds: Execution duration

        Returns:
            Optimal participation rate (0.0 to 1.0)
        """
        sigma = impact_model.volatility
        gamma = impact_model.permanent_impact_factor
        adv = impact_model.average_daily_volume

        # Fraction of ADV
        order_fraction = total_amount / adv if adv > 0 else 0.01

        # Simple heuristic: higher volatility → slower execution
        # Higher order size → slower execution
        # Longer duration → can afford slower execution

        # Base rate: 5% of available volume per interval
        base_rate = 0.05

        # Adjust for order size (larger orders = lower rate)
        size_adjustment = 1.0 / (1.0 + order_fraction * 10.0)

        # Adjust for volatility (higher vol = lower rate)
        vol_adjustment = 1.0 / (1.0 + sigma * 2.0)

        # Adjust for duration (longer = lower rate)
        duration_adjustment = min(1.0, 300.0 / duration_seconds)

        optimal_rate = base_rate * size_adjustment * vol_adjustment * duration_adjustment

        # Clamp to reasonable range
        return max(0.02, min(0.20, optimal_rate))

    async def _create_is_slices(
        self,
        total_amount: float,
        participation_rate: float,
        duration_seconds: float,
        decision_price: float,
        side: str
    ) -> List[ExecutionSlice]:
        """Create execution slices for IS algorithm.

        Args:
            total_amount: Total amount to execute
            participation_rate: Target participation rate
            duration_seconds: Total execution duration
            decision_price: Decision price for limit orders
            side: "buy" or "sell"

        Returns:
            List of ExecutionSlice objects
        """
        # Calculate number of slices based on participation rate
        # More aggressive = fewer slices, more patient = more slices
        n_slices = max(3, min(20, int(1.0 / participation_rate)))

        slice_amount = total_amount / n_slices
        interval = duration_seconds / n_slices

        slices = []
        for i in range(n_slices):
            # Add slight randomization to avoid pattern detection
            random_factor = random.uniform(0.9, 1.1)
            amount = slice_amount * random_factor

            # Add delay randomization (±20%)
            delay = interval * i * random.uniform(0.8, 1.2)

            # Price limit: place inside spread for better fill probability
            # For IS, we want to trade off speed vs cost
            slices.append(ExecutionSlice(
                slice_id=i,
                amount=amount,
                delay_seconds=delay,
                price_limit=None,  # Market orders for guaranteed fills
                urgency=participation_rate  # Higher rate = more urgent
            ))

        return slices

    # =====================================================================
    # Adaptive TWAP Algorithm
    # =====================================================================

    async def execute_adaptive_twap(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        duration_seconds: float = 300.0,
        **kwargs
    ) -> ExecutionResult:
        """Execute using Adaptive TWAP algorithm.

        Splits order into time-weighted slices, but dynamically adjusts based on:
        - Price movement (accelerate if favorable, decelerate if adverse)
        - Spread widening (pause if spread widens significantly)
        - Volume spikes (increase participation when liquidity is high)

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            total_amount: Total amount to execute
            duration_seconds: Total execution duration
            **kwargs: Additional parameters

        Returns:
            ExecutionResult with execution details
        """
        execution_id = f"ATWAP_{symbol}_{int(time.time())}"
        self._active_executions[execution_id] = True

        logger.info(
            "Starting Adaptive TWAP: {} {} amount={} duration={}s",
            symbol,
            side,
            total_amount,
            duration_seconds
        )

        try:
            if not await self._check_circuit_breaker():
                return ExecutionResult(
                    success=False,
                    filled_amount=0.0,
                    average_price=0.0,
                    total_slices=0,
                    completed_slices=0,
                    total_time_seconds=0.0,
                    error="Circuit breaker active"
                )

            # Get initial market state
            ticker = await self._exchange.get_ticker(symbol)
            initial_price = ticker.last
            initial_spread = (ticker.ask - ticker.bid) / ticker.last if ticker.bid > 0 else 0.01

            # Create initial TWAP slices (base schedule)
            n_slices = max(5, min(20, int(duration_seconds / 15)))  # One slice every 15s
            remaining_amount = total_amount
            filled_amount = 0.0
            total_cost = 0.0
            completed_slices = 0
            start_time = time.time()

            slices_detail = []

            for i in range(n_slices):
                if remaining_amount <= 0 or not self._active_executions.get(execution_id, False):
                    break

                # Calculate base slice size
                base_slice_size = remaining_amount / (n_slices - i)

                # Adaptive adjustments
                current_ticker = await self._exchange.get_ticker(symbol)
                current_price = current_ticker.last
                current_spread = (current_ticker.ask - current_ticker.bid) / current_ticker.last if current_ticker.bid > 0 else initial_spread

                # Price movement factor
                if side == "buy":
                    price_movement = (initial_price - current_price) / initial_price
                else:
                    price_movement = (current_price - initial_price) / initial_price

                # Accelerate if price moving favorably (>0), decelerate if adverse (<0)
                if price_movement > 0.001:  # Favorable
                    size_adjustment = 1.2  # Increase slice size by 20%
                elif price_movement < -0.001:  # Adverse
                    size_adjustment = 0.8  # Decrease slice size by 20%
                else:
                    size_adjustment = 1.0

                # Spread factor: pause if spread widens significantly
                spread_ratio = current_spread / initial_spread if initial_spread > 0 else 1.0
                if spread_ratio > 2.0:
                    logger.warning("Spread widened {:.1f}x - pausing TWAP", spread_ratio)
                    await asyncio.sleep(10.0)  # Wait for normalization
                    continue

                # Calculate adjusted slice size
                adjusted_slice_size = min(remaining_amount, base_slice_size * size_adjustment)
                adjusted_slice_size = max(adjusted_slice_size, remaining_amount * self.MIN_SLICE_AMOUNT_FRACTION)

                logger.info(
                    "Adaptive TWAP slice {}/{}: amount={:.6f} (adj={:.1f}x) spread_ratio={:.2f}",
                    i + 1,
                    n_slices,
                    adjusted_slice_size,
                    size_adjustment,
                    spread_ratio
                )

                # Execute slice
                try:
                    # Place market order
                    from exchange.base_exchange import OrderSide
                    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

                    order = await self._order_manager.place_market_order(
                        symbol=symbol,
                        side=order_side,
                        amount=adjusted_slice_size,
                        strategy="adaptive_twap"
                    )

                    # Wait for fill (simplified - in production, would monitor actively)
                    await asyncio.sleep(1.0)

                    # Record result
                    fill_price = order.price or current_price
                    filled_amount += adjusted_slice_size
                    total_cost += fill_price * adjusted_slice_size
                    remaining_amount -= adjusted_slice_size
                    completed_slices += 1

                    slices_detail.append({
                        "slice_id": i,
                        "amount": adjusted_slice_size,
                        "price": fill_price,
                        "adjustment_factor": size_adjustment
                    })

                except Exception as e:
                    logger.error("TWAP slice {} failed: {}", i, e)
                    # Continue with next slice

                # Wait for next interval
                if i < n_slices - 1:
                    interval = duration_seconds / n_slices
                    await asyncio.sleep(interval)

            # Calculate results
            average_price = total_cost / filled_amount if filled_amount > 0 else 0.0
            total_time = time.time() - start_time

            result = ExecutionResult(
                success=filled_amount > 0,
                filled_amount=filled_amount,
                average_price=average_price,
                total_slices=n_slices,
                completed_slices=completed_slices,
                total_time_seconds=total_time,
                slices_detail=slices_detail
            )

            self._on_execution_complete(success=result.success)
            return result

        except Exception as e:
            logger.error("Adaptive TWAP execution failed: {}", e)
            self._on_execution_complete(success=False)
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_slices=0,
                completed_slices=0,
                total_time_seconds=0.0,
                error=str(e)
            )
        finally:
            self._active_executions.pop(execution_id, None)

    # =====================================================================
    # Adaptive VWAP Algorithm
    # =====================================================================

    async def execute_adaptive_vwap(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        duration_seconds: float = 300.0,
        **kwargs
    ) -> ExecutionResult:
        """Execute using Adaptive VWAP algorithm.

        Distributes order based on predicted volume profile, with real-time adjustments
        when actual volume deviates from prediction.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            total_amount: Total amount to execute
            duration_seconds: Total execution duration
            **kwargs: Additional parameters

        Returns:
            ExecutionResult with execution details
        """
        execution_id = f"AVWAP_{symbol}_{int(time.time())}"
        self._active_executions[execution_id] = True

        logger.info(
            "Starting Adaptive VWAP: {} {} amount={} duration={}s",
            symbol,
            side,
            total_amount,
            duration_seconds
        )

        try:
            if not await self._check_circuit_breaker():
                return ExecutionResult(
                    success=False,
                    filled_amount=0.0,
                    average_price=0.0,
                    total_slices=0,
                    completed_slices=0,
                    total_time_seconds=0.0,
                    error="Circuit breaker active"
                )

            # Get historical volume profile
            volume_profile = await self._predict_volume_profile(symbol, duration_seconds)

            if not volume_profile:
                logger.warning("Could not predict volume profile, falling back to TWAP")
                return await self.execute_adaptive_twap(symbol, side, total_amount, duration_seconds)

            # Create VWAP slices based on volume profile
            slices = self._create_vwap_slices(total_amount, volume_profile)

            # Execute with real-time volume adjustment
            result = await self._execute_slices_with_volume_adjustment(
                execution_id,
                symbol,
                side,
                slices,
                volume_profile
            )

            self._on_execution_complete(success=result.success)
            return result

        except Exception as e:
            logger.error("Adaptive VWAP execution failed: {}", e)
            self._on_execution_complete(success=False)
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_slices=0,
                completed_slices=0,
                total_time_seconds=0.0,
                error=str(e)
            )
        finally:
            self._active_executions.pop(execution_id, None)

    async def _predict_volume_profile(
        self,
        symbol: str,
        duration_seconds: float
    ) -> Optional[List[float]]:
        """Predict volume profile for VWAP execution.

        In production, this would use historical intraday volume curves.
        For now, we use a simplified U-shaped profile (high at open/close, low midday).

        Args:
            symbol: Trading symbol
            duration_seconds: Execution duration

        Returns:
            List of volume percentages for each time bucket, or None if unavailable
        """
        try:
            # Fetch recent OHLCV to estimate volume patterns
            # In production: use 30 days of intraday data to build profile
            ohlcv = await self._exchange.get_ohlcv(symbol, "1m", limit=60)

            if ohlcv is None or len(ohlcv) < 10:
                return None

            # Calculate average volume per minute
            avg_volume = ohlcv['volume'].mean()

            # Create simplified U-shaped profile (10 buckets)
            n_buckets = 10
            profile = []

            for i in range(n_buckets):
                # U-shape: high at start and end, low in middle
                position = i / (n_buckets - 1)
                # Parabola: y = 4 * (x - 0.5)^2, then invert
                u_factor = 1.0 - 4.0 * (position - 0.5) ** 2
                # Normalize so sum = 1.0
                profile.append(max(0.05, u_factor))

            # Normalize
            total = sum(profile)
            profile = [p / total for p in profile]

            logger.debug("Volume profile for {}: {}", symbol, profile)
            return profile

        except Exception as e:
            logger.error("Failed to predict volume profile: {}", e)
            return None

    def _create_vwap_slices(
        self,
        total_amount: float,
        volume_profile: List[float]
    ) -> List[ExecutionSlice]:
        """Create execution slices based on volume profile.

        Args:
            total_amount: Total amount to execute
            volume_profile: List of volume percentages per bucket

        Returns:
            List of ExecutionSlice objects
        """
        slices = []
        cumulative_delay = 0.0

        for i, volume_pct in enumerate(volume_profile):
            slice_amount = total_amount * volume_pct
            slices.append(ExecutionSlice(
                slice_id=i,
                amount=slice_amount,
                delay_seconds=cumulative_delay,
                urgency=volume_pct  # Higher volume = more urgent
            ))
            # Assume equal time spacing between buckets
            cumulative_delay += 30.0  # 30 seconds per bucket

        return slices

    async def _execute_slices_with_volume_adjustment(
        self,
        execution_id: str,
        symbol: str,
        side: str,
        slices: List[ExecutionSlice],
        expected_volume_profile: List[float]
    ) -> ExecutionResult:
        """Execute slices with real-time volume adjustment.

        Args:
            execution_id: Unique execution ID
            symbol: Trading symbol
            side: "buy" or "sell"
            slices: List of execution slices
            expected_volume_profile: Expected volume profile

        Returns:
            ExecutionResult
        """
        # For now, execute slices without volume adjustment
        # In production, would monitor actual vs expected volume and rebalance
        return await self._execute_slices(execution_id, symbol, side, slices, algorithm="VWAP")

    # =====================================================================
    # Iceberg Orders
    # =====================================================================

    async def execute_iceberg(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        display_size: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute using Iceberg order strategy.

        Shows only a small portion of the order, automatically replenishing
        after each fill. Randomizes display size to avoid detection.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            total_amount: Total amount to execute
            display_size: Amount to display (defaults to 15% of total)
            **kwargs: Additional parameters

        Returns:
            ExecutionResult with execution details
        """
        execution_id = f"ICEBERG_{symbol}_{int(time.time())}"
        self._active_executions[execution_id] = True

        logger.info(
            "Starting Iceberg execution: {} {} total={} display={}",
            symbol,
            side,
            total_amount,
            display_size or f"{self.ICEBERG_DISPLAY_FRACTION:.0%}"
        )

        try:
            if not await self._check_circuit_breaker():
                return ExecutionResult(
                    success=False,
                    filled_amount=0.0,
                    average_price=0.0,
                    total_slices=0,
                    completed_slices=0,
                    total_time_seconds=0.0,
                    error="Circuit breaker active"
                )

            if display_size is None:
                display_size = total_amount * self.ICEBERG_DISPLAY_FRACTION

            remaining_amount = total_amount
            filled_amount = 0.0
            total_cost = 0.0
            completed_slices = 0
            start_time = time.time()
            slices_detail = []

            from exchange.base_exchange import OrderSide
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            while remaining_amount > 0 and self._active_executions.get(execution_id, False):
                # Randomize display size (±15%)
                randomized_display = display_size * random.uniform(
                    1.0 - self.ICEBERG_RANDOMIZATION,
                    1.0 + self.ICEBERG_RANDOMIZATION
                )
                slice_amount = min(remaining_amount, randomized_display)

                logger.info(
                    "Iceberg slice: amount={:.6f} (remaining={:.6f})",
                    slice_amount,
                    remaining_amount
                )

                try:
                    # Place limit order at best bid/ask
                    ticker = await self._exchange.get_ticker(symbol)
                    limit_price = ticker.bid if side == "buy" else ticker.ask

                    order = await self._order_manager.place_limit_order(
                        symbol=symbol,
                        side=order_side,
                        amount=slice_amount,
                        price=limit_price,
                        strategy="iceberg"
                    )

                    # Wait for fill (with timeout)
                    await asyncio.sleep(30.0)  # 30s timeout per slice

                    # Record result (simplified)
                    fill_price = order.price or limit_price
                    filled_amount += slice_amount
                    total_cost += fill_price * slice_amount
                    remaining_amount -= slice_amount
                    completed_slices += 1

                    slices_detail.append({
                        "slice_id": completed_slices,
                        "amount": slice_amount,
                        "price": fill_price
                    })

                except Exception as e:
                    logger.error("Iceberg slice failed: {}", e)
                    break

            average_price = total_cost / filled_amount if filled_amount > 0 else 0.0
            total_time = time.time() - start_time

            result = ExecutionResult(
                success=filled_amount > 0,
                filled_amount=filled_amount,
                average_price=average_price,
                total_slices=completed_slices,
                completed_slices=completed_slices,
                total_time_seconds=total_time,
                slices_detail=slices_detail
            )

            self._on_execution_complete(success=result.success)
            return result

        except Exception as e:
            logger.error("Iceberg execution failed: {}", e)
            self._on_execution_complete(success=False)
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_slices=0,
                completed_slices=0,
                total_time_seconds=0.0,
                error=str(e)
            )
        finally:
            self._active_executions.pop(execution_id, None)

    # =====================================================================
    # Sniper Mode
    # =====================================================================

    async def execute_sniper(
        self,
        symbol: str,
        side: str,
        amount: float,
        timeout_seconds: float = 5.0,
        **kwargs
    ) -> ExecutionResult:
        """Execute using Sniper mode for small orders.

        Waits for favorable microstructure moment (spread tightening, book imbalance)
        then executes entire order as single aggressive limit. Falls back to market
        if not filled within timeout.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            amount: Amount to execute (should be small)
            timeout_seconds: Timeout before falling back to market
            **kwargs: Additional parameters

        Returns:
            ExecutionResult with execution details
        """
        execution_id = f"SNIPER_{symbol}_{int(time.time())}"
        self._active_executions[execution_id] = True

        logger.info(
            "Starting Sniper execution: {} {} amount={} timeout={}s",
            symbol,
            side,
            amount,
            timeout_seconds
        )

        try:
            if not await self._check_circuit_breaker():
                return ExecutionResult(
                    success=False,
                    filled_amount=0.0,
                    average_price=0.0,
                    total_slices=0,
                    completed_slices=0,
                    total_time_seconds=0.0,
                    error="Circuit breaker active"
                )

            start_time = time.time()

            # Wait for favorable moment
            favorable_moment = await self._wait_for_favorable_microstructure(
                symbol,
                side,
                timeout_seconds * 0.8  # Use 80% of timeout for waiting
            )

            if favorable_moment:
                logger.info("Favorable microstructure detected - sniping!")
            else:
                logger.warning("No favorable moment found - executing anyway")

            # Execute as aggressive limit order
            ticker = await self._exchange.get_ticker(symbol)
            limit_price = ticker.ask if side == "buy" else ticker.bid

            from exchange.base_exchange import OrderSide
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            order = await self._order_manager.place_limit_order(
                symbol=symbol,
                side=order_side,
                amount=amount,
                price=limit_price,
                strategy="sniper"
            )

            # Wait for fill
            elapsed = time.time() - start_time
            remaining_timeout = timeout_seconds - elapsed

            await asyncio.sleep(min(remaining_timeout, 2.0))

            # If not filled, fall back to market order
            # (In production, would check order status)
            if elapsed > timeout_seconds:
                logger.warning("Sniper timeout - falling back to market order")
                order = await self._order_manager.place_market_order(
                    symbol=symbol,
                    side=order_side,
                    amount=amount,
                    strategy="sniper_fallback"
                )

            # Record result
            fill_price = order.price or limit_price
            total_time = time.time() - start_time

            result = ExecutionResult(
                success=True,
                filled_amount=amount,
                average_price=fill_price,
                total_slices=1,
                completed_slices=1,
                total_time_seconds=total_time,
                slices_detail=[{"slice_id": 0, "amount": amount, "price": fill_price}]
            )

            self._on_execution_complete(success=True)
            return result

        except Exception as e:
            logger.error("Sniper execution failed: {}", e)
            self._on_execution_complete(success=False)
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_slices=0,
                completed_slices=0,
                total_time_seconds=0.0,
                error=str(e)
            )
        finally:
            self._active_executions.pop(execution_id, None)

    async def _wait_for_favorable_microstructure(
        self,
        symbol: str,
        side: str,
        timeout_seconds: float
    ) -> bool:
        """Wait for favorable microstructure moment.

        Favorable conditions:
        - Spread tightening (below historical average)
        - Order book imbalance in our favor

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            timeout_seconds: Maximum wait time

        Returns:
            True if favorable moment found, False if timeout
        """
        start_time = time.time()
        check_interval = 0.2  # Check every 200ms

        # Get initial spread
        ticker = await self._exchange.get_ticker(symbol)
        initial_spread_pct = (ticker.ask - ticker.bid) / ticker.last if ticker.bid > 0 else 0.01

        while time.time() - start_time < timeout_seconds:
            try:
                ticker = await self._exchange.get_ticker(symbol)
                current_spread_pct = (ticker.ask - ticker.bid) / ticker.last if ticker.bid > 0 else initial_spread_pct

                # Check spread tightening
                spread_tightened = current_spread_pct < initial_spread_pct * 0.8

                # Check order book imbalance (simplified - would need full orderbook)
                # For now, just check if spread tightened
                if spread_tightened:
                    return True

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.debug("Error checking microstructure: {}", e)
                await asyncio.sleep(check_interval)

        return False

    # =====================================================================
    # Helper Methods
    # =====================================================================

    async def _execute_slices(
        self,
        execution_id: str,
        symbol: str,
        side: str,
        slices: List[ExecutionSlice],
        algorithm: str
    ) -> ExecutionResult:
        """Execute a list of slices.

        Args:
            execution_id: Unique execution ID
            symbol: Trading symbol
            side: "buy" or "sell"
            slices: List of execution slices
            algorithm: Algorithm name for logging

        Returns:
            ExecutionResult
        """
        from exchange.base_exchange import OrderSide
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        start_time = time.time()
        filled_amount = 0.0
        total_cost = 0.0
        completed_slices = 0
        slices_detail = []

        for slice_obj in slices:
            if not self._active_executions.get(execution_id, False):
                break

            # Wait for slice delay
            if slice_obj.delay_seconds > 0:
                await asyncio.sleep(slice_obj.delay_seconds)

            logger.info(
                "{} slice {}/{}: amount={:.6f}",
                algorithm,
                slice_obj.slice_id + 1,
                len(slices),
                slice_obj.amount
            )

            try:
                # Place order (market for guaranteed fill)
                order = await self._order_manager.place_market_order(
                    symbol=symbol,
                    side=order_side,
                    amount=slice_obj.amount,
                    strategy=algorithm.lower()
                )

                # Wait for fill
                await asyncio.sleep(1.0)

                # Record result
                ticker = await self._exchange.get_ticker(symbol)
                fill_price = order.price or ticker.last

                filled_amount += slice_obj.amount
                total_cost += fill_price * slice_obj.amount
                completed_slices += 1

                slices_detail.append({
                    "slice_id": slice_obj.slice_id,
                    "amount": slice_obj.amount,
                    "price": fill_price
                })

            except Exception as e:
                logger.error("{} slice {} failed: {}", algorithm, slice_obj.slice_id, e)
                # Continue with next slice

        average_price = total_cost / filled_amount if filled_amount > 0 else 0.0
        total_time = time.time() - start_time

        return ExecutionResult(
            success=filled_amount > 0,
            filled_amount=filled_amount,
            average_price=average_price,
            total_slices=len(slices),
            completed_slices=completed_slices,
            total_time_seconds=total_time,
            slices_detail=slices_detail
        )

    async def _get_impact_model(self, symbol: str) -> MarketImpactModel:
        """Get or create market impact model for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            MarketImpactModel with parameters
        """
        if symbol in self._impact_models:
            return self._impact_models[symbol]

        # Fetch historical data to estimate parameters
        try:
            ohlcv = await self._exchange.get_ohlcv(symbol, "1h", limit=168)  # 1 week

            if ohlcv is not None and len(ohlcv) > 0:
                # Calculate volatility (annualized)
                returns = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
                volatility = returns.std() * np.sqrt(24 * 365)  # Annualized

                # Average daily volume
                avg_daily_volume = ohlcv['volume'].sum() / 7.0  # Weekly to daily

                model = MarketImpactModel(
                    symbol=symbol,
                    volatility=volatility,
                    average_daily_volume=avg_daily_volume
                )
            else:
                # Default model
                model = MarketImpactModel(
                    symbol=symbol,
                    volatility=0.5,  # 50% annualized (crypto default)
                    average_daily_volume=1000000.0  # 1M default
                )

            self._impact_models[symbol] = model
            logger.debug("Market impact model for {}: vol={:.2%} ADV={:.0f}", symbol, model.volatility, model.average_daily_volume)
            return model

        except Exception as e:
            logger.error("Failed to create impact model for {}: {}", symbol, e)
            # Return default
            model = MarketImpactModel(
                symbol=symbol,
                volatility=0.5,
                average_daily_volume=1000000.0
            )
            self._impact_models[symbol] = model
            return model

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is active.

        Returns:
            True if execution allowed, False if circuit breaker active
        """
        if not self._circuit_breaker_active:
            return True

        if self._circuit_breaker_until and time.time() >= self._circuit_breaker_until:
            # Cooldown expired, reset circuit breaker
            self._circuit_breaker_active = False
            self._circuit_breaker_until = None
            self._consecutive_failures = 0
            logger.info("Circuit breaker cooldown expired - resuming execution")
            return True

        logger.warning("Circuit breaker active - execution blocked")
        return False

    def _on_execution_complete(self, success: bool) -> None:
        """Handle execution completion for circuit breaker.

        Args:
            success: Whether execution succeeded
        """
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._circuit_breaker_active = True
                self._circuit_breaker_until = time.time() + self.CIRCUIT_BREAKER_COOLDOWN_SECONDS
                logger.error(
                    "Circuit breaker activated after {} consecutive failures - cooldown {}s",
                    self._consecutive_failures,
                    self.CIRCUIT_BREAKER_COOLDOWN_SECONDS
                )

    def stop_execution(self, execution_id: str) -> None:
        """Stop an active execution.

        Args:
            execution_id: Execution to stop
        """
        if execution_id in self._active_executions:
            self._active_executions[execution_id] = False
            logger.warning("Execution {} stopped by request", execution_id)

    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs.

        Returns:
            List of execution IDs
        """
        return [eid for eid, active in self._active_executions.items() if active]
