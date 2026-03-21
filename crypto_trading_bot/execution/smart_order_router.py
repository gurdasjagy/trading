"""Smart order routing system for multi-exchange execution.

Routes orders intelligently across multiple exchanges to achieve best execution
based on liquidity, fees, spreads, and historical fill performance.

In the upgraded architecture the **actual order placement** is performed by the
Rust ``trading_engine`` binary (see ``rust_engine/src/execution_gateway.rs``).
This module now acts as a **routing config generator**: it scores venues,
computes optimal allocation percentages, and pushes the resulting routing config
to the Rust engine via ZeroMQ PUSH.  The Rust process consumes the config on
its PULL socket and executes the orders using its own connection pool.

Direct order placement via ``_execute_market_order`` / ``_execute_limit_order``
is retained only for back-testing and paper-trading (when the Rust binary is
not running).
"""

import asyncio
import json
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from exchange.base_exchange import BaseExchange, Order, OrderSide

try:
    from rust_trading_engine.execution_prep import score_venues as rust_score_venues
    _USE_RUST_SCORING = True
except ImportError:
    _USE_RUST_SCORING = False


@dataclass
class ExchangeVenue:
    """Represents an exchange venue for order routing."""

    name: str
    exchange: BaseExchange
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    enabled: bool = True
    latency_ms: float = 100.0
    reliability_score: float = 1.0  # 0.0 to 1.0
    historical_fill_rate: float = 0.95  # 95%


@dataclass
class RoutingDecision:
    """Order routing decision."""

    venue_name: str
    allocation_pct: float
    amount: float
    expected_price: float
    expected_fee: float
    reason: str


@dataclass
class ExecutionReport:
    """Execution report after routing."""

    total_filled: float
    average_price: float
    total_fees: float
    venue_fills: Dict[str, float] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    # True when the routing config was pushed to the Rust engine for async execution.
    # In this case total_filled/average_price are placeholders; actual fill
    # confirmations arrive via ZeroMQ telemetry (topic "fill").
    routed_to_rust: bool = False


class SmartOrderRouter:
    """Smart order routing engine for best execution.

    Routes orders across multiple exchanges based on:
    - Liquidity (order book depth)
    - Fees (maker/taker)
    - Spreads
    - Historical execution quality
    - Venue reliability

    In the upgraded architecture this class computes an optimal routing config
    and forwards it to the Rust ``trading_engine`` via ZeroMQ PUSH (see
    ``rust_engine/src/execution_gateway.rs``).  Direct order placement is
    retained as a fallback for paper-trading and back-testing.

    Args:
        venues: List of exchange venues
        min_venue_allocation: Minimum % to allocate to a venue (default 5%)
        max_venues_per_order: Maximum venues to split order across
        rust_config_push_addr: ZeroMQ PUSH address of the Rust engine's
            config PULL socket (default ``tcp://127.0.0.1:5556``).
    """

    def __init__(
        self,
        venues: List[ExchangeVenue],
        min_venue_allocation: float = 0.05,
        max_venues_per_order: int = 3,
        rust_config_push_addr: str = "tcp://127.0.0.1:5556",
    ):
        self.venues = {v.name: v for v in venues if v.enabled}
        self.min_venue_allocation = min_venue_allocation
        self.max_venues_per_order = max_venues_per_order
        self._rust_config_push_addr = rust_config_push_addr

        # ZeroMQ PUSH socket (lazy init on first use)
        self._zmq_push_sock: Optional[object] = None
        self._zmq_ctx: Optional[object] = None

        # Track execution history for learning
        self._execution_history: deque = deque(maxlen=1000)

        logger.info(
            f"SmartOrderRouter initialized with {len(self.venues)} venues: "
            f"{list(self.venues.keys())}"
        )

    # ------------------------------------------------------------------
    # ZeroMQ helpers
    # ------------------------------------------------------------------

    def _get_zmq_push_sock(self) -> Optional[object]:
        """Lazily initialise and return the ZeroMQ PUSH socket."""
        if self._zmq_push_sock is not None:
            return self._zmq_push_sock
        try:
            import zmq
            self._zmq_ctx = zmq.Context()
            sock = self._zmq_ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 1000)
            sock.connect(self._rust_config_push_addr)
            self._zmq_push_sock = sock
            return sock
        except ImportError:
            return None

    def _push_routing_config_to_rust(self, config: dict) -> bool:
        """Send routing config to Rust engine via ZeroMQ PUSH.

        Returns True if successfully sent, False otherwise.
        """
        sock = self._get_zmq_push_sock()
        if sock is None:
            return False
        try:
            import zmq
            payload = json.dumps(config).encode()
            sock.send(payload, flags=zmq.NOBLOCK)
            return True
        except Exception as exc:
            logger.debug("Failed to push routing config to Rust: {}", exc)
            return False

    async def compute_routing_config(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        """Compute an optimal routing config dict without placing any orders.

        This is the primary entry point in the upgraded architecture.  Call
        this method to derive venue scores and allocation percentages, then
        either push the result to the Rust engine via :meth:`route_order` or
        consume it directly in your strategy code.

        Returns:
            Dict with keys:
              ``symbol``, ``side``, ``amount``, ``order_type``,
              ``limit_price``, ``venues`` (list of
              ``{"name", "allocation_pct", "amount", "expected_fee"}``).
        """
        venue_data = await self._gather_venue_data(symbol)
        if not venue_data:
            raise ValueError(f"No venue data available for {symbol}")

        venue_scores = await self._score_venues(venue_data, side, amount)
        routing_decisions = self._calculate_routing_allocation(venue_scores, amount)
        if not routing_decisions:
            raise ValueError("Unable to determine routing allocation")

        return {
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "order_type": order_type,
            "limit_price": limit_price,
            "venues": [
                {
                    "name": d.venue_name,
                    "allocation_pct": d.allocation_pct,
                    "amount": d.amount,
                    "expected_fee": d.expected_fee,
                    "reason": d.reason,
                }
                for d in routing_decisions
            ],
        }

    async def route_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> ExecutionReport:
        """Route an order: push routing config to Rust engine, fall back to Python.

        In the upgraded architecture this method:
        1. Computes optimal venue allocation via :meth:`compute_routing_config`.
        2. Pushes the routing config to the Rust ``trading_engine`` via ZeroMQ.
        3. If the Rust engine is not reachable (e.g. paper-trading / tests),
           falls back to direct Python order placement.

        Args:
            symbol: Trading pair symbol
            side: Buy or sell
            amount: Total amount to execute
            order_type: 'market' or 'limit'
            limit_price: Limit price if order_type is 'limit'

        Returns:
            ExecutionReport with execution details
        """
        logger.info(
            f"Routing {order_type} order: {side.value} {amount} {symbol} "
            f"across {len(self.venues)} venues"
        )

        start_time = datetime.utcnow()

        # Step 1: Gather market data from all venues
        venue_data = await self._gather_venue_data(symbol)

        if not venue_data:
            raise ValueError(f"No venue data available for {symbol}")

        # Step 2: Analyze and score venues
        venue_scores = await self._score_venues(venue_data, side, amount)

        # Step 3: Determine optimal routing allocation
        routing_decisions = self._calculate_routing_allocation(
            venue_scores, amount
        )

        if not routing_decisions:
            raise ValueError("Unable to determine routing allocation")

        logger.info(
            f"Routing plan: {len(routing_decisions)} venues, "
            f"allocations: {[(r.venue_name, f'{r.allocation_pct*100:.1f}%') for r in routing_decisions]}"
        )

        # Step 4a: Push routing config to the Rust engine.
        # The Rust binary executes the orders using its own connection pool.
        routing_config = {
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "order_type": order_type,
            "limit_price": limit_price,
            "venues": [
                {
                    "name": d.venue_name,
                    "allocation_pct": d.allocation_pct,
                    "amount": d.amount,
                    "expected_fee": d.expected_fee,
                }
                for d in routing_decisions
            ],
        }
        rust_sent = self._push_routing_config_to_rust(routing_config)

        if rust_sent:
            logger.info(
                "Routing config pushed to Rust engine for {} {} {} ({})",
                order_type, side.value, symbol, amount,
            )
            # Return a placeholder report with routed_to_rust=True.
            # Callers MUST check this flag before using total_filled/average_price,
            # as they are placeholder zeros.  Actual fill confirmations arrive
            # via ZeroMQ telemetry (topic "fill") consumed by
            # EventDrivenEngine._consume_rust_telemetry().
            return ExecutionReport(
                total_filled=0.0,
                average_price=0.0,
                total_fees=0.0,
                venue_fills={d.venue_name: 0.0 for d in routing_decisions},
                orders=[],
                execution_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                routed_to_rust=True,
            )

        # Step 4b: Rust engine not reachable — fall back to Python direct placement.
        logger.debug(
            "Rust engine not reachable — falling back to Python order placement for {}.",
            symbol,
        )
        warnings.warn(
            "SmartOrderRouter is falling back to direct Python order placement. "
            "Ensure the Rust trading_engine binary is running for production use.",
            RuntimeWarning,
            stacklevel=2,
        )
        execution_tasks = []
        for decision in routing_decisions:
            venue = self.venues[decision.venue_name]

            if order_type == "market":
                task = self._execute_market_order(
                    venue, symbol, side, decision.amount
                )
            else:
                task = self._execute_limit_order(
                    venue, symbol, side, decision.amount, limit_price
                )

            execution_tasks.append((decision.venue_name, task))

        # Execute in parallel
        results = await asyncio.gather(
            *[task for _, task in execution_tasks],
            return_exceptions=True,
        )

        # Step 5: Aggregate results
        report = self._aggregate_execution_results(
            routing_decisions, execution_tasks, results, start_time
        )

        # Step 6: Record for learning
        self._record_execution(symbol, side, amount, routing_decisions, report)

        logger.info(
            f"Order routing complete: filled {report.total_filled:.6f} / {amount:.6f} "
            f"@ avg {report.average_price:.2f}, fees {report.total_fees:.4f} "
            f"in {report.execution_time_seconds:.2f}s"
        )

        return report

    async def _gather_venue_data(self, symbol: str) -> Dict[str, Dict]:
        """Gather market data from all venues concurrently."""
        tasks = {}
        for venue_name, venue in self.venues.items():
            tasks[venue_name] = self._fetch_venue_market_data(venue, symbol)

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        venue_data = {}
        for venue_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch data from {venue_name}: {result}")
            elif result:
                venue_data[venue_name] = result

        return venue_data

    async def _fetch_venue_market_data(
        self, venue: ExchangeVenue, symbol: str
    ) -> Optional[Dict]:
        """Fetch market data from a single venue."""
        try:
            # Fetch ticker and order book concurrently
            ticker_task = venue.exchange.get_ticker(symbol)
            orderbook_task = venue.exchange.get_orderbook(symbol, limit=20)

            ticker, orderbook = await asyncio.gather(ticker_task, orderbook_task)

            # Calculate available liquidity
            bid_liquidity = sum(bid[1] for bid in orderbook.get("bids", [])[:10])
            ask_liquidity = sum(ask[1] for ask in orderbook.get("asks", [])[:10])

            return {
                "ticker": ticker,
                "orderbook": orderbook,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "spread_pct": (
                    (ticker.ask - ticker.bid) / ticker.last * 100
                    if ticker.last > 0
                    else 0.0
                ),
            }

        except Exception as exc:
            logger.error(f"Error fetching market data from {venue.name}: {exc}")
            return None

    async def _score_venues(
        self,
        venue_data: Dict[str, Dict],
        side: OrderSide,
        amount: float,
    ) -> Dict[str, float]:
        """Score venues based on multiple criteria.

        Scoring factors:
        - Liquidity (40%)
        - Fees (25%)
        - Spread (20%)
        - Reliability (10%)
        - Historical performance (5%)
        """
        if _USE_RUST_SCORING:
            try:
                venues_input = []
                for venue_name, data in venue_data.items():
                    venue = self.venues[venue_name]
                    relevant_liq = (
                        data["ask_liquidity"] if side == OrderSide.BUY else data["bid_liquidity"]
                    )
                    venues_input.append((
                        venue_name,
                        venue.taker_fee,
                        relevant_liq,
                        data["spread_pct"],
                        venue.reliability_score,
                        venue.historical_fill_rate,
                    ))
                results = rust_score_venues(
                    venues_input,
                    amount,
                    self.min_venue_allocation,
                    self.max_venues_per_order,
                )
                # Convert list of (name, alloc_pct, expected_fee) to scores dict
                # Re-derive scores as allocation percentages (proportional to original scores)
                return {name: alloc_pct for name, alloc_pct, _ in results}
            except Exception as exc:
                logger.debug("Rust venue scoring failed: {} — falling back to Python", exc)

        venue_scores = {}

        for venue_name, data in venue_data.items():
            venue = self.venues[venue_name]

            # Liquidity score (higher = better)
            relevant_liquidity = (
                data["ask_liquidity"] if side == OrderSide.BUY else data["bid_liquidity"]
            )
            liquidity_score = min(1.0, relevant_liquidity / (amount * 2.0))

            # Fee score (lower fees = better)
            fee_rate = venue.taker_fee  # Assuming market orders
            fee_score = 1.0 - (fee_rate / 0.002)  # Normalize to 0.2% max
            fee_score = max(0.0, min(1.0, fee_score))

            # Spread score (tighter spread = better)
            spread_pct = data["spread_pct"]
            spread_score = 1.0 - min(1.0, spread_pct / 0.5)  # 0.5% max spread

            # Reliability score
            reliability_score = venue.reliability_score

            # Historical performance score
            history_score = venue.historical_fill_rate

            # Weighted total score
            total_score = (
                liquidity_score * 0.40
                + fee_score * 0.25
                + spread_score * 0.20
                + reliability_score * 0.10
                + history_score * 0.05
            )

            venue_scores[venue_name] = total_score

            logger.debug(
                f"{venue_name} score: {total_score:.3f} "
                f"(liq={liquidity_score:.2f}, fee={fee_score:.2f}, "
                f"spread={spread_score:.2f}, rel={reliability_score:.2f})"
            )

        return venue_scores

    def _calculate_routing_allocation(
        self,
        venue_scores: Dict[str, float],
        total_amount: float,
    ) -> List[RoutingDecision]:
        """Calculate optimal allocation across venues."""
        if not venue_scores:
            return []

        # Sort venues by score (descending)
        sorted_venues = sorted(
            venue_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top N venues
        top_venues = sorted_venues[: self.max_venues_per_order]

        if not top_venues:
            return []

        # Calculate allocation based on scores
        total_score = sum(score for _, score in top_venues)

        if total_score == 0:
            # Equal allocation if all scores are 0
            allocation_pct = 1.0 / len(top_venues)
            allocations = {name: allocation_pct for name, _ in top_venues}
        else:
            # Proportional to scores
            allocations = {
                name: score / total_score for name, score in top_venues
            }

        # Filter out venues below minimum allocation
        allocations = {
            name: pct
            for name, pct in allocations.items()
            if pct >= self.min_venue_allocation
        }

        # Renormalize if we filtered some out
        total_pct = sum(allocations.values())
        if total_pct > 0:
            allocations = {name: pct / total_pct for name, pct in allocations.items()}

        # Create routing decisions
        decisions = []
        for venue_name, allocation_pct in allocations.items():
            venue = self.venues[venue_name]
            amount = total_amount * allocation_pct

            decisions.append(
                RoutingDecision(
                    venue_name=venue_name,
                    allocation_pct=allocation_pct,
                    amount=amount,
                    expected_price=0.0,  # Will be filled on execution
                    expected_fee=amount * venue.taker_fee,
                    reason=f"Score: {venue_scores[venue_name]:.3f}",
                )
            )

        return decisions

    async def _execute_market_order(
        self,
        venue: ExchangeVenue,
        symbol: str,
        side: OrderSide,
        amount: float,
    ) -> Order:
        """Execute a market order on a specific venue."""
        try:
            logger.debug(
                f"Executing on {venue.name}: {side.value} {amount:.6f} {symbol}"
            )

            order = await venue.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
            )

            logger.info(
                f"{venue.name} filled: {order.filled:.6f} @ {order.price:.2f}"
            )

            return order

        except Exception as exc:
            logger.error(f"Order execution failed on {venue.name}: {exc}")
            raise

    async def _execute_limit_order(
        self,
        venue: ExchangeVenue,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: Optional[float],
    ) -> Order:
        """Execute a limit order on a specific venue."""
        if price is None:
            raise ValueError("Limit price required for limit orders")

        try:
            logger.debug(
                f"Executing limit on {venue.name}: {side.value} {amount:.6f} "
                f"{symbol} @ {price:.2f}"
            )

            order = await venue.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
            )

            logger.info(
                f"{venue.name} limit order placed: {order.id}, "
                f"filled: {order.filled:.6f}"
            )

            return order

        except Exception as exc:
            logger.error(f"Limit order failed on {venue.name}: {exc}")
            raise

    def _aggregate_execution_results(
        self,
        routing_decisions: List[RoutingDecision],
        execution_tasks: List[Tuple[str, asyncio.Task]],
        results: List,
        start_time: datetime,
    ) -> ExecutionReport:
        """Aggregate execution results from all venues."""
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        venue_fills = {}
        orders = []

        for (venue_name, _), result in zip(execution_tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Execution on {venue_name} failed: {result}")
                venue_fills[venue_name] = 0.0
                continue

            if isinstance(result, Order):
                order = result
                filled = order.filled
                price = order.price if order.price else 0.0
                fee = order.fee

                total_filled += filled
                total_cost += filled * price
                total_fees += fee
                venue_fills[venue_name] = filled
                orders.append(order)

        average_price = total_cost / total_filled if total_filled > 0 else 0.0
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        return ExecutionReport(
            total_filled=total_filled,
            average_price=average_price,
            total_fees=total_fees,
            venue_fills=venue_fills,
            orders=orders,
            execution_time_seconds=execution_time,
        )

    def _record_execution(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        routing_decisions: List[RoutingDecision],
        report: ExecutionReport,
    ) -> None:
        """Record execution for learning and optimization."""
        execution_record = {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "filled": report.total_filled,
            "average_price": report.average_price,
            "total_fees": report.total_fees,
            "execution_time": report.execution_time_seconds,
            "venue_allocations": {
                d.venue_name: d.allocation_pct for d in routing_decisions
            },
            "venue_fills": report.venue_fills,
        }

        self._execution_history.append(execution_record)

        # Update venue fill rates
        for venue_name in self.venues:
            if venue_name in report.venue_fills:
                filled = report.venue_fills[venue_name]
                expected = next(
                    (d.amount for d in routing_decisions if d.venue_name == venue_name),
                    0.0,
                )

                if expected > 0:
                    fill_rate = filled / expected
                    # Exponential moving average
                    alpha = 0.1
                    current_rate = self.venues[venue_name].historical_fill_rate
                    new_rate = alpha * fill_rate + (1 - alpha) * current_rate
                    self.venues[venue_name].historical_fill_rate = new_rate

    def get_routing_statistics(self) -> Dict:
        """Get statistics on historical routing decisions."""
        if not self._execution_history:
            return {}

        total_executions = len(self._execution_history)
        total_volume = sum(e["amount"] for e in self._execution_history)
        avg_execution_time = np.mean([e["execution_time"] for e in self._execution_history])

        # Venue usage
        venue_usage = {}
        for venue_name in self.venues:
            usage_count = sum(
                1 for e in self._execution_history if venue_name in e["venue_fills"]
            )
            venue_usage[venue_name] = {
                "usage_pct": usage_count / total_executions * 100,
                "fill_rate": self.venues[venue_name].historical_fill_rate * 100,
            }

        return {
            "total_executions": total_executions,
            "total_volume": total_volume,
            "avg_execution_time_seconds": avg_execution_time,
            "venue_usage": venue_usage,
        }
