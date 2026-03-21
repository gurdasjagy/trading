"""Order flow execution engine - uses real-time order flow for optimal execution timing."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange


@dataclass
class OrderFlowState:
    """Current order flow state for a symbol."""

    symbol: str
    timestamp: float
    buy_volume: float
    sell_volume: float
    delta: float  # buy_volume - sell_volume
    cumulative_delta: float
    vpin: float  # Volume-Synchronized Probability of Informed Trading
    bid_absorption: float  # Large bids absorbing sells
    ask_absorption: float  # Large asks absorbing buys
    toxic_flow_detected: bool


class OrderFlowExecutionEngine:
    """Uses real-time order flow to optimize execution timing.

    Features:
    - Wait for favorable order flow before executing
    - Detect and avoid toxic flow (informed trading)
    - Momentum-aligned execution based on delta
    - Integration with order flow analysis from market data

    Execution Logic:
    - BUY: Wait for bid-side absorption (large resting bids absorbing sells)
    - SELL: Wait for ask-side absorption (large resting asks absorbing buys)
    - Avoid execution when VPIN > 0.7 and flow is against us
    - Use aggressive orders when momentum aligns, passive when neutral
    """

    # Flow thresholds
    VPIN_TOXIC_THRESHOLD = 0.7
    DELTA_STRONG_THRESHOLD = 100000.0  # Strong directional flow (USDT)
    ABSORPTION_THRESHOLD = 50000.0  # Large absorption volume (USDT)
    MAX_WAIT_SECONDS = 60.0  # Maximum wait for favorable flow

    def __init__(
        self,
        exchange: BaseExchange,
        order_flow_analyzer: Optional[Any] = None
    ):
        """Initialize order flow execution engine.

        Args:
            exchange: Exchange interface
            order_flow_analyzer: Order flow analyzer (if available)
        """
        self._exchange = exchange
        self._order_flow_analyzer = order_flow_analyzer

        # Order flow state tracking
        self._flow_states: Dict[str, OrderFlowState] = {}
        self._flow_history: Dict[str, deque] = {}  # symbol -> deque of recent states

        logger.info("OrderFlowExecutionEngine initialized")

    async def should_execute_now(
        self,
        symbol: str,
        side: str,
        urgency: str = "normal"
    ) -> Tuple[bool, str]:
        """Determine if execution should proceed based on order flow.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            urgency: "low", "normal", or "high"

        Returns:
            Tuple of (should_execute, reason)
        """
        # High urgency always executes
        if urgency == "high":
            return True, "High urgency execution"

        # Get current order flow state
        flow_state = await self._get_order_flow_state(symbol)

        if flow_state is None:
            # No order flow data available
            return True, "No order flow data available"

        # Check for toxic flow
        if flow_state.toxic_flow_detected and urgency == "low":
            return False, f"Toxic flow detected (VPIN={flow_state.vpin:.2f})"

        # Check VPIN threshold
        if flow_state.vpin > self.VPIN_TOXIC_THRESHOLD:
            # Check if flow is against our direction
            if side == "buy" and flow_state.delta < -self.DELTA_STRONG_THRESHOLD:
                return False, f"Strong sell flow detected (delta={flow_state.delta:.0f}, VPIN={flow_state.vpin:.2f})"
            elif side == "sell" and flow_state.delta > self.DELTA_STRONG_THRESHOLD:
                return False, f"Strong buy flow detected (delta={flow_state.delta:.0f}, VPIN={flow_state.vpin:.2f})"

        # Check for favorable absorption
        if side == "buy":
            if flow_state.bid_absorption > self.ABSORPTION_THRESHOLD:
                return True, f"Bid absorption detected ({flow_state.bid_absorption:.0f} USDT)"
        else:
            if flow_state.ask_absorption > self.ABSORPTION_THRESHOLD:
                return True, f"Ask absorption detected ({flow_state.ask_absorption:.0f} USDT)"

        # Check momentum alignment
        if abs(flow_state.delta) < self.DELTA_STRONG_THRESHOLD * 0.1:
            # Neutral flow - safe to execute
            return True, "Neutral order flow"

        if side == "buy" and flow_state.delta > 0:
            return True, f"Buy momentum aligned (delta={flow_state.delta:.0f})"
        elif side == "sell" and flow_state.delta < 0:
            return True, f"Sell momentum aligned (delta={flow_state.delta:.0f})"

        # Flow not strongly favorable, but not unfavorable
        if urgency == "normal":
            return True, "Normal urgency - proceeding despite neutral flow"

        # Low urgency waits for better conditions
        return False, "Waiting for more favorable order flow"

    async def wait_for_favorable_flow(
        self,
        symbol: str,
        side: str,
        max_wait_seconds: Optional[float] = None
    ) -> bool:
        """Wait for favorable order flow conditions.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            max_wait_seconds: Maximum wait time (defaults to MAX_WAIT_SECONDS)

        Returns:
            True if favorable conditions found, False if timeout
        """
        max_wait = max_wait_seconds or self.MAX_WAIT_SECONDS
        start_time = time.time()
        check_interval = 1.0  # Check every second

        logger.info("Waiting for favorable order flow: {} {}", symbol, side)

        while time.time() - start_time < max_wait:
            should_execute, reason = await self.should_execute_now(symbol, side, urgency="low")

            if should_execute:
                logger.info("Favorable flow found: {}", reason)
                return True

            await asyncio.sleep(check_interval)

        logger.warning("Order flow wait timeout after {:.1f}s", max_wait)
        return False

    async def get_execution_recommendation(
        self,
        symbol: str,
        side: str
    ) -> Dict:
        """Get execution recommendation based on order flow.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"

        Returns:
            Dict with execution recommendation
        """
        flow_state = await self._get_order_flow_state(symbol)

        if flow_state is None:
            return {
                "order_type": "market",
                "urgency": "normal",
                "wait_for_flow": False,
                "reason": "No order flow data"
            }

        # Determine order type based on flow
        if abs(flow_state.delta) > self.DELTA_STRONG_THRESHOLD:
            # Strong momentum - use market orders
            if (side == "buy" and flow_state.delta > 0) or (side == "sell" and flow_state.delta < 0):
                return {
                    "order_type": "market",
                    "urgency": "high",
                    "wait_for_flow": False,
                    "reason": f"Strong momentum aligned (delta={flow_state.delta:.0f})"
                }
            else:
                return {
                    "order_type": "iceberg",
                    "urgency": "low",
                    "wait_for_flow": True,
                    "reason": f"Momentum against us (delta={flow_state.delta:.0f})"
                }
        else:
            # Neutral flow - use limit orders
            return {
                "order_type": "limit",
                "urgency": "normal",
                "wait_for_flow": False,
                "reason": "Neutral flow - limit orders preferred"
            }

    async def _get_order_flow_state(self, symbol: str) -> Optional[OrderFlowState]:
        """Get current order flow state for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            OrderFlowState or None if unavailable
        """
        # If we have an order flow analyzer, use it
        if self._order_flow_analyzer:
            try:
                flow_data = await self._order_flow_analyzer.get_flow_state(symbol)
                if flow_data:
                    return self._parse_flow_data(symbol, flow_data)
            except Exception as e:
                logger.debug("Failed to get flow state from analyzer: {}", e)

        # Otherwise, calculate from recent trades (simplified)
        return await self._calculate_flow_state(symbol)

    async def _calculate_flow_state(self, symbol: str) -> Optional[OrderFlowState]:
        """Calculate order flow state from market data.

        This is a simplified version. In production, would use:
        - Real-time trade stream
        - Order book deltas
        - More sophisticated VPIN calculation

        Args:
            symbol: Trading symbol

        Returns:
            OrderFlowState or None if data unavailable
        """
        try:
            # Get recent orderbook
            orderbook = await self._exchange.get_orderbook(symbol, limit=20)
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return None

            # Calculate volumes from orderbook
            bid_volume = sum(bid[1] * bid[0] for bid in orderbook['bids'][:10])
            ask_volume = sum(ask[1] * ask[0] for ask in orderbook['asks'][:10])

            # Calculate delta (simplified - should use actual trade flow)
            delta = bid_volume - ask_volume

            # Get historical state for cumulative delta
            if symbol not in self._flow_history:
                self._flow_history[symbol] = deque(maxlen=60)  # 1 minute of history
                cumulative_delta = delta
            else:
                recent_deltas = [state.delta for state in self._flow_history[symbol]]
                cumulative_delta = sum(recent_deltas) + delta

            # Calculate VPIN (simplified)
            # VPIN = volume-weighted probability of informed trading
            # Real calculation requires bulk volume classification
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                vpin = abs(delta) / total_volume
            else:
                vpin = 0.0

            # Detect absorption (large resting orders being hit)
            bid_absorption = 0.0
            ask_absorption = 0.0

            # Check for large resting bids/asks (top 3 levels)
            if orderbook['bids']:
                top_bids = orderbook['bids'][:3]
                bid_absorption = sum(bid[1] * bid[0] for bid in top_bids if bid[1] > 10.0)

            if orderbook['asks']:
                top_asks = orderbook['asks'][:3]
                ask_absorption = sum(ask[1] * ask[0] for ask in top_asks if ask[1] > 10.0)

            # Detect toxic flow
            toxic_flow_detected = vpin > self.VPIN_TOXIC_THRESHOLD

            state = OrderFlowState(
                symbol=symbol,
                timestamp=time.time(),
                buy_volume=bid_volume,
                sell_volume=ask_volume,
                delta=delta,
                cumulative_delta=cumulative_delta,
                vpin=vpin,
                bid_absorption=bid_absorption,
                ask_absorption=ask_absorption,
                toxic_flow_detected=toxic_flow_detected
            )

            # Store state
            self._flow_states[symbol] = state
            self._flow_history[symbol].append(state)

            return state

        except Exception as e:
            logger.debug("Failed to calculate flow state: {}", e)
            return None

    def _parse_flow_data(self, symbol: str, flow_data: Dict) -> OrderFlowState:
        """Parse flow data from order flow analyzer.

        Args:
            symbol: Trading symbol
            flow_data: Flow data dict

        Returns:
            OrderFlowState
        """
        return OrderFlowState(
            symbol=symbol,
            timestamp=flow_data.get('timestamp', time.time()),
            buy_volume=flow_data.get('buy_volume', 0.0),
            sell_volume=flow_data.get('sell_volume', 0.0),
            delta=flow_data.get('delta', 0.0),
            cumulative_delta=flow_data.get('cumulative_delta', 0.0),
            vpin=flow_data.get('vpin', 0.0),
            bid_absorption=flow_data.get('bid_absorption', 0.0),
            ask_absorption=flow_data.get('ask_absorption', 0.0),
            toxic_flow_detected=flow_data.get('toxic_flow_detected', False)
        )

    def get_flow_summary(self, symbol: str) -> Dict:
        """Get order flow summary for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with flow summary
        """
        if symbol not in self._flow_states:
            return {"status": "no_data", "symbol": symbol}

        state = self._flow_states[symbol]

        return {
            "symbol": symbol,
            "timestamp": state.timestamp,
            "delta": state.delta,
            "cumulative_delta": state.cumulative_delta,
            "vpin": state.vpin,
            "bid_absorption": state.bid_absorption,
            "ask_absorption": state.ask_absorption,
            "toxic_flow": state.toxic_flow_detected,
            "buy_recommendation": "execute" if state.delta > 0 and not state.toxic_flow_detected else "wait",
            "sell_recommendation": "execute" if state.delta < 0 and not state.toxic_flow_detected else "wait"
        }
