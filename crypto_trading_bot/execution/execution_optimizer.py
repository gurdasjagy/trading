"""Execution optimiser — decides timing, order type, and splitting strategy."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from .fee_calculator import FeeCalculator
from .slippage_estimator import SlippageEstimator


class ExecutionOptimizer:
    """Optimises order execution timing, type, and splitting."""

    # Order size above this fraction of average volume triggers splitting
    _SPLIT_VOLUME_THRESHOLD = 0.05  # 5 % of avg daily volume

    def __init__(self) -> None:
        self._fee_calc = FeeCalculator()
        self._slippage = SlippageEstimator()

    def should_use_limit(self, symbol: str, urgency: str = "normal") -> bool:
        """Return True if a limit order is preferred over a market order.

        Market orders are preferred by default for instant execution.
        Limit orders are only used when explicitly requested (urgency="low").

        Args:
            symbol: Trading symbol (for logging).
            urgency: ``"low"`` forces limit orders, otherwise market orders are used.

        Returns:
            ``True`` if a limit order should be used.
        """
        use_limit = urgency == "low"
        logger.debug("Use limit for {}: {} (urgency={})", symbol, use_limit, urgency)
        return use_limit

    def calculate_optimal_limit_price(
        self,
        symbol: str,
        side: str,
        spread: float,
        mid_price: float,
    ) -> float:
        """Calculate an aggressive-but-inside limit price.

        Places the order just inside the spread (25 % of spread from mid).

        Args:
            symbol: Trading symbol.
            side: ``"buy"`` or ``"sell"``.
            spread: Current bid-ask spread in price units.
            mid_price: Current mid-price.

        Returns:
            Optimal limit price.
        """
        if mid_price <= 0:
            return mid_price
        offset = spread * 0.25
        if side == "buy":
            price = mid_price - offset
        else:
            price = mid_price + offset
        logger.debug(
            "Optimal limit price for {} {}: mid={} spread={} price={:.6f}",
            symbol,
            side,
            mid_price,
            spread,
            price,
        )
        return price

    def should_split_order(self, amount: float, avg_volume: float) -> bool:
        """Return True if the order is large enough to warrant splitting.

        Args:
            amount: Order size in base currency.
            avg_volume: Average daily trading volume in base currency.

        Returns:
            ``True`` if the order should be split into chunks.
        """
        if avg_volume <= 0:
            return False
        ratio = amount / avg_volume
        split = ratio >= self._SPLIT_VOLUME_THRESHOLD
        logger.debug(
            "Split order: amount={} avg_vol={} ratio={:.4%} split={}",
            amount,
            avg_volume,
            ratio,
            split,
        )
        return split

    def get_execution_plan(
        self,
        signal: dict,
        market_data: Dict[str, Any],
    ) -> dict:
        """Build a complete execution plan for a trade signal.

        Args:
            signal: Trade signal dict with at least ``symbol``, ``side``,
                ``amount``, and ``urgency`` keys.
            market_data: Market context dict with ``mid_price``, ``spread``,
                ``avg_volume``, and ``orderbook`` keys.

        Returns:
            Execution plan dict with ``order_type``, ``limit_price``,
            ``split_orders``, ``chunks``, and ``estimated_slippage`` keys.
        """
        symbol = signal.get("symbol", "")
        side = signal.get("side", "buy")
        amount = signal.get("amount", 0.0)
        urgency = signal.get("urgency", "normal")

        mid_price = market_data.get("mid_price", 0.0)
        spread = market_data.get("spread", 0.0)
        avg_volume = market_data.get("avg_volume", 0.0)
        orderbook = market_data.get("orderbook", {})

        use_limit = self.should_use_limit(symbol, urgency)
        order_type = "limit" if use_limit else "market"
        limit_price = (
            self.calculate_optimal_limit_price(symbol, side, spread, mid_price)
            if use_limit
            else 0.0
        )
        split = self.should_split_order(amount, avg_volume)
        chunks: list = []
        if split and amount > 0:
            chunk_size = max(amount / 4, amount * 0.1)
            remaining = amount
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                chunks.append(round(chunk, 8))
                remaining -= chunk

        estimated_slippage = self._slippage.estimate_slippage(symbol, amount, side, orderbook)

        plan = {
            "symbol": symbol,
            "side": side,
            "total_amount": amount,
            "order_type": order_type,
            "limit_price": limit_price,
            "split_orders": split,
            "chunks": chunks if split else [amount],
            "estimated_slippage": estimated_slippage,
            "urgency": urgency,
        }
        logger.info(
            "Execution plan for {}: type={} split={} chunks={}",
            symbol,
            order_type,
            split,
            len(chunks),
        )
        return plan
