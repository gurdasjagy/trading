"""Slippage estimation before order placement."""

from __future__ import annotations

import math
from typing import Any, Dict

from loguru import logger


class SlippageEstimator:
    """Estimates expected slippage before placing orders."""

    def estimate_slippage(
        self,
        symbol: str,
        amount: float,
        side: str,
        orderbook: Dict[str, Any],
    ) -> float:
        """Estimate the slippage fraction for filling *amount* on the given *side*.

        Walks the order-book levels and calculates the volume-weighted average
        price (VWAP) of the filled portion, then returns the fractional
        deviation from the best price.

        Args:
            symbol: Trading symbol (for logging).
            amount: Order size in base currency.
            side: ``"buy"`` or ``"sell"``.
            orderbook: Order-book dict with ``"asks"`` and ``"bids"`` keys,
                each a list of ``[price, size]`` pairs sorted best-first.

        Returns:
            Estimated slippage as a fraction (e.g. ``0.001`` = 0.1 %).
        """
        try:
            levels = orderbook.get("asks" if side == "buy" else "bids", [])
            if not levels:
                logger.warning("Empty order book for {} side={}", symbol, side)
                return 0.001  # default 0.1 %

            best_price = float(levels[0][0])
            if best_price <= 0:
                return 0.001

            remaining = amount
            total_cost = 0.0
            for level in levels:
                level_price, level_size = float(level[0]), float(level[1])
                fill = min(remaining, level_size)
                total_cost += fill * level_price
                remaining -= fill
                if remaining <= 0:
                    break

            filled = amount - remaining
            if filled <= 0:
                return 0.005  # thin book — assume 0.5 % slippage

            vwap = total_cost / filled
            slippage = abs(vwap - best_price) / best_price
            logger.debug("Slippage estimate for {} {}: {:.4%}", symbol, side, slippage)
            return slippage
        except Exception as exc:
            logger.error("Slippage estimation failed: {}", exc)
            return 0.001

    def is_slippage_acceptable(
        self,
        estimated: float,
        max_allowed: float = 0.001,
    ) -> bool:
        """Return True if the estimated slippage is within the acceptable bound.

        Args:
            estimated: Estimated slippage fraction.
            max_allowed: Maximum acceptable slippage fraction (default 0.1 %).

        Returns:
            ``True`` if slippage is acceptable.
        """
        acceptable = estimated <= max_allowed
        if not acceptable:
            logger.warning(
                "Slippage unacceptable: estimated={:.4%} max={:.4%}", estimated, max_allowed
            )
        return acceptable

    def calculate_market_impact(
        self,
        amount: float,
        volume: float,
        volatility: float,
    ) -> float:
        """Estimate market impact using the square-root model.

        Impact ≈ volatility × sqrt(amount / volume)

        Args:
            amount: Order size in base currency.
            volume: Average daily trading volume in base currency.
            volatility: Annualised or intraday volatility fraction.

        Returns:
            Estimated market impact as a fraction.
        """
        if volume <= 0 or volatility <= 0 or amount <= 0:
            return 0.0
        try:
            impact = volatility * math.sqrt(amount / volume)
            logger.debug(
                "Market impact: amount={} volume={} vol={} impact={:.4%}",
                amount,
                volume,
                volatility,
                impact,
            )
            return impact
        except Exception as exc:
            logger.error("Market impact calculation failed: {}", exc)
            return 0.0
