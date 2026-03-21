"""Microstructure-aware position sizing adjustments.

Applies multipliers to position sizes based on real-time market microstructure
indicators from the Rust engine: VPIN, order book depth, and spread percentiles.
"""

from __future__ import annotations

from typing import List

from loguru import logger


class MicrostructureSizer:
    """Calculate position size multipliers based on market microstructure.

    Integrates with Rust microstructure engine outputs (VPIN, depth, spread)
    to reduce position sizes when market conditions are adverse.
    """

    def __init__(self) -> None:
        pass

    def get_vpin_multiplier(self, vpin: float) -> float:
        """Return a position size multiplier based on VPIN (Volume-Synchronized Probability of Informed Trading).

        VPIN ranges from 0 to 1, where higher values indicate more informed
        trading (toxic flow) and increased adverse selection risk.

        Args:
            vpin: VPIN value from Rust microstructure engine (0-1).

        Returns:
            Multiplier in range [0.7, 1.0]:
                - vpin < 0.5  → 1.0  (normal conditions)
                - 0.5-0.7     → 0.85 (moderate toxicity)
                - > 0.7       → 0.7  (high toxicity, reduce size)
        """
        if vpin < 0.5:
            return 1.0
        elif vpin < 0.7:
            return 0.85
        else:
            return 0.7

    def get_depth_multiplier(self, depth_usdt: float, position_notional: float) -> float:
        """Return a position size multiplier based on order book depth.

        Reduces position size when the order is large relative to visible
        book depth to avoid excessive slippage and market impact.

        Args:
            depth_usdt: Available liquidity in USDT at the best bid/ask.
            position_notional: Intended position size in USDT.

        Returns:
            Multiplier in range [0.0, 1.0]:
                - depth >= 2× position → 1.0 (sufficient liquidity)
                - depth < 2× position  → depth / (2 × position) (scale down)
        """
        if depth_usdt <= 0 or position_notional <= 0:
            return 1.0

        required_depth = position_notional * 2.0
        if depth_usdt >= required_depth:
            return 1.0

        multiplier = depth_usdt / required_depth
        return min(1.0, max(0.0, multiplier))

    def get_spread_multiplier(self, spread_bps: float, spread_history: List[float]) -> float:
        """Return a position size multiplier based on spread percentile.

        Reduces position size when the current spread is abnormally wide
        (indicating poor liquidity or market stress).

        Args:
            spread_bps: Current bid-ask spread in basis points.
            spread_history: Rolling history of spread_bps values (last 100 samples).

        Returns:
            Multiplier in range [0.8, 1.0]:
                - spread < 75th percentile → 1.0  (normal spread)
                - 75-90th percentile      → 0.9  (wide spread)
                - > 90th percentile       → 0.8  (extreme spread)
        """
        if not spread_history or len(spread_history) < 10:
            # Insufficient history — use conservative default
            return 1.0 if spread_bps < 10.0 else 0.9

        sorted_spreads = sorted(spread_history)
        n = len(sorted_spreads)

        # Calculate percentile rank of current spread
        rank = sum(1 for s in sorted_spreads if s < spread_bps)
        percentile = (rank / n) * 100.0

        if percentile < 75.0:
            return 1.0
        elif percentile < 90.0:
            return 0.9
        else:
            return 0.8
