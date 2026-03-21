"""Order router — selects the optimal exchange for each order."""

from __future__ import annotations

from typing import Dict, List

from loguru import logger

from .fee_calculator import FeeCalculator


class OrderRouter:
    """Routes orders to the optimal exchange based on fees and liquidity."""

    def __init__(self) -> None:
        self._fee_calc = FeeCalculator()

    def select_exchange(self, symbol: str, order_type: str, available_exchanges: List[str]) -> str:
        """Select the best exchange for the given symbol and order type.

        Selection priority: lowest fee → then best liquidity (simplified).

        Args:
            symbol: Trading symbol.
            order_type: ``"limit"`` or ``"market"``.
            available_exchanges: List of connected exchange identifiers.

        Returns:
            Chosen exchange identifier.
        """
        if not available_exchanges:
            logger.warning("No exchanges available for routing; defaulting to 'mexc'")
            return "mexc"
        chosen = self._fee_calc.get_cheapest_exchange(available_exchanges, order_type)
        logger.info("Order router: {} {} → {}", symbol, order_type, chosen)
        return chosen

    def route_order(self, signal: dict, exchanges: List[str]) -> str:
        """Route an order signal to the optimal exchange.

        Args:
            signal: Trade signal dict with at minimum ``symbol`` and ``order_type`` keys.
            exchanges: List of available exchange identifiers.

        Returns:
            Chosen exchange identifier.
        """
        symbol = signal.get("symbol", "")
        order_type = signal.get("order_type", "limit")
        return self.select_exchange(symbol, order_type, exchanges)

    def compare_fees(self, exchanges: List[str], symbol: str) -> Dict[str, float]:
        """Return a fee comparison dict for each exchange.

        Args:
            exchanges: Exchange identifiers to compare.
            symbol: Trading symbol (for logging).

        Returns:
            Dict mapping exchange → estimated round-trip fee fraction.
        """
        from .fee_calculator import _DEFAULT_FEE, _FEE_TABLE

        result: Dict[str, float] = {}
        for ex in exchanges:
            fee_rates = _FEE_TABLE.get(ex.lower(), _DEFAULT_FEE)
            # Round-trip = 2 × limit maker rate
            result[ex] = fee_rates.get("limit", 0.0002) * 2
        logger.debug("Fee comparison for {}: {}", symbol, result)
        return result

    def compare_liquidity(self, exchanges: List[str], symbol: str) -> Dict[str, float]:
        """Return a mock liquidity score for each exchange.

        In production this should query real order-book depth from each exchange.

        Args:
            exchanges: Exchange identifiers to compare.
            symbol: Trading symbol.

        Returns:
            Dict mapping exchange → relative liquidity score (higher is better).
        """
        # Placeholder relative scores — replace with real order-book depth data
        liquidity_scores: Dict[str, float] = {
            "mexc": 1.0,
            "gateio": 0.9,
            "bingx": 0.8,
            "bitget": 0.85,
        }
        result = {ex: liquidity_scores.get(ex.lower(), 0.5) for ex in exchanges}
        logger.debug("Liquidity comparison for {}: {}", symbol, result)
        return result
