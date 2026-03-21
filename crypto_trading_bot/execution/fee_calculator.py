"""Fee calculation and optimisation across exchanges."""

from __future__ import annotations

from loguru import logger

# Exchange fee structures: {exchange: {order_type: taker/maker rate}}
_FEE_TABLE: dict[str, dict[str, float]] = {
    "mexc": {"limit": 0.0002, "market": 0.0006},
    "gateio": {"limit": 0.00015, "market": 0.0005},
    "bingx": {"limit": 0.0002, "market": 0.0005},
    "bitget": {"limit": 0.0002, "market": 0.0006},
}
_DEFAULT_FEE = {"limit": 0.0002, "market": 0.0006}


class FeeCalculator:
    """Calculates and optimises trading fees across exchanges."""

    def calculate_fee(
        self,
        amount: float,
        price: float,
        exchange: str,
        order_type: str = "limit",
    ) -> float:
        """Calculate the trading fee for a single order.

        Args:
            amount: Order size in base currency.
            price: Order price in quote currency.
            exchange: Exchange identifier (e.g. ``"mexc"``).
            order_type: ``"limit"`` (maker) or ``"market"`` (taker).

        Returns:
            Estimated fee in quote currency.
        """
        fee_rates = _FEE_TABLE.get(exchange.lower(), _DEFAULT_FEE)
        rate = fee_rates.get(order_type, 0.0006)
        notional = amount * price
        fee = notional * rate
        logger.debug(
            "Fee: exchange={} type={} notional={:.2f} rate={:.4%} fee={:.4f}",
            exchange,
            order_type,
            notional,
            rate,
            fee,
        )
        return fee

    def optimize_order_type(
        self,
        symbol: str,
        exchange: str,
        urgency: str = "normal",
    ) -> str:
        """Recommend limit or market order based on urgency and fee savings.

        Args:
            symbol: Trading symbol (for logging).
            exchange: Exchange identifier.
            urgency: ``"high"`` forces market, ``"normal"`` prefers limit.

        Returns:
            ``"limit"`` or ``"market"``.
        """
        if urgency == "high":
            logger.debug("High urgency — using market order for {}", symbol)
            return "market"
        fee_rates = _FEE_TABLE.get(exchange.lower(), _DEFAULT_FEE)
        savings = fee_rates.get("market", 0.0006) - fee_rates.get("limit", 0.0002)
        if savings > 0:
            logger.debug(
                "Limit order preferred for {} on {} (fee saving {:.4%})", symbol, exchange, savings
            )
            return "limit"
        return "market"

    def calculate_round_trip_cost(
        self,
        amount: float,
        price: float,
        exchange: str,
        order_type: str = "limit",
    ) -> float:
        """Calculate total round-trip fee (entry + exit).

        Args:
            amount: Order size in base currency.
            price: Reference price in quote currency.
            exchange: Exchange identifier.
            order_type: Order type for both legs.

        Returns:
            Total round-trip fee in quote currency.
        """
        one_way = self.calculate_fee(amount, price, exchange, order_type)
        round_trip = one_way * 2.0
        logger.debug("Round-trip cost: {:.4f} (2 × {:.4f})", round_trip, one_way)
        return round_trip

    def get_cheapest_exchange(self, exchanges: list[str], order_type: str = "limit") -> str:
        """Return the exchange with the lowest fee for the given *order_type*.

        Args:
            exchanges: List of exchange identifiers to compare.
            order_type: ``"limit"`` or ``"market"``.

        Returns:
            Exchange identifier with the lowest fee rate.
        """
        best_exchange = exchanges[0] if exchanges else "mexc"
        best_rate = float("inf")
        for ex in exchanges:
            rate = _FEE_TABLE.get(ex.lower(), _DEFAULT_FEE).get(order_type, 0.0006)
            if rate < best_rate:
                best_rate = rate
                best_exchange = ex
        logger.debug(
            "Cheapest exchange for {} orders: {} ({:.4%})", order_type, best_exchange, best_rate
        )
        return best_exchange
