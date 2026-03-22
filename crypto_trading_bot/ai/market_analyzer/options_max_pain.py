"""Options Max Pain Calculator — calculates BTC/ETH options max pain from Deribit data."""

from typing import Dict, List, Optional

from loguru import logger


class OptionsMaxPainCalculator:
    """Calculates BTC/ETH options max pain price from Deribit open interest data.
    
    Max pain is the strike price where option sellers have minimum loss,
    calculated as the price that minimizes total payout to option holders.
    """

    def calculate_max_pain(
        self,
        strikes: List[float],
        call_oi: List[float],
        put_oi: List[float],
        price_range: Optional[tuple] = None,
        step: float = 100.0,
    ) -> Dict[str, float]:
        """Calculate max pain price from strike prices and open interest.

        Args:
            strikes: List of strike prices
            call_oi: List of call open interest at each strike
            put_oi: List of put open interest at each strike
            price_range: Optional (min_price, max_price) to search, defaults to strike range
            step: Price increment for search (default 100)

        Returns:
            Dict with 'max_pain_price', 'total_payout', and 'call_payout', 'put_payout'
        """
        if not strikes or len(strikes) != len(call_oi) or len(strikes) != len(put_oi):
            logger.warning("Invalid input: strikes and OI lists must have same length")
            return {
                "max_pain_price": 0.0,
                "total_payout": 0.0,
                "call_payout": 0.0,
                "put_payout": 0.0,
            }

        try:
            # Determine price range to search
            if price_range:
                min_price, max_price = price_range
            else:
                min_price = min(strikes)
                max_price = max(strikes)

            # Search for price that minimizes total payout
            min_payout = float("inf")
            max_pain_price = 0.0
            best_call_payout = 0.0
            best_put_payout = 0.0

            current_price = min_price
            while current_price <= max_price:
                total_payout = self._calculate_total_payout(
                    current_price, strikes, call_oi, put_oi
                )

                if total_payout < min_payout:
                    min_payout = total_payout
                    max_pain_price = current_price
                    # Calculate individual payouts for the max pain price
                    best_call_payout = sum(
                        call_oi[i] * max(0, current_price - strikes[i])
                        for i in range(len(strikes))
                    )
                    best_put_payout = sum(
                        put_oi[i] * max(0, strikes[i] - current_price)
                        for i in range(len(strikes))
                    )

                current_price += step

            logger.debug(
                f"Max pain calculated: ${max_pain_price:,.0f} "
                f"(total payout: ${min_payout:,.0f})"
            )

            return {
                "max_pain_price": max_pain_price,
                "total_payout": min_payout,
                "call_payout": best_call_payout,
                "put_payout": best_put_payout,
            }

        except Exception as exc:
            logger.error(f"Max pain calculation error: {exc}")
            return {
                "max_pain_price": 0.0,
                "total_payout": 0.0,
                "call_payout": 0.0,
                "put_payout": 0.0,
            }

    def _calculate_total_payout(
        self,
        price: float,
        strikes: List[float],
        call_oi: List[float],
        put_oi: List[float],
    ) -> float:
        """Calculate total payout at a given price.

        Formula: sum of (call_oi * max(0, price - strike) + put_oi * max(0, strike - price))
        """
        total = 0.0
        for i in range(len(strikes)):
            strike = strikes[i]
            # Call payout: in-the-money when price > strike
            call_value = max(0, price - strike)
            # Put payout: in-the-money when price < strike
            put_value = max(0, strike - price)
            total += call_oi[i] * call_value + put_oi[i] * put_value
        return total

    def calculate_max_pain_from_deribit(
        self,
        deribit_data: Dict,
        asset: str = "BTC",
    ) -> Dict[str, float]:
        """Calculate max pain from Deribit options data structure.

        Args:
            deribit_data: Dict with 'strikes', 'call_oi', 'put_oi' keys
            asset: Asset symbol (BTC or ETH)

        Returns:
            Dict with max pain calculation results
        """
        try:
            strikes = deribit_data.get("strikes", [])
            call_oi = deribit_data.get("call_oi", [])
            put_oi = deribit_data.get("put_oi", [])

            if not strikes:
                logger.warning(f"No Deribit data for {asset}")
                return {
                    "max_pain_price": 0.0,
                    "total_payout": 0.0,
                    "call_payout": 0.0,
                    "put_payout": 0.0,
                }

            # Use appropriate step size based on asset
            step = 100.0 if asset == "BTC" else 10.0

            result = self.calculate_max_pain(strikes, call_oi, put_oi, step=step)
            result["asset"] = asset
            return result

        except Exception as exc:
            logger.error(f"Deribit max pain calculation error for {asset}: {exc}")
            return {
                "max_pain_price": 0.0,
                "total_payout": 0.0,
                "call_payout": 0.0,
                "put_payout": 0.0,
            }
