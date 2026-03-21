"""Funding rate arbitrage strategy — exploits extreme funding rates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class FundingRateArbStrategy(BaseStrategy):
    """Delta-neutral strategy that captures extreme perpetual funding rates.

    Logic
    -----
    * Funding > +0.03 %: short the perp, long spot → collect positive funding.
    * Funding < -0.03 %: long the perp, short spot → collect negative funding.
    * Auto-close when funding rate normalises below 0.01 %.
    * Runs in ALL market regimes (market-neutral strategy).
    * Monitors funding rates across all configured exchanges and picks
      the most profitable opportunity.
    """

    # Enhanced thresholds (annualised >30 % = 0.03% per 8h funding period)
    LONG_THRESHOLD = -0.0003   # -0.03 %
    SHORT_THRESHOLD = 0.0003   # +0.03 %
    # Auto-close when rate normalises below this level
    CLOSE_THRESHOLD = 0.0001   # 0.01 %

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="funding_rate_arb",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._funding_cache: Dict[str, float] = {}
        # Multi-exchange funding rates: exchange_name -> symbol -> rate
        self._exchange_rates: Dict[str, Dict[str, float]] = {}
        # Fee estimates per exchange (maker fee as fraction)
        self._exchange_fees: Dict[str, float] = {}

    def update_funding_rate(self, symbol: str, rate: float) -> None:
        """Inject a fresh funding rate for *symbol*."""
        self._funding_cache[symbol] = rate

    def update_exchange_funding_rate(
        self, exchange: str, symbol: str, rate: float, fee: float = 0.0001
    ) -> None:
        """Store a funding rate from a specific exchange.

        Args:
            exchange: Exchange name (e.g. ``"binance"``).
            symbol: Trading symbol.
            rate: Funding rate as a decimal fraction.
            fee: Maker fee for the exchange (default 0.01 %).
        """
        if exchange not in self._exchange_rates:
            self._exchange_rates[exchange] = {}
        self._exchange_rates[exchange][symbol] = rate
        self._exchange_fees[exchange] = fee

    def get_best_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Find the highest net-profit funding opportunity across all exchanges.

        Compares funding rates across configured exchanges, deducts estimated
        fees, and returns the best opportunity if it exceeds the threshold.

        Args:
            symbol: Trading symbol to check.

        Returns:
            Dict with ``exchange``, ``rate``, ``net_rate``, ``direction``, or
            ``None`` if no profitable opportunity exists.
        """
        best: Optional[Dict[str, Any]] = None
        best_net = 0.0

        all_rates: Dict[str, float] = {}
        # Collect rates from all exchanges
        for exch, rates in self._exchange_rates.items():
            if symbol in rates:
                all_rates[exch] = rates[symbol]
        # Also include the default cache under "default"
        if symbol in self._funding_cache:
            all_rates["default"] = self._funding_cache[symbol]

        for exch, rate in all_rates.items():
            fee = self._exchange_fees.get(exch, 0.0001)
            # Round trip cost: 2 × maker fee (entry + exit)
            round_trip_cost = fee * 2
            net_rate = abs(rate) - round_trip_cost
            if net_rate <= 0:
                continue
            if abs(rate) >= self.SHORT_THRESHOLD and net_rate > best_net:
                best_net = net_rate
                direction = "short" if rate > 0 else "long"
                best = {
                    "exchange": exch,
                    "rate": rate,
                    "net_rate": round(net_rate, 6),
                    "direction": direction,
                }

        return best

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            rate = self._funding_cache.get(symbol)
            if rate is None:
                # Try fetching from exchange directly
                if self._exchange is not None:
                    try:
                        rate = await self._exchange.get_funding_rate(symbol)
                        self._funding_cache[symbol] = rate
                    except Exception as exc:
                        logger.warning(f"[{self.name}] Could not fetch funding rate: {exc}")
                        return self._neutral_signal(symbol, "Funding rate unavailable")
                else:
                    return self._neutral_signal(symbol, "Funding rate unavailable")

            if rate > self.SHORT_THRESHOLD:
                strength = min(1.0, (rate - self.SHORT_THRESHOLD) / self.SHORT_THRESHOLD)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.8,
                    strategy_name=self.name,
                    reasoning=(
                        f"Funding rate {rate:.4%} > threshold — "
                        "short perp / long spot (delta-neutral)"
                    ),
                    leverage=1,
                )

            if rate < self.LONG_THRESHOLD:
                strength = min(
                    1.0, (abs(rate) - abs(self.LONG_THRESHOLD)) / abs(self.LONG_THRESHOLD)
                )
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.8,
                    strategy_name=self.name,
                    reasoning=(
                        f"Funding rate {rate:.4%} < threshold — "
                        "long perp / short spot (delta-neutral)"
                    ),
                    leverage=1,
                )

            return self._neutral_signal(symbol, f"Funding rate {rate:.4%} within normal range")
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when funding rate normalises below 0.01 % (auto-close threshold)."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        rate = self._funding_cache.get(symbol, 0.0)
        side = str(getattr(position, "side", "long")).lower()
        # Use tighter normalisation threshold
        if side == "short" and rate < self.CLOSE_THRESHOLD:
            return True
        if side == "long" and rate > -self.CLOSE_THRESHOLD:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        return {
            "position_size_pct": 0.10,  # larger — delta-neutral reduces risk
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
            "leverage": 1,
        }
