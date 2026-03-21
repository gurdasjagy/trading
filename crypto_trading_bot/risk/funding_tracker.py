"""Funding rate tracker — monitors perpetual funding costs for open positions."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


class FundingTracker:
    """Tracks funding rate payments for all open positions.

    Funding rates on perpetual futures contracts are exchanged every 8 hours
    between longs and shorts.  A negative funding rate means longs pay shorts;
    a positive rate means shorts pay longs.

    This tracker:
    * Fetches current funding rates from the exchange for each position.
    * Calculates the expected funding payment based on position notional value.
    * Records actual funding payments when they occur.
    * Provides a daily funding cost total and a recommendation on whether a
      position should be closed because funding is eroding its profitability.
    """

    def __init__(self, funding_rate_tolerance: float = -0.05) -> None:
        """Initialise the tracker.

        Args:
            funding_rate_tolerance: Funding rate threshold expressed as a
                percentage value (e.g. ``-0.05`` means ``-0.05%``).  When a
                position's funding rate is worse (more negative) than this
                threshold, :meth:`should_close_for_funding` may recommend
                closing the position.
        """
        self._tolerance = funding_rate_tolerance
        # Records of actual payments: {symbol: [{"ts": ..., "amount": ...}, ...]}
        self._payment_history: Dict[str, List[dict]] = {}
        # Accumulated today's costs per symbol
        self._daily_costs: Dict[str, float] = {}
        self._today: date = date.today()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_funding_rates(
        self,
        exchange: Any,
        positions: List[Any],
    ) -> List[dict]:
        """Fetch current funding rates and estimate upcoming payments.

        Args:
            exchange: An exchange adapter that exposes a
                ``fetch_funding_rate(symbol)`` coroutine (CCXT-compatible).
            positions: List of position objects with at least ``symbol``,
                ``amount``, and ``entry_price`` attributes.

        Returns:
            List of dicts, one per position::

                {
                    "symbol": "BTC/USDT",
                    "rate": -0.0001,           # current funding rate (decimal)
                    "rate_pct": -0.01,         # funding rate as percentage
                    "expected_payment": -2.50, # expected USD payment (+ve = receive)
                    "next_funding_time": "2026-01-01T08:00:00+00:00",
                }
        """
        results: List[dict] = []
        for pos in positions:
            symbol = getattr(pos, "symbol", None) or pos.get("symbol", "")
            if not symbol:
                continue
            try:
                # Use the bot's wrapper (get_funding_rate) instead of the raw
                # CCXT method (fetch_funding_rate) so that swap-symbol resolution
                # and rate-limiting are applied consistently.
                rate: float = await exchange.get_funding_rate(symbol)

                # Notional position value (in USDT)
                amount = float(getattr(pos, "amount", 0.0) or pos.get("amount", 0.0))
                entry_price = float(
                    getattr(pos, "entry_price", 0.0) or pos.get("entry_price", 0.0)
                )
                notional = amount * entry_price

                # Positive notional × positive rate = longs pay (negative for longs)
                side_str = (
                    getattr(pos, "side", None)
                    or pos.get("side", "long")
                )
                # side.value if it's an enum, else the string directly
                if hasattr(side_str, "value"):
                    side_str = side_str.value
                side_str = str(side_str).lower()
                # Long pays when rate > 0; short pays when rate < 0
                expected_payment = (
                    -notional * rate if side_str == "long" else notional * rate
                )

                results.append(
                    {
                        "symbol": symbol,
                        "rate": rate,
                        "rate_pct": round(rate * 100, 6),
                        "expected_payment": round(expected_payment, 4),
                        # next_funding_time is not available via the get_funding_rate()
                        # wrapper (which returns a plain float).  Use None here; callers
                        # that need the exact next-funding timestamp should call
                        # exchange.fetch_funding_rate() directly.
                        "next_funding_time": None,
                    }
                )
                logger.debug(
                    "Funding rate {}: rate={:.6f}% expected_payment={:.4f}",
                    symbol,
                    rate * 100,
                    expected_payment,
                )
            except Exception as exc:
                logger.debug("Could not fetch funding rate for {}: {}", symbol, exc)

        return results

    def record_funding_payment(self, symbol: str, amount: float) -> None:
        """Record an actual funding payment for *symbol*.

        Args:
            symbol: Trading pair symbol.
            amount: Payment amount in USDT.  Negative means the position *paid*
                funding; positive means it *received* funding.
        """
        self._reset_daily_if_needed()
        ts = datetime.now(tz=timezone.utc).isoformat()
        self._payment_history.setdefault(symbol, []).append(
            {"ts": ts, "amount": amount}
        )
        self._daily_costs[symbol] = self._daily_costs.get(symbol, 0.0) + amount
        logger.info("Funding payment recorded: {} amount={:.4f}", symbol, amount)

    def get_daily_funding_cost(self) -> float:
        """Return the total (net) funding cost paid today across all symbols.

        A negative return value means the bot paid net funding costs.
        """
        self._reset_daily_if_needed()
        return sum(self._daily_costs.values())

    def get_daily_funding_cost_by_symbol(self) -> Dict[str, float]:
        """Return funding costs today broken down by symbol."""
        self._reset_daily_if_needed()
        return dict(self._daily_costs)

    def should_close_for_funding(
        self,
        symbol: str,
        position_pnl: float,
        funding_rate: float,
    ) -> bool:
        """Recommend closing *symbol* if funding costs are eroding its profitability.

        Args:
            symbol: Trading pair symbol.
            position_pnl: Current unrealised P&L of the position (USDT).
            funding_rate: Current funding rate as a percentage (e.g. ``-0.05``
                for ``-0.05%``).

        Returns:
            ``True`` if the position should be closed due to adverse funding.
        """
        if funding_rate >= self._tolerance:
            # Funding rate is acceptable — no action needed
            return False

        # Funding rate is worse than tolerance
        daily_cost = self._daily_costs.get(symbol, 0.0)

        # If position is profitable and covers at least 3× today's funding cost,
        # keep it open; otherwise recommend closing.
        if position_pnl > 0 and position_pnl > abs(daily_cost) * 3:
            return False

        logger.warning(
            "Recommending close for {} due to adverse funding: "
            "rate={:.4f}% pnl={:.4f} daily_cost={:.4f}",
            symbol,
            funding_rate,
            position_pnl,
            daily_cost,
        )
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if the UTC date has changed."""
        today = date.today()
        if today != self._today:
            self._today = today
            self._daily_costs = {}
