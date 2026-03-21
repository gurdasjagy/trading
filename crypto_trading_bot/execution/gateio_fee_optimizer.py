"""Gate.io fee optimizer — minimize execution cost and choose optimal order type."""

from __future__ import annotations

from typing import Dict

# ------------------------------------------------------------------
# Decision thresholds (extracted for readability and configurability)
# ------------------------------------------------------------------
_TYPICAL_FUNDING_RATE_PER_PERIOD = 0.0001  # 0.01 % per funding period (Gate.io default)
_FUNDING_PERIODS_PER_DAY = 3               # Gate.io settles funding 3× per day
_MAX_SPREAD_FOR_MAKER_BPS = 5.0            # Wider spread → maker orders are risky
_HIGH_CONFIDENCE_THRESHOLD = 0.75          # Signal confidence above which to hit market
_MEDIUM_CONFIDENCE_THRESHOLD = 0.50        # Below which, fall back to passive limit
_IMBALANCE_THRESHOLD = 0.3                 # |imbalance| > this = "favorable"
_TIGHT_SPREAD_BPS = 3.0                    # Spread below which post-only is preferred


class GateioFeeOptimizer:
    """Computes trade costs and recommends the optimal Gate.io order type.

    Gate.io futures fee schedule (VIP 0 defaults):
        Maker rebate : -0.025% (-0.00025)  ← exchange *pays* you
        Taker fee    :  0.075%  (0.00075)

    Higher VIP tiers have lower taker fees and larger maker rebates.
    Pass the current rates from :meth:`GateIOClient.get_fee_tier` to
    override the defaults at runtime.
    """

    def __init__(
        self,
        maker_fee: float = -0.00025,
        taker_fee: float = 0.00075,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    def calculate_trade_cost(
        self,
        entry_price: float,
        exit_price: float,
        amount: float,
        leverage: float,
        direction: str,
        use_maker_entry: bool = False,
    ) -> Dict[str, float]:
        """Return a breakdown of the full round-trip trade cost.

        Args:
            entry_price  : Expected fill price at entry.
            exit_price   : Expected fill price at exit.
            amount       : Position size in base currency (e.g. BTC).
            leverage     : Applied leverage (used for margin, not fee calc).
            direction    : "long" or "short" (reserved for future asymmetric fees).
            use_maker_entry: If True, entry uses maker (rebate) rate; otherwise taker.

        Returns:
            dict with keys:
                entry_fee         : cost (+) or rebate (-) at entry
                exit_fee          : cost at exit (always taker for safety)
                funding_estimate  : rough 3-period funding cost
                total_cost        : sum of the above (may be negative = profit)
                break_even_pct    : total_cost / notional × 100
        """
        if amount <= 0 or entry_price <= 0 or exit_price <= 0:
            return {
                "entry_fee": 0.0,
                "exit_fee": 0.0,
                "funding_estimate": 0.0,
                "total_cost": 0.0,
                "break_even_pct": 0.0,
            }

        entry_fee_rate = self.maker_fee if use_maker_entry else self.taker_fee
        entry_fee = amount * entry_price * entry_fee_rate
        exit_fee = amount * exit_price * self.taker_fee
        # Rough estimate: 3 funding periods per day at typical 0.01 % per period
        funding_estimate = (
            amount * entry_price * _TYPICAL_FUNDING_RATE_PER_PERIOD * _FUNDING_PERIODS_PER_DAY
        )
        total_cost = entry_fee + exit_fee + funding_estimate

        notional = amount * entry_price
        break_even_pct = (total_cost / notional * 100.0) if notional > 0 else 0.0

        return {
            "entry_fee": round(entry_fee, 8),
            "exit_fee": round(exit_fee, 8),
            "funding_estimate": round(funding_estimate, 8),
            "total_cost": round(total_cost, 8),
            "break_even_pct": round(break_even_pct, 6),
        }

    # ------------------------------------------------------------------
    # Order type decision helpers
    # ------------------------------------------------------------------

    def should_use_maker(self, urgency: str, spread_bps: float) -> bool:
        """Decide whether a maker (limit/post-only) order is worthwhile.

        Args:
            urgency    : "high" → need immediate fill; anything else → ok to wait.
            spread_bps : Current bid-ask spread in basis points.

        Returns:
            True  → place a maker/post-only order to capture the rebate.
            False → use a taker/market order for immediate execution.
        """
        if urgency == "high":
            return False
        if spread_bps > _MAX_SPREAD_FOR_MAKER_BPS:
            # Wide spread means limit orders are risky (might not fill)
            return False
        # Tight spread → maker is worth the potential non-fill risk
        return True

    def get_optimal_order_type(
        self,
        signal_confidence: float,
        book_imbalance: float,
        spread_bps: float,
    ) -> str:
        """Return the recommended Gate.io order type string.

        Logic:
        - High confidence + favorable imbalance → "market"  (get in fast)
        - Medium confidence + tight spread      → "post_only" (save fees)
        - Low confidence or wide spread         → "limit_passive"

        Args:
            signal_confidence : Strategy confidence in [0, 1].
            book_imbalance    : Imbalance from GateioBookAnalyzer in [-1, +1].
            spread_bps        : Current spread in basis points.

        Returns:
            One of: "market", "post_only", "limit_passive"
        """
        favorable_imbalance = abs(book_imbalance) > _IMBALANCE_THRESHOLD
        high_confidence = signal_confidence >= _HIGH_CONFIDENCE_THRESHOLD
        medium_confidence = _MEDIUM_CONFIDENCE_THRESHOLD <= signal_confidence < _HIGH_CONFIDENCE_THRESHOLD
        tight_spread = spread_bps < _TIGHT_SPREAD_BPS

        if high_confidence and favorable_imbalance:
            return "market"
        if medium_confidence and tight_spread:
            return "post_only"
        return "limit_passive"

    # ------------------------------------------------------------------
    # Trade viability gate
    # ------------------------------------------------------------------

    def trade_is_viable(
        self,
        cost_breakdown: Dict[str, float],
        expected_profit_pct: float,
        min_profit_to_cost_ratio: float = 2.0,
    ) -> bool:
        """Return True when expected profit covers costs by the required ratio.

        Args:
            cost_breakdown         : Result from :meth:`calculate_trade_cost`.
            expected_profit_pct    : Estimated profit as a percentage of notional.
            min_profit_to_cost_ratio: Reject if expected_profit < ratio × break_even.

        Returns:
            True  → trade passes the fee filter.
            False → trade would likely be unprofitable after fees.
        """
        break_even_pct = cost_breakdown.get("break_even_pct", 0.0)
        if break_even_pct <= 0:
            return True
        return expected_profit_pct >= break_even_pct * min_profit_to_cost_ratio

    # ------------------------------------------------------------------
    # Phase 3: System 3 - Smart execution routing
    # ------------------------------------------------------------------

    def calculate_optimal_execution_route(
        self,
        signal_confidence: float,
        spread_bps: float,
        book_state: Dict,
        position_notional: float,
        maker_fee_bps: float,
        taker_fee_bps: float,
    ) -> Dict[str, Any]:
        """Calculate the optimal execution route based on market conditions.

        Decision logic:
        1. High confidence + tight spread → market order (fast execution)
        2. Wide spread (> 2× fee cost) → post_only limit (capture rebate)
        3. Large order (> 5% visible depth) → TWAP with iceberg (reduce impact)

        Args:
            signal_confidence: Strategy confidence in [0, 1]
            spread_bps: Current bid-ask spread in basis points
            book_state: Dict with bid_depth_usdt, ask_depth_usdt, mid_price
            position_notional: Position size in USDT
            maker_fee_bps: Maker fee in basis points (negative for rebate)
            taker_fee_bps: Taker fee in basis points

        Returns:
            Dict with keys:
                - order_type: "market", "limit_passive", or "twap"
                - use_iceberg: bool
                - chunk_count: int (for TWAP)
                - reasoning: str
        """
        from typing import Any

        bid_depth = book_state.get("bid_depth_usdt", 0.0)
        ask_depth = book_state.get("ask_depth_usdt", 0.0)
        visible_depth = min(bid_depth, ask_depth) if bid_depth > 0 and ask_depth > 0 else 0.0

        # Calculate fee cost in bps (absolute value)
        fee_cost_bps = abs(maker_fee_bps) + abs(taker_fee_bps)

        # Rule 1: High confidence + tight spread → market order
        if signal_confidence > 0.85 and spread_bps < 3.0:
            return {
                "order_type": "market",
                "use_iceberg": False,
                "chunk_count": 1,
                "reasoning": f"High confidence ({signal_confidence:.2f}) + tight spread ({spread_bps:.1f}bps) → market order",
            }

        # Rule 2: Wide spread → post_only limit to capture rebate
        if spread_bps > 2 * fee_cost_bps:
            return {
                "order_type": "limit_passive",
                "use_iceberg": False,
                "chunk_count": 1,
                "reasoning": f"Wide spread ({spread_bps:.1f}bps > 2×{fee_cost_bps:.1f}bps fee) → post_only limit",
            }

        # Rule 3: Large order → TWAP with iceberg
        if visible_depth > 0 and position_notional > visible_depth * 0.05:
            # Calculate chunk count: aim for 2% of visible depth per chunk
            chunk_count = min(10, max(2, int(position_notional / (visible_depth * 0.02))))
            return {
                "order_type": "twap",
                "use_iceberg": True,
                "chunk_count": chunk_count,
                "reasoning": f"Large order ({position_notional:.0f} USDT > 5% of {visible_depth:.0f} depth) → TWAP with {chunk_count} chunks",
            }

        # Default: passive limit order
        return {
            "order_type": "limit_passive",
            "use_iceberg": False,
            "chunk_count": 1,
            "reasoning": f"Default routing: conf={signal_confidence:.2f} spread={spread_bps:.1f}bps → limit_passive",
        }
