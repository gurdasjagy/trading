"""Partial profit taking manager — tiered TP levels with stop management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class PartialCloseEvent:
    """Record of a partial position close."""

    symbol: str
    tp_level: int          # 1, 2, 3, or 4 (final)
    close_fraction: float  # fraction of original position closed
    price: float
    new_stop_loss: float
    reason: str = ""


@dataclass
class TakeProfitTier:
    """Single take-profit tier specification."""

    r_multiple: float    # distance from entry in units of initial risk (R)
    close_fraction: float  # fraction of position to close at this tier
    move_sl_to: Optional[float] = None  # new SL as price (set after entry)
    trailing_pct: Optional[float] = None  # trailing stop % for the final tier


class PartialProfitManager:
    """Manages tiered partial profit taking with dynamic stop management.

    Standard tiers (non-trending / default)
    ----------------------------------------
    * TP1 at 1.5R → close 25 %, move SL to breakeven
    * TP2 at 2.5R → close 35 %, move SL to TP1 level
    * TP3 at 4.0R → close 25 %, trail remaining 15 % with 1.5 % trailing stop
    * Final 15 % → ride with 1.5 % tight trailing stop

    Ranging market tiers (tighter)
    --------------------------------
    * TP1 at 1.2R → close 25 %
    * TP2 at 1.8R → close 35 %
    * TP3 at 2.5R → close 25 %

    Trending market tiers (wider)
    --------------------------------
    * TP1 at 2.0R → close 25 %
    * TP2 at 3.5R → close 35 %
    * TP3 at 6.0R → close 25 %
    """

    # Default tiers keyed by market regime
    _TIER_CONFIGS: Dict[str, List[Tuple[float, float, Optional[float]]]] = {
        # (r_multiple, close_fraction, trailing_pct)
        "default": [
            (1.5, 0.25, None),
            (2.5, 0.35, None),
            (4.0, 0.25, None),
            (float("inf"), 0.15, 0.015),  # final slice — trailing stop
        ],
        "ranging": [
            (1.2, 0.25, None),
            (1.8, 0.35, None),
            (2.5, 0.25, None),
            (float("inf"), 0.15, 0.015),
        ],
        "trending": [
            (2.0, 0.25, None),
            (3.5, 0.35, None),
            (6.0, 0.25, None),
            (float("inf"), 0.15, 0.015),
        ],
    }

    def __init__(self) -> None:
        # Track which TP levels have already been triggered per position
        # key: position_id, value: set of triggered tier indices (0-based)
        self._triggered: Dict[str, set] = {}
        # Track partial close history per position
        self._history: Dict[str, List[PartialCloseEvent]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tp_levels(
        self,
        entry: float,
        stop_loss: float,
        direction: str,
        market_regime: str = "default",
    ) -> List[Dict[str, Any]]:
        """Calculate take-profit price levels for the given position parameters.

        Args:
            entry: Entry price.
            stop_loss: Initial stop-loss price.
            direction: ``"long"`` or ``"short"``.
            market_regime: One of ``"default"``, ``"ranging"``, ``"trending"``.

        Returns:
            List of dicts with keys ``price``, ``close_fraction``,
            ``r_multiple``, and optionally ``trailing_pct``.
        """
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit <= 0:
            logger.warning("[PartialProfit] Zero risk (entry={} sl={})", entry, stop_loss)
            return []

        config_key = market_regime if market_regime in self._TIER_CONFIGS else "default"
        tiers = self._TIER_CONFIGS[config_key]

        levels = []
        for r_mult, frac, trail_pct in tiers:
            if r_mult == float("inf"):
                price = None  # final tier — no fixed TP price
            else:
                if direction == "long":
                    price = round(entry + r_mult * risk_per_unit, 8)
                else:
                    price = round(entry - r_mult * risk_per_unit, 8)

            tier_dict: Dict[str, Any] = {
                "price": price,
                "close_fraction": frac,
                "r_multiple": r_mult,
            }
            if trail_pct is not None:
                tier_dict["trailing_pct"] = trail_pct
            levels.append(tier_dict)

        logger.debug(
            "[PartialProfit] {} levels for {} entry={} sl={} regime={}",
            len(levels),
            direction,
            entry,
            stop_loss,
            config_key,
        )
        return levels

    def check_triggers(
        self,
        position_id: str,
        current_price: float,
        entry: float,
        stop_loss: float,
        direction: str,
        remaining_fraction: float = 1.0,
        market_regime: str = "default",
    ) -> Optional[PartialCloseEvent]:
        """Check whether any TP tier has been reached and return a close event.

        Only the *lowest untriggered* tier is returned per call so the caller
        can process one partial close at a time.

        Args:
            position_id: Unique position identifier.
            current_price: Latest market price.
            entry: Entry price of the position.
            stop_loss: Initial stop-loss price.
            direction: ``"long"`` or ``"short"``.
            remaining_fraction: What fraction of the original position is still open.
            market_regime: Market regime label for tier selection.

        Returns:
            :class:`PartialCloseEvent` if a tier was triggered, else ``None``.
        """
        levels = self.get_tp_levels(entry, stop_loss, direction, market_regime)
        triggered = self._triggered.setdefault(position_id, set())
        risk_per_unit = abs(entry - stop_loss)

        for i, level in enumerate(levels):
            if i in triggered:
                continue
            tp_price = level.get("price")
            if tp_price is None:
                continue  # final trailing tier — handled separately

            if direction == "long" and current_price >= tp_price:
                pass  # triggered
            elif direction == "short" and current_price <= tp_price:
                pass
            else:
                continue

            triggered.add(i)
            # Calculate new SL
            if i == 0:
                new_sl = entry  # move to breakeven at TP1
            elif i == 1:
                # Move SL to TP1 level (lock in profit)
                tp1_price = levels[0].get("price", entry)
                new_sl = tp1_price if tp1_price else entry
            else:
                new_sl = stop_loss  # keep existing for TP3+

            close_frac = level["close_fraction"] * remaining_fraction
            event = PartialCloseEvent(
                symbol=position_id,
                tp_level=i + 1,
                close_fraction=close_frac,
                price=current_price,
                new_stop_loss=new_sl,
                reason=f"TP{i+1} at {level['r_multiple']}R reached",
            )
            self._history.setdefault(position_id, []).append(event)
            logger.info(
                "[PartialProfit] TP{} triggered for {} at {:.4f} — close {:.0%} new_sl={:.4f}",
                i + 1,
                position_id,
                current_price,
                close_frac,
                new_sl,
            )
            return event

        return None

    def reset_position(self, position_id: str) -> None:
        """Clear tracking state for a closed position."""
        self._triggered.pop(position_id, None)
        self._history.pop(position_id, None)

    def get_history(self, position_id: str) -> List[PartialCloseEvent]:
        """Return all partial close events for *position_id*."""
        return list(self._history.get(position_id, []))

    def get_trailing_stop_for_final(
        self,
        direction: str,
        current_price: float,
        market_regime: str = "default",
    ) -> float:
        """Calculate the trailing stop price for the final position slice.

        Args:
            direction: ``"long"`` or ``"short"``.
            current_price: Current market price.
            market_regime: Market regime label.

        Returns:
            Trailing stop price.
        """
        config_key = market_regime if market_regime in self._TIER_CONFIGS else "default"
        tiers = self._TIER_CONFIGS[config_key]
        # Final tier is the last entry with r_multiple == inf
        final_tier = tiers[-1]
        trail_pct = final_tier[2] or 0.015

        if direction == "long":
            return round(current_price * (1.0 - trail_pct), 8)
        else:
            return round(current_price * (1.0 + trail_pct), 8)
