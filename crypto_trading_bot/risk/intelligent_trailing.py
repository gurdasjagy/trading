"""Intelligent ATR-based trailing stop engine.

Replaces the fixed-distance trailing stop with a dynamic system that adapts to:
* Current volatility regime (tighter in low vol, wider in high vol)
* Profit multiple (trail tightens as profit grows — parabolic acceleration)
* Chandelier-exit logic (trail from highest high / lowest low)
* Time-based tightening (after 6 h in profit, reduce trail distance by 50 %)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

# Volatility regime → ATR multiplier
_VOL_ADJUSTMENT: Dict[str, float] = {
    "low": 0.75,
    "normal": 1.0,
    "medium": 1.0,
    "high": 1.5,
    "extreme": 2.0,
}

# Profit-multiple → (min ATR multiplier)
#   At 1R profit: trail at 2 ATR
#   At 2R profit: trail at 1.5 ATR
#   At 3R profit: trail at 1 ATR
_PROFIT_TIGHTENING_STEPS = [
    (3.0, 1.0),   # profit_multiple >= 3 → cap at 1.0×
    (2.0, 1.5),   # profit_multiple >= 2 → cap at 1.5×
    (1.0, 2.0),   # profit_multiple >= 1 → cap at 2.0×
]

# After this many hours in profit, reduce trail distance by 50 %
_TIME_TIGHTENING_HOURS: float = 6.0
_TIME_TIGHTENING_FACTOR: float = 0.50


class IntelligentTrailingStop:
    """ATR-based trailing stop with parabolic tightening and chandelier exit.

    Usage::

        trailing = IntelligentTrailingStop()
        new_sl = trailing.update(
            symbol="BTC/USDT",
            current_price=50_000,
            highest_price=51_000,
            atr=500,
            vol_regime="normal",
            entry_price=48_000,
        )
        if new_sl is not None:
            # Update the stop-loss order on the exchange
            ...
    """

    def __init__(self, base_atr_multiplier: float = 2.0) -> None:
        self._base_atr_multiplier = base_atr_multiplier
        # Per-symbol tracking: symbol → {"highest": float, "lowest": float, "current_sl": float}
        self._symbol_state: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_trail_distance(
        self,
        atr: float,
        volatility_regime: str,
        profit_multiple: float,
        hours_in_profit: float = 0.0,
    ) -> float:
        """Calculate the adaptive trailing-stop distance in price units.

        Args:
            atr: Current Average True Range.
            volatility_regime: One of ``"low"``, ``"normal"``, ``"high"``, ``"extreme"``.
            profit_multiple: Current profit expressed as a multiple of the initial
                risk (e.g. 2.0 means 2R profit).
            hours_in_profit: How long the position has been in profit (hours).
                After *_TIME_TIGHTENING_HOURS* the distance is halved.

        Returns:
            Trail distance in price units (positive number).
        """
        if atr <= 0:
            return 0.0

        base = atr * self._base_atr_multiplier

        # Volatility adjustment
        vol_adj = _VOL_ADJUSTMENT.get(volatility_regime, 1.0)

        # Profit tightening (parabolic acceleration)
        profit_tightening = max(0.5, 1.0 - (profit_multiple - 1.0) * 0.2)

        # Override with fixed-step caps for well-defined profit multiples
        for threshold, cap_mult in _PROFIT_TIGHTENING_STEPS:
            if profit_multiple >= threshold:
                profit_tightening = min(profit_tightening, cap_mult)
                break

        distance = base * vol_adj * profit_tightening

        # Time-based tightening
        if hours_in_profit >= _TIME_TIGHTENING_HOURS:
            distance *= _TIME_TIGHTENING_FACTOR

        logger.debug(
            "TrailDistance: atr={:.4f} vol_regime={} profit_mult={:.2f} "
            "hours_in_profit={:.1f} → distance={:.4f}",
            atr,
            volatility_regime,
            profit_multiple,
            hours_in_profit,
            distance,
        )
        return distance

    def update(
        self,
        symbol: str,
        current_price: float,
        highest_price: float,
        atr: float,
        vol_regime: str,
        entry_price: float,
        direction: str = "long",
        opened_at: Optional[datetime] = None,
        stop_distance_initial: Optional[float] = None,
    ) -> Optional[float]:
        """Calculate the new trailing stop-loss price using chandelier logic.

        The stop trails below the highest observed price (for longs) or above
        the lowest observed price (for shorts).  The stop only moves in the
        direction of profit — it never widens.

        Args:
            symbol: Trading symbol (used for per-symbol state tracking).
            current_price: Latest market price.
            highest_price: Highest price seen since the position was opened
                (for longs).  Callers should pass the lowest price for shorts
                via this argument.
            atr: Current ATR.
            vol_regime: Volatility regime label.
            entry_price: Original entry price.
            direction: ``"long"`` (default) or ``"short"``.
            opened_at: When the position was opened.  Used for time-based
                tightening.  Defaults to now when ``None``.
            stop_distance_initial: Initial stop distance (abs difference between
                entry and initial SL).  Used to compute profit multiple.  When
                ``None``, defaults to ``2 × ATR``.

        Returns:
            New stop-loss price if the stop should be moved upward (for longs) /
            downward (for shorts), or ``None`` if no update is needed.
        """
        if entry_price <= 0 or atr <= 0 or current_price <= 0:
            return None

        # Resolve per-symbol state
        state = self._symbol_state.setdefault(
            symbol, {"highest": highest_price, "lowest": highest_price, "current_sl": None}
        )

        # Update chandelier anchor
        if direction == "long":
            if current_price > state["highest"]:
                state["highest"] = current_price
            anchor = state["highest"]
        else:
            if state["lowest"] == 0.0 or current_price < state["lowest"]:
                state["lowest"] = current_price
            anchor = state["lowest"]

        # Compute profit multiple
        init_risk = stop_distance_initial if (stop_distance_initial and stop_distance_initial > 0) else atr * 2.0
        if direction == "long":
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price
        profit_multiple = max(0.0, profit / init_risk)

        # Time in profit
        if opened_at is None:
            opened_at = datetime.now(tz=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        hours_in_profit = (now - opened_at).total_seconds() / 3600.0 if profit > 0 else 0.0

        trail_dist = self.calculate_trail_distance(atr, vol_regime, profit_multiple, hours_in_profit)

        if direction == "long":
            new_sl = anchor - trail_dist
        else:
            new_sl = anchor + trail_dist

        new_sl = round(new_sl, 8)

        # Only move stop in the direction of profit
        existing_sl = state["current_sl"]
        if existing_sl is not None:
            if direction == "long" and new_sl <= existing_sl:
                return None  # Stop didn't move up
            if direction == "short" and new_sl >= existing_sl:
                return None  # Stop didn't move down

        state["current_sl"] = new_sl
        logger.debug(
            "IntelligentTrail {}: anchor={:.4f} trail_dist={:.4f} new_sl={:.4f} "
            "(profit_mult={:.2f} vol={})",
            symbol,
            anchor,
            trail_dist,
            new_sl,
            profit_multiple,
            vol_regime,
        )
        return new_sl

    def reset(self, symbol: str) -> None:
        """Remove per-symbol state (call when a position is closed)."""
        self._symbol_state.pop(symbol, None)
