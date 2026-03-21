"""Dynamic multi-level take-profit engine.

Replaces fixed 3-level TP with adaptive levels based on ATR, support/resistance,
Fibonacci extensions, and market regime.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

# Regime ATR multiplier adjustments
_REGIME_MULTIPLIERS: Dict[str, float] = {
    "trending_up": 1.5,
    "trending_down": 1.5,
    "ranging": 0.7,
    "low_volatility": 0.8,
    "high": 1.2,
    "crash": 0.5,
    "unknown": 1.0,
}

# Fibonacci extension ratios used for TP3
_FIB_EXTENSIONS = [1.618, 2.618, 4.236]

# Default ATR multipliers for the three TP levels
_DEFAULT_ATR_MULTIPLIERS = [1.5, 3.0, 5.0]

# Default partial-close percentages for each level
_DEFAULT_PERCENTAGES = [0.30, 0.30, 0.20]

# Trailing portion (remaining 20 %) closes with trailing stop only
_TRAILING_REMAINDER_PCT = 0.20


class DynamicTakeProfitEngine:
    """Calculate adaptive take-profit levels driven by ATR, S/R, and market regime.

    Replaces the rigid 25/50/25 split with:
    * TP1 (30 %): entry ± 1.5 × ATR  — lock in early profit
    * TP2 (30 %): entry ± 3.0 × ATR  — significant profit locked
    * TP3 (20 %): entry ± 5.0 × ATR  — let the rest ride
    * Remaining 20 %: trailing stop only

    All distances are scaled by a regime multiplier:
    * Trending regime  → ×1.5 (wider targets)
    * Ranging regime   → ×0.7 (tighter targets)
    """

    def __init__(
        self,
        atr_multipliers: Optional[List[float]] = None,
        tp_percentages: Optional[List[float]] = None,
    ) -> None:
        self._atr_multipliers: List[float] = atr_multipliers or list(_DEFAULT_ATR_MULTIPLIERS)
        self._tp_percentages: List[float] = tp_percentages or list(_DEFAULT_PERCENTAGES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_tp_levels(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        market_regime: str = "unknown",
        support_resistance_levels: Optional[List[float]] = None,
        adx: Optional[float] = None,
    ) -> List[dict]:
        """Calculate adaptive take-profit levels.

        Args:
            entry_price: Trade entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range for the current timeframe.
            market_regime: Label such as ``"trending_up"``, ``"ranging"``, etc.
            support_resistance_levels: Optional list of key price levels used to
                snap TP prices to nearby S/R zones.
            adx: ADX value used for dynamic partial-close percentage adjustment.
                When ADX > 40 (strong trend), TP1 portion is reduced to 20 % and
                the trailing remainder is increased to 30 %.
                When ADX < 20 (ranging), TP1 is raised to 40 % and trailing is 10 %.

        Returns:
            List of dicts with keys ``price`` (float), ``percentage`` (float),
            and ``type`` ("fixed" | "trailing").
        """
        if entry_price <= 0 or atr <= 0:
            logger.warning(
                "DynamicTP: invalid entry ({}) or ATR ({}) — returning empty levels",
                entry_price,
                atr,
            )
            return []

        regime_mult = _REGIME_MULTIPLIERS.get(market_regime, 1.0)

        # Determine partial-close percentages based on ADX / regime
        pcts = self._resolve_percentages(adx)

        levels: List[dict] = []
        for i, (mult, pct) in enumerate(zip(self._atr_multipliers, pcts)):
            distance = atr * mult * regime_mult
            if direction == "long":
                raw_price = entry_price + distance
            else:
                raw_price = entry_price - distance
                # Guard: short TP must be strictly positive
                if raw_price <= 0.0:
                    raw_price = entry_price * 0.001  # 0.1% floor

            # Snap to nearest S/R level if within 0.5 ATR
            price = self._snap_to_sr(
                raw_price,
                support_resistance_levels or [],
                atr,
                direction,
                level_index=i,
            )

            levels.append(
                {
                    "price": round(price, 8),
                    "percentage": pct,
                    "type": "fixed",
                }
            )

        # Add trailing-stop-only remainder level
        trailing_pct = _TRAILING_REMAINDER_PCT
        if adx is not None:
            if adx > 40:
                trailing_pct = 0.30
            elif adx < 20:
                trailing_pct = 0.10

        levels.append(
            {
                "price": 0.0,  # managed by IntelligentTrailingStop
                "percentage": trailing_pct,
                "type": "trailing",
            }
        )

        logger.debug(
            "DynamicTP levels for {} entry={} regime={} atr={:.4f}: {}",
            direction,
            entry_price,
            market_regime,
            atr,
            [(lvl["price"], lvl["percentage"], lvl["type"]) for lvl in levels],
        )
        return levels

    def recalculate(
        self,
        entry_price: float,
        direction: str,
        current_price: float,
        new_atr: float,
        market_regime: str = "unknown",
        new_sr_levels: Optional[List[float]] = None,
        adx: Optional[float] = None,
    ) -> List[dict]:
        """Recalculate TP levels with updated market data.

        Called every 15 minutes to adapt to changing volatility and S/R zones.

        Args:
            entry_price: Original trade entry price.
            direction: ``"long"`` or ``"short"``.
            current_price: Current market price (used for sanity check only).
            new_atr: Latest ATR value.
            market_regime: Current market regime label.
            new_sr_levels: Updated support/resistance levels.
            adx: Latest ADX value.

        Returns:
            Updated list of TP level dicts.
        """
        logger.debug(
            "DynamicTP recalculate: entry={} current={} new_atr={:.4f} regime={}",
            entry_price,
            current_price,
            new_atr,
            market_regime,
        )
        return self.calculate_tp_levels(
            entry_price=entry_price,
            direction=direction,
            atr=new_atr,
            market_regime=market_regime,
            support_resistance_levels=new_sr_levels,
            adx=adx,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_percentages(self, adx: Optional[float]) -> List[float]:
        """Return the three fixed-level close percentages adjusted for ADX."""
        if adx is not None:
            if adx > 40:
                # Strong trend — reduce TP1, let more ride
                return [0.20, 0.30, 0.20]
            if adx < 20:
                # Ranging market — take more profit early
                return [0.40, 0.30, 0.20]
        return list(self._tp_percentages)

    def _snap_to_sr(
        self,
        raw_price: float,
        sr_levels: List[float],
        atr: float,
        direction: str,
        level_index: int,
    ) -> float:
        """Snap *raw_price* to the nearest S/R level within 0.5 ATR.

        For long positions only S/R levels above entry are considered; for
        shorts only levels below entry.  Returns *raw_price* unchanged when
        no suitable level is found.
        """
        if not sr_levels or atr <= 0:
            return raw_price

        snap_threshold = atr * 0.5
        best: Optional[float] = None
        best_dist = float("inf")

        for level in sr_levels:
            dist = abs(level - raw_price)
            if dist < snap_threshold and dist < best_dist:
                best = level
                best_dist = dist

        if best is not None:
            logger.debug(
                "DynamicTP: snapped TP{} from {:.4f} to S/R level {:.4f} (dist={:.4f})",
                level_index + 1,
                raw_price,
                best,
                best_dist,
            )
            return best
        return raw_price
