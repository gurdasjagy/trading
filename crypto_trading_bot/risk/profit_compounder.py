"""Profit compounding system.

Increases position sizes when the bot is profitable and reduces them during
drawdown periods, creating a compounding effect on winning streaks.

Rules:
  Daily P&L > 5%  → size multiplier 1.2  (increase by 20%)
  Daily P&L > 2%  → size multiplier 1.1  (increase by 10%)
  Daily P&L < -1% → size multiplier 0.75 (reduce by 25%)
  Daily P&L < -3% → size multiplier 0.50 (reduce by 50%)
  Weekly P&L > 5% → increase base capital allocation by 5% (compounding)
  Cap: never exceed 2× the base position size.
"""

from __future__ import annotations

from loguru import logger


class ProfitCompounder:
    """Adaptive position-size multiplier driven by recent P&L performance.

    Usage::

        compounder = ProfitCompounder()
        multiplier = compounder.get_size_multiplier(daily_pnl_pct=3.5, weekly_pnl_pct=6.0)
        position_size *= multiplier
    """

    def __init__(
        self,
        base_size_pct: float = 0.03,
        max_compound_multiplier: float = 2.0,
    ) -> None:
        """
        Args:
            base_size_pct: Base position size as a fraction of capital (default 3 %).
            max_compound_multiplier: Maximum allowed multiplier relative to base size.
        """
        self._base_size_pct = base_size_pct
        self._max_multiplier = max_compound_multiplier
        # Weekly compounding accumulator: how much extra allocation has been unlocked
        self._extra_allocation_pct: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_size_multiplier(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float = 0.0,
    ) -> float:
        """Return the position-size multiplier for the next trade.

        Args:
            daily_pnl_pct: Today's realised P&L as a percentage of capital
                (e.g. 3.5 for +3.5 %, -2.0 for −2 %).
            weekly_pnl_pct: This week's realised P&L as a percentage of capital.
                Used for weekly compounding logic.

        Returns:
            A float multiplier in the range [0.25, ``max_compound_multiplier``].
        """
        # Determine base multiplier from daily P&L
        if daily_pnl_pct > 5.0:
            multiplier = 1.2
        elif daily_pnl_pct > 2.0:
            multiplier = 1.1
        elif daily_pnl_pct < -3.0:
            multiplier = 0.5
        elif daily_pnl_pct < -1.0:
            multiplier = 0.75
        else:
            multiplier = 1.0

        # Weekly compounding bonus
        if weekly_pnl_pct > 5.0:
            # Unlock an extra +5 % base allocation (one-time per qualifying week)
            if self._extra_allocation_pct < (self._max_multiplier - 1.0) * 100:
                self._extra_allocation_pct += 5.0
                logger.info(
                    "ProfitCompounder: weekly P&L > 5%% — unlocking +5%% base allocation "
                    "(total extra: {:.0f}%%)",
                    self._extra_allocation_pct,
                )

        # Apply weekly extra allocation as an additive bonus on the multiplier
        if self._extra_allocation_pct > 0:
            multiplier *= 1.0 + self._extra_allocation_pct / 100.0

        # Clamp
        multiplier = max(0.25, min(multiplier, self._max_multiplier))

        logger.debug(
            "ProfitCompounder: daily_pnl={:.2f}%% weekly_pnl={:.2f}%% → multiplier={:.3f}",
            daily_pnl_pct,
            weekly_pnl_pct,
            multiplier,
        )
        return multiplier

    def reset_weekly_compounding(self) -> None:
        """Call at the start of each new week to reset the compounding accumulator."""
        self._extra_allocation_pct = 0.0
        logger.debug("ProfitCompounder: weekly compounding accumulator reset.")

    @property
    def extra_allocation_pct(self) -> float:
        """Accumulated extra capital allocation percentage from weekly compounding."""
        return self._extra_allocation_pct
