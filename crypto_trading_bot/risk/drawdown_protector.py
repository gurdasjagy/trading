"""Drawdown protection — monitors equity and reduces exposure when limits approach."""

from __future__ import annotations

from loguru import logger


class DrawdownProtector:
    """Monitors and protects against excessive portfolio drawdowns."""

    def __init__(self, max_drawdown_pct: float = 10.0) -> None:
        self._max_drawdown_pct = max_drawdown_pct
        self._equity_peak: float = 0.0

    def record_equity_peak(self, equity: float) -> None:
        """Update the all-time equity peak if *equity* is a new high.

        Args:
            equity: Current portfolio equity value.
        """
        if equity > self._equity_peak:
            self._equity_peak = equity
            logger.debug("New equity peak recorded: {:.2f}", equity)

    def calculate_current_drawdown(self, portfolio: dict) -> float:
        """Calculate the current percentage drawdown from the equity peak.

        Args:
            portfolio: Portfolio dict containing at least ``{"equity": float}``.

        Returns:
            Current drawdown as a percentage (0–100).
        """
        equity = portfolio.get("equity", 0.0)
        if self._equity_peak <= 0:
            self.record_equity_peak(equity)
            return 0.0
        if equity >= self._equity_peak:
            self.record_equity_peak(equity)
            return 0.0
        drawdown = (self._equity_peak - equity) / self._equity_peak * 100.0
        logger.debug(
            "Current drawdown: peak={:.2f} equity={:.2f} dd={:.2f}%",
            self._equity_peak,
            equity,
            drawdown,
        )
        return drawdown

    def should_reduce_exposure(self, drawdown: float) -> bool:
        """Return True if exposure should be reduced due to drawdown.

        Exposure reduction is recommended when drawdown exceeds half the
        maximum allowed drawdown.

        Args:
            drawdown: Current drawdown percentage.

        Returns:
            ``True`` if exposure should be reduced.
        """
        threshold = self._max_drawdown_pct / 2.0
        return drawdown >= threshold

    def get_exposure_multiplier(self, drawdown: float) -> float:
        """Return a scaling factor (0–1) for position sizing based on drawdown.

        - Below 50 % of max → full exposure (1.0)
        - 50–75 % of max → 50 % exposure (0.5)
        - 75–100 % of max → 25 % exposure (0.25)
        - Above max → 0 % exposure (0.0)

        Args:
            drawdown: Current drawdown percentage.

        Returns:
            Exposure multiplier between 0.0 and 1.0.
        """
        max_dd = self._max_drawdown_pct
        if drawdown < max_dd * 0.5:
            return 1.0
        if drawdown < max_dd * 0.75:
            return 0.5
        if drawdown < max_dd:
            return 0.25
        logger.warning("Maximum drawdown exceeded: dd={:.2f}% max={:.2f}%", drawdown, max_dd)
        return 0.0

    def check_max_drawdown_breach(self, equity: float) -> bool:
        """Return True if equity has breached the maximum allowed drawdown.

        Args:
            equity: Current portfolio equity value.

        Returns:
            ``True`` if the maximum drawdown has been breached.
        """
        if self._equity_peak <= 0:
            return False
        drawdown = (self._equity_peak - equity) / self._equity_peak * 100.0
        breached = drawdown >= self._max_drawdown_pct
        if breached:
            logger.error(
                "Maximum drawdown breached: dd={:.2f}% limit={:.2f}% peak={:.2f} equity={:.2f}",
                drawdown,
                self._max_drawdown_pct,
                self._equity_peak,
                equity,
            )
        return breached
