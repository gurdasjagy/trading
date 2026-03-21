"""Bayesian Kelly Criterion — conservative position sizing with Bayesian updates.

Replaces the naive Kelly implementation with a Bayesian approach that:

* Starts with an informative Beta prior (configurable alpha/beta)
* Updates the posterior win-rate after each trade
* Uses the posterior mean for sizing, which is more conservative with
  limited data than the MLE win-rate
* Implements fractional Kelly (default 0.25×) with a hard cap
* Supports per-strategy instances so each strategy has its own state
"""

from __future__ import annotations

from loguru import logger


class BayesianKelly:
    """Bayesian Kelly position sizer with Beta-distribution prior.

    Args:
        prior_alpha: Initial successes (prior). Higher values assume a better
            base win-rate.  Default 2 → prior win-rate ≈ 0.40.
        prior_beta: Initial failures (prior). Default 3.
        kelly_fraction: Fractional Kelly multiplier (e.g. 0.25 for quarter-Kelly).
        max_fraction: Hard cap on the Kelly fraction before applying
            ``kelly_fraction``.  Prevents outsized bets even when the
            Bayesian win-rate is high.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 3.0,
        kelly_fraction: float = 0.25,
        max_fraction: float = 0.20,
    ) -> None:
        self._alpha = prior_alpha
        self._beta = prior_beta
        self.kelly_fraction = kelly_fraction
        self.max_fraction = max_fraction

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update(self, won: bool) -> None:
        """Update the Beta posterior with the outcome of one trade.

        Args:
            won: True if the trade was profitable, False otherwise.
        """
        if won:
            self._alpha += 1.0
        else:
            self._beta += 1.0
        logger.debug(
            "BayesianKelly updated: won={} → α={:.1f} β={:.1f} "
            "posterior_win_rate={:.4f}",
            won,
            self._alpha,
            self._beta,
            self.posterior_win_rate,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def posterior_win_rate(self) -> float:
        """Posterior mean win-rate: α / (α + β)."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def total_trades(self) -> int:
        """Approximate number of trades observed (posterior updates - prior)."""
        return max(0, int(round(self._alpha + self._beta - 5.0)))

    # ------------------------------------------------------------------
    # Core Kelly computation
    # ------------------------------------------------------------------

    def get_kelly_fraction(self, avg_win: float, avg_loss: float) -> float:
        """Compute the fractional Kelly fraction for the current posterior.

        Kelly formula: f* = (p × b − q) / b
        where p = win-rate, q = 1−p, b = avg_win / avg_loss.

        The result is scaled by ``kelly_fraction`` (quarter-Kelly by default)
        and clamped to [0, max_fraction].

        Args:
            avg_win: Average win return (e.g. 0.03 for 3 % average win).
            avg_loss: Average loss return (e.g. 0.02 for 2 % average loss).

        Returns:
            Recommended fraction of capital to risk (0–max_fraction).
        """
        if avg_win <= 0 or avg_loss <= 0:
            logger.debug(
                "BayesianKelly: invalid avg_win={} avg_loss={}, returning 0",
                avg_win,
                avg_loss,
            )
            return 0.0

        p = self.posterior_win_rate
        q = 1.0 - p
        b = avg_win / avg_loss  # odds ratio

        kelly_full = (p * b - q) / b  # standard Kelly formula

        if kelly_full <= 0:
            return 0.0

        # Apply fractional Kelly and hard cap
        fraction = min(kelly_full * self.kelly_fraction, self.max_fraction)

        logger.debug(
            "BayesianKelly: p={:.4f} b={:.4f} kelly_full={:.4f} → "
            "fraction={:.4f} ({}x Kelly, cap={:.4f})",
            p,
            b,
            kelly_full,
            fraction,
            self.kelly_fraction,
            self.max_fraction,
        )
        return fraction

    def get_position_size(
        self,
        capital: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Return recommended position size in USDT.

        Args:
            capital: Available capital in USDT.
            avg_win: Average win return fraction (e.g. 0.03).
            avg_loss: Average loss return fraction (e.g. 0.02).

        Returns:
            Position size in USDT (≥ 0).
        """
        if capital <= 0:
            return 0.0
        fraction = self.get_kelly_fraction(avg_win, avg_loss)
        size = capital * fraction
        logger.debug(
            "BayesianKelly position size: capital={:.2f} fraction={:.4f} → {:.2f} USDT",
            capital,
            fraction,
            size,
        )
        return size
