"""Mean-Variance Portfolio Optimizer.

Implements Markowitz mean-variance optimization to compute optimal
position weights for the active trading universe.

* Objective: maximise Sharpe ratio (return / risk)
* Constraints: per-symbol cap (default 10 %), total exposure cap (default 50 %)
* Solver: scipy.optimize.minimize with SLSQP
* Rebalance cadence: every 4 hours (enforced externally via the scheduler)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger


class PortfolioOptimizer:
    """Markowitz mean-variance portfolio optimizer.

    Args:
        max_position_pct: Maximum weight per symbol (fraction, e.g. 0.10 = 10 %).
        max_total_exposure: Maximum total portfolio exposure (fraction, e.g. 0.50 = 50 %).
        risk_free_rate: Annual risk-free rate used in Sharpe computation (fraction).
        allow_short: Whether to allow negative (short) weights.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_total_exposure: float = 0.50,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        symbols: List[str],
    ) -> Dict[str, float]:
        """Compute optimal portfolio weights.

        Args:
            expected_returns: Map of symbol → expected return (fraction,
                e.g. ``{"BTC/USDT": 0.02, "ETH/USDT": 0.015}``).
            covariance_matrix: n×n covariance matrix (must match ``symbols`` order).
            symbols: List of symbols in the same order as ``covariance_matrix``.

        Returns:
            Dict mapping each symbol to its optimal weight (0–``max_position_pct``).
            Returns an equal-weight fallback if optimisation fails.
        """
        try:
            from scipy.optimize import minimize  # noqa: PLC0415
        except ImportError:
            logger.error("scipy is required for PortfolioOptimizer but is not installed.")
            return self._equal_weight(symbols)

        n = len(symbols)
        if n == 0:
            return {}
        if covariance_matrix.shape != (n, n):
            logger.error(
                "PortfolioOptimizer: covariance matrix shape {} does not match {} symbols",
                covariance_matrix.shape,
                n,
            )
            return self._equal_weight(symbols)

        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
        sigma = covariance_matrix
        rf = self.risk_free_rate

        def neg_sharpe(w: np.ndarray) -> float:
            port_return = float(w @ mu)
            port_vol = float(np.sqrt(max(1e-12, w @ sigma @ w)))
            return -(port_return - rf) / port_vol

        # Initial guess: equal weight clipped to constraint
        w0 = np.full(n, min(self.max_position_pct, self.max_total_exposure / n))

        # Bounds per symbol
        if self.allow_short:
            bounds = [(-self.max_position_pct, self.max_position_pct)] * n
        else:
            bounds = [(0.0, self.max_position_pct)] * n

        # Constraints: sum of weights <= max_total_exposure
        constraints = [
            {
                "type": "ineq",
                "fun": lambda w: self.max_total_exposure - np.sum(np.abs(w)),
            }
        ]

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning(
                "PortfolioOptimizer: optimisation did not converge ({}); "
                "using equal-weight fallback.",
                result.message,
            )
            return self._equal_weight(symbols)

        weights = result.x
        # Clean up near-zero weights
        weights = np.where(np.abs(weights) < 1e-6, 0.0, weights)

        portfolio = {sym: float(weights[i]) for i, sym in enumerate(symbols)}
        logger.info(
            "PortfolioOptimizer result: Sharpe={:.4f} total_exposure={:.2%} weights={}",
            -result.fun,
            float(np.sum(np.abs(weights))),
            {k: f"{v:.4f}" for k, v in portfolio.items()},
        )
        return portfolio

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Return equal portfolio weights respecting the total exposure cap."""
        n = len(symbols)
        if n == 0:
            return {}
        w = min(self.max_position_pct, self.max_total_exposure / n)
        return {sym: w for sym in symbols}

    def target_sizes(
        self,
        weights: Dict[str, float],
        equity: float,
    ) -> Dict[str, float]:
        """Convert portfolio weights to absolute position sizes in USDT.

        Args:
            weights: Symbol → weight from :meth:`optimize`.
            equity: Total portfolio equity in USDT.

        Returns:
            Symbol → USDT position size.
        """
        return {sym: abs(w) * equity for sym, w in weights.items()}
