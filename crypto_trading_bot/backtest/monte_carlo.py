"""Monte Carlo simulation for strategy validation and risk analysis.

Runs N paths of bootstrapped or shuffled trade returns to estimate:
* Distribution of final equity
* Value at Risk (VaR) and Conditional Value at Risk (CVaR) at multiple
  confidence levels
* Maximum drawdown distribution
* Probability of ruin (equity < 0)
* Statistical significance of Sharpe ratio
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class MonteCarloAnalyzer:
    """Run Monte Carlo simulations over backtest trade returns.

    Args:
        n_simulations: Number of simulation paths (default 1 000).
        confidence_levels: VaR/CVaR confidence levels (default [0.90, 0.95, 0.99]).
        seed: Random seed for reproducibility.
        bootstrap: When ``True``, sample returns with replacement (bootstrap);
            when ``False``, use random permutation of the observed returns.
    """

    def __init__(
        self,
        n_simulations: int = 1_000,
        confidence_levels: Optional[List[float]] = None,
        seed: Optional[int] = None,
        bootstrap: bool = True,
    ) -> None:
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.bootstrap = bootstrap

        self._rng = np.random.default_rng(seed)

        logger.info(
            "MonteCarloAnalyzer initialised: n_simulations={}, bootstrap={}",
            n_simulations,
            bootstrap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        trade_returns: List[float],
        initial_capital: float = 10_000.0,
    ) -> Dict[str, Any]:
        """Run the Monte Carlo simulation on *trade_returns*.

        Args:
            trade_returns: List of per-trade P&L percentages (e.g. 2.5 for +2.5 %).
            initial_capital: Starting equity for each path.

        Returns:
            Comprehensive result dict.  See docstring of
            :meth:`_compile_results` for the full schema.
        """
        if len(trade_returns) < 5:
            logger.warning(
                "Monte Carlo: insufficient returns ({} < 5), returning defaults",
                len(trade_returns),
            )
            return self._empty_result()

        returns_arr = np.array(trade_returns, dtype=float)
        n_trades = len(returns_arr)

        # Run simulations
        final_equities: List[float] = []
        max_drawdowns: List[float] = []
        all_equity_curves: List[np.ndarray] = []

        for _ in range(self.n_simulations):
            if self.bootstrap:
                sim_returns = self._rng.choice(returns_arr, size=n_trades, replace=True)
            else:
                sim_returns = self._rng.permutation(returns_arr)

            equity_curve, max_dd = self._simulate_path(sim_returns, initial_capital)
            final_equities.append(equity_curve[-1])
            max_drawdowns.append(max_dd)
            if len(all_equity_curves) < 100:  # Store only first 100 paths
                all_equity_curves.append(equity_curve)

        return self._compile_results(
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            initial_capital=initial_capital,
            trade_returns=trade_returns,
            equity_curves=all_equity_curves,
        )

    def estimate_sharpe_significance(
        self,
        observed_sharpe: float,
        trade_returns: List[float],
        n_permutations: int = 1_000,
    ) -> Dict[str, float]:
        """Estimate the statistical significance of an observed Sharpe ratio.

        Uses a permutation test: shuffle returns N times and count how often
        the shuffled Sharpe exceeds the observed one.

        Args:
            observed_sharpe: The Sharpe ratio from the actual backtest.
            trade_returns: List of per-trade returns.
            n_permutations: Number of permutations for the p-value estimate.

        Returns:
            Dict with ``p_value``, ``z_score``, ``significant_at_5pct``.
        """
        arr = np.array(trade_returns, dtype=float)
        perm_sharpes: List[float] = []
        for _ in range(n_permutations):
            perm = self._rng.permutation(arr)
            perm_sharpes.append(self._sharpe(perm.tolist()))

        perm_arr = np.array(perm_sharpes)
        p_value = float(np.mean(perm_arr >= observed_sharpe))
        perm_mean = float(np.mean(perm_arr))
        perm_std = float(np.std(perm_arr))
        z_score = (
            (observed_sharpe - perm_mean) / perm_std if perm_std > 0 else 0.0
        )
        return {
            "observed_sharpe": observed_sharpe,
            "p_value": p_value,
            "z_score": z_score,
            "permutation_mean_sharpe": perm_mean,
            "significant_at_5pct": p_value < 0.05,
            "significant_at_1pct": p_value < 0.01,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_path(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> Tuple[np.ndarray, float]:
        """Simulate one equity path.

        Returns:
            ``(equity_curve, max_drawdown_pct)``.
        """
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0
        curve = [initial_capital]

        for r in returns:
            equity *= 1.0 + r / 100.0
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            curve.append(equity)

        return np.array(curve), max_dd

    def _compile_results(
        self,
        final_equities: List[float],
        max_drawdowns: List[float],
        initial_capital: float,
        trade_returns: List[float],
        equity_curves: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Build the full result dict."""
        fe_arr = np.array(final_equities)
        dd_arr = np.array(max_drawdowns)

        # Return distribution in % terms
        path_returns_pct = ((fe_arr - initial_capital) / initial_capital) * 100.0

        var_results: Dict[str, float] = {}
        cvar_results: Dict[str, float] = {}
        for cl in self.confidence_levels:
            var_results[f"var_{int(cl * 100)}"] = float(
                np.percentile(path_returns_pct, (1 - cl) * 100)
            )
            tail = path_returns_pct[path_returns_pct <= var_results[f"var_{int(cl * 100)}"]]
            cvar_results[f"cvar_{int(cl * 100)}"] = float(
                np.mean(tail) if len(tail) > 0 else var_results[f"var_{int(cl * 100)}"]
            )

        ruin_prob = float(np.mean(fe_arr <= 0.0))
        mean_return = float(np.mean(path_returns_pct))
        median_return = float(np.median(path_returns_pct))
        std_return = float(np.std(path_returns_pct))
        best_return = float(np.max(path_returns_pct))
        worst_return = float(np.min(path_returns_pct))

        observed_sharpe = self._sharpe(trade_returns)
        sig = self.estimate_sharpe_significance(observed_sharpe, trade_returns)

        return {
            "n_simulations": self.n_simulations,
            "n_trades": len(trade_returns),
            "initial_capital": initial_capital,
            # Return stats
            "mean_return_pct": round(mean_return, 2),
            "median_return_pct": round(median_return, 2),
            "std_return_pct": round(std_return, 2),
            "best_return_pct": round(best_return, 2),
            "worst_return_pct": round(worst_return, 2),
            # Risk metrics
            "var": var_results,
            "cvar": cvar_results,
            "probability_of_ruin": round(ruin_prob, 4),
            # Drawdown
            "mean_max_drawdown_pct": round(float(np.mean(dd_arr)), 2),
            "worst_max_drawdown_pct": round(float(np.max(dd_arr)), 2),
            "median_max_drawdown_pct": round(float(np.median(dd_arr)), 2),
            # Percentiles of final equity
            "equity_p5": round(float(np.percentile(fe_arr, 5)), 2),
            "equity_p25": round(float(np.percentile(fe_arr, 25)), 2),
            "equity_p50": round(float(np.percentile(fe_arr, 50)), 2),
            "equity_p75": round(float(np.percentile(fe_arr, 75)), 2),
            "equity_p95": round(float(np.percentile(fe_arr, 95)), 2),
            # Sharpe significance
            "sharpe_significance": sig,
            # Sample equity curves (first 100)
            "sample_equity_curves": [c.tolist() for c in equity_curves],
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return a zero-filled result when there is insufficient data."""
        return {
            "n_simulations": 0,
            "n_trades": 0,
            "mean_return_pct": 0.0,
            "probability_of_ruin": 1.0,
            "var": {},
            "cvar": {},
            "sharpe_significance": {"p_value": 1.0, "significant_at_5pct": False},
        }

    @staticmethod
    def _sharpe(returns: List[float], annualisation: float = 252.0) -> float:
        """Annualised Sharpe from a list of per-trade return percentages."""
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-9:
            return 0.0
        return (mean / std) * math.sqrt(annualisation)
