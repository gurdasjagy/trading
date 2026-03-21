"""Strategy parameter optimizer using grid and random search."""

from __future__ import annotations

import itertools
import random
from datetime import datetime
from typing import Any, Dict, List, Type

from loguru import logger


class StrategyOptimizer:
    """Optimizes strategy parameters using grid search or random search."""

    def __init__(self, data_loader=None) -> None:
        self._data_loader = data_loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def optimize(
        self,
        strategy_class: Type,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        param_ranges: Dict[str, Any],
        method: str = "grid",
    ) -> dict:
        """Run parameter optimisation for *strategy_class*.

        Args:
            strategy_class: Strategy class to instantiate and backtest.
            symbol: Trading pair symbol (e.g. ``"BTC/USDT"``).
            start_date: Optimisation window start.
            end_date: Optimisation window end.
            param_ranges: Dict of parameter name → list of values (grid search)
                or ``(min, max)`` tuple (random search).
            method: ``"grid"`` or ``"random"``.

        Returns:
            Best parameter set dict with ``params`` and ``metrics`` keys.
        """
        logger.info(
            "Starting {} optimisation for {} ({} → {})",
            method,
            strategy_class.__name__ if hasattr(strategy_class, "__name__") else strategy_class,
            start_date.date(),
            end_date.date(),
        )

        if method == "grid":
            results = await self.grid_search(
                strategy_class, symbol, start_date, end_date, param_ranges
            )
        elif method == "random":
            results = await self.random_search(
                strategy_class, symbol, start_date, end_date, param_ranges
            )
        else:
            raise ValueError(f"Unknown optimisation method: {method!r}. Use 'grid' or 'random'.")

        best = self._select_best(results)
        logger.info("Optimisation complete. Best params: {}", best.get("params", {}))
        return best

    async def grid_search(
        self,
        strategy_class: Type,
        symbol: str,
        start: datetime,
        end: datetime,
        param_grid: Dict[str, List[Any]],
    ) -> List[dict]:
        """Exhaustive grid search over all parameter combinations.

        Args:
            strategy_class: Strategy class to evaluate.
            symbol: Trading pair.
            start: Start date.
            end: End date.
            param_grid: Dict of parameter name → list of candidate values.

        Returns:
            List of result dicts (``params`` + ``metrics``), sorted best-first.
        """
        combinations = self._generate_param_combinations(param_grid)
        logger.info("Grid search: {} parameter combinations", len(combinations))
        results = await self._evaluate_combinations(
            strategy_class, symbol, start, end, combinations
        )
        return results

    async def random_search(
        self,
        strategy_class: Type,
        symbol: str,
        start: datetime,
        end: datetime,
        param_ranges: Dict[str, Any],
        n_trials: int = 50,
    ) -> List[dict]:
        """Random search over the parameter space.

        Args:
            strategy_class: Strategy class to evaluate.
            symbol: Trading pair.
            start: Start date.
            end: End date.
            param_ranges: Dict of parameter name → list of values or
                ``(min, max)`` tuple (for numeric ranges).
            n_trials: Number of random trials to evaluate.

        Returns:
            List of result dicts sorted best-first.
        """
        logger.info("Random search: {} trials", n_trials)
        combinations = []
        for _ in range(n_trials):
            params = {}
            for key, value_spec in param_ranges.items():
                if (
                    isinstance(value_spec, (list, tuple))
                    and len(value_spec) == 2
                    and isinstance(value_spec[0], (int, float))
                ):
                    lo, hi = value_spec
                    params[key] = (
                        random.uniform(lo, hi)
                        if isinstance(lo, float)
                        else random.randint(int(lo), int(hi))
                    )
                elif isinstance(value_spec, list):
                    params[key] = random.choice(value_spec)
                else:
                    params[key] = value_spec
            combinations.append(params)

        return await self._evaluate_combinations(strategy_class, symbol, start, end, combinations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[dict]:
        """Cartesian product of all parameter values.

        Args:
            param_grid: Dict of parameter name → list of candidate values.

        Returns:
            List of dicts, each representing one parameter combination.
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def _select_best(self, results: List[dict], metric: str = "sharpe_ratio") -> dict:
        """Return the result with the highest value for *metric*.

        Args:
            results: List of result dicts with a ``metrics`` sub-dict.
            metric: Metric key to sort by.

        Returns:
            Best result dict, or empty dict if *results* is empty.
        """
        if not results:
            return {}
        valid = [r for r in results if r.get("metrics") is not None]
        if not valid:
            return {}
        return max(valid, key=lambda r: r["metrics"].get(metric, float("-inf")))

    async def _evaluate_combinations(
        self,
        strategy_class: Type,
        symbol: str,
        start: datetime,
        end: datetime,
        combinations: List[dict],
    ) -> List[dict]:
        """Run a backtest for each parameter combination and collect results.

        Args:
            strategy_class: Strategy class accepting keyword parameters.
            symbol: Trading pair.
            start: Start date.
            end: End date.
            combinations: List of parameter dicts to evaluate.

        Returns:
            List of result dicts with ``params`` and ``metrics`` keys.
        """
        from backtest.backtester import Backtester  # lazy import to avoid circular

        results: List[dict] = []
        for i, params in enumerate(combinations):
            try:
                strategy = strategy_class(**params)
                backtester = Backtester(data_loader=self._data_loader)
                result = await backtester.run(strategy, symbol, start, end)
                results.append(
                    {
                        "params": params,
                        "metrics": {
                            "sharpe_ratio": result.sharpe_ratio,
                            "sortino_ratio": result.sortino_ratio,
                            "max_drawdown_pct": result.max_drawdown_pct,
                            "total_return_pct": result.total_return_pct,
                            "win_rate": result.win_rate,
                            "profit_factor": result.profit_factor,
                            "calmar_ratio": result.calmar_ratio,
                            "total_trades": result.total_trades,
                        },
                        "result": result,
                    }
                )
            except Exception as exc:
                logger.warning("Evaluation failed for params {}: {}", params, exc)
                results.append({"params": params, "metrics": None, "error": str(exc)})

            if (i + 1) % 10 == 0:
                logger.info("Optimisation progress: {}/{}", i + 1, len(combinations))

        results.sort(
            key=lambda r: (
                r["metrics"].get("sharpe_ratio", float("-inf"))
                if r.get("metrics")
                else float("-inf")
            ),
            reverse=True,
        )
        return results
