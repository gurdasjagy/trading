"""Walk-forward optimizer for parameter validation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from skopt import Optimizer
from skopt.space import Real, Integer

from backtest.backtest_engine import BacktestEngine
from strategy.base_strategy import BaseStrategy


class WalkForwardOptimizer:
    """
    Implements walk-forward analysis for parameter validation.

    Splits historical data into training/validation windows, optimizes
    on training, validates on out-of-sample data, then steps forward.

    This prevents overfitting and provides robust parameter ranges.
    """

    def __init__(
        self,
        training_window_days: int = 60,
        validation_window_days: int = 15,
        step_forward_days: int = 15,
    ) -> None:
        """
        Initialize walk-forward optimizer.

        Args:
            training_window_days: Training window size
            validation_window_days: Validation window size
            step_forward_days: Step forward increment
        """
        self._training_window = training_window_days
        self._validation_window = validation_window_days
        self._step_forward = step_forward_days

        self._backtest_engine = BacktestEngine()

        logger.info(
            "WalkForwardOptimizer initialized: train={}d validation={}d step={}d",
            training_window_days,
            validation_window_days,
            step_forward_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def optimize(
        self,
        strategy_class: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        param_space: List[Any],
        n_trials: int = 20,
    ) -> Dict:
        """Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            symbol: Trading symbol
            start_date: Overall start date
            end_date: Overall end date
            param_space: Parameter search space (skopt dimensions)
            n_trials: Trials per window

        Returns:
            Optimization results dict with recommended parameter ranges
        """
        logger.info(
            "Starting walk-forward optimization: {} {} → {}",
            symbol,
            start_date.date(),
            end_date.date(),
        )

        # Generate windows
        windows = self._generate_windows(start_date, end_date)
        logger.info("Generated {} walk-forward windows", len(windows))

        # Optimize each window
        results = []
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            logger.info(
                "Window {}/{}: train {} → {} | val {} → {}",
                i + 1,
                len(windows),
                train_start.date(),
                train_end.date(),
                val_start.date(),
                val_end.date(),
            )

            window_result = await self._optimize_window(
                strategy_class,
                symbol,
                train_start,
                train_end,
                val_start,
                val_end,
                param_space,
                n_trials,
            )

            results.append(window_result)

        # Aggregate results
        summary = self._aggregate_results(results, param_space)

        logger.info(
            "Walk-forward optimization complete: avg validation Sharpe {:.3f}",
            summary["avg_validation_sharpe"],
        )

        return summary

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def _generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward windows.

        Args:
            start_date: Overall start
            end_date: Overall end

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        windows = []
        current = start_date

        while True:
            train_start = current
            train_end = train_start + timedelta(days=self._training_window)
            val_start = train_end
            val_end = val_start + timedelta(days=self._validation_window)

            # Stop if validation window exceeds end date
            if val_end > end_date:
                break

            windows.append((train_start, train_end, val_start, val_end))

            # Step forward
            current += timedelta(days=self._step_forward)

        return windows

    # ------------------------------------------------------------------
    # Window optimization
    # ------------------------------------------------------------------

    async def _optimize_window(
        self,
        strategy_class: Any,
        symbol: str,
        train_start: datetime,
        train_end: datetime,
        val_start: datetime,
        val_end: datetime,
        param_space: List[Any],
        n_trials: int,
    ) -> Dict:
        """Optimize a single window.

        Args:
            strategy_class: Strategy class
            symbol: Symbol
            train_start: Training start
            train_end: Training end
            val_start: Validation start
            val_end: Validation end
            param_space: Parameter space
            n_trials: Number of trials

        Returns:
            Window result dict
        """
        # Create optimizer
        optimizer = Optimizer(
            dimensions=param_space,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=min(5, n_trials // 2),
            random_state=42,
        )

        best_params = None
        best_train_sharpe = -np.inf

        # Bayesian optimization on training data
        for trial in range(n_trials):
            # Ask optimizer for next parameters
            params_vector = optimizer.ask()

            # Convert to dict
            params_dict = self._vector_to_dict(params_vector, param_space)

            # Run backtest on training data
            try:
                strategy = strategy_class(symbols=[symbol], **params_dict)
                train_result = await self._backtest_engine.run(
                    strategy=strategy,
                    symbol=symbol,
                    timeframe="1h",
                    start_date=train_start,
                    end_date=train_end,
                )

                train_sharpe = train_result.metrics.get("sharpe_ratio", 0.0)

                # Tell optimizer
                optimizer.tell(params_vector, -train_sharpe)  # Negative to minimize

                # Track best
                if train_sharpe > best_train_sharpe:
                    best_train_sharpe = train_sharpe
                    best_params = params_dict

                logger.debug(
                    "Trial {}/{}: train Sharpe {:.3f}",
                    trial + 1,
                    n_trials,
                    train_sharpe,
                )

            except Exception as exc:
                logger.error("Trial {} failed: {}", trial + 1, exc)
                # Tell optimizer with poor score
                optimizer.tell(params_vector, 10.0)

        # Validate best params on out-of-sample data
        if best_params:
            try:
                strategy = strategy_class(symbols=[symbol], **best_params)
                val_result = await self._backtest_engine.run(
                    strategy=strategy,
                    symbol=symbol,
                    timeframe="1h",
                    start_date=val_start,
                    end_date=val_end,
                )

                val_sharpe = val_result.metrics.get("sharpe_ratio", 0.0)
                val_max_dd = val_result.metrics.get("max_drawdown", 0.0)
                val_win_rate = val_result.metrics.get("win_rate", 0.0)

                logger.info(
                    "Best params validation: Sharpe {:.3f} DD {:.2f}% WR {:.1f}%",
                    val_sharpe,
                    val_max_dd,
                    val_win_rate * 100,
                )

            except Exception as exc:
                logger.error("Validation failed: {}", exc)
                val_sharpe = 0.0
                val_max_dd = 100.0
                val_win_rate = 0.0

        else:
            val_sharpe = 0.0
            val_max_dd = 100.0
            val_win_rate = 0.0

        return {
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "best_params": best_params or {},
            "train_sharpe": best_train_sharpe,
            "val_sharpe": val_sharpe,
            "val_max_drawdown": val_max_dd,
            "val_win_rate": val_win_rate,
        }

    # ------------------------------------------------------------------
    # Results aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(self, results: List[Dict], param_space: List[Any]) -> Dict:
        """Aggregate walk-forward results.

        Args:
            results: List of window results
            param_space: Parameter space

        Returns:
            Aggregated summary
        """
        if not results:
            return {
                "avg_validation_sharpe": 0.0,
                "robust_parameters": False,
                "recommended_ranges": {},
            }

        # Aggregate validation metrics
        val_sharpes = [r["val_sharpe"] for r in results]
        val_drawdowns = [r["val_max_drawdown"] for r in results]
        val_win_rates = [r["val_win_rate"] for r in results]

        avg_val_sharpe = float(np.mean(val_sharpes))
        std_val_sharpe = float(np.std(val_sharpes))
        avg_val_dd = float(np.mean(val_drawdowns))
        avg_val_wr = float(np.mean(val_win_rates))

        # Check robustness: Sharpe should be consistent
        robust = avg_val_sharpe > 1.0 and std_val_sharpe < 0.5

        # Extract parameter distributions
        param_names = [dim.name for dim in param_space if hasattr(dim, "name")]
        param_values = {name: [] for name in param_names}

        for result in results:
            best_params = result.get("best_params", {})
            for name in param_names:
                if name in best_params:
                    param_values[name].append(best_params[name])

        # Calculate recommended ranges (mean ± std)
        recommended_ranges = {}
        for name, values in param_values.items():
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                recommended_ranges[name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "recommended_min": max(mean_val - 2 * std_val, float(np.min(values))),
                    "recommended_max": min(mean_val + 2 * std_val, float(np.max(values))),
                }

        return {
            "total_windows": len(results),
            "avg_validation_sharpe": avg_val_sharpe,
            "std_validation_sharpe": std_val_sharpe,
            "avg_validation_drawdown": avg_val_dd,
            "avg_validation_win_rate": avg_val_wr,
            "robust_parameters": robust,
            "recommended_ranges": recommended_ranges,
            "window_results": results,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vector_to_dict(self, vector: List[float], param_space: List[Any]) -> Dict:
        """Convert parameter vector to dict.

        Args:
            vector: Parameter vector
            param_space: Parameter space

        Returns:
            Parameter dict
        """
        params = {}
        for i, dim in enumerate(param_space):
            if hasattr(dim, "name"):
                name = dim.name
                value = vector[i]

                # Convert to int if Integer dimension
                if isinstance(dim, Integer):
                    value = int(value)

                params[name] = value

        return params
