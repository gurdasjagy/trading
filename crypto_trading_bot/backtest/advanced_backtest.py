"""Advanced backtesting features including Monte Carlo simulation,
walk-forward optimization, and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import itertools


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""

    mean_return: float
    median_return: float
    std_return: float
    best_case_return: float  # 95th percentile
    worst_case_return: float  # 5th percentile
    probability_of_profit: float
    var_95: float
    cvar_95: float
    simulated_returns: List[float]


@dataclass
class WalkForwardResult:
    """Walk-forward optimization result."""

    in_sample_sharpe: float
    out_of_sample_sharpe: float
    best_parameters: Dict[str, Any]
    stability_score: float  # How consistent parameters are across windows
    overfitting_score: float  # Difference between IS and OOS performance


class MonteCarloSimulator:
    """Monte Carlo simulation for backtesting robustness analysis.

    Simulates thousands of potential trading outcomes by resampling
    historical returns to understand strategy robustness.
    """

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate_strategy_returns(
        self,
        historical_returns: pd.Series,
        n_trades: int,
        starting_capital: float = 10000.0,
    ) -> MonteCarloResult:
        """Simulate strategy returns using bootstrap resampling.

        Args:
            historical_returns: Historical per-trade returns (%)
            n_trades: Number of trades to simulate per path
            starting_capital: Starting capital in USDT

        Returns:
            MonteCarloResult with simulation statistics
        """
        logger.info(
            f"Running Monte Carlo simulation: {self.n_simulations} paths, "
            f"{n_trades} trades each"
        )

        if len(historical_returns) < 10:
            raise ValueError("Need at least 10 historical returns for simulation")

        simulated_returns = []

        for _ in range(self.n_simulations):
            # Bootstrap sample returns
            sampled_returns = np.random.choice(
                historical_returns.values,
                size=n_trades,
                replace=True,
            )

            # Calculate path return
            path_return = ((1 + sampled_returns / 100).prod() - 1) * 100
            simulated_returns.append(path_return)

        simulated_returns = np.array(simulated_returns)

        # Calculate statistics
        mean_return = np.mean(simulated_returns)
        median_return = np.median(simulated_returns)
        std_return = np.std(simulated_returns)
        best_case = np.percentile(simulated_returns, 95)
        worst_case = np.percentile(simulated_returns, 5)
        probability_of_profit = np.mean(simulated_returns > 0) * 100

        # VaR and CVaR at 95% confidence
        var_95 = abs(np.percentile(simulated_returns, 5))
        tail_losses = simulated_returns[simulated_returns < -var_95]
        cvar_95 = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

        logger.info(
            f"Monte Carlo results: mean={mean_return:.2f}%, "
            f"median={median_return:.2f}%, P(profit)={probability_of_profit:.1f}%, "
            f"VaR95={var_95:.2f}%"
        )

        return MonteCarloResult(
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            best_case_return=best_case,
            worst_case_return=worst_case,
            probability_of_profit=probability_of_profit,
            var_95=var_95,
            cvar_95=cvar_95,
            simulated_returns=simulated_returns.tolist(),
        )

    def simulate_drawdown_distribution(
        self,
        historical_returns: pd.Series,
        n_periods: int = 252,
    ) -> Dict[str, float]:
        """Simulate distribution of maximum drawdowns.

        Args:
            historical_returns: Historical returns series
            n_periods: Number of periods to simulate (default 252 = 1 year)

        Returns:
            Dict with drawdown statistics
        """
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Bootstrap sample returns
            sampled_returns = np.random.choice(
                historical_returns.values,
                size=n_periods,
                replace=True,
            )

            # Calculate max drawdown for this path
            cumulative = (1 + sampled_returns / 100).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max * 100
            max_dd = abs(drawdown.min())
            max_drawdowns.append(max_dd)

        max_drawdowns = np.array(max_drawdowns)

        return {
            "mean_max_drawdown": np.mean(max_drawdowns),
            "median_max_drawdown": np.median(max_drawdowns),
            "95th_percentile_drawdown": np.percentile(max_drawdowns, 95),
            "99th_percentile_drawdown": np.percentile(max_drawdowns, 99),
        }


class WalkForwardOptimizer:
    """Walk-forward optimization for parameter robustness testing.

    Divides data into multiple train/test windows and optimizes parameters
    on training data, then tests on unseen data to detect overfitting.
    """

    def __init__(
        self,
        in_sample_periods: int = 120,  # Training window (days)
        out_of_sample_periods: int = 30,  # Testing window (days)
        step_size: int = 30,  # How much to roll forward (days)
    ):
        self.in_sample_periods = in_sample_periods
        self.out_of_sample_periods = out_of_sample_periods
        self.step_size = step_size

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        optimization_metric: str = "sharpe",
    ) -> WalkForwardResult:
        """Perform walk-forward optimization.

        Args:
            data: Historical price data with OHLCV columns
            strategy_func: Function(data, params) -> returns_series
            parameter_grid: Dict of param_name -> list of values to test
            optimization_metric: Metric to optimize ('sharpe', 'sortino', 'calmar')

        Returns:
            WalkForwardResult with optimization statistics
        """
        logger.info(
            f"Starting walk-forward optimization: "
            f"IS={self.in_sample_periods}d, OOS={self.out_of_sample_periods}d, "
            f"step={self.step_size}d"
        )

        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))

        logger.info(
            f"Testing {len(param_combinations)} parameter combinations "
            f"across multiple windows"
        )

        # Track results for each window
        window_results = []
        best_params_per_window = []

        # Walk forward through data
        start_idx = 0
        window_num = 0

        while start_idx + self.in_sample_periods + self.out_of_sample_periods <= len(data):
            window_num += 1

            # Define train/test split
            is_end = start_idx + self.in_sample_periods
            oos_end = is_end + self.out_of_sample_periods

            train_data = data.iloc[start_idx:is_end]
            test_data = data.iloc[is_end:oos_end]

            # Optimize on training data
            best_params, best_is_score = self._optimize_on_window(
                train_data,
                strategy_func,
                param_combinations,
                param_names,
                optimization_metric,
            )

            # Test on out-of-sample data
            test_returns = strategy_func(test_data, best_params)
            oos_score = self._calculate_metric(test_returns, optimization_metric)

            window_results.append({
                "window": window_num,
                "is_score": best_is_score,
                "oos_score": oos_score,
                "best_params": best_params,
            })

            best_params_per_window.append(best_params)

            logger.debug(
                f"Window {window_num}: IS {optimization_metric}={best_is_score:.3f}, "
                f"OOS {optimization_metric}={oos_score:.3f}, params={best_params}"
            )

            # Move to next window
            start_idx += self.step_size

        # Calculate aggregate statistics
        avg_is_sharpe = np.mean([w["is_score"] for w in window_results])
        avg_oos_sharpe = np.mean([w["oos_score"] for w in window_results])

        # Parameter stability: how often does each parameter value appear
        stability_score = self._calculate_stability_score(best_params_per_window)

        # Overfitting score: difference between IS and OOS
        overfitting_score = avg_is_sharpe - avg_oos_sharpe

        # Most common parameter set
        best_parameters = self._find_most_common_params(best_params_per_window)

        logger.info(
            f"Walk-forward complete: {window_num} windows, "
            f"avg IS {optimization_metric}={avg_is_sharpe:.3f}, "
            f"avg OOS {optimization_metric}={avg_oos_sharpe:.3f}, "
            f"stability={stability_score:.2f}, overfitting={overfitting_score:.3f}"
        )

        return WalkForwardResult(
            in_sample_sharpe=avg_is_sharpe,
            out_of_sample_sharpe=avg_oos_sharpe,
            best_parameters=best_parameters,
            stability_score=stability_score,
            overfitting_score=overfitting_score,
        )

    def _optimize_on_window(
        self,
        train_data: pd.DataFrame,
        strategy_func: Callable,
        param_combinations: List[Tuple],
        param_names: List[str],
        metric: str,
    ) -> Tuple[Dict, float]:
        """Optimize parameters on a single training window."""
        best_score = -np.inf
        best_params = {}

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            try:
                returns = strategy_func(train_data, params)
                score = self._calculate_metric(returns, metric)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as exc:
                logger.debug(f"Error testing params {params}: {exc}")
                continue

        return best_params, best_score

    def _calculate_metric(self, returns: pd.Series, metric: str) -> float:
        """Calculate optimization metric."""
        if len(returns) < 2:
            return -np.inf

        if metric == "sharpe":
            mean_ret = returns.mean()
            std_ret = returns.std()
            return (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else -np.inf

        elif metric == "sortino":
            mean_ret = returns.mean()
            downside = returns[returns < 0].std()
            return (mean_ret / downside * np.sqrt(252)) if downside > 0 else -np.inf

        elif metric == "calmar":
            total_ret = (1 + returns).prod() - 1
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min())
            return (total_ret / max_dd) if max_dd > 0 else -np.inf

        else:
            return -np.inf

    def _calculate_stability_score(
        self, param_sets: List[Dict]
    ) -> float:
        """Calculate parameter stability score (0-1)."""
        if not param_sets:
            return 0.0

        # For each parameter, calculate entropy of its distribution
        param_names = param_sets[0].keys()
        entropies = []

        for param_name in param_names:
            values = [p[param_name] for p in param_sets]
            unique_values = set(values)

            # Calculate value frequencies
            frequencies = [values.count(v) / len(values) for v in unique_values]

            # Calculate normalized entropy
            if len(unique_values) > 1:
                entropy = -sum(f * np.log(f) for f in frequencies if f > 0)
                max_entropy = np.log(len(unique_values))
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0.0  # All same value = very stable

            entropies.append(normalized_entropy)

        # Stability is inverse of average entropy
        avg_entropy = np.mean(entropies)
        stability = 1.0 - avg_entropy

        return stability

    def _find_most_common_params(self, param_sets: List[Dict]) -> Dict:
        """Find most frequently occurring parameter set."""
        if not param_sets:
            return {}

        # Convert param dicts to hashable tuples
        param_tuples = [tuple(sorted(p.items())) for p in param_sets]

        # Find most common
        from collections import Counter
        counter = Counter(param_tuples)
        most_common = counter.most_common(1)[0][0]

        return dict(most_common)


class SensitivityAnalyzer:
    """Analyze strategy sensitivity to parameter changes."""

    @staticmethod
    def analyze_parameter_sensitivity(
        data: pd.DataFrame,
        strategy_func: Callable,
        base_params: Dict[str, Any],
        param_to_test: str,
        test_range: List[Any],
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """Analyze how strategy performance varies with parameter changes.

        Args:
            data: Historical data
            strategy_func: Strategy function
            base_params: Base parameter set
            param_to_test: Parameter name to vary
            test_range: Range of values to test
            metric: Performance metric

        Returns:
            DataFrame with parameter values and corresponding metrics
        """
        results = []

        for value in test_range:
            test_params = base_params.copy()
            test_params[param_to_test] = value

            try:
                returns = strategy_func(data, test_params)
                score = WalkForwardOptimizer._calculate_metric(None, returns, metric)

                results.append({
                    "parameter_value": value,
                    "metric_value": score,
                })

            except Exception as exc:
                logger.debug(f"Error testing {param_to_test}={value}: {exc}")
                results.append({
                    "parameter_value": value,
                    "metric_value": np.nan,
                })

        df = pd.DataFrame(results)
        logger.info(
            f"Sensitivity analysis for {param_to_test}: "
            f"best value={df.loc[df['metric_value'].idxmax(), 'parameter_value']}"
        )

        return df
