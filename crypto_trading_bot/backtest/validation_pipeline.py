"""Strategy validation pipeline.

Orchestrates the full validation workflow for a strategy before deploying it
to live trading:

1. **In-sample backtest** — confirms the strategy can fit historical data.
2. **Walk-forward optimisation** — ensures parameters are not over-fitted.
3. **Monte Carlo stress test** — estimates probability of ruin and VaR.
4. **Final accept/reject decision** — applies minimum quality thresholds.

A strategy must pass all gates to be recommended for live deployment.

Minimum acceptance criteria (configurable):
* Walk-forward validation Sharpe ≥ 0.5
* Walk-forward max drawdown ≤ 30 %
* Walk-forward win rate ≥ 40 %
* Monte Carlo P(ruin) ≤ 5 %
* Monte Carlo VaR (95 %) ≥ -20 %
* Sharpe statistically significant at 5 % level
* In-sample Sharpe ≥ 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from backtest.backtest_engine import BacktestEngine
from backtest.monte_carlo import MonteCarloAnalyzer
from backtest.walk_forward_optimizer import WalkForwardOptimizer

# ------------------------------------------------------------------
# Threshold configuration
# ------------------------------------------------------------------

@dataclass
class ValidationThresholds:
    """Configurable acceptance thresholds for the validation pipeline."""

    min_insample_sharpe: float = 1.0
    min_wf_validation_sharpe: float = 0.5
    max_wf_max_drawdown_pct: float = 30.0
    min_wf_win_rate: float = 0.40
    max_mc_ruin_probability: float = 0.05
    min_mc_var_95_pct: float = -20.0  # VaR must not be worse than -20 %
    require_sharpe_significance: bool = True


# ------------------------------------------------------------------
# Validation result
# ------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Structured result from the full validation pipeline."""

    strategy_name: str
    symbol: str
    passed: bool

    # Gate results
    insample_passed: bool = False
    walk_forward_passed: bool = False
    monte_carlo_passed: bool = False
    sharpe_significance_passed: bool = False

    # Metrics
    insample_sharpe: float = 0.0
    wf_avg_validation_sharpe: float = 0.0
    wf_avg_max_drawdown_pct: float = 0.0
    wf_avg_win_rate: float = 0.0
    mc_ruin_probability: float = 1.0
    mc_var_95_pct: float = -100.0
    sharpe_p_value: float = 1.0

    # Recommended parameters from walk-forward
    recommended_params: Dict[str, Any] = field(default_factory=dict)

    # Failure reasons
    failure_reasons: List[str] = field(default_factory=list)

    # Full result data (for downstream analysis)
    insample_result: Optional[Dict] = None
    wf_result: Optional[Dict] = None
    mc_result: Optional[Dict] = None

    # Timestamp
    validated_at: datetime = field(default_factory=datetime.utcnow)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class StrategyValidator:
    """End-to-end strategy validation pipeline.

    Args:
        thresholds: Configurable acceptance thresholds.
        n_mc_simulations: Number of Monte Carlo paths (default 1 000).
        wf_training_days: Walk-forward training window in days.
        wf_validation_days: Walk-forward validation window in days.
    """

    def __init__(
        self,
        thresholds: Optional[ValidationThresholds] = None,
        n_mc_simulations: int = 1_000,
        wf_training_days: int = 60,
        wf_validation_days: int = 15,
    ) -> None:
        self.thresholds = thresholds or ValidationThresholds()
        self._backtest_engine = BacktestEngine()
        self._wf_optimizer = WalkForwardOptimizer(
            training_window_days=wf_training_days,
            validation_window_days=wf_validation_days,
            step_forward_days=wf_validation_days,
        )
        self._mc_analyzer = MonteCarloAnalyzer(n_simulations=n_mc_simulations)

        logger.info(
            "StrategyValidator initialised: n_mc={}, wf_train={}d, wf_val={}d",
            n_mc_simulations,
            wf_training_days,
            wf_validation_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(
        self,
        strategy_class: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        param_space: Optional[List[Any]] = None,
        initial_capital: float = 10_000.0,
    ) -> ValidationResult:
        """Run the full validation pipeline.

        Args:
            strategy_class: Strategy class (not instance) to validate.
            symbol: Trading symbol (e.g. ``"BTC/USDT"``).
            start_date: Start of the historical data window.
            end_date: End of the historical data window.
            param_space: Optional skopt parameter space for walk-forward.
                When ``None``, uses default parameters.
            initial_capital: Starting capital for Monte Carlo paths.

        Returns:
            :class:`ValidationResult` with gate outcomes and metrics.
        """
        strategy_name = getattr(strategy_class, "__name__", str(strategy_class))
        logger.info(
            "Starting validation pipeline: {} / {} [{} → {}]",
            strategy_name,
            symbol,
            start_date.date(),
            end_date.date(),
        )

        result = ValidationResult(
            strategy_name=strategy_name,
            symbol=symbol,
            passed=False,
        )

        # ── Gate 1: In-sample backtest ────────────────────────────────
        try:
            strategy_instance = strategy_class(symbols=[symbol])
            insample = await self._backtest_engine.run(
                strategy=strategy_instance,
                symbol=symbol,
                timeframe="1h",
                start_date=start_date,
                end_date=end_date,
            )
            result.insample_result = insample.metrics if hasattr(insample, "metrics") else {}
            result.insample_sharpe = float(
                (insample.metrics if hasattr(insample, "metrics") else {}).get("sharpe_ratio", 0.0)
            )
            result.insample_passed = result.insample_sharpe >= self.thresholds.min_insample_sharpe
            if not result.insample_passed:
                result.failure_reasons.append(
                    f"In-sample Sharpe {result.insample_sharpe:.3f} < "
                    f"{self.thresholds.min_insample_sharpe}"
                )

            # Extract trade returns for Monte Carlo
            trade_returns = self._extract_returns(insample)
        except Exception as exc:
            logger.error("In-sample backtest failed for {}: {}", strategy_name, exc)
            result.failure_reasons.append(f"In-sample backtest error: {exc}")
            return result

        # ── Gate 2: Walk-forward optimisation ────────────────────────
        try:
            space = param_space or []
            if space:
                wf_result = await self._wf_optimizer.optimize(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    param_space=space,
                    n_trials=20,
                )
            else:
                # No param space: just check robustness via held-out test
                wf_result = await self._quick_out_of_sample_check(
                    strategy_class, symbol, start_date, end_date
                )

            result.wf_result = wf_result
            result.wf_avg_validation_sharpe = float(
                wf_result.get("avg_validation_sharpe", 0.0)
            )
            result.wf_avg_max_drawdown_pct = float(
                wf_result.get("avg_validation_drawdown", 100.0)
            )
            result.wf_avg_win_rate = float(
                wf_result.get("avg_validation_win_rate", 0.0)
            )
            result.recommended_params = wf_result.get("recommended_ranges", {})

            wf_ok = (
                result.wf_avg_validation_sharpe >= self.thresholds.min_wf_validation_sharpe
                and result.wf_avg_max_drawdown_pct <= self.thresholds.max_wf_max_drawdown_pct
                and result.wf_avg_win_rate >= self.thresholds.min_wf_win_rate
            )
            result.walk_forward_passed = wf_ok
            if not wf_ok:
                result.failure_reasons.append(
                    f"Walk-forward failed: sharpe={result.wf_avg_validation_sharpe:.3f}, "
                    f"dd={result.wf_avg_max_drawdown_pct:.1f}%, "
                    f"wr={result.wf_avg_win_rate:.1%}"
                )
        except Exception as exc:
            logger.error("Walk-forward optimisation failed: {}", exc)
            result.failure_reasons.append(f"Walk-forward error: {exc}")
            result.walk_forward_passed = False

        # ── Gate 3: Monte Carlo stress test ──────────────────────────
        if trade_returns:
            try:
                mc_result = self._mc_analyzer.run(
                    trade_returns=trade_returns,
                    initial_capital=initial_capital,
                )
                result.mc_result = mc_result
                result.mc_ruin_probability = float(
                    mc_result.get("probability_of_ruin", 1.0)
                )
                result.mc_var_95_pct = float(
                    mc_result.get("var", {}).get("var_95", -100.0)
                )

                mc_ok = (
                    result.mc_ruin_probability <= self.thresholds.max_mc_ruin_probability
                    and result.mc_var_95_pct >= self.thresholds.min_mc_var_95_pct
                )
                result.monte_carlo_passed = mc_ok
                if not mc_ok:
                    result.failure_reasons.append(
                        f"Monte Carlo failed: ruin_prob={result.mc_ruin_probability:.1%}, "
                        f"VaR95={result.mc_var_95_pct:.1f}%"
                    )

                # Sharpe significance
                sig = mc_result.get("sharpe_significance", {})
                result.sharpe_p_value = float(sig.get("p_value", 1.0))
                if self.thresholds.require_sharpe_significance:
                    result.sharpe_significance_passed = sig.get(
                        "significant_at_5pct", False
                    )
                    if not result.sharpe_significance_passed:
                        result.failure_reasons.append(
                            f"Sharpe not statistically significant (p={result.sharpe_p_value:.3f})"
                        )
                else:
                    result.sharpe_significance_passed = True
            except Exception as exc:
                logger.error("Monte Carlo analysis failed: {}", exc)
                result.failure_reasons.append(f"Monte Carlo error: {exc}")
                result.monte_carlo_passed = False
        else:
            logger.warning(
                "No trades generated in backtest — Monte Carlo skipped for {}",
                strategy_name,
            )
            result.monte_carlo_passed = False
            result.failure_reasons.append("No trades generated in backtest")

        # ── Final decision ────────────────────────────────────────────
        result.passed = (
            result.insample_passed
            and result.walk_forward_passed
            and result.monte_carlo_passed
            and result.sharpe_significance_passed
        )

        status = "✓ PASSED" if result.passed else "✗ FAILED"
        logger.info(
            "Validation {} for {} / {}: insample={}, wf={}, mc={}, sig={}",
            status,
            strategy_name,
            symbol,
            result.insample_passed,
            result.walk_forward_passed,
            result.monte_carlo_passed,
            result.sharpe_significance_passed,
        )
        if result.failure_reasons:
            for reason in result.failure_reasons:
                logger.warning("  Failure: {}", reason)

        return result

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    async def validate_portfolio(
        self,
        strategy_classes: List[Any],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, ValidationResult]:
        """Validate multiple strategies sequentially.

        Args:
            strategy_classes: List of strategy class objects.
            symbol: Symbol to validate against.
            start_date: Historical start.
            end_date: Historical end.

        Returns:
            ``{strategy_name: ValidationResult}`` mapping.
        """
        results: Dict[str, ValidationResult] = {}
        for cls in strategy_classes:
            name = getattr(cls, "__name__", str(cls))
            try:
                result = await self.validate(
                    strategy_class=cls,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[name] = result
            except Exception as exc:
                logger.error("Portfolio validation failed for {}: {}", name, exc)
                results[name] = ValidationResult(
                    strategy_name=name,
                    symbol=symbol,
                    passed=False,
                    failure_reasons=[str(exc)],
                )

        passed = [n for n, r in results.items() if r.passed]
        logger.info(
            "Portfolio validation: {}/{} strategies passed",
            len(passed),
            len(strategy_classes),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _quick_out_of_sample_check(
        self,
        strategy_class: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Perform a simple 80/20 in-sample/out-of-sample check.

        Returns a WalkForwardOptimizer-compatible result dict.
        """
        total_days = (end_date - start_date).days
        split_days = int(total_days * 0.8)
        from datetime import timedelta

        train_end = start_date + timedelta(days=split_days)
        val_start = train_end
        val_end = end_date

        try:
            strategy = strategy_class(symbols=[symbol])
            val_result = await self._backtest_engine.run(
                strategy=strategy,
                symbol=symbol,
                timeframe="1h",
                start_date=val_start,
                end_date=val_end,
            )
            metrics = val_result.metrics if hasattr(val_result, "metrics") else {}
            return {
                "avg_validation_sharpe": metrics.get("sharpe_ratio", 0.0),
                "avg_validation_drawdown": metrics.get("max_drawdown", 100.0),
                "avg_validation_win_rate": metrics.get("win_rate", 0.0),
                "robust_parameters": metrics.get("sharpe_ratio", 0.0) > 0.5,
                "recommended_ranges": {},
            }
        except Exception as exc:
            logger.error("Quick OOS check failed: {}", exc)
            return {
                "avg_validation_sharpe": 0.0,
                "avg_validation_drawdown": 100.0,
                "avg_validation_win_rate": 0.0,
                "robust_parameters": False,
                "recommended_ranges": {},
            }

    @staticmethod
    def _extract_returns(backtest_result: Any) -> List[float]:
        """Extract per-trade return percentages from a backtest result."""
        if backtest_result is None:
            return []
        # Try standard BacktestResult API
        if hasattr(backtest_result, "trades"):
            return [
                float(t.get("pnl_pct", 0.0))
                for t in (backtest_result.trades or [])
                if isinstance(t, dict)
            ]
        # Try dict API
        if isinstance(backtest_result, dict):
            trades = backtest_result.get("trades", [])
            return [float(t.get("pnl_pct", 0.0)) for t in trades if isinstance(t, dict)]
        return []

    def get_summary_table(
        self, results: Dict[str, "ValidationResult"]
    ) -> List[Dict[str, Any]]:
        """Render a compact summary table for all validation results.

        Returns:
            List of dicts with columns: strategy, symbol, passed,
            insample_sharpe, wf_sharpe, mc_ruin_prob.
        """
        rows = []
        for name, r in results.items():
            rows.append(
                {
                    "strategy": name,
                    "symbol": r.symbol,
                    "passed": r.passed,
                    "insample_sharpe": round(r.insample_sharpe, 3),
                    "wf_sharpe": round(r.wf_avg_validation_sharpe, 3),
                    "wf_drawdown_pct": round(r.wf_avg_max_drawdown_pct, 1),
                    "mc_ruin_prob": round(r.mc_ruin_probability, 3),
                    "mc_var_95": round(r.mc_var_95_pct, 1),
                    "sharpe_pvalue": round(r.sharpe_p_value, 4),
                    "failure_reasons": r.failure_reasons,
                }
            )
        return sorted(rows, key=lambda x: x["passed"], reverse=True)
