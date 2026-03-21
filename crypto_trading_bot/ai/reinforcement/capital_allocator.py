"""Strategy capital allocator using Sharpe-ratio-weighted softmax allocation.

Allocates a capital budget across the top-K strategies each cycle.
Strategies with higher expected Sharpe ratios receive proportionally more
capital, subject to minimum (5 %) and maximum (30 %) per-strategy bounds.
Allocations are rebalanced every 4 hours based on rolling performance.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class StrategyCapitalAllocator:
    """Allocates capital across strategies proportional to their Sharpe ratios.

    Algorithm
    ---------
    1. Compute the Sharpe ratio for each strategy from ``rolling_metrics``.
    2. Apply softmax (temperature=0.5) to the Sharpe ratios to obtain raw
       allocation weights.
    3. Clamp each weight to ``[min_alloc, max_alloc]``.
    4. Re-normalise to ensure allocations sum to 1.0.
    5. Multiply by ``total_capital`` to get absolute amounts.

    Rebalancing is triggered when :meth:`rebalance` is called or when
    ``auto_rebalance=True`` and more than ``rebalance_interval_hours`` have
    passed since the last rebalance.

    Args:
        strategy_names: Names of all strategies that will be considered.
        min_alloc: Minimum allocation fraction per active strategy (default 5 %).
        max_alloc: Maximum allocation fraction per strategy (default 30 %).
        softmax_temperature: Controls how sharply capital concentrates on the
            best-performing strategies (lower = more concentrated).
        rebalance_interval_hours: Hours between automatic rebalances.
        history_maxlen: Maximum number of allocation snapshots to keep.
    """

    def __init__(
        self,
        strategy_names: List[str],
        min_alloc: float = 0.05,
        max_alloc: float = 0.30,
        softmax_temperature: float = 0.5,
        rebalance_interval_hours: float = 4.0,
        history_maxlen: int = 200,
    ) -> None:
        if not (0.0 < min_alloc < max_alloc <= 1.0):
            raise ValueError(
                f"Invalid allocation bounds: min={min_alloc}, max={max_alloc}"
            )
        self.strategy_names = list(strategy_names)
        self.min_alloc = min_alloc
        self.max_alloc = max_alloc
        self.softmax_temperature = softmax_temperature
        self.rebalance_interval_hours = rebalance_interval_hours

        # Current normalised weights (strategy_name -> fraction)
        self._current_weights: Dict[str, float] = {
            name: 1.0 / len(strategy_names) for name in strategy_names
        }

        # Allocation history for analysis
        self._history: deque = deque(maxlen=history_maxlen)

        # Timestamps
        self._last_rebalance_ts: float = time.time()

        logger.info(
            "StrategyCapitalAllocator initialised: {} strategies, "
            "min_alloc={:.0%}, max_alloc={:.0%}",
            len(strategy_names),
            min_alloc,
            max_alloc,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        rolling_metrics: Dict[str, Dict[str, Any]],
        total_capital: float,
        active_strategies: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute capital amounts for each strategy.

        Args:
            rolling_metrics: Mapping of strategy_name → metrics dict.
                Expected keys in the inner dict:
                ``win_rate`` (float), ``avg_profit`` (float),
                ``avg_loss`` (float), ``sharpe`` (float).
            total_capital: Total capital to distribute (quote currency).
            active_strategies: If provided, only allocate to these strategies
                (others receive 0).

        Returns:
            ``{strategy_name: capital_amount}`` mapping.  Capital amounts
            sum to ``total_capital``.
        """
        candidates = active_strategies or self.strategy_names
        candidates = [n for n in candidates if n in self.strategy_names]

        if not candidates:
            logger.warning("No valid candidates for capital allocation — returning empty")
            return {}

        sharpes = self._compute_sharpes(rolling_metrics, candidates)
        weights = self._softmax_weights(sharpes, candidates)
        weights = self._clamp_and_renormalise(weights, candidates)

        self._current_weights.update(weights)

        allocation = {name: weights[name] * total_capital for name in candidates}

        # Zero-out non-candidates
        for name in self.strategy_names:
            if name not in allocation:
                allocation[name] = 0.0

        # Record in history
        self._history.append(
            {
                "timestamp": time.time(),
                "total_capital": total_capital,
                "weights": dict(weights),
                "allocation": {k: v for k, v in allocation.items() if v > 0},
            }
        )

        logger.debug(
            "Capital allocated: total={:.2f}, top_3={}",
            total_capital,
            sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3],
        )
        return allocation

    def rebalance(
        self,
        rolling_metrics: Dict[str, Dict[str, Any]],
        total_capital: float,
        active_strategies: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Force a rebalance and return the new allocation.

        This is an alias for :meth:`allocate` that also updates the
        rebalance timestamp.

        Should be called every ``rebalance_interval_hours`` hours from the
        scheduler.
        """
        allocation = self.allocate(rolling_metrics, total_capital, active_strategies)
        self._last_rebalance_ts = time.time()
        logger.info(
            "Portfolio rebalanced at {}",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return allocation

    def should_rebalance(self) -> bool:
        """Return True if the rebalance interval has elapsed."""
        elapsed_hours = (time.time() - self._last_rebalance_ts) / 3600.0
        return elapsed_hours >= self.rebalance_interval_hours

    def get_current_weights(self) -> Dict[str, float]:
        """Return a copy of the current normalised weights."""
        return dict(self._current_weights)

    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """Return the full allocation history as a list of snapshots."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sharpes(
        self,
        rolling_metrics: Dict[str, Dict[str, Any]],
        candidates: List[str],
    ) -> Dict[str, float]:
        """Extract or estimate Sharpe ratio for each candidate strategy."""
        sharpes: Dict[str, float] = {}
        for name in candidates:
            metrics = rolling_metrics.get(name, {})
            if "sharpe" in metrics and not math.isnan(metrics["sharpe"]):
                sharpe = float(metrics["sharpe"])
            else:
                # Estimate from win_rate, avg_profit, avg_loss
                win_rate = float(metrics.get("win_rate", 0.5))
                avg_profit = float(metrics.get("avg_profit", 0.0))
                avg_loss = float(metrics.get("avg_loss", 0.0))
                total_trades = int(metrics.get("total_trades", 0))
                if total_trades < 5:
                    sharpe = 0.0
                else:
                    expected_pnl = win_rate * avg_profit + (1 - win_rate) * avg_loss
                    variance = (
                        win_rate * (avg_profit - expected_pnl) ** 2
                        + (1 - win_rate) * (avg_loss - expected_pnl) ** 2
                    )
                    std_pnl = math.sqrt(variance) if variance > 0 else 1e-6
                    sharpe = (expected_pnl / std_pnl) * math.sqrt(252)
            sharpes[name] = max(sharpe, -3.0)  # floor at -3
        return sharpes

    def _softmax_weights(
        self,
        sharpes: Dict[str, float],
        candidates: List[str],
    ) -> Dict[str, float]:
        """Apply softmax with ``softmax_temperature`` to Sharpe values."""
        values = np.array([sharpes[n] for n in candidates], dtype=float)
        # Shift for numerical stability
        values = values / self.softmax_temperature
        values -= values.max()
        exp_vals = np.exp(values)
        exp_sum = exp_vals.sum()
        if exp_sum == 0 or math.isnan(exp_sum):
            uniform = 1.0 / len(candidates)
            return {n: uniform for n in candidates}
        weights = exp_vals / exp_sum
        return dict(zip(candidates, weights.tolist()))

    def _clamp_and_renormalise(
        self,
        weights: Dict[str, float],
        candidates: List[str],
    ) -> Dict[str, float]:
        """Clamp each weight to ``[min_alloc, max_alloc]`` then renormalise."""
        # Ensure min_alloc doesn't exceed 1/n for feasibility
        n = len(candidates)
        effective_min = min(self.min_alloc, 1.0 / n)

        clamped = {
            name: max(effective_min, min(self.max_alloc, weights[name]))
            for name in candidates
        }

        total = sum(clamped.values())
        if total == 0:
            uniform = 1.0 / n
            return {name: uniform for name in candidates}

        return {name: v / total for name, v in clamped.items()}
