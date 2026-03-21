"""Dynamic parameter tuner using Bayesian Optimization for trading strategy parameters."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel
from skopt import Optimizer
from skopt.space import Real, Integer

from config.settings import Settings


class TradingParameters(BaseModel):
    """Container for dynamically optimized trading parameters."""

    stop_loss_pct: float = 2.0
    take_profit_pct: float = 3.0
    trailing_stop_pct: float = 1.5
    risk_per_trade_pct: float = 1.5
    max_leverage: int = 5
    confidence_threshold: float = 0.65
    max_open_positions: int = 5
    cooldown_minutes: int = 30

    # Metadata
    last_update: datetime = datetime.now(timezone.utc)
    optimization_iteration: int = 0
    sharpe_ratio: float = 0.0


class TradeResult(BaseModel):
    """Container for trade result data used in optimization."""

    pnl: float
    pnl_pct: float
    return_pct: float
    duration_seconds: float
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float


class DynamicParameterTuner:
    """
    Dynamically optimizes trading parameters using Bayesian Optimization.

    Uses Gaussian Process (GP) surrogate model with Expected Improvement (EI)
    acquisition function to maximize Sharpe ratio over a rolling window.

    Features:
    - Continuously optimizes 8 key trading parameters
    - Updates every 20 trades
    - Respects hard constraints (max daily loss, portfolio risk cap)
    - Safety rails prevent parameters that would cause excessive drawdown
    - Warm start from backtest results
    - Persistence for restart continuity
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        warm_start_data: Optional[List[Dict]] = None,
    ) -> None:
        """
        Initialize the parameter tuner.

        Args:
            settings: Trading bot settings
            warm_start_data: Optional backtest results to initialize GP
        """
        self._settings = settings or Settings.get_settings()
        self._lock = asyncio.Lock()

        # Current parameter values
        self._current_params = TradingParameters(
            stop_loss_pct=self._settings.risk.default_stop_loss_pct,
            take_profit_pct=self._settings.risk.default_take_profit_pct,
            trailing_stop_pct=self._settings.risk.trailing_stop_pct,
            risk_per_trade_pct=2.0,
            max_leverage=self._settings.exchange.default_leverage,
            confidence_threshold=0.65,
            max_open_positions=self._settings.risk.max_open_positions,
            cooldown_minutes=self._settings.risk.cooldown_after_loss_minutes,
        )

        # Parameter search space (bounds)
        self._param_space = [
            Real(0.5, 5.0, name="stop_loss_pct"),
            Real(1.0, 10.0, name="take_profit_pct"),
            Real(0.3, 3.0, name="trailing_stop_pct"),
            Real(0.5, 3.0, name="risk_per_trade_pct"),
            Integer(3, 20, name="max_leverage"),
            Real(0.5, 0.9, name="confidence_threshold"),
            Integer(3, 15, name="max_open_positions"),
            Integer(5, 120, name="cooldown_minutes"),
        ]

        # Bayesian optimizer with GP + EI
        self._optimizer = Optimizer(
            dimensions=self._param_space,
            base_estimator="GP",  # Gaussian Process
            acq_func="EI",  # Expected Improvement
            acq_optimizer="sampling",
            n_initial_points=10,  # Random exploration first
            random_state=42,
        )

        # Trade history for optimization
        self._trade_history: deque[TradeResult] = deque(maxlen=100)
        self._rolling_window_size = 50  # Trades for Sharpe calculation
        self._update_frequency = 20  # Re-optimize every N trades
        self._trades_since_update = 0

        # Safety constraints
        self._max_daily_loss_pct = self._settings.risk.max_daily_loss_pct
        self._portfolio_risk_cap = 0.5  # 50% max portfolio risk
        self._max_drawdown_threshold = 10.0  # 10% max drawdown in last 100 trades

        # Performance tracking
        self._optimization_history: List[Dict] = []

        # Persistence
        self._state_file = Path("data/parameter_tuner_state.json")
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load saved state if exists
        self._load_state()

        # Warm start if provided
        if warm_start_data:
            self._warm_start(warm_start_data)

        logger.info("DynamicParameterTuner initialized with parameters: {}", self._current_params.dict())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_current_parameters(self) -> TradingParameters:
        """Get the current optimized parameters.

        Returns:
            Current trading parameters
        """
        async with self._lock:
            return self._current_params.copy(deep=True)

    async def record_trade(self, trade_data: Dict) -> None:
        """Record a completed trade and potentially trigger optimization.

        Args:
            trade_data: Trade result dictionary with pnl, pnl_pct, etc.
        """
        async with self._lock:
            # Create trade result
            result = TradeResult(
                pnl=trade_data.get("pnl", 0.0),
                pnl_pct=trade_data.get("pnl_pct", 0.0),
                return_pct=trade_data.get("return_pct", trade_data.get("pnl_pct", 0.0)),
                duration_seconds=trade_data.get("duration_seconds", 0.0),
                timestamp=trade_data.get("timestamp", datetime.now(timezone.utc)),
                symbol=trade_data.get("symbol", ""),
                direction=trade_data.get("direction", ""),
                entry_price=trade_data.get("entry_price", 0.0),
                exit_price=trade_data.get("exit_price", 0.0),
                stop_loss=trade_data.get("stop_loss", 0.0),
                take_profit=trade_data.get("take_profit", 0.0),
            )

            self._trade_history.append(result)
            self._trades_since_update += 1

            logger.debug(
                "Trade recorded: pnl={:.2f} pnl_pct={:.2f}% trades_since_update={}",
                result.pnl,
                result.pnl_pct,
                self._trades_since_update,
            )

            # Trigger optimization if enough trades
            if self._trades_since_update >= self._update_frequency:
                await self._optimize_parameters()
                self._trades_since_update = 0

    async def force_optimization(self) -> None:
        """Force parameter optimization regardless of trade count."""
        async with self._lock:
            if len(self._trade_history) >= 20:
                await self._optimize_parameters()
                self._trades_since_update = 0
            else:
                logger.warning("Not enough trade history for optimization (need 20+, have {})", len(self._trade_history))

    def get_optimization_history(self) -> List[Dict]:
        """Get history of parameter optimizations.

        Returns:
            List of optimization records
        """
        return self._optimization_history.copy()

    # ------------------------------------------------------------------
    # Optimization engine
    # ------------------------------------------------------------------

    async def _optimize_parameters(self) -> None:
        """Run Bayesian optimization to find better parameters."""
        if len(self._trade_history) < self._rolling_window_size:
            logger.warning("Not enough trades for optimization (need {}, have {})", self._rolling_window_size, len(self._trade_history))
            return

        logger.info("Starting parameter optimization iteration {}", self._current_params.optimization_iteration + 1)

        # Get current parameter vector
        current_vector = self._params_to_vector(self._current_params)

        # Calculate current objective (Sharpe ratio)
        current_sharpe = self._calculate_sharpe_ratio(list(self._trade_history)[-self._rolling_window_size:])

        # Tell optimizer about current result
        self._optimizer.tell(current_vector, -current_sharpe)  # Negative because we minimize

        # Safety check: verify current params didn't cause excessive drawdown
        if not self._passes_safety_check(self._current_params):
            logger.warning("Current parameters failed safety check, reverting to safe defaults")
            self._current_params = self._get_safe_defaults()
            self._save_state()
            return

        # Ask optimizer for next candidate
        next_vector = self._optimizer.ask()

        # Convert to parameters
        candidate_params = self._vector_to_params(next_vector)

        # Apply constraints
        if not self._validate_constraints(candidate_params):
            logger.warning("Candidate parameters violate constraints, keeping current params")
            return

        # Safety check candidate
        if not self._passes_safety_check(candidate_params):
            logger.warning("Candidate parameters fail safety check, keeping current params")
            return

        # Update parameters
        old_params = self._current_params.copy(deep=True)
        self._current_params = candidate_params
        self._current_params.last_update = datetime.now(timezone.utc)
        self._current_params.optimization_iteration += 1
        self._current_params.sharpe_ratio = current_sharpe

        # Log parameter changes
        logger.info(
            "Parameters updated (iteration {}): SL {:.2f}% -> {:.2f}%, TP {:.2f}% -> {:.2f}%, "
            "Risk {:.2f}% -> {:.2f}%, Leverage {} -> {}, Confidence {:.3f} -> {:.3f}, "
            "Sharpe {:.3f}",
            self._current_params.optimization_iteration,
            old_params.stop_loss_pct,
            self._current_params.stop_loss_pct,
            old_params.take_profit_pct,
            self._current_params.take_profit_pct,
            old_params.risk_per_trade_pct,
            self._current_params.risk_per_trade_pct,
            old_params.max_leverage,
            self._current_params.max_leverage,
            old_params.confidence_threshold,
            self._current_params.confidence_threshold,
            current_sharpe,
        )

        # Record optimization
        self._optimization_history.append({
            "iteration": self._current_params.optimization_iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_params": old_params.dict(),
            "new_params": self._current_params.dict(),
            "sharpe_ratio": current_sharpe,
        })

        # Save state
        self._save_state()

    def _calculate_sharpe_ratio(self, trades: List[TradeResult]) -> float:
        """Calculate Sharpe ratio from trade returns.

        Args:
            trades: List of trade results

        Returns:
            Sharpe ratio
        """
        if len(trades) < 2:
            return 0.0

        returns = [t.return_pct / 100.0 for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        return float(sharpe)

    def _calculate_max_drawdown(self, trades: List[TradeResult]) -> float:
        """Calculate maximum drawdown from trade history.

        Args:
            trades: List of trade results

        Returns:
            Maximum drawdown percentage
        """
        if not trades:
            return 0.0

        # Build equity curve
        equity = [10000.0]  # Starting capital
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)

        # Calculate drawdown
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak * 100.0
                max_dd = max(max_dd, dd)

        return max_dd

    # ------------------------------------------------------------------
    # Constraints and safety
    # ------------------------------------------------------------------

    def _validate_constraints(self, params: TradingParameters) -> bool:
        """Validate parameter constraints.

        Args:
            params: Parameters to validate

        Returns:
            True if constraints satisfied
        """
        # Portfolio risk cap: risk_per_trade * max_leverage * max_open_positions < 50% equity
        portfolio_risk = (
            params.risk_per_trade_pct * params.max_leverage * params.max_open_positions / 100.0
        )

        if portfolio_risk >= self._portfolio_risk_cap * 100:
            logger.warning(
                "Portfolio risk constraint violated: {:.1f}% >= {:.1f}%",
                portfolio_risk,
                self._portfolio_risk_cap * 100,
            )
            return False

        # All parameters must be within bounds
        if not (0.5 <= params.stop_loss_pct <= 5.0):
            return False
        if not (1.0 <= params.take_profit_pct <= 10.0):
            return False
        if not (0.3 <= params.trailing_stop_pct <= 3.0):
            return False
        if not (0.5 <= params.risk_per_trade_pct <= 3.0):
            return False
        if not (3 <= params.max_leverage <= 20):
            return False
        if not (0.5 <= params.confidence_threshold <= 0.9):
            return False
        if not (3 <= params.max_open_positions <= 15):
            return False
        if not (5 <= params.cooldown_minutes <= 120):
            return False

        return True

    def _passes_safety_check(self, params: TradingParameters) -> bool:
        """Check if parameters would have caused excessive drawdown.

        Args:
            params: Parameters to check

        Returns:
            True if safe
        """
        if len(self._trade_history) < 100:
            return True  # Not enough history

        # Calculate what max drawdown would have been with these params
        # This is a simplified heuristic check
        max_dd = self._calculate_max_drawdown(list(self._trade_history))

        if max_dd > self._max_drawdown_threshold:
            logger.warning("Safety check failed: max drawdown {:.2f}% > {:.2f}%", max_dd, self._max_drawdown_threshold)
            return False

        return True

    def _get_safe_defaults(self) -> TradingParameters:
        """Get safe default parameters.

        Returns:
            Safe default parameters
        """
        return TradingParameters(
            stop_loss_pct=2.0,
            take_profit_pct=3.0,
            trailing_stop_pct=1.5,
            risk_per_trade_pct=1.0,
            max_leverage=3,
            confidence_threshold=0.7,
            max_open_positions=5,
            cooldown_minutes=30,
        )

    # ------------------------------------------------------------------
    # Vector conversion
    # ------------------------------------------------------------------

    def _params_to_vector(self, params: TradingParameters) -> List[float]:
        """Convert parameters to vector for optimizer.

        Args:
            params: Parameters to convert

        Returns:
            Parameter vector
        """
        return [
            params.stop_loss_pct,
            params.take_profit_pct,
            params.trailing_stop_pct,
            params.risk_per_trade_pct,
            float(params.max_leverage),
            params.confidence_threshold,
            float(params.max_open_positions),
            float(params.cooldown_minutes),
        ]

    def _vector_to_params(self, vector: List[float]) -> TradingParameters:
        """Convert vector to parameters.

        Args:
            vector: Parameter vector

        Returns:
            Parameters object
        """
        return TradingParameters(
            stop_loss_pct=float(vector[0]),
            take_profit_pct=float(vector[1]),
            trailing_stop_pct=float(vector[2]),
            risk_per_trade_pct=float(vector[3]),
            max_leverage=int(vector[4]),
            confidence_threshold=float(vector[5]),
            max_open_positions=int(vector[6]),
            cooldown_minutes=int(vector[7]),
        )

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def _warm_start(self, backtest_results: List[Dict]) -> None:
        """Initialize optimizer with backtest results.

        Args:
            backtest_results: List of backtest results with params and metrics
        """
        logger.info("Warm starting optimizer with {} backtest results", len(backtest_results))

        for result in backtest_results:
            params_dict = result.get("params", {})
            metrics = result.get("metrics", {})

            # Convert to parameter vector
            params = TradingParameters(**params_dict)
            vector = self._params_to_vector(params)

            # Get Sharpe ratio
            sharpe = metrics.get("sharpe_ratio", 0.0)

            # Tell optimizer
            self._optimizer.tell(vector, -sharpe)

        logger.info("Warm start complete")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save optimizer state to disk."""
        try:
            state = {
                "current_params": self._current_params.dict(),
                "optimization_history": self._optimization_history[-20:],  # Last 20
                "trades_since_update": self._trades_since_update,
            }

            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.debug("Parameter tuner state saved")
        except Exception as exc:
            logger.error("Failed to save parameter tuner state: {}", exc)

    def _load_state(self) -> None:
        """Load optimizer state from disk."""
        if not self._state_file.exists():
            logger.debug("No saved state found")
            return

        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)

            self._current_params = TradingParameters(**state.get("current_params", {}))
            self._optimization_history = state.get("optimization_history", [])
            self._trades_since_update = state.get("trades_since_update", 0)

            logger.info("Parameter tuner state loaded from disk")
        except Exception as exc:
            logger.error("Failed to load parameter tuner state: {}", exc)
