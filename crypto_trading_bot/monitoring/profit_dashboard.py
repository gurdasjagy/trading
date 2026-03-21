"""Profit maximization dashboard endpoints and real-time data."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter
from loguru import logger

from ai.profit_maximizer import ProfitMaximizer
from ai.reinforcement.parameter_tuner import DynamicParameterTuner
from ai.reinforcement.meta_learner import MetaLearner


def create_profit_routes(
    profit_maximizer: Optional[ProfitMaximizer] = None,
    parameter_tuner: Optional[DynamicParameterTuner] = None,
    meta_learner: Optional[MetaLearner] = None,
) -> APIRouter:
    """Create FastAPI routes for profit maximization dashboard.

    Args:
        profit_maximizer: ProfitMaximizer instance
        parameter_tuner: DynamicParameterTuner instance
        meta_learner: MetaLearner instance

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/profit", tags=["profit_maximization"])

    @router.get("/status")
    async def get_profit_status() -> Dict:
        """Get profit maximization system status."""
        if not profit_maximizer:
            return {"error": "ProfitMaximizer not initialized"}

        status = profit_maximizer.get_status()

        # Add parameter tuner info
        if parameter_tuner:
            params = await parameter_tuner.get_current_parameters()
            status["current_parameters"] = params.dict()

        # Add meta-learner info
        if meta_learner:
            status["strategy_weights_available"] = True
        else:
            status["strategy_weights_available"] = False

        return status

    @router.get("/parameters")
    async def get_current_parameters() -> Dict:
        """Get current optimized parameters."""
        if not parameter_tuner:
            return {"error": "ParameterTuner not initialized"}

        params = await parameter_tuner.get_current_parameters()
        return params.dict()

    @router.get("/parameters/history")
    async def get_parameter_history() -> List[Dict]:
        """Get parameter optimization history."""
        if not parameter_tuner:
            return []

        return parameter_tuner.get_optimization_history()

    @router.get("/strategy-weights/{regime}")
    async def get_strategy_weights(regime: str) -> Dict:
        """Get strategy weights for a regime.

        Args:
            regime: Market regime

        Returns:
            Strategy weights dict
        """
        if not meta_learner:
            return {"error": "MetaLearner not initialized"}

        weights = await meta_learner.get_strategy_weights(regime)
        return {"regime": regime, "weights": weights}

    @router.get("/strategy-performance")
    async def get_strategy_performance(regime: Optional[str] = None) -> Dict:
        """Get strategy performance summary.

        Args:
            regime: Optional regime filter

        Returns:
            Performance summary
        """
        if not meta_learner:
            return {"error": "MetaLearner not initialized"}

        return meta_learner.get_performance_summary(regime)

    @router.get("/hourly-profitability")
    async def get_hourly_profitability() -> Dict:
        """Get profitability by hour of day."""
        if not profit_maximizer:
            return {"error": "ProfitMaximizer not initialized"}

        return profit_maximizer.get_hourly_profitability()

    @router.get("/pair-profitability")
    async def get_pair_profitability() -> Dict:
        """Get profitability by trading pair."""
        if not profit_maximizer:
            return {"error": "ProfitMaximizer not initialized"}

        return profit_maximizer.get_pair_profitability()

    @router.get("/drawdown-state")
    async def get_drawdown_state() -> Dict:
        """Get current drawdown and recovery state."""
        if not profit_maximizer:
            return {"error": "ProfitMaximizer not initialized"}

        return profit_maximizer.get_drawdown_state()

    @router.get("/compound-state")
    async def get_compound_state() -> Dict:
        """Get compound growth state."""
        if not profit_maximizer:
            return {"error": "ProfitMaximizer not initialized"}

        state = profit_maximizer.get_drawdown_state()  # Contains compound info
        return {
            "equity": state["equity"],
            "peak_equity": state["peak_equity"],
            "compound_factor": state["compound_factor"],
            "drawdown_pct": state["drawdown_pct"],
        }

    @router.get("/dashboard-data")
    async def get_dashboard_data() -> Dict:
        """Get all dashboard data in one call."""
        data = {
            "timestamp": None,
            "status": {},
            "parameters": {},
            "hourly_profitability": {},
            "pair_profitability": {},
            "drawdown_state": {},
        }

        if profit_maximizer:
            data["status"] = profit_maximizer.get_status()
            data["hourly_profitability"] = profit_maximizer.get_hourly_profitability()
            data["pair_profitability"] = profit_maximizer.get_pair_profitability()
            data["drawdown_state"] = profit_maximizer.get_drawdown_state()

        if parameter_tuner:
            params = await parameter_tuner.get_current_parameters()
            data["parameters"] = params.dict()

        # Add timestamp
        from datetime import datetime, timezone
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return data

    logger.info("Profit maximization routes created")
    return router


class ProfitDashboardData:
    """Container for real-time profit dashboard data.

    This class is used by the realtime hub to broadcast updates.
    """

    def __init__(
        self,
        profit_maximizer: Optional[ProfitMaximizer] = None,
        parameter_tuner: Optional[DynamicParameterTuner] = None,
        meta_learner: Optional[MetaLearner] = None,
    ) -> None:
        """Initialize profit dashboard data provider.

        Args:
            profit_maximizer: ProfitMaximizer instance
            parameter_tuner: ParameterTuner instance
            meta_learner: MetaLearner instance
        """
        self._profit_maximizer = profit_maximizer
        self._parameter_tuner = parameter_tuner
        self._meta_learner = meta_learner

    async def get_dashboard_update(self) -> Dict:
        """Get dashboard update data.

        Returns:
            Dashboard data dict
        """
        data = {
            "type": "profit_update",
            "timestamp": None,
            "mode": "normal",
            "equity": 0.0,
            "drawdown_pct": 0.0,
            "quality_threshold": 0.6,
        }

        if self._profit_maximizer:
            status = self._profit_maximizer.get_status()
            data.update(status)

        if self._parameter_tuner:
            params = await self._parameter_tuner.get_current_parameters()
            data["parameters"] = {
                "stop_loss_pct": params.stop_loss_pct,
                "take_profit_pct": params.take_profit_pct,
                "risk_per_trade_pct": params.risk_per_trade_pct,
                "max_leverage": params.max_leverage,
                "confidence_threshold": params.confidence_threshold,
            }

        from datetime import datetime, timezone
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return data
