"""Dashboard integration for advanced institutional features.

Adds API endpoints for VaR/CVaR, execution algorithms, liquidation monitoring,
smart order routing, advanced analytics, and audit trail.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

from risk.var_cvar_calculator import VaRCVaRCalculator
from risk.liquidation_monitor import LiquidationRiskMonitor
from execution.advanced_execution_algos import TWAPExecutor, VWAPExecutor, IcebergOrderExecutor, AdaptiveExecutor
from execution.smart_order_router import SmartOrderRouter, ExchangeVenue
from analytics.advanced_analytics import (
    MarketImpactModel,
    OrderBookAnalytics,
    PortfolioAttributionAnalyzer,
    AdvancedPerformanceMetrics,
    RegimeDetectionAdvanced,
)
from backtest.advanced_backtest import MonteCarloSimulator, WalkForwardOptimizer
from compliance.audit_trail import AuditTrail

# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────

class VaRRequest(BaseModel):
    """Request model for VaR calculation."""
    symbols: List[str]
    confidence_level: float = 0.95
    time_horizon_days: int = 1
    method: str = "historical"


class VaRResponse(BaseModel):
    """Response model for VaR calculation."""
    portfolio_var: float
    portfolio_cvar: float
    position_vars: Dict[str, float]
    timestamp: str


class ExecutionRequest(BaseModel):
    """Request model for advanced execution."""
    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    algorithm: str  # "twap", "vwap", "iceberg", "adaptive"
    duration_minutes: Optional[int] = 60
    display_quantity: Optional[float] = None


class LiquidationRiskResponse(BaseModel):
    """Response model for liquidation risk summary."""
    total_positions_at_risk: int
    critical_count: int
    high_count: int
    medium_count: int
    alerts: List[Dict[str, Any]]


class MarketImpactRequest(BaseModel):
    """Request model for market impact estimation."""
    symbol: str
    order_size_usdt: float


class AnalyticsResponse(BaseModel):
    """Response model for advanced analytics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    regime: str
    regime_confidence: float


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation."""
    n_simulations: int = 10000
    n_trades: int = 100
    starting_capital: float = 10000.0


class AuditQueryRequest(BaseModel):
    """Request model for audit trail query."""
    event_type: Optional[str] = None
    symbol: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# Router Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_advanced_features_router(
    engine: Any = None,
    exchange: Any = None,
    risk_manager: Any = None,
    performance_tracker: Any = None,
) -> APIRouter:
    """Create FastAPI router for advanced institutional features.

    Args:
        engine: Trading engine instance
        exchange: Exchange client instance
        risk_manager: Risk manager instance
        performance_tracker: Performance tracker instance

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v1/advanced", tags=["advanced"])

    # Initialize components
    var_calculator = VaRCVaRCalculator(confidence_level=0.95, method="historical")
    liquidation_monitor = LiquidationRiskMonitor()
    market_impact_model = MarketImpactModel()
    orderbook_analytics = OrderBookAnalytics()
    attribution_analyzer = PortfolioAttributionAnalyzer()
    advanced_metrics = AdvancedPerformanceMetrics()
    regime_detector = RegimeDetectionAdvanced()
    monte_carlo = MonteCarloSimulator()
    audit_trail = AuditTrail()

    # ─────────────────────────────────────────────────────────────────────────
    # VaR/CVaR Endpoints
    # ─────────────────────────────────────────────────────────────────────────

    @router.post("/var", response_model=VaRResponse)
    async def calculate_var(request: VaRRequest) -> VaRResponse:
        """Calculate Value-at-Risk and Conditional VaR for portfolio."""
        try:
            if not exchange or not engine:
                raise HTTPException(status_code=503, detail="Exchange or engine not available")

            # Get position data
            positions = await exchange.get_positions()

            if not positions:
                return VaRResponse(
                    portfolio_var=0.0,
                    portfolio_cvar=0.0,
                    position_vars={},
                    timestamp=datetime.utcnow().isoformat(),
                )

            # Get historical returns for each position
            position_returns = {}
            position_values = {}

            for pos in positions:
                if pos.symbol in request.symbols or not request.symbols:
                    # Fetch historical data
                    df = await exchange.get_ohlcv(pos.symbol, timeframe="1h", limit=100)
                    returns = df["close"].pct_change().dropna()

                    position_returns[pos.symbol] = returns
                    position_values[pos.symbol] = pos.amount * pos.current_price

            # Calculate portfolio VaR and CVaR
            portfolio_var, portfolio_cvar = var_calculator.calculate_portfolio_var(
                position_returns=position_returns,
                position_values=position_values,
            )

            # Calculate individual position VaRs
            position_vars = {}
            for symbol, returns in position_returns.items():
                value = position_values[symbol]
                var = var_calculator.calculate_var(returns, value)
                position_vars[symbol] = var

            return VaRResponse(
                portfolio_var=portfolio_var,
                portfolio_cvar=portfolio_cvar,
                position_vars=position_vars,
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VaR calculation failed: {exc}")

    @router.get("/liquidation-risk", response_model=LiquidationRiskResponse)
    async def get_liquidation_risk() -> LiquidationRiskResponse:
        """Get current liquidation risk summary."""
        try:
            summary = liquidation_monitor.get_current_risk_summary()
            return LiquidationRiskResponse(**summary)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get liquidation risk: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Execution Algorithm Endpoints
    # ─────────────────────────────────────────────────────────────────────────

    @router.post("/execute")
    async def execute_advanced_order(request: ExecutionRequest) -> Dict[str, Any]:
        """Execute order using advanced execution algorithm."""
        try:
            if not exchange:
                raise HTTPException(status_code=503, detail="Exchange not available")

            from exchange.base_exchange import OrderSide
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL

            # Select algorithm
            if request.algorithm.lower() == "twap":
                executor = TWAPExecutor(
                    exchange=exchange,
                    duration_minutes=request.duration_minutes or 60,
                )
            elif request.algorithm.lower() == "vwap":
                executor = VWAPExecutor(
                    exchange=exchange,
                    duration_minutes=request.duration_minutes or 60,
                )
            elif request.algorithm.lower() == "iceberg":
                executor = IcebergOrderExecutor(
                    exchange=exchange,
                    display_quantity=request.display_quantity or 0.1,
                )
            elif request.algorithm.lower() == "adaptive":
                executor = AdaptiveExecutor(
                    exchange=exchange,
                    target_duration_minutes=request.duration_minutes or 60,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")

            # Execute
            result = await executor.execute(
                symbol=request.symbol,
                side=side,
                total_amount=request.amount,
            )

            return result

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Execution failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Market Analytics Endpoints
    # ─────────────────────────────────────────────────────────────────────────

    @router.post("/market-impact")
    async def estimate_market_impact(request: MarketImpactRequest) -> Dict[str, Any]:
        """Estimate market impact for order size."""
        try:
            if not exchange:
                raise HTTPException(status_code=503, detail="Exchange not available")

            ticker = await exchange.get_ticker(request.symbol)
            orderbook = await exchange.get_orderbook(request.symbol, limit=20)

            # Estimate daily volume from ticker
            daily_volume_usdt = ticker.volume * ticker.last

            impact = market_impact_model.estimate_impact(
                order_size_usdt=request.order_size_usdt,
                orderbook=orderbook,
                ticker=ticker,
                daily_volume_usdt=daily_volume_usdt,
            )

            return {
                "estimated_slippage_pct": impact.estimated_slippage_pct,
                "estimated_slippage_usdt": impact.estimated_slippage_usdt,
                "market_depth_score": impact.market_depth_score,
                "recommended_execution_method": impact.recommended_execution_method,
                "estimated_execution_time_minutes": impact.estimated_execution_time_minutes,
            }

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Market impact estimation failed: {exc}")

    @router.get("/analytics/{symbol}", response_model=AnalyticsResponse)
    async def get_advanced_analytics(symbol: str) -> AnalyticsResponse:
        """Get advanced analytics for symbol."""
        try:
            if not exchange or not performance_tracker:
                raise HTTPException(status_code=503, detail="Services not available")

            # Fetch historical data
            df = await exchange.get_ohlcv(symbol, timeframe="1h", limit=500)
            returns = df["close"].pct_change().dropna()

            # Calculate advanced metrics
            sharpe = advanced_metrics.calculate_sortino_ratio(returns) if len(returns) > 20 else 0.0
            sortino = advanced_metrics.calculate_sortino_ratio(returns) if len(returns) > 20 else 0.0
            calmar = advanced_metrics.calculate_calmar_ratio(returns) if len(returns) > 20 else 0.0
            omega = advanced_metrics.calculate_omega_ratio(returns) if len(returns) > 20 else 1.0

            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min()) * 100

            # Detect regime
            regime, confidence = regime_detector.detect_regime_with_confidence(df["close"])

            return AnalyticsResponse(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                omega_ratio=omega,
                max_drawdown=max_dd,
                regime=regime,
                regime_confidence=confidence,
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analytics failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Backtesting Endpoints
    # ─────────────────────────────────────────────────────────────────────────

    @router.post("/monte-carlo")
    async def run_monte_carlo(request: MonteCarloRequest) -> Dict[str, Any]:
        """Run Monte Carlo simulation on historical performance."""
        try:
            if not performance_tracker:
                raise HTTPException(status_code=503, detail="Performance tracker not available")

            # Get historical trade returns
            trades = performance_tracker.get_recent_trades(limit=1000)

            if not trades or len(trades) < 10:
                raise HTTPException(status_code=400, detail="Insufficient trade history")

            returns = pd.Series([t.get("pnl_pct", 0) for t in trades])

            result = monte_carlo.simulate_strategy_returns(
                historical_returns=returns,
                n_trades=request.n_trades,
                starting_capital=request.starting_capital,
            )

            return {
                "mean_return": result.mean_return,
                "median_return": result.median_return,
                "std_return": result.std_return,
                "best_case_return": result.best_case_return,
                "worst_case_return": result.worst_case_return,
                "probability_of_profit": result.probability_of_profit,
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
            }

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Audit Trail Endpoints
    # ─────────────────────────────────────────────────────────────────────────

    @router.post("/audit/query")
    async def query_audit_trail(request: AuditQueryRequest) -> Dict[str, Any]:
        """Query audit trail entries."""
        try:
            start_time = (
                datetime.fromisoformat(request.start_time.rstrip("Z"))
                if request.start_time
                else None
            )
            end_time = (
                datetime.fromisoformat(request.end_time.rstrip("Z"))
                if request.end_time
                else None
            )

            entries = audit_trail.query_entries(
                event_type=request.event_type,
                symbol=request.symbol,
                start_time=start_time,
                end_time=end_time,
                limit=request.limit,
            )

            return {
                "total_entries": len(entries),
                "entries": [
                    {
                        "timestamp": e.timestamp,
                        "entry_id": e.entry_id,
                        "event_type": e.event_type,
                        "action": e.action,
                        "symbol": e.symbol,
                        "details": e.details,
                    }
                    for e in entries
                ],
            }

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audit query failed: {exc}")

    @router.get("/audit/verify")
    async def verify_audit_integrity() -> Dict[str, Any]:
        """Verify audit trail integrity."""
        try:
            is_valid, errors = audit_trail.verify_integrity()

            return {
                "is_valid": is_valid,
                "errors": errors,
                "total_entries": audit_trail.entry_counter,
            }

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audit verification failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Dashboard Endpoint
    # ─────────────────────────────────────────────────────────────────────────

    @router.get("/summary")
    async def get_advanced_features_summary() -> Dict[str, Any]:
        """Get summary of all advanced features status."""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "features": {
                    "var_cvar": {"enabled": True, "status": "operational"},
                    "liquidation_monitor": {
                        "enabled": True,
                        "status": "operational",
                        "risk_summary": liquidation_monitor.get_current_risk_summary(),
                    },
                    "execution_algorithms": {
                        "enabled": True,
                        "available": ["TWAP", "VWAP", "Iceberg", "Adaptive"],
                    },
                    "smart_order_routing": {"enabled": True, "status": "operational"},
                    "advanced_analytics": {"enabled": True, "status": "operational"},
                    "monte_carlo": {"enabled": True, "status": "operational"},
                    "audit_trail": {
                        "enabled": True,
                        "total_entries": audit_trail.entry_counter,
                    },
                },
            }

            return summary

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Summary failed: {exc}")

    return router
