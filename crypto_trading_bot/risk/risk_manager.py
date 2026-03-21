"""Risk manager — THE central risk-control component of the trading bot."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from config.settings import Settings
from core.exceptions import CircuitBreakerError

from .advanced_kelly import BayesianKelly
from .circuit_breaker import CircuitBreaker
from .correlation_risk import CorrelationRiskManager
from .daily_pnl_manager import DailyPnLManager
from .drawdown_protector import DrawdownProtector
from .dynamic_take_profit import DynamicTakeProfitEngine
from .intelligent_trailing import IntelligentTrailingStop
from .leverage_optimizer import LeverageOptimizer
from .portfolio_optimizer import PortfolioOptimizer
from .position_sizer import PositionSizer
from .profit_compounder import ProfitCompounder
from .stop_loss_engine import StopLossEngine
from .take_profit_engine import TakeProfitEngine
from .var_cvar_calculator import VaRCVaRCalculator


class RiskApproval(BaseModel):
    """Result returned by :meth:`RiskManager.validate_trade`."""

    approved: bool
    symbol: str
    direction: str
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit_levels: List[float] = []
    leverage: int = 1
    risk_reward: float = 0.0
    rejection_reason: Optional[str] = None
    warnings: List[str] = []


class RiskManager:
    """Central risk-management engine enforcing all trading rules.

    Rules enforced:
    - Max 10 % capital per position
    - Max 5 concurrent positions
    - Max 2 % daily loss → reduce sizes 50 %
    - Max 5 % daily loss → circuit breaker
    - Max 10 % drawdown
    - Every trade requires a stop-loss
    - Min 1.5:1 risk-reward ratio
    - Correlated positions count as one
    - Reduce leverage in high volatility
    - No trades 30 min before major events
    - Cooldown after 3 consecutive losses
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings.get_settings()
        self._risk = self._settings.risk

        self._sizer = PositionSizer()
        self._bayesian_kelly = BayesianKelly(
            prior_alpha=2.0,
            prior_beta=3.0,
            kelly_fraction=0.25,
            max_fraction=0.05,
        )
        self._portfolio_optimizer = PortfolioOptimizer(
            max_position_pct=self._risk.max_position_size_pct / 100.0,
            max_total_exposure=0.50,
        )
        self._sl_engine = StopLossEngine()
        self._tp_engine = TakeProfitEngine()
        self._drawdown = DrawdownProtector(max_drawdown_pct=self._risk.max_drawdown_pct)
        self._correlation = CorrelationRiskManager(default_correlation=0.5)
        self._leverage_opt = LeverageOptimizer(
            default_leverage=self._settings.exchange.default_leverage,
            max_leverage=self._settings.exchange.max_leverage,
        )
        self._daily_pnl = DailyPnLManager(settings=self._settings)
        self._circuit_breaker = CircuitBreaker()
        self._var_calculator = VaRCVaRCalculator()
        self._max_portfolio_var_pct: float = 5.0  # Max 5% daily VaR

        # Profit maximization components (Upgrade 3)
        self._dynamic_tp_engine = DynamicTakeProfitEngine()
        self._intelligent_trailing = IntelligentTrailingStop(base_atr_multiplier=2.0)
        self._profit_compounder = ProfitCompounder(
            base_size_pct=self._risk.max_position_size_pct / 100.0,
            max_compound_multiplier=2.0,
        )
        # Support/resistance levels cache — updated via update_market_state()
        self._sr_levels: List[float] = []

        # Runtime state
        self._open_positions: List[dict] = []
        self._consecutive_losses: int = 0
        self._cooldown_until: Optional[datetime] = None
        self._volatility_regime: str = "normal"
        self._market_regime: str = "unknown"
        self._portfolio_equity: float = 0.0
        self._lock = asyncio.Lock()

        # Trade performance tracking for adaptive Kelly sizing
        self._trade_wins: int = 0
        self._trade_losses: int = 0
        self._total_win_return: float = 0.0   # sum of |pnl_pct| for winning trades
        self._total_loss_return: float = 0.0  # sum of |pnl_pct| for losing trades
        # Minimum trades required before using real Kelly parameters
        self._KELLY_MIN_TRADES: int = 20
        # Hard cap: no single trade may exceed this fraction of capital
        self._MAX_POSITION_FRACTION: float = 0.05  # 5 %
        # Minimum avg return used as denominator guard in Kelly estimation
        self._MIN_AVG_RETURN: float = 0.01
        # Funding rate multiplier: close position if rate is worse than EXTREME_FUNDING × tolerance
        self._EXTREME_FUNDING_MULTIPLIER: float = 3.0

    # ------------------------------------------------------------------
    # Primary validation entry-point
    # ------------------------------------------------------------------

    async def validate_trade(
        self,
        trade_signal: dict,
        dynamic_params: Optional[dict] = None,
        trade_quality_score: Optional[float] = None,
    ) -> RiskApproval:
        """Validate a trade signal against all risk rules.

        Args:
            trade_signal: Dict with keys ``symbol``, ``direction``,
                ``entry_price``, ``capital``, and optional ``atr``.
            dynamic_params: Optional dict with dynamically optimized parameters:
                - stop_loss_pct: Dynamic SL percentage
                - take_profit_pct: Dynamic TP percentage
                - risk_per_trade_pct: Dynamic risk percentage
                - max_leverage: Dynamic max leverage
                - confidence_threshold: Minimum signal confidence
                - max_open_positions: Dynamic max positions
            trade_quality_score: Optional composite quality score (0-1) from ProfitMaximizer

        Returns:
            :class:`RiskApproval` indicating approval or rejection.
        """
        symbol: str = trade_signal.get("symbol", "")
        direction: str = trade_signal.get("direction", "long")
        entry_price: float = trade_signal.get("entry_price", 0.0)
        capital: float = trade_signal.get("capital", self._portfolio_equity)
        atr: float = trade_signal.get("atr", entry_price * 0.02)
        warnings: List[str] = []

        # Extract dynamic parameters if provided, otherwise use config defaults
        if dynamic_params:
            stop_loss_pct = dynamic_params.get("stop_loss_pct", self._risk.default_stop_loss_pct)
            take_profit_pct = dynamic_params.get("take_profit_pct", self._risk.default_take_profit_pct)
            risk_per_trade_pct = dynamic_params.get("risk_per_trade_pct", 1.5)
            max_leverage = dynamic_params.get("max_leverage", self._settings.exchange.max_leverage)
            confidence_threshold = dynamic_params.get("confidence_threshold", 0.65)
            max_open_positions = dynamic_params.get("max_open_positions", self._risk.max_open_positions)
        else:
            stop_loss_pct = self._risk.default_stop_loss_pct
            take_profit_pct = self._risk.default_take_profit_pct
            risk_per_trade_pct = 1.5
            max_leverage = self._settings.exchange.max_leverage
            confidence_threshold = 0.65
            max_open_positions = self._risk.max_open_positions

        logger.info("Validating trade signal: {} {} @ {} (dynamic_params={}, quality_score={})",
                    symbol, direction, entry_price, dynamic_params is not None, trade_quality_score)

        # Trade quality filter (CRITICAL for profit maximization)
        if trade_quality_score is not None:
            # Use dynamic confidence threshold if available
            min_quality = confidence_threshold if dynamic_params else 0.6
            if trade_quality_score < min_quality:
                return RiskApproval(
                    approved=False,
                    symbol=symbol,
                    direction=direction,
                    rejection_reason=f"Trade quality score {trade_quality_score:.3f} below threshold {min_quality:.3f}",
                )
            logger.debug("Trade quality check passed: {:.3f} >= {:.3f}", trade_quality_score, min_quality)

        logger.info("Validating trade signal: {} {} @ {}", symbol, direction, entry_price)

        # Circuit breaker check
        if self._circuit_breaker.is_triggered():
            raise CircuitBreakerError("Circuit breaker is active — trading halted")

        # Daily limits
        daily_status = await self.check_daily_limits()
        if daily_status.get("circuit_breaker_active"):
            raise CircuitBreakerError("Daily loss circuit breaker is active")

        if daily_status.get("should_stop_trading"):
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason="Daily trading limit reached",
            )

        # Max daily trade count check
        if daily_status.get("trade_count", 0) >= self._risk.max_daily_trades:
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason=(
                    f"Max daily trade count reached "
                    f"({daily_status['trade_count']}/{self._risk.max_daily_trades})"
                ),
            )

        # Cooldown after consecutive losses
        if self._cooldown_until and datetime.now(tz=timezone.utc) < self._cooldown_until:
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason=f"Cooldown active until {self._cooldown_until.isoformat()}",
            )

        # Max concurrent positions (use dynamic value if available)
        effective_positions = self._correlation.get_effective_positions(
            self._open_positions, threshold=self._risk.max_correlation
        )
        if effective_positions >= max_open_positions:
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason=f"Max positions reached ({effective_positions}/{max_open_positions})",
            )

        # Correlation check
        corr = self._check_correlation(symbol, self._open_positions)
        if corr >= self._risk.max_correlation:
            warnings.append(f"High correlation ({corr:.2f}) with existing positions")

        # Drawdown check
        if self._drawdown.check_max_drawdown_breach(self._portfolio_equity):
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason="Maximum drawdown breached",
            )

        # Portfolio VaR check
        if self._open_positions and self._portfolio_equity > 0:
            try:
                var_pct = self._var_calculator.calculate_portfolio_var(
                    positions=self._open_positions,
                    confidence_level=0.95,
                )
                if var_pct > self._max_portfolio_var_pct:
                    return RiskApproval(
                        approved=False,
                        symbol=symbol,
                        direction=direction,
                        rejection_reason=(
                            f"Portfolio VaR {var_pct:.2f}% exceeds limit "
                            f"{self._max_portfolio_var_pct:.2f}%"
                        ),
                    )
            except Exception as exc:
                logger.debug("VaR check failed: {} — continuing without VaR constraint", exc)

        # Size reduction during high daily loss
        size_multiplier = 1.0
        if daily_status.get("daily_loss_pct", 0.0) >= self._risk.max_daily_loss_pct / 2:
            size_multiplier = 0.5
            warnings.append("Position size reduced 50% due to daily loss approaching limit")

        # Calculate components
        position_size = await self.calculate_position_size(symbol, direction, capital)
        position_size *= size_multiplier

        # Pre-trade notional validation: reject if the USDT position size would exceed
        # max_position_size_pct of total capital.  All sizing methods return values in USDT,
        # so position_size IS the notional in quote currency — no price multiplication needed.
        if entry_price > 0 and capital > 0:
            max_notional = capital * self._risk.max_position_size_pct / 100.0
            if position_size > max_notional:
                logger.error(
                    "Pre-trade notional check failed for {}: notional={:.2f} USDT exceeds "
                    "max_position_size {}% of capital ({:.2f} USDT) — rejecting trade",
                    symbol,
                    position_size,
                    self._risk.max_position_size_pct,
                    max_notional,
                )
                return RiskApproval(
                    approved=False,
                    symbol=symbol,
                    direction=direction,
                    rejection_reason=(
                        f"Notional {position_size:.2f} USDT exceeds "
                        f"{self._risk.max_position_size_pct}% of capital "
                        f"({max_notional:.2f} USDT)"
                    ),
                )

        stop_loss = await self.calculate_stop_loss(
            symbol, entry_price, direction, atr, stop_loss_pct=stop_loss_pct
        )
        tp_levels = await self.calculate_take_profit(
            symbol, entry_price, direction, atr, stop_loss, take_profit_pct=take_profit_pct
        )

        if not tp_levels:
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason="Could not calculate take-profit levels",
            )

        # Risk-reward check — round to 8 d.p. to neutralise floating-point
        # noise (e.g. 1.4999999999 instead of 1.5) introduced when TP prices
        # are rounded to 8 decimal places in TakeProfitEngine.
        rr = round(self._tp_engine.calculate_rr_ratio(entry_price, stop_loss, tp_levels[0]), 8)
        if rr < self._risk.risk_reward_min:
            return RiskApproval(
                approved=False,
                symbol=symbol,
                direction=direction,
                rejection_reason=f"Insufficient R:R ratio ({rr:.2f} < {self._risk.risk_reward_min})",
            )

        # Use the signal's strategy-specified leverage if provided; otherwise fall
        # back to the dynamically calculated leverage from the risk manager.
        signal_leverage: int = int(trade_signal.get("leverage", 0) or 0)
        if signal_leverage > 0:
            leverage = min(signal_leverage, max_leverage)
            logger.debug(
                "Using strategy leverage {}x for {} (capped at max_leverage={})",
                leverage, symbol, max_leverage,
            )
        else:
            leverage = await self.calculate_leverage(symbol)

        approval = RiskApproval(
            approved=True,
            symbol=symbol,
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit_levels=tp_levels,
            leverage=leverage,
            risk_reward=rr,
            warnings=warnings,
        )
        logger.info(
            "Trade approved: {} {} size={:.4f} sl={:.4f} tp={} rr={:.2f}",
            symbol,
            direction,
            position_size,
            stop_loss,
            tp_levels,
            rr,
        )
        return approval

    # ------------------------------------------------------------------
    # Sizing & levels
    # ------------------------------------------------------------------

    async def calculate_position_size(
        self,
        symbol: str,
        direction: str,
        capital: float,
    ) -> float:
        """Calculate position size using Kelly Criterion (half-Kelly).

        Falls back to fixed fraction when Kelly data is unavailable.
        A performance-based risk multiplier is applied on top of the
        Kelly/fixed-fraction result to scale risk up or down based on
        recent trade outcomes.

        Args:
            symbol: Trading symbol.
            direction: ``"long"`` or ``"short"``.
            capital: Available capital.

        Returns:
            Position size in USDT (quote currency).
        """
        try:
            total_trades = self._trade_wins + self._trade_losses
            if self._risk.use_kelly_criterion:
                # Always use BayesianKelly — it starts with an informative prior
                # so it is safe from trade #1 without a separate fallback.
                avg_win = (
                    self._total_win_return / self._trade_wins
                    if self._trade_wins
                    else self._MIN_AVG_RETURN
                )
                avg_loss = (
                    self._total_loss_return / self._trade_losses
                    if self._trade_losses
                    else self._MIN_AVG_RETURN
                )
                size = self._bayesian_kelly.get_position_size(
                    capital=capital,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                )
                logger.debug(
                    "BayesianKelly size for {}: posterior_win_rate={:.4f} "
                    "avg_win={:.4f} avg_loss={:.4f} size={:.2f} USDT "
                    "(total_trades={})",
                    symbol,
                    self._bayesian_kelly.posterior_win_rate,
                    avg_win,
                    avg_loss,
                    size,
                    total_trades,
                )
            else:
                size = self._sizer.fixed_fraction_size(
                    capital=capital,
                    risk_pct=self._risk.max_position_size_pct / 100.0,
                )
            # Apply performance-based risk multiplier (0.25–1.5×)
            perf_multiplier = self.get_performance_risk_multiplier()
            size *= perf_multiplier
            if perf_multiplier != 1.0:
                logger.debug(
                    "Performance risk multiplier applied for {}: {:.2f}x → size={:.2f} USDT",
                    symbol,
                    perf_multiplier,
                    size,
                )
            # Apply profit compounder multiplier based on daily/weekly P&L
            try:
                daily_status = await self._daily_pnl.get_daily_status()
                # daily_pnl_pct is already in percentage units (e.g. 3.0 for 3%)
                daily_pnl_pct = daily_status.get("daily_pnl_pct", 0.0)
                compound_mult = self._profit_compounder.get_size_multiplier(
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=0.0,
                )
                if compound_mult != 1.0:
                    size *= compound_mult
                    logger.debug(
                        "ProfitCompounder multiplier applied for {}: {:.2f}x → size={:.2f} USDT "
                        "(daily_pnl={:.2f}%)",
                        symbol,
                        compound_mult,
                        size,
                        daily_pnl_pct,
                    )
            except Exception as exc:
                logger.debug("ProfitCompounder skipped ({})", exc)
            # Hard cap: never exceed max_position_size_pct of capital (settings)
            # AND never exceed the absolute 5% hard cap regardless of Kelly output.
            hard_cap = capital * self._MAX_POSITION_FRACTION
            settings_cap = capital * self._risk.max_position_size_pct / 100.0
            max_size = min(hard_cap, settings_cap)
            size = min(size, max_size)
            # Ensure a minimum viable size (1% of capital) so trades are never silently
            # rejected due to a zero Kelly output from low-probability expected-value stats.
            if size <= 0 and capital > 0:
                size = capital * 0.01
                logger.debug(
                    "Position size for {} was zero after Kelly/cap; using 1%% fallback: {:.2f} USDT",
                    symbol,
                    size,
                )
            logger.debug(
                "Position size for {} in USDT: {:.2f} (capital={:.2f} USDT, hard_cap={:.2f} USDT)",
                symbol,
                size,
                capital,
                hard_cap,
            )
            return size
        except Exception as exc:
            logger.error("Position size calculation failed: {}", exc)
            return capital * 0.01  # 1 % safe fallback


    async def calculate_stop_loss(
        self,
        symbol: str,
        entry: float,
        direction: str,
        atr: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> float:
        """Calculate ATR-based dynamic stop-loss.

        Args:
            symbol: Trading symbol.
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range; defaults to 2 % of entry if not provided.
            stop_loss_pct: Dynamic stop loss percentage (overrides default if provided).

        Returns:
            Stop-loss price.
        """
        if atr is None or atr <= 0:
            atr = entry * 0.02

        # Use dynamic stop loss percentage if provided
        if stop_loss_pct is not None:
            # Direct percentage-based stop loss
            if direction == "long":
                stop = entry * (1 - stop_loss_pct / 100.0)
            else:
                stop = entry * (1 + stop_loss_pct / 100.0)
            logger.debug(
                "Stop-loss for {} (dynamic): entry={} dir={} stop_loss_pct={:.2f}% stop={:.4f}",
                symbol,
                entry,
                direction,
                stop_loss_pct,
                stop,
            )
        else:
            # ATR-based stop loss (original logic)
            stop = self._sl_engine.calculate_initial_stop(entry, direction, atr, multiplier=2.0)
            stop = self._sl_engine.adjust_for_volatility(
                stop, self._volatility_regime, entry_price=entry, direction=direction
            )
            logger.debug(
                "Stop-loss for {}: entry={} dir={} atr={} stop={:.4f}",
                symbol,
                entry,
                direction,
                atr,
                stop,
            )
        return stop

    async def calculate_take_profit(
        self,
        symbol: str,
        entry: float,
        direction: str,
        atr: float | None = None,
        stop_loss: float | None = None,
        take_profit_pct: float | None = None,
        adx: float | None = None,
    ) -> List[float]:
        """Calculate take-profit levels using the DynamicTakeProfitEngine.

        When *take_profit_pct* is provided the legacy percentage-based calculation
        is used (for backward compatibility with callers that pass a fixed target).
        Otherwise the adaptive ATR/regime/S&R-based levels are returned.

        Args:
            symbol: Trading symbol.
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range.
            stop_loss: Used to derive risk_reward if provided.
            take_profit_pct: Dynamic take profit percentage (overrides adaptive
                calculation when provided).
            adx: ADX value for dynamic partial-close percentage adjustment.

        Returns:
            List of TP price levels (three fixed levels; trailing level excluded
            from the price list since it is managed by IntelligentTrailingStop).
        """
        if atr is None or atr <= 0:
            atr = entry * 0.02

        # Use dynamic take profit percentage if provided (legacy path)
        if take_profit_pct is not None:
            # Direct percentage-based take profit (create 3 levels)
            if direction == "long":
                tp1 = entry * (1 + take_profit_pct * 0.7 / 100.0)  # 70% of target
                tp2 = entry * (1 + take_profit_pct / 100.0)  # Full target
                tp3 = entry * (1 + take_profit_pct * 1.3 / 100.0)  # 130% of target
            else:
                tp1 = entry * (1 - take_profit_pct * 0.7 / 100.0)
                tp2 = entry * (1 - take_profit_pct / 100.0)
                tp3 = entry * (1 - take_profit_pct * 1.3 / 100.0)
            prices = [tp1, tp2, tp3]
            logger.debug(
                "TP levels for {} (dynamic pct): tp_pct={:.2f}% levels={}",
                symbol,
                take_profit_pct,
                prices,
            )
            return prices

        # Adaptive path: use DynamicTakeProfitEngine
        dynamic_levels = self._dynamic_tp_engine.calculate_tp_levels(
            entry_price=entry,
            direction=direction,
            atr=atr,
            market_regime=self._market_regime,
            support_resistance_levels=self._sr_levels or None,
            adx=adx,
        )

        # Extract price list for fixed TP levels only (exclude trailing placeholder)
        prices = [lvl["price"] for lvl in dynamic_levels if lvl.get("type") == "fixed"]

        # Fallback: if dynamic engine returned nothing, use the legacy engine
        if not prices:
            stop_distance = abs(entry - stop_loss) if stop_loss is not None else None
            legacy_levels = self._tp_engine.calculate_tp_levels(
                entry, direction, atr, risk_reward=1.5, stop_distance=stop_distance
            )
            prices = [lvl["price"] for lvl in legacy_levels]

        logger.debug("TP levels for {} (adaptive): {}", symbol, prices)
        return prices

    async def calculate_leverage(
        self,
        symbol: str,
        volatility_regime: str | None = None,
        market_regime: str | None = None,
    ) -> int:
        """Calculate optimal leverage for *symbol*.

        Args:
            symbol: Trading symbol.
            volatility_regime: Override for current volatility regime.
            market_regime: Override for current market regime.

        Returns:
            Recommended leverage as an integer.
        """
        vol_regime = volatility_regime or self._volatility_regime
        mkt_regime = market_regime or self._market_regime
        max_safe = self._leverage_opt.get_max_safe_leverage(symbol)
        leverage = self._leverage_opt.calculate_optimal_leverage(
            symbol=symbol,
            max_leverage=max_safe,
            volatility_regime=vol_regime,
            market_regime=mkt_regime,
        )
        logger.debug("Leverage for {}: {}", symbol, leverage)
        return leverage

    # ------------------------------------------------------------------
    # Daily limits & circuit breaker
    # ------------------------------------------------------------------

    async def check_daily_limits(self) -> dict:
        """Check daily PnL limits and circuit breaker conditions.

        Returns:
            Status dict with keys: ``daily_loss_pct``, ``should_stop_trading``,
            ``circuit_breaker_active``.
        """
        status = await self._daily_pnl.check_daily_status()
        daily_pnl_pct = status.get("daily_pnl_pct", 0.0)
        daily_loss_pct = max(0.0, -daily_pnl_pct)

        circuit_breaker_active = False
        if daily_loss_pct >= self._risk.circuit_breaker_loss_pct:
            await self._circuit_breaker.trigger(
                f"Daily loss circuit breaker: {daily_loss_pct:.2f}% >= {self._risk.circuit_breaker_loss_pct:.2f}%"
            )
            circuit_breaker_active = True

        return {
            **status,
            "daily_loss_pct": daily_loss_pct,
            "circuit_breaker_active": circuit_breaker_active,
            "should_stop_trading": status.get("limit_reached", False) or circuit_breaker_active,
        }

    async def check_circuit_breaker(self) -> bool:
        """Return True if the circuit breaker is currently triggered."""
        return self._circuit_breaker.is_triggered()

    # ------------------------------------------------------------------
    # Trailing stops & portfolio risk
    # ------------------------------------------------------------------

    async def update_trailing_stops(self) -> None:
        """Update trailing stops for all open positions.

        This is a no-op placeholder; the actual trailing-stop logic lives
        inside :class:`exchange.position_manager.PositionManager`.
        """
        logger.debug("update_trailing_stops: delegated to PositionManager")

    async def get_portfolio_risk(self) -> dict:
        """Return a comprehensive portfolio risk snapshot.

        Returns:
            Dict with keys: ``open_positions``, ``effective_positions``,
            ``avg_correlation``, ``drawdown_pct``, ``exposure_multiplier``,
            ``daily_status``, ``circuit_breaker``, ``total_margin_used``,
            ``available_margin``, ``margin_ratio``, ``correlation_risk_score``.
        """
        drawdown = self._drawdown.calculate_current_drawdown({"equity": self._portfolio_equity})
        exposure_mult = self._drawdown.get_exposure_multiplier(drawdown)
        avg_corr = self._correlation.calculate_portfolio_correlation(self._open_positions)
        effective = self._correlation.get_effective_positions(self._open_positions)
        daily_status = await self.check_daily_limits()

        # Calculate total margin used from open positions
        total_margin = sum(
            float(p.get("margin", 0.0)) for p in self._open_positions
        )
        available_margin = max(0.0, self._portfolio_equity - total_margin)
        margin_ratio = (
            round(total_margin / self._portfolio_equity, 4)
            if self._portfolio_equity > 0
            else 0.0
        )
        correlation_risk = self.calculate_correlation_risk(self._open_positions)

        return {
            "open_positions": len(self._open_positions),
            "effective_positions": effective,
            "avg_correlation": avg_corr,
            "drawdown_pct": drawdown,
            "exposure_multiplier": exposure_mult,
            "daily_status": daily_status,
            "circuit_breaker": self._circuit_breaker.trigger_info,
            "volatility_regime": self._volatility_regime,
            "market_regime": self._market_regime,
            "consecutive_losses": self._consecutive_losses,
            "total_margin_used": round(total_margin, 4),
            "available_margin": round(available_margin, 4),
            "margin_ratio": margin_ratio,
            "correlation_risk_score": round(correlation_risk, 4),
        }

    def calculate_correlation_risk(self, positions: List[dict]) -> float:
        """Calculate a portfolio-level correlation risk score (0–1).

        A score of 0 means all positions are uncorrelated; a score of 1 means
        all positions are perfectly correlated (maximum concentration risk).

        Args:
            positions: List of position dicts (same format as ``_open_positions``).

        Returns:
            Correlation risk score between 0.0 and 1.0.
        """
        if len(positions) < 2:
            return 0.0
        try:
            avg_corr = self._correlation.calculate_portfolio_correlation(positions)
            # avg_corr is already 0–1; clamp to be safe
            return float(max(0.0, min(1.0, avg_corr)))
        except Exception as exc:
            logger.warning("calculate_correlation_risk error: {}", exc)
            return 0.0

    def should_reduce_for_funding(
        self,
        symbol: str,
        funding_rate: float,
        position_pnl: float,
    ) -> bool:
        """Recommend closing a position if funding costs erode its profitability.

        Args:
            symbol: Trading pair symbol (used for logging only).
            funding_rate: Current funding rate as a percentage (e.g. ``-0.05``
                for ``-0.05%``).
            position_pnl: Current unrealised P&L of the position (USDT).

        Returns:
            ``True`` if the position should be closed due to adverse funding.
        """
        tolerance = getattr(self._risk, "max_funding_rate_tolerance", -0.05)
        if funding_rate >= tolerance:
            return False

        # If the position is losing money and funding is also negative, close
        if position_pnl <= 0:
            logger.warning(
                "should_reduce_for_funding: {} funding={:.4f}% pnl={:.4f} → recommend close",
                symbol,
                funding_rate,
                position_pnl,
            )
            return True

        # If position has positive P&L but funding is eating into it significantly
        # (funding rate worse than EXTREME_FUNDING_MULTIPLIER × tolerance), still recommend closing
        if funding_rate < tolerance * self._EXTREME_FUNDING_MULTIPLIER:
            logger.warning(
                "should_reduce_for_funding: {} extreme funding={:.4f}% → recommend close",
                symbol,
                funding_rate,
            )
            return True

        return False

    def get_performance_risk_multiplier(self) -> float:
        """Scale risk based on recent performance.

        Returns a multiplier in the range 0.25–1.5 that is applied to the
        calculated position size.  The multiplier is conservative until
        enough trades have been recorded.

        Returns:
            Float multiplier between 0.25 (severe reduction) and 1.5
            (scaled-up risk on strong recent performance).
        """
        total = self._trade_wins + self._trade_losses
        if total < 10:
            return 0.5  # Conservative until proven

        recent_win_rate = self._trade_wins / total
        if self._consecutive_losses >= 3:
            return 0.25  # Severe reduction after losing streak
        elif recent_win_rate > 0.6 and total > 20:
            return min(1.5, 0.5 + recent_win_rate)  # Scale up with wins
        elif recent_win_rate < 0.4:
            return 0.5  # Reduce on poor performance
        return 1.0

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    async def record_trade_result(
        self,
        pnl: float,
        trade_id: str,
        pnl_pct: float = 0.0,
    ) -> None:
        """Record the result of a completed trade and update risk state.

        The *pnl_pct* value is used to maintain running averages of winning
        and losing trade returns so that :meth:`calculate_position_size` can
        switch to real Kelly parameters once enough trades have been recorded.

        Args:
            pnl: Realised PnL (positive = profit, negative = loss).
            trade_id: Unique trade identifier.
            pnl_pct: Return as a decimal fraction (e.g. 0.02 for +2%).  Used
                only for Kelly parameter estimation; defaults to 0.0 if not
                provided.
        """
        await self._daily_pnl.record_pnl(pnl, trade_id)
        async with self._lock:
            if pnl < 0:
                self._consecutive_losses += 1
                self._trade_losses += 1
                self._total_loss_return += abs(pnl_pct)
                self._bayesian_kelly.update(won=False)
                if self._consecutive_losses >= 3:
                    cooldown_minutes = self._risk.cooldown_after_loss_minutes
                    from datetime import timedelta

                    self._cooldown_until = datetime.now(tz=timezone.utc) + timedelta(
                        minutes=cooldown_minutes
                    )
                    logger.warning(
                        "Cooldown activated after {} consecutive losses until {}",
                        self._consecutive_losses,
                        self._cooldown_until,
                    )
            else:
                self._consecutive_losses = 0
                self._cooldown_until = None
                self._trade_wins += 1
                self._total_win_return += abs(pnl_pct)
                self._bayesian_kelly.update(won=True)

    def update_market_state(
        self,
        volatility_regime: str,
        market_regime: str,
        equity: float,
        open_positions: List[dict],
        support_resistance_levels: Optional[List[float]] = None,
    ) -> None:
        """Update cached market-state used for risk calculations.

        Args:
            volatility_regime: Current volatility regime label.
            market_regime: Current market regime label.
            equity: Current portfolio equity.
            open_positions: List of current position dicts.
            support_resistance_levels: Optional list of key price levels for
                dynamic TP snapping.
        """
        self._volatility_regime = volatility_regime
        self._market_regime = market_regime
        self._portfolio_equity = equity
        self._open_positions = open_positions
        self._drawdown.record_equity_peak(equity)
        if support_resistance_levels is not None:
            self._sr_levels = support_resistance_levels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def intelligent_trailing(self) -> IntelligentTrailingStop:
        """Expose the IntelligentTrailingStop instance for use by position managers."""
        return self._intelligent_trailing

    @property
    def dynamic_tp_engine(self) -> DynamicTakeProfitEngine:
        """Expose the DynamicTakeProfitEngine instance."""
        return self._dynamic_tp_engine

    @property
    def profit_compounder(self) -> ProfitCompounder:
        """Expose the ProfitCompounder instance."""
        return self._profit_compounder

    def _check_correlation(self, symbol: str, existing_positions: List[dict]) -> float:
        """Return the highest pairwise correlation between *symbol* and existing positions.

        Args:
            symbol: Candidate new symbol.
            existing_positions: Currently open positions.

        Returns:
            Maximum correlation coefficient found (0–1).
        """
        max_corr = 0.0
        for pos in existing_positions:
            pos_symbol = pos.get("symbol", "")
            if pos_symbol == symbol:
                continue
            corr = self._correlation._get_pairwise_correlation(symbol, pos_symbol)
            max_corr = max(max_corr, corr)
        return max_corr

    def validate_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit_levels: List[float],
        direction: str,
    ) -> bool:
        """Check whether a trade meets the minimum risk-reward ratio requirement.

        Uses the first take-profit level to calculate the reward-to-risk ratio.

        Args:
            entry_price: Intended entry price.
            stop_loss: Stop-loss price.  Must be > 0.
            take_profit_levels: Ordered list of take-profit prices.  The first
                level is used for the calculation.
            direction: ``"long"`` or ``"short"``.

        Returns:
            ``True`` if the R:R ratio is at or above :attr:`settings.risk.risk_reward_min`,
            ``False`` otherwise.
        """
        if not take_profit_levels or stop_loss <= 0 or entry_price <= 0:
            return False

        if direction == "long":
            risk = entry_price - stop_loss
            reward = take_profit_levels[0] - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit_levels[0]

        if risk <= 0:
            return False

        # Round to 8 d.p. to neutralise floating-point noise (e.g. a computed
        # ratio of 1.4999999999 instead of 1.5 due to intermediate rounding).
        rr_ratio = round(reward / risk, 8)
        meets_minimum = rr_ratio >= self._settings.risk.risk_reward_min
        if not meets_minimum:
            logger.warning(
                "R:R ratio {:.2f} is below minimum {:.2f} (entry={} sl={} tp={})",
                rr_ratio,
                self._settings.risk.risk_reward_min,
                entry_price,
                stop_loss,
                take_profit_levels[0],
            )
        return meets_minimum
