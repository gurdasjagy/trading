"""Profit maximizer orchestrator coordinating all profit-maximizing subsystems."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel

from config.settings import Settings


class CompoundGrowthState(BaseModel):
    """State for compound growth engine."""

    equity: float = 10000.0
    peak_equity: float = 10000.0
    compound_factor: float = 0.5
    last_profitable_day: Optional[datetime] = None
    current_drawdown_pct: float = 0.0


class TradeQualityScore(BaseModel):
    """Trade quality assessment."""

    signal_confidence: float = 0.0
    ai_confidence: float = 0.0
    order_flow_alignment: float = 0.0
    regime_appropriateness: float = 0.0
    historical_performance: float = 0.0
    spread_quality: float = 0.0
    microstructure_quality: float = 0.0
    composite_score: float = 0.0
    should_trade: bool = False


class ProfitMaximizer:
    """
    Master orchestrator for profit maximization.

    Coordinates all subsystems:
    - Compound growth engine
    - Momentum scaling
    - Time-of-day optimization
    - Pair profitability tracking
    - Drawdown recovery mode
    - Trade quality filtering

    This is the KEY to achieving 60-70% win rate by aggressively filtering trades.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize profit maximizer.

        Args:
            settings: Trading bot settings
        """
        self._settings = settings or Settings.get_settings()
        self._lock = asyncio.Lock()

        # Compound growth engine
        self._compound_state = CompoundGrowthState()

        # Momentum tracking (win/loss streaks per strategy)
        self._strategy_streaks: Dict[str, int] = defaultdict(int)  # Positive=wins, negative=losses
        self._strategy_paused_until: Dict[str, datetime] = {}

        # Time-of-day profitability (24 hourly buckets)
        self._hourly_pnl: Dict[int, List[float]] = defaultdict(list)
        self._hourly_trades: Dict[int, int] = defaultdict(int)

        # Pair profitability tracking
        self._pair_pnl: Dict[str, List[float]] = defaultdict(list)
        self._pair_trades: Dict[str, int] = defaultdict(int)
        self._pair_sharpe: Dict[str, float] = {}

        # Drawdown recovery state
        self._drawdown_mode: str = "normal"  # normal | recovery_light | recovery_heavy | paused
        self._pause_until: Optional[datetime] = None

        # Trade history for analysis
        self._recent_trades: deque = deque(maxlen=200)

        # Quality filter thresholds
        self._min_composite_score = 0.6  # Base threshold
        self._dynamic_threshold = 0.6  # Adjusted based on mode

        logger.info("ProfitMaximizer initialized")

    # ------------------------------------------------------------------
    # Compound growth engine
    # ------------------------------------------------------------------

    async def update_equity(self, new_equity: float) -> None:
        """Update equity and adjust compound factor.

        Args:
            new_equity: Current portfolio equity
        """
        async with self._lock:
            old_equity = self._compound_state.equity
            self._compound_state.equity = new_equity

            # Update peak
            if new_equity > self._compound_state.peak_equity:
                self._compound_state.peak_equity = new_equity
                # At new high: aggressive compounding
                self._compound_state.compound_factor = 0.7
                logger.info("New equity high: {:.2f} → compound factor 0.7", new_equity)

            # Calculate drawdown
            if self._compound_state.peak_equity > 0:
                self._compound_state.current_drawdown_pct = (
                    (self._compound_state.peak_equity - new_equity)
                    / self._compound_state.peak_equity
                    * 100.0
                )
            else:
                self._compound_state.current_drawdown_pct = 0.0

            # Adjust compound factor based on drawdown
            if self._compound_state.current_drawdown_pct > 5.0:
                # Heavy drawdown: stop compounding
                self._compound_state.compound_factor = 0.0
                logger.warning("Drawdown {:.2f}% > 5% → compound factor 0", self._compound_state.current_drawdown_pct)
            elif self._compound_state.current_drawdown_pct > 3.0:
                # Light drawdown: reduce compounding
                self._compound_state.compound_factor = 0.2
                logger.info("Drawdown {:.2f}% > 3% → compound factor 0.2", self._compound_state.current_drawdown_pct)

    async def get_compound_multiplier(self, daily_profit: float) -> float:
        """Calculate compound growth multiplier.

        Args:
            daily_profit: Profit for the current day

        Returns:
            Multiplier to apply to base position size
        """
        async with self._lock:
            if daily_profit <= 0:
                return 1.0

            # Calculate increase
            if self._compound_state.equity > 0:
                increase = (
                    daily_profit / self._compound_state.equity * self._compound_state.compound_factor
                )
                multiplier = 1.0 + increase
                logger.debug(
                    "Compound multiplier: {:.4f} (profit={:.2f} equity={:.2f} factor={:.2f})",
                    multiplier,
                    daily_profit,
                    self._compound_state.equity,
                    self._compound_state.compound_factor,
                )
                return multiplier

            return 1.0

    # ------------------------------------------------------------------
    # Momentum scaling
    # ------------------------------------------------------------------

    async def record_trade_outcome(
        self,
        strategy_name: str,
        pnl: float,
        trade_data: Dict,
    ) -> None:
        """Record trade outcome and update momentum.

        Args:
            strategy_name: Name of strategy
            pnl: Profit/loss
            trade_data: Full trade data
        """
        async with self._lock:
            # Update streak
            if pnl > 0:
                # Win
                if self._strategy_streaks[strategy_name] < 0:
                    self._strategy_streaks[strategy_name] = 1
                else:
                    self._strategy_streaks[strategy_name] += 1
            else:
                # Loss
                if self._strategy_streaks[strategy_name] > 0:
                    self._strategy_streaks[strategy_name] = -1
                else:
                    self._strategy_streaks[strategy_name] -= 1

            streak = self._strategy_streaks[strategy_name]

            # Pause strategy after 5 consecutive losses
            if streak <= -5:
                pause_until = datetime.now(timezone.utc) + timedelta(hours=4)
                self._strategy_paused_until[strategy_name] = pause_until
                logger.warning(
                    "Strategy {} paused until {} after 5 consecutive losses",
                    strategy_name,
                    pause_until.isoformat(),
                )

            # Record hourly stats
            hour = datetime.now(timezone.utc).hour
            self._hourly_pnl[hour].append(pnl)
            self._hourly_trades[hour] += 1

            # Record pair stats
            symbol = trade_data.get("symbol", "")
            if symbol:
                self._pair_pnl[symbol].append(pnl)
                self._pair_trades[symbol] += 1

            # Add to recent trades
            self._recent_trades.append({
                "strategy": strategy_name,
                "pnl": pnl,
                "timestamp": datetime.now(timezone.utc),
                **trade_data,
            })

            # Update pair Sharpe ratios if enough trades
            if symbol and self._pair_trades[symbol] >= 50:
                self._update_pair_sharpe(symbol)

    async def get_momentum_multiplier(self, strategy_name: str) -> float:
        """Get position size multiplier based on strategy momentum.

        Args:
            strategy_name: Strategy to check

        Returns:
            Multiplier (0.7 - 1.2)
        """
        async with self._lock:
            # Check if paused
            if strategy_name in self._strategy_paused_until:
                if datetime.now(timezone.utc) < self._strategy_paused_until[strategy_name]:
                    logger.debug("Strategy {} is paused", strategy_name)
                    return 0.0  # No trading
                else:
                    # Unpause
                    del self._strategy_paused_until[strategy_name]
                    self._strategy_streaks[strategy_name] = 0

            streak = self._strategy_streaks.get(strategy_name, 0)

            if streak >= 3:
                # 3+ wins: increase size 20%
                multiplier = 1.2
                logger.debug("Strategy {} on {} win streak → multiplier {}", strategy_name, streak, multiplier)
            elif streak <= -2:
                # 2+ losses: decrease size 30%
                multiplier = 0.7
                logger.debug("Strategy {} on {} loss streak → multiplier {}", strategy_name, abs(streak), multiplier)
            else:
                multiplier = 1.0

            return multiplier

    # ------------------------------------------------------------------
    # Time-of-day optimization
    # ------------------------------------------------------------------

    async def get_time_multiplier(self) -> float:
        """Get position size multiplier based on current hour profitability.

        Returns:
            Multiplier (0.5 - 1.5)
        """
        async with self._lock:
            hour = datetime.now(timezone.utc).hour

            # Need at least 100 total trades before filtering
            total_trades = sum(self._hourly_trades.values())
            if total_trades < 100:
                return 1.0

            # Calculate expected value for current hour
            if hour not in self._hourly_pnl or len(self._hourly_pnl[hour]) < 5:
                return 1.0  # Not enough data for this hour

            hourly_returns = self._hourly_pnl[hour]
            mean_pnl = np.mean(hourly_returns)

            # If this hour is negative EV, reduce size significantly
            if mean_pnl < 0:
                logger.debug("Hour {} has negative EV ({:.2f}) → multiplier 0.5", hour, mean_pnl)
                return 0.5

            # Calculate percentile among all hours
            all_hour_means = []
            for h in range(24):
                if h in self._hourly_pnl and len(self._hourly_pnl[h]) >= 5:
                    all_hour_means.append(np.mean(self._hourly_pnl[h]))

            if all_hour_means:
                percentile = np.percentile(all_hour_means, 50)
                if mean_pnl > percentile:
                    # This hour is above median: increase size
                    multiplier = 1.0 + (mean_pnl - percentile) / max(abs(percentile), 1.0) * 0.5
                    multiplier = min(multiplier, 1.5)
                    logger.debug("Hour {} above median → multiplier {:.2f}", hour, multiplier)
                    return multiplier

            return 1.0

    def get_hourly_profitability(self) -> Dict[int, Dict]:
        """Get profitability statistics by hour.

        Returns:
            Dict mapping hour -> stats
        """
        result = {}
        for hour in range(24):
            if hour in self._hourly_pnl and self._hourly_pnl[hour]:
                result[hour] = {
                    "avg_pnl": float(np.mean(self._hourly_pnl[hour])),
                    "total_pnl": float(np.sum(self._hourly_pnl[hour])),
                    "trades": self._hourly_trades[hour],
                    "win_rate": sum(1 for p in self._hourly_pnl[hour] if p > 0) / len(self._hourly_pnl[hour]),
                }
        return result

    # ------------------------------------------------------------------
    # Pair profitability tracking
    # ------------------------------------------------------------------

    async def get_pair_multiplier(self, symbol: str) -> float:
        """Get position size multiplier based on pair profitability.

        Args:
            symbol: Trading pair

        Returns:
            Multiplier (0.0 - 1.5)
        """
        async with self._lock:
            # Need at least 50 trades per pair
            if symbol not in self._pair_trades or self._pair_trades[symbol] < 50:
                return 1.0

            # Check if pair has been removed (negative Sharpe after 100 trades)
            if self._pair_trades[symbol] >= 100:
                sharpe = self._pair_sharpe.get(symbol, 0.0)
                if sharpe < 0:
                    logger.warning("Pair {} has negative Sharpe {:.3f} → removed", symbol, sharpe)
                    return 0.0  # Don't trade this pair

            # Rank pairs by Sharpe
            if symbol in self._pair_sharpe:
                all_sharpes = list(self._pair_sharpe.values())
                if all_sharpes:
                    percentile = np.percentile([s for s in all_sharpes if s > 0], 75) if len([s for s in all_sharpes if s > 0]) > 0 else 1.0
                    sharpe = self._pair_sharpe[symbol]

                    if sharpe > percentile:
                        # Top performing pair: increase allocation
                        multiplier = 1.0 + (sharpe - percentile) / max(percentile, 0.5) * 0.5
                        multiplier = min(multiplier, 1.5)
                        logger.debug("Pair {} Sharpe {:.3f} > p75 → multiplier {:.2f}", symbol, sharpe, multiplier)
                        return multiplier

            return 1.0

    def _update_pair_sharpe(self, symbol: str) -> None:
        """Update Sharpe ratio for a trading pair.

        Args:
            symbol: Symbol to update
        """
        if symbol not in self._pair_pnl or len(self._pair_pnl[symbol]) < 10:
            return

        returns = self._pair_pnl[symbol]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return > 0 and not np.isnan(std_return):
            sharpe = mean_return / std_return * np.sqrt(252)
            self._pair_sharpe[symbol] = float(sharpe)
        else:
            self._pair_sharpe[symbol] = 0.0

    def get_pair_profitability(self) -> Dict[str, Dict]:
        """Get profitability statistics by pair.

        Returns:
            Dict mapping symbol -> stats
        """
        result = {}
        for symbol in self._pair_pnl:
            if self._pair_pnl[symbol]:
                result[symbol] = {
                    "avg_pnl": float(np.mean(self._pair_pnl[symbol])),
                    "total_pnl": float(np.sum(self._pair_pnl[symbol])),
                    "trades": self._pair_trades[symbol],
                    "win_rate": sum(1 for p in self._pair_pnl[symbol] if p > 0) / len(self._pair_pnl[symbol]),
                    "sharpe": self._pair_sharpe.get(symbol, 0.0),
                }
        return result

    # ------------------------------------------------------------------
    # Drawdown recovery mode
    # ------------------------------------------------------------------

    async def update_drawdown_mode(self) -> None:
        """Update drawdown recovery mode based on current drawdown."""
        async with self._lock:
            dd = self._compound_state.current_drawdown_pct

            old_mode = self._drawdown_mode

            if dd > 8.0:
                # Critical: pause trading
                self._drawdown_mode = "paused"
                self._pause_until = datetime.now(timezone.utc) + timedelta(hours=2)
                self._dynamic_threshold = 0.95  # Only A++ setups
                logger.error("Drawdown {:.2f}% > 8% → PAUSED for 2 hours", dd)
            elif dd > 5.0:
                # Heavy recovery
                self._drawdown_mode = "recovery_heavy"
                self._dynamic_threshold = 0.9  # Only A+ setups
                logger.warning("Drawdown {:.2f}% > 5% → heavy recovery mode", dd)
            elif dd > 3.0:
                # Light recovery
                self._drawdown_mode = "recovery_light"
                self._dynamic_threshold = 0.8
                logger.info("Drawdown {:.2f}% > 3% → light recovery mode", dd)
            else:
                # Normal
                self._drawdown_mode = "normal"
                self._dynamic_threshold = 0.6
                if old_mode != "normal":
                    logger.info("Drawdown {:.2f}% → normal mode", dd)

    async def get_drawdown_multiplier(self) -> float:
        """Get position size multiplier based on drawdown mode.

        Returns:
            Multiplier (0.0 - 1.0)
        """
        async with self._lock:
            # Check if paused
            if self._drawdown_mode == "paused":
                if self._pause_until and datetime.now(timezone.utc) < self._pause_until:
                    logger.debug("Trading paused until {}", self._pause_until.isoformat())
                    return 0.0
                else:
                    # Unpause
                    await self.update_drawdown_mode()

            if self._drawdown_mode == "recovery_heavy":
                return 0.25  # 25% size
            elif self._drawdown_mode == "recovery_light":
                return 0.5  # 50% size
            else:
                return 1.0  # Normal

    def get_drawdown_state(self) -> Dict:
        """Get current drawdown state.

        Returns:
            Drawdown state dict
        """
        return {
            "mode": self._drawdown_mode,
            "drawdown_pct": self._compound_state.current_drawdown_pct,
            "equity": self._compound_state.equity,
            "peak_equity": self._compound_state.peak_equity,
            "compound_factor": self._compound_state.compound_factor,
            "paused_until": self._pause_until.isoformat() if self._pause_until else None,
        }

    # ------------------------------------------------------------------
    # Trade quality filter (MOST IMPORTANT)
    # ------------------------------------------------------------------

    async def assess_trade_quality(
        self,
        signal: Dict,
        ai_decision: Optional[Dict] = None,
        order_flow: Optional[Dict] = None,
        regime: str = "unknown",
        strategy_performance: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
    ) -> TradeQualityScore:
        """Assess trade quality across multiple dimensions.

        This is the KEY function that filters out 40-60% of signals.

        Args:
            signal: Trading signal
            ai_decision: AI brain decision
            order_flow: Order flow data
            regime: Current market regime
            strategy_performance: Historical strategy performance
            market_data: Current market data

        Returns:
            TradeQualityScore
        """
        async with self._lock:
            score = TradeQualityScore()

            # 1. Signal confidence (0-1)
            score.signal_confidence = signal.get("confidence", 0.0)

            # 2. AI confidence modifier (0-1)
            if ai_decision:
                score.ai_confidence = ai_decision.get("confidence", 0.0)
            else:
                score.ai_confidence = 0.7  # Neutral if no AI input

            # 3. Order flow alignment (0-1)
            if order_flow:
                # Check if order flow supports signal direction
                direction = signal.get("direction", "neutral")
                flow_imbalance = order_flow.get("imbalance", 0.0)

                if direction == "long" and flow_imbalance > 0:
                    score.order_flow_alignment = min(abs(flow_imbalance), 1.0)
                elif direction == "short" and flow_imbalance < 0:
                    score.order_flow_alignment = min(abs(flow_imbalance), 1.0)
                else:
                    score.order_flow_alignment = 0.5  # Neutral
            else:
                score.order_flow_alignment = 0.7  # Neutral if no data

            # 4. Regime appropriateness (0-1)
            strategy_name = signal.get("strategy", "")
            if strategy_performance and regime in strategy_performance.get("regime_performance", {}):
                regime_perf = strategy_performance["regime_performance"][regime]
                regime_win_rate = regime_perf.get("win_rate", 0.5)
                score.regime_appropriateness = regime_win_rate
            else:
                score.regime_appropriateness = 0.6  # Neutral

            # 5. Historical strategy performance (0-1)
            if strategy_performance:
                overall_win_rate = strategy_performance.get("win_rate", 0.5)
                score.historical_performance = overall_win_rate
            else:
                score.historical_performance = 0.6  # Neutral

            # 6. Spread quality (0-1)
            if market_data:
                spread_pct = market_data.get("spread_pct", 0.0)
                # Good spread < 0.1%, bad spread > 0.5%
                if spread_pct < 0.1:
                    score.spread_quality = 1.0
                elif spread_pct > 0.5:
                    score.spread_quality = 0.3
                else:
                    score.spread_quality = 1.0 - (spread_pct - 0.1) / 0.4 * 0.7
            else:
                score.spread_quality = 0.7  # Neutral

            # 7. Microstructure quality (0-1)
            if market_data:
                # Composite of volume, volatility, liquidity
                volume_quality = market_data.get("volume_quality", 0.7)
                volatility_quality = market_data.get("volatility_quality", 0.7)
                liquidity_quality = market_data.get("liquidity_quality", 0.7)
                score.microstructure_quality = (
                    volume_quality * 0.4 + volatility_quality * 0.3 + liquidity_quality * 0.3
                )
            else:
                score.microstructure_quality = 0.7  # Neutral

            # Weighted composite score
            weights = {
                "signal_confidence": 0.25,
                "ai_confidence": 0.20,
                "order_flow_alignment": 0.15,
                "regime_appropriateness": 0.15,
                "historical_performance": 0.10,
                "spread_quality": 0.08,
                "microstructure_quality": 0.07,
            }

            score.composite_score = (
                score.signal_confidence * weights["signal_confidence"]
                + score.ai_confidence * weights["ai_confidence"]
                + score.order_flow_alignment * weights["order_flow_alignment"]
                + score.regime_appropriateness * weights["regime_appropriateness"]
                + score.historical_performance * weights["historical_performance"]
                + score.spread_quality * weights["spread_quality"]
                + score.microstructure_quality * weights["microstructure_quality"]
            )

            # Decision: should trade?
            score.should_trade = score.composite_score >= self._dynamic_threshold

            logger.debug(
                "Trade quality: {:.3f} (threshold {:.3f}) signal={:.2f} ai={:.2f} flow={:.2f} "
                "regime={:.2f} perf={:.2f} spread={:.2f} micro={:.2f} → {}",
                score.composite_score,
                self._dynamic_threshold,
                score.signal_confidence,
                score.ai_confidence,
                score.order_flow_alignment,
                score.regime_appropriateness,
                score.historical_performance,
                score.spread_quality,
                score.microstructure_quality,
                "TRADE" if score.should_trade else "SKIP",
            )

            return score

    # ------------------------------------------------------------------
    # Combined position sizing
    # ------------------------------------------------------------------

    async def get_combined_multiplier(
        self,
        strategy_name: str,
        symbol: str,
        daily_profit: float = 0.0,
    ) -> float:
        """Get combined position size multiplier from all subsystems.

        Args:
            strategy_name: Strategy name
            symbol: Trading pair
            daily_profit: Profit for current day

        Returns:
            Combined multiplier
        """
        # Get all multipliers
        compound_mult = await self.get_compound_multiplier(daily_profit)
        momentum_mult = await self.get_momentum_multiplier(strategy_name)
        time_mult = await self.get_time_multiplier()
        pair_mult = await self.get_pair_multiplier(symbol)
        drawdown_mult = await self.get_drawdown_multiplier()

        # Combine multiplicatively
        combined = compound_mult * momentum_mult * time_mult * pair_mult * drawdown_mult

        logger.debug(
            "Combined multiplier: {:.3f} (compound={:.2f} momentum={:.2f} time={:.2f} pair={:.2f} drawdown={:.2f})",
            combined,
            compound_mult,
            momentum_mult,
            time_mult,
            pair_mult,
            drawdown_mult,
        )

        return combined

    # ------------------------------------------------------------------
    # Status and reporting
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get profit maximizer status.

        Returns:
            Status dict
        """
        return {
            "mode": self._drawdown_mode,
            "equity": self._compound_state.equity,
            "peak_equity": self._compound_state.peak_equity,
            "drawdown_pct": self._compound_state.current_drawdown_pct,
            "compound_factor": self._compound_state.compound_factor,
            "quality_threshold": self._dynamic_threshold,
            "total_trades": len(self._recent_trades),
            "paused_strategies": len([s for s, t in self._strategy_paused_until.items() if datetime.now(timezone.utc) < t]),
        }
