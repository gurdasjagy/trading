"""Smart exit engine with intelligent exit strategies beyond simple SL/TP."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger
import numpy as np

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange
    from exchange.position_manager import Position, PositionTracker


@dataclass
class ExitSignal:
    """Exit signal for a position."""

    position_id: str
    symbol: str
    exit_type: str  # "stop_loss", "take_profit", "chandelier", "time_based", etc.
    reason: str
    should_exit: bool
    partial_exit_fraction: float = 1.0  # 1.0 = full exit, 0.5 = 50% exit
    new_stop_loss: Optional[float] = None  # Updated SL if not exiting
    urgency: str = "normal"  # "normal", "urgent", "immediate"


@dataclass
class MAEStats:
    """Maximum Adverse Excursion statistics for a strategy."""

    strategy_name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=100))
    p50_mae: float = 0.0
    p90_mae: float = 0.0
    p95_mae: float = 0.0


class SmartExitEngine:
    """Intelligent exit management with advanced strategies.

    Exit Strategies:
    1. Chandelier Exit - ATR-based trailing stop from highest high
    2. Parabolic SAR - Accelerating trailing stop
    3. Time-based Exit - Close dead trades (no profit after N candles)
    4. Volatility Expansion Exit - Exit on regime change
    5. Partial Profit Taking - Structured profit taking at R multiples
    6. MAE Analysis - Exit if adverse excursion exceeds historical 90th percentile
    7. Correlation-based Exit - Exit when correlated assets break down

    Features:
    - Replaces simple SL/TP with intelligent management
    - Lets winners run with trailing stops
    - Cuts losers early using MAE analysis
    - Adapts to changing volatility
    """

    # Time-based exit configuration
    TIME_EXIT_CANDLES = 20  # Close if no profit after 20 candles
    TIME_EXIT_TIMEFRAME = "15m"  # Timeframe for candle counting

    # Volatility expansion threshold
    VOL_EXPANSION_MULTIPLIER = 2.0  # Exit if volatility doubles

    # Partial profit taking schedule (R multiples)
    PROFIT_SCHEDULE = [
        (1.0, 0.25, True),   # At 1R: take 25%, move SL to breakeven
        (2.0, 0.25, False),  # At 2R: take 25%, trail SL to 1R
        (3.0, 0.0, False),   # At 3R: let remaining 50% run with trailing stop
    ]

    def __init__(
        self,
        exchange: BaseExchange,
        position_manager: Any
    ):
        """Initialize smart exit engine.

        Args:
            exchange: Exchange interface
            position_manager: Position manager for tracking
        """
        self._exchange = exchange
        self._position_manager = position_manager

        # MAE statistics per strategy
        self._mae_stats: Dict[str, MAEStats] = defaultdict(
            lambda: MAEStats(strategy_name="unknown")
        )

        # Position tracking for exits
        self._position_entry_times: Dict[str, float] = {}
        self._position_highest_prices: Dict[str, float] = {}
        self._position_lowest_prices: Dict[str, float] = {}
        self._position_peak_profit: Dict[str, float] = {}  # For MAE calculation
        self._position_initial_volatility: Dict[str, float] = {}

        # Partial profit taking tracking
        self._profit_levels_taken: Dict[str, List[float]] = defaultdict(list)

        logger.info("SmartExitEngine initialized")

    async def evaluate_exits(
        self,
        positions: List[PositionTracker]
    ) -> List[ExitSignal]:
        """Evaluate all exit conditions for open positions.

        Args:
            positions: List of open positions to evaluate

        Returns:
            List of ExitSignal objects with exit recommendations
        """
        signals = []

        for position in positions:
            try:
                # Run all exit strategies in parallel
                chandelier = asyncio.create_task(self._check_chandelier_exit(position))
                parabolic_sar = asyncio.create_task(self._check_parabolic_sar_exit(position))
                time_based = asyncio.create_task(self._check_time_based_exit(position))
                vol_expansion = asyncio.create_task(self._check_volatility_expansion_exit(position))
                partial_profit = asyncio.create_task(self._check_partial_profit_taking(position))
                mae_exit = asyncio.create_task(self._check_mae_exit(position))

                results = await asyncio.gather(
                    chandelier,
                    parabolic_sar,
                    time_based,
                    vol_expansion,
                    partial_profit,
                    mae_exit,
                    return_exceptions=True
                )

                # Collect all signals
                for result in results:
                    if isinstance(result, ExitSignal) and result is not None:
                        signals.append(result)
                    elif isinstance(result, Exception):
                        logger.error("Exit evaluation error: {}", result)

            except Exception as e:
                logger.error("Failed to evaluate exits for {}: {}", position.position.symbol, e)

        return signals

    async def _check_chandelier_exit(
        self,
        position: PositionTracker
    ) -> Optional[ExitSignal]:
        """Check Chandelier Exit condition.

        Trailing stop based on ATR from highest high since entry.

        Formula:
        - Long: Exit if price < highest_high - (ATR * multiplier)
        - Short: Exit if price > lowest_low + (ATR * multiplier)

        Args:
            position: Position to evaluate

        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        symbol = position.position.symbol
        current_price = await self._get_current_price(symbol)

        if current_price is None:
            return None

        # Get ATR
        atr = await self._calculate_atr(symbol, period=14)
        if atr is None or atr <= 0:
            return None

        # ATR multiplier (typically 2-3)
        multiplier = 3.0

        # Track highest/lowest prices
        if symbol not in self._position_highest_prices:
            self._position_highest_prices[symbol] = current_price
            self._position_lowest_prices[symbol] = current_price
        else:
            if position.position.side == "long":
                self._position_highest_prices[symbol] = max(
                    self._position_highest_prices[symbol],
                    current_price
                )
            else:
                self._position_lowest_prices[symbol] = min(
                    self._position_lowest_prices[symbol],
                    current_price
                )

        # Check exit condition
        if position.position.side == "long":
            chandelier_stop = self._position_highest_prices[symbol] - (atr * multiplier)
            should_exit = current_price < chandelier_stop

            if should_exit:
                return ExitSignal(
                    position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                    symbol=symbol,
                    exit_type="chandelier",
                    reason=f"Price {current_price:.2f} < Chandelier stop {chandelier_stop:.2f}",
                    should_exit=True,
                    urgency="normal"
                )
            else:
                # Update trailing stop
                return ExitSignal(
                    position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                    symbol=symbol,
                    exit_type="chandelier",
                    reason="Trailing stop update",
                    should_exit=False,
                    new_stop_loss=chandelier_stop
                )
        else:
            # Short position
            chandelier_stop = self._position_lowest_prices[symbol] + (atr * multiplier)
            should_exit = current_price > chandelier_stop

            if should_exit:
                return ExitSignal(
                    position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                    symbol=symbol,
                    exit_type="chandelier",
                    reason=f"Price {current_price:.2f} > Chandelier stop {chandelier_stop:.2f}",
                    should_exit=True,
                    urgency="normal"
                )
            else:
                return ExitSignal(
                    position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                    symbol=symbol,
                    exit_type="chandelier",
                    reason="Trailing stop update",
                    should_exit=False,
                    new_stop_loss=chandelier_stop
                )

        return None

    async def _check_parabolic_sar_exit(
        self,
        position: PositionTracker
    ) -> Optional[ExitSignal]:
        """Check Parabolic SAR exit condition.

        Accelerating trailing stop that tightens as profit increases.

        Args:
            position: Position to evaluate

        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        symbol = position.position.symbol
        current_price = await self._get_current_price(symbol)

        if current_price is None:
            return None

        # Fetch OHLCV for SAR calculation
        try:
            ohlcv = await self._exchange.get_ohlcv(symbol, "15m", limit=50)
            if ohlcv is None or len(ohlcv) < 5:
                return None

            # Calculate Parabolic SAR
            sar = self._calculate_parabolic_sar(
                ohlcv['high'].values,
                ohlcv['low'].values,
                ohlcv['close'].values
            )

            if sar is None:
                return None

            current_sar = sar[-1]

            # Check exit condition
            if position.position.side == "long":
                should_exit = current_price < current_sar
            else:
                should_exit = current_price > current_sar

            if should_exit:
                return ExitSignal(
                    position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                    symbol=symbol,
                    exit_type="parabolic_sar",
                    reason=f"Price {current_price:.2f} crossed SAR {current_sar:.2f}",
                    should_exit=True,
                    urgency="normal"
                )

        except Exception as e:
            logger.debug("Parabolic SAR calculation failed: {}", e)

        return None

    async def _check_time_based_exit(
        self,
        position: PositionTracker
    ) -> Optional[ExitSignal]:
        """Check time-based exit condition.

        Exit if position has been open for N candles without profit.

        Args:
            position: Position to evaluate

        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        symbol = position.position.symbol

        # Track entry time
        if symbol not in self._position_entry_times:
            self._position_entry_times[symbol] = position.opened_at.timestamp() if hasattr(position.opened_at, 'timestamp') else time.time()

        entry_time = self._position_entry_times[symbol]
        current_time = time.time()

        # Calculate time elapsed in candles
        timeframe_seconds = 15 * 60  # 15 minutes
        candles_elapsed = (current_time - entry_time) / timeframe_seconds

        if candles_elapsed < self.TIME_EXIT_CANDLES:
            return None

        # Check if position is profitable
        current_price = await self._get_current_price(symbol)
        if current_price is None:
            return None

        entry_price = position.position.entry_price
        is_profitable = (
            (current_price > entry_price and position.position.side == "long") or
            (current_price < entry_price and position.position.side == "short")
        )

        if not is_profitable:
            return ExitSignal(
                position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                symbol=symbol,
                exit_type="time_based",
                reason=f"No profit after {int(candles_elapsed)} candles",
                should_exit=True,
                urgency="normal"
            )

        return None

    async def _check_volatility_expansion_exit(
        self,
        position: PositionTracker
    ) -> Optional[ExitSignal]:
        """Check volatility expansion exit condition.

        Exit if volatility suddenly expands (regime change).

        Args:
            position: Position to evaluate

        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        symbol = position.position.symbol

        # Calculate current volatility
        current_vol = await self._calculate_volatility(symbol, period=20)
        if current_vol is None:
            return None

        # Track initial volatility at entry
        if symbol not in self._position_initial_volatility:
            self._position_initial_volatility[symbol] = current_vol
            return None

        initial_vol = self._position_initial_volatility[symbol]

        # Check for expansion
        vol_ratio = current_vol / initial_vol if initial_vol > 0 else 1.0

        if vol_ratio > self.VOL_EXPANSION_MULTIPLIER:
            return ExitSignal(
                position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                symbol=symbol,
                exit_type="volatility_expansion",
                reason=f"Volatility expanded {vol_ratio:.1f}x (regime change detected)",
                should_exit=True,
                urgency="immediate"
            )

        return None

    async def _check_partial_profit_taking(
        self,
        position: PositionTracker
    ) -> Optional[ExitSignal]:
        """Check partial profit taking schedule.

        Take partial profits at R multiples and move SL.

        Args:
            position: Position to evaluate

        Returns:
            ExitSignal with partial exit, None otherwise
        """
        symbol = position.position.symbol
        current_price = await self._get_current_price(symbol)

        if current_price is None or position.stop_loss is None:
            return None

        entry_price = position.position.entry_price
        stop_loss = position.stop_loss

        # Calculate R (initial risk)
        r = abs(entry_price - stop_loss)
        if r <= 0:
            return None

        # Calculate current profit in R multiples
        if position.position.side == "long":
            profit_r = (current_price - entry_price) / r
        else:
            profit_r = (entry_price - current_price) / r

        # Check profit schedule
        for r_level, take_fraction, move_to_breakeven in self.PROFIT_SCHEDULE:
            if profit_r >= r_level:
                # Check if this level already taken
                if symbol not in self._profit_levels_taken:
                    self._profit_levels_taken[symbol] = []

                if r_level in self._profit_levels_taken[symbol]:
                    continue

                # Mark level as taken
                self._profit_levels_taken[symbol].append(r_level)

                # Calculate new stop loss
                if move_to_breakeven:
                    new_sl = entry_price
                elif r_level >= 2.0:
                    # Trail to 1R profit
                    new_sl = entry_price + r if position.position.side == "long" else entry_price - r
                else:
                    new_sl = None

                reason = f"Profit taking at {r_level}R"
                if move_to_breakeven:
                    reason += ", moving SL to breakeven"
                elif new_sl:
                    reason += f", trailing SL to {new_sl:.2f}"

                if take_fraction > 0:
                    return ExitSignal(
                        position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                        symbol=symbol,
                        exit_type="partial_profit",
                        reason=reason,
                        should_exit=True,
                        partial_exit_fraction=take_fraction,
                        new_stop_loss=new_sl,
                        urgency="normal"
                    )
                elif new_sl:
                    return ExitSignal(
                        position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                        symbol=symbol,
                        exit_type="partial_profit",
                        reason=reason,
                        should_exit=False,
                        new_stop_loss=new_sl
                    )

        return None

    async def _check_mae_exit(
        self,
        position: PositionTracker
    ) -> Optional[ExitSignal]:
        """Check Maximum Adverse Excursion exit condition.

        Exit if current drawdown exceeds historical 90th percentile MAE.
        This prevents holding losers that are statistically unlikely to recover.

        Args:
            position: Position to evaluate

        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        symbol = position.position.symbol
        strategy = position.strategy
        current_price = await self._get_current_price(symbol)

        if current_price is None:
            return None

        entry_price = position.position.entry_price

        # Calculate current adverse excursion
        if position.position.side == "long":
            current_ae = max(0.0, (entry_price - current_price) / entry_price)
        else:
            current_ae = max(0.0, (current_price - entry_price) / entry_price)

        # Track peak profit for this position
        if position.position.side == "long":
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price

        if symbol not in self._position_peak_profit:
            self._position_peak_profit[symbol] = current_profit
        else:
            self._position_peak_profit[symbol] = max(
                self._position_peak_profit[symbol],
                current_profit
            )

        # Get MAE statistics for strategy
        mae_stats = self._mae_stats[strategy]

        if len(mae_stats.samples) < 10:
            # Not enough data yet
            return None

        # Calculate percentiles
        samples_sorted = sorted(mae_stats.samples)
        p90_mae = samples_sorted[int(len(samples_sorted) * 0.90)]

        # Exit if current AE exceeds 90th percentile
        if current_ae > p90_mae:
            return ExitSignal(
                position_id=str(position.position.id) if hasattr(position.position, 'id') else symbol,
                symbol=symbol,
                exit_type="mae_exit",
                reason=f"Adverse excursion {current_ae:.2%} exceeds 90th percentile {p90_mae:.2%}",
                should_exit=True,
                urgency="immediate"
            )

        return None

    async def record_trade_mae(
        self,
        strategy: str,
        entry_price: float,
        exit_price: float,
        worst_price: float,
        side: str
    ) -> None:
        """Record MAE for a completed trade.

        Args:
            strategy: Strategy name
            entry_price: Entry price
            exit_price: Exit price
            worst_price: Worst price during trade
            side: "long" or "short"
        """
        if side == "long":
            mae = (entry_price - worst_price) / entry_price
        else:
            mae = (worst_price - entry_price) / entry_price

        mae = max(0.0, mae)  # MAE is always positive

        mae_stats = self._mae_stats[strategy]
        mae_stats.samples.append(mae)

        # Update percentiles
        if len(mae_stats.samples) >= 10:
            samples_sorted = sorted(mae_stats.samples)
            mae_stats.p50_mae = samples_sorted[int(len(samples_sorted) * 0.50)]
            mae_stats.p90_mae = samples_sorted[int(len(samples_sorted) * 0.90)]
            mae_stats.p95_mae = samples_sorted[int(len(samples_sorted) * 0.95)]

            logger.debug(
                "MAE stats for {}: p50={:.2%} p90={:.2%} p95={:.2%}",
                strategy,
                mae_stats.p50_mae,
                mae_stats.p90_mae,
                mae_stats.p95_mae
            )

    def get_mae_stats(self, strategy: str) -> Dict:
        """Get MAE statistics for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Dict with MAE statistics
        """
        if strategy not in self._mae_stats:
            return {
                "strategy": strategy,
                "sample_count": 0,
                "p50_mae": 0.0,
                "p90_mae": 0.0,
                "p95_mae": 0.0
            }

        mae_stats = self._mae_stats[strategy]
        return {
            "strategy": strategy,
            "sample_count": len(mae_stats.samples),
            "p50_mae": mae_stats.p50_mae,
            "p90_mae": mae_stats.p90_mae,
            "p95_mae": mae_stats.p95_mae
        }

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            ticker = await self._exchange.get_ticker(symbol)
            return ticker.last
        except Exception as e:
            logger.debug("Failed to get current price for {}: {}", symbol, e)
            return None

    async def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range.

        Args:
            symbol: Trading symbol
            period: ATR period

        Returns:
            ATR value or None if calculation fails
        """
        try:
            ohlcv = await self._exchange.get_ohlcv(symbol, "1h", limit=period + 10)
            if ohlcv is None or len(ohlcv) < period:
                return None

            high = ohlcv['high'].values
            low = ohlcv['low'].values
            close = ohlcv['close'].values

            # Calculate True Range
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - np.roll(close, 1)),
                    abs(low - np.roll(close, 1))
                )
            )

            # Average True Range
            atr = np.mean(tr[-period:])
            return float(atr)

        except Exception as e:
            logger.debug("ATR calculation failed: {}", e)
            return None

    async def _calculate_volatility(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate historical volatility.

        Args:
            symbol: Trading symbol
            period: Lookback period

        Returns:
            Volatility or None if calculation fails
        """
        try:
            ohlcv = await self._exchange.get_ohlcv(symbol, "1h", limit=period + 10)
            if ohlcv is None or len(ohlcv) < period:
                return None

            returns = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
            volatility = returns.std()

            return float(volatility)

        except Exception as e:
            logger.debug("Volatility calculation failed: {}", e)
            return None

    def _calculate_parabolic_sar(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2
    ) -> Optional[np.ndarray]:
        """Calculate Parabolic SAR.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            af_start: Starting acceleration factor
            af_increment: AF increment
            af_max: Maximum AF

        Returns:
            SAR array or None if calculation fails
        """
        try:
            n = len(close)
            sar = np.zeros(n)
            ep = np.zeros(n)
            trend = np.ones(n)
            af = np.zeros(n)

            # Initialize
            sar[0] = close[0]
            ep[0] = high[0]
            trend[0] = 1
            af[0] = af_start

            for i in range(1, n):
                # Calculate SAR
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

                # Trend detection
                if trend[i-1] == 1:  # Uptrend
                    if low[i] < sar[i]:
                        # Trend reversal
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        ep[i] = low[i]
                        af[i] = af_start
                    else:
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + af_increment, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:  # Downtrend
                    if high[i] > sar[i]:
                        # Trend reversal
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        ep[i] = high[i]
                        af[i] = af_start
                    else:
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + af_increment, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]

            return sar

        except Exception as e:
            logger.debug("Parabolic SAR calculation failed: {}", e)
            return None

    def cleanup_position(self, symbol: str) -> None:
        """Clean up tracking data for closed position.

        Args:
            symbol: Trading symbol
        """
        self._position_entry_times.pop(symbol, None)
        self._position_highest_prices.pop(symbol, None)
        self._position_lowest_prices.pop(symbol, None)
        self._position_peak_profit.pop(symbol, None)
        self._position_initial_volatility.pop(symbol, None)
        self._profit_levels_taken.pop(symbol, None)
