"""Dynamic stop-loss management — initial, trailing, time, and volatility-adjusted stops."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from exchange.position_manager import PositionTracker


class StopLossEngine:
    """Calculates and manages dynamic stop-loss levels."""

    def calculate_initial_stop(
        self,
        entry: float,
        direction: str,
        atr: float,
        multiplier: float = 2.0,
    ) -> float:
        """Calculate the initial ATR-based stop-loss price.

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range.
            multiplier: ATR multiplier for the stop distance.

        Returns:
            Stop-loss price.
        """
        if atr <= 0 or entry <= 0:
            logger.warning("Invalid ATR ({}) or entry ({}) for stop calculation", atr, entry)
            return entry * (0.98 if direction == "long" else 1.02)
        distance = atr * multiplier
        if direction == "long":
            stop = entry - distance
        else:
            stop = entry + distance
        logger.debug(
            "Initial stop: entry={} dir={} atr={} mult={} stop={:.4f}",
            entry,
            direction,
            atr,
            multiplier,
            stop,
        )
        return max(stop, 0.0)

    def calculate_trailing_stop(
        self,
        entry: float,
        current_price: float,
        direction: str,
        trail_pct: float,
    ) -> float:
        """Calculate a percentage-based trailing stop price.

        For longs the stop trails below the *current_price*; for shorts
        it trails above.

        Args:
            entry: Original entry price (used as fallback).
            current_price: Latest market price.
            direction: ``"long"`` or ``"short"``.
            trail_pct: Trail distance as a decimal (e.g. ``0.015`` for 1.5 %).

        Returns:
            Trailing stop price.
        """
        if current_price <= 0:
            return entry
        distance = current_price * trail_pct
        if direction == "long":
            stop = current_price - distance
        else:
            stop = current_price + distance
        logger.debug(
            "Trailing stop: current={} dir={} trail_pct={} stop={:.4f}",
            current_price,
            direction,
            trail_pct,
            stop,
        )
        return max(stop, 0.0)

    def should_move_stop(
        self,
        position: "PositionTracker",
        current_price: float,
        new_stop: float,
    ) -> bool:
        """Return True if *new_stop* is more favourable than the existing stop.

        For longs, the stop should only move up; for shorts, only down.

        Args:
            position: Existing position tracker with current stop-loss.
            current_price: Latest market price.
            new_stop: Proposed new stop-loss price.

        Returns:
            ``True`` if the stop should be moved.
        """
        existing_sl = position.stop_loss
        if existing_sl is None:
            return True
        direction = position.position.side.value.lower()
        if direction == "long":
            return new_stop > existing_sl
        else:
            return new_stop < existing_sl

    def adjust_for_volatility(
        self,
        stop: float,
        volatility_regime: str,
        entry_price: float = 0.0,
        direction: str = "long",
    ) -> float:
        """Widen or tighten the stop based on the current volatility regime.

        The stop DISTANCE from entry is scaled by the volatility multiplier,
        then the new stop PRICE is recomputed.  This prevents nonsensical
        results (e.g. a stop above entry for a long) that arise from naïvely
        multiplying the raw stop price.

        Args:
            stop: Base stop-loss price (already calculated relative to entry).
            volatility_regime: One of ``"low"``, ``"normal"``, ``"high"``, ``"extreme"``.
            entry_price: Entry price of the trade.  Required for correct
                distance-based adjustment.  Falls back to price-multiplication
                when 0 (backward-compatible, but not recommended).
            direction: ``"long"`` or ``"short"``.  Used to determine the sign
                of the stop distance.

        Returns:
            Adjusted stop price.
        """
        multipliers = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.3,
            "extreme": 1.6,
        }
        mult = multipliers.get(volatility_regime, 1.0)

        if entry_price > 0 and stop > 0:
            # Work with the stop DISTANCE so the result stays on the correct
            # side of entry regardless of the multiplier value.
            distance = abs(entry_price - stop)
            adjusted_distance = distance * mult
            if direction == "long":
                adjusted = entry_price - adjusted_distance
            else:
                adjusted = entry_price + adjusted_distance
            adjusted = max(adjusted, 0.0)
        else:
            # Legacy fallback: directly scale the stop price (not recommended)
            adjusted = stop * mult

        logger.debug(
            "Volatility-adjusted stop: base={:.4f} regime={} mult={} adjusted={:.4f}",
            stop,
            volatility_regime,
            mult,
            adjusted,
        )
        return adjusted

    def calculate_support_based_stop(
        self,
        entry: float,
        direction: str,
        support_levels: list,
    ) -> float:
        """Place a stop-loss just beyond the nearest support or resistance level.

        For **long** positions the stop is placed just below the nearest support
        level that is *below* the entry price.  For **short** positions the stop
        is placed just above the nearest resistance level that is *above* entry.

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            support_levels: List of price levels representing key support /
                resistance zones (unsorted).

        Returns:
            Stop-loss price, or a 2 % fallback when no suitable level is found.
        """
        if not support_levels or entry <= 0:
            return entry * (0.98 if direction == "long" else 1.02)

        # Small buffer (0.1 %) below/above the level to avoid premature fills
        _LEVEL_BUFFER_PCT = 0.001

        if direction == "long":
            # Find the highest support below entry
            candidates = [lvl for lvl in support_levels if lvl < entry]
            if not candidates:
                return entry * 0.98
            nearest = max(candidates)
            stop = nearest * (1 - _LEVEL_BUFFER_PCT)
        else:
            # Find the lowest resistance above entry
            candidates = [lvl for lvl in support_levels if lvl > entry]
            if not candidates:
                return entry * 1.02
            nearest = min(candidates)
            stop = nearest * (1 + _LEVEL_BUFFER_PCT)

        logger.debug(
            "Support-based stop: entry={} dir={} nearest_level={} stop={:.4f}",
            entry,
            direction,
            nearest,
            stop,
        )
        return max(stop, 0.0)

    def calculate_volatility_scaled_stop(
        self,
        entry: float,
        direction: str,
        atr: float,
        volatility_regime: str,
    ) -> float:
        """Calculate a stop scaled by ATR multiplier appropriate for *volatility_regime*.

        Multipliers:
            * ``"low"``     → 1.5×
            * ``"normal"`` / ``"medium"`` → 2.0×
            * ``"high"``    → 2.5×
            * ``"extreme"`` → 3.0×

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range.
            volatility_regime: Current volatility regime label.

        Returns:
            Stop-loss price.
        """
        _MULTIPLIERS = {
            "low": 1.5,
            "normal": 2.0,
            "medium": 2.0,
            "high": 2.5,
            "extreme": 3.0,
        }
        multiplier = _MULTIPLIERS.get(volatility_regime, 2.0)
        return self.calculate_initial_stop(entry, direction, atr, multiplier)

    def calculate_smart_stop(
        self,
        entry: float,
        direction: str,
        atr: float,
        volatility_regime: str,
        support_levels: Optional[list] = None,
    ) -> float:
        """Combine ATR-based and support-based stops, returning the tighter of the two.

        Using the tighter stop provides better risk management by limiting the
        maximum loss per trade.

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range.
            volatility_regime: Current volatility regime label.
            support_levels: Optional list of key support / resistance levels.

        Returns:
            The tighter (more conservative) stop-loss price.
        """
        vol_stop = self.calculate_volatility_scaled_stop(entry, direction, atr, volatility_regime)

        if support_levels:
            sup_stop = self.calculate_support_based_stop(entry, direction, support_levels)
        else:
            sup_stop = None

        if sup_stop is None:
            smart = vol_stop
        elif direction == "long":
            # For longs, a higher stop is tighter (closer to entry)
            smart = max(vol_stop, sup_stop)
        else:
            # For shorts, a lower stop is tighter
            smart = min(vol_stop, sup_stop)

        logger.debug(
            "Smart stop: entry={} dir={} vol_stop={:.4f} sup_stop={} chosen={:.4f}",
            entry,
            direction,
            vol_stop,
            f"{sup_stop:.4f}" if sup_stop is not None else "N/A",
            smart,
        )
        return max(smart, 0.0)

    def calculate_time_stop(
        self,
        position: "PositionTracker",
        max_hours: float = 24.0,
    ) -> bool:
        """Return True if the position has exceeded its maximum hold time.

        Args:
            position: Position tracker with ``opened_at`` timestamp.
            max_hours: Maximum allowed hold time in hours.

        Returns:
            ``True`` if the time-stop has been reached.
        """
        try:
            now = datetime.now(tz=timezone.utc)
            elapsed = (now - position.opened_at).total_seconds() / 3600.0
            triggered = elapsed >= max_hours
            if triggered:
                logger.info(
                    "Time stop triggered for {}: elapsed={:.1f}h max={:.1f}h",
                    position.position.symbol,
                    elapsed,
                    max_hours,
                )
            return triggered
        except Exception as exc:
            logger.error("Time stop check failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Advanced trailing stop methods
    # ------------------------------------------------------------------

    def calculate_chandelier_stop(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 22,
        multiplier: float = 3.0,
    ) -> float:
        """Calculate the Chandelier Exit trailing stop.

        The Chandelier Exit trails from the highest high over *period* candles
        minus ATR × *multiplier* (for longs).  For shorts it uses the lowest
        low plus ATR × multiplier.

        This method returns the **long** Chandelier stop.  For shorts, negate
        the logic in the caller or use the returned value as the upper boundary.

        Args:
            highs: List of high prices (most recent last).
            lows: List of low prices.
            closes: List of close prices.
            period: Lookback period for highest high and ATR.
            multiplier: ATR multiplier.

        Returns:
            Chandelier stop price (for a long position).
        """
        if len(closes) < period or len(highs) < period or len(lows) < period:
            return 0.0
        try:
            highest_high = max(highs[-period:])
            atr = self._calculate_atr(highs, lows, closes, period)
            stop = highest_high - multiplier * atr
            logger.debug(
                "Chandelier stop: highest_high={:.4f} atr={:.4f} mult={} stop={:.4f}",
                highest_high,
                atr,
                multiplier,
                stop,
            )
            return max(stop, 0.0)
        except Exception as exc:
            logger.error("Chandelier stop calculation failed: {}", exc)
            return 0.0

    def calculate_parabolic_sar_stop(
        self,
        highs: List[float],
        lows: List[float],
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.20,
    ) -> float:
        """Calculate the Parabolic SAR trailing stop.

        Implements the classic Parabolic Stop and Reverse (SAR) algorithm.
        Returns the SAR value for the most recent candle.

        Args:
            highs: List of high prices (oldest first, most recent last).
            lows: List of low prices.
            af_start: Initial acceleration factor.
            af_increment: AF increment per new extreme.
            af_max: Maximum acceleration factor.

        Returns:
            Parabolic SAR price for the latest candle.
        """
        if len(highs) < 3 or len(lows) < 3:
            return 0.0
        try:
            # Determine initial trend: uptrend if close[-1] > close[-2]
            is_uptrend = highs[-1] >= highs[-2]
            sar = lows[0] if is_uptrend else highs[0]
            ep = highs[0] if is_uptrend else lows[0]
            af = af_start

            for i in range(1, len(highs)):
                sar = sar + af * (ep - sar)
                if is_uptrend:
                    sar = min(sar, lows[i - 1], lows[max(0, i - 2)])
                    if lows[i] < sar:
                        is_uptrend = False
                        sar = ep
                        ep = lows[i]
                        af = af_start
                    else:
                        if highs[i] > ep:
                            ep = highs[i]
                            af = min(af + af_increment, af_max)
                else:
                    sar = max(sar, highs[i - 1], highs[max(0, i - 2)])
                    if highs[i] > sar:
                        is_uptrend = True
                        sar = ep
                        ep = highs[i]
                        af = af_start
                    else:
                        if lows[i] < ep:
                            ep = lows[i]
                            af = min(af + af_increment, af_max)

            logger.debug("Parabolic SAR: sar={:.4f} uptrend={}", sar, is_uptrend)
            return max(sar, 0.0)
        except Exception as exc:
            logger.error("Parabolic SAR calculation failed: {}", exc)
            return 0.0

    def calculate_keltner_stop(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        period: int = 20,
        multiplier: float = 1.5,
    ) -> float:
        """Calculate the Keltner Channel lower band as a dynamic stop.

        The lower Keltner Channel band = EMA(close, period) - multiplier × ATR.
        Use this as a stop-loss for long positions in ranging markets.

        Args:
            closes: List of close prices (most recent last).
            highs: List of high prices.
            lows: List of low prices.
            period: EMA and ATR period.
            multiplier: ATR multiplier for the channel width.

        Returns:
            Lower Keltner Channel price (for long stop placement).
        """
        if len(closes) < period:
            return 0.0
        try:
            # EMA of closes
            ema = closes[-period]
            k = 2.0 / (period + 1)
            for price in closes[-period + 1:]:
                ema = price * k + ema * (1 - k)

            atr = self._calculate_atr(highs, lows, closes, period)
            lower_band = ema - multiplier * atr
            logger.debug(
                "Keltner stop: ema={:.4f} atr={:.4f} mult={} lower={:.4f}",
                ema,
                atr,
                multiplier,
                lower_band,
            )
            return max(lower_band, 0.0)
        except Exception as exc:
            logger.error("Keltner stop calculation failed: {}", exc)
            return 0.0

    def select_best_trailing_stop(
        self,
        entry: float,
        direction: str,
        current_price: float,
        atr: float,
        volatility_regime: str,
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        closes: Optional[List[float]] = None,
    ) -> float:
        """Automatically select and calculate the best trailing stop method.

        Selection logic:
        * Trending + low vol:  Parabolic SAR (tight trail, captures trends)
        * Trending + high vol: Chandelier Exit (wider, avoids whipsaws)
        * Ranging market:      Keltner Channel (adapts to range)
        * Fallback:            ATR-based trailing stop

        Args:
            entry: Entry price of the trade.
            direction: ``"long"`` or ``"short"``.
            current_price: Current market price.
            atr: Average True Range.
            volatility_regime: ``"low"``, ``"normal"``, ``"high"``, or ``"extreme"``.
            highs: Optional list of recent high prices for advanced methods.
            lows: Optional list of recent low prices.
            closes: Optional list of recent close prices.

        Returns:
            Selected trailing stop price.
        """
        _trending_regimes = {"low", "normal"}
        _ranging_regimes = {"ranging"}
        _high_vol_regimes = {"high", "extreme"}

        has_price_data = highs and lows and closes and len(closes) >= 22

        if volatility_regime in _ranging_regimes and has_price_data:
            stop = self.calculate_keltner_stop(closes, highs, lows)  # type: ignore[arg-type]
            if stop > 0:
                logger.debug("Selected Keltner stop for ranging market: {:.4f}", stop)
                return stop

        if volatility_regime in _high_vol_regimes and has_price_data:
            if direction == "long":
                stop = self.calculate_chandelier_stop(highs, lows, closes)  # type: ignore[arg-type]
            else:
                # For shorts, use the lowest low + ATR * multiplier variant
                period = 22
                multiplier = 3.0
                if len(lows) >= period:
                    lowest_low = min(lows[-period:])
                    stop = lowest_low + multiplier * atr
                else:
                    stop = current_price + multiplier * atr
            if stop > 0:
                logger.debug(
                    "Selected Chandelier stop for high-vol trending: {:.4f}", stop
                )
                return stop

        if volatility_regime in _trending_regimes and has_price_data:
            stop = self.calculate_parabolic_sar_stop(highs, lows)  # type: ignore[arg-type]
            if stop > 0:
                logger.debug("Selected Parabolic SAR stop for trending: {:.4f}", stop)
                return stop

        # Fallback to ATR-based trailing stop
        stop = self.calculate_trailing_stop(entry, current_price, direction, trail_pct=atr / current_price if current_price > 0 else 0.02)
        logger.debug("Fallback to ATR trailing stop: {:.4f}", stop)
        return stop

    def tighten_stop_over_time(
        self,
        entry: float,
        current_stop: float,
        direction: str,
        elapsed_hours: float,
        atr: float,
        tighten_after_hours: float = 12.0,
        tighten_pct: float = 0.25,
    ) -> float:
        """Tighten the stop-loss after a trade has been open for a set duration.

        After *tighten_after_hours* have elapsed, the stop distance from entry is
        reduced by *tighten_pct* (default 25 %), moving the stop closer to the
        current price to lock in more of any unrealised profit.

        Args:
            entry: Entry price of the trade.
            current_stop: Existing stop-loss price.
            direction: ``"long"`` or ``"short"``.
            elapsed_hours: Time in hours since the trade was opened.
            atr: Average True Range (used as minimum stop distance to avoid
                setting a stop so tight it triggers on noise).
            tighten_after_hours: Hours after which tightening begins.
            tighten_pct: Fraction by which to reduce the stop distance (0–1).

        Returns:
            Updated stop-loss price, never tighter than 1× ATR from entry.
        """
        if elapsed_hours < tighten_after_hours or current_stop <= 0 or entry <= 0:
            return current_stop

        current_distance = abs(entry - current_stop)
        min_distance = atr if atr > 0 else current_distance * 0.5
        new_distance = max(current_distance * (1.0 - tighten_pct), min_distance)

        if direction == "long":
            new_stop = entry - new_distance
        else:
            new_stop = entry + new_distance

        new_stop = max(new_stop, 0.0)

        if (direction == "long" and new_stop > current_stop) or (
            direction == "short" and new_stop < current_stop
        ):
            logger.info(
                "Time-tightened stop: elapsed={:.1f}h old={:.4f} new={:.4f}",
                elapsed_hours,
                current_stop,
                new_stop,
            )
            return round(new_stop, 8)

        return current_stop

    def calculate_breakeven_stop(
        self,
        entry: float,
        direction: str,
        buffer_pct: float = 0.001,
    ) -> float:
        """Return the breakeven stop price (entry ± a small buffer).

        Moving the stop to breakeven after the trade reaches 1R profit
        ensures the worst-case outcome is a scratch trade rather than a loss.

        Args:
            entry: Trade entry price.
            direction: ``"long"`` or ``"short"``.
            buffer_pct: Small buffer fraction above/below entry to avoid
                immediate stop-outs on spread (default 0.1 %).

        Returns:
            Breakeven stop price.
        """
        if entry <= 0:
            return 0.0
        if direction == "long":
            stop = entry * (1 + buffer_pct)
        else:
            stop = entry * (1 - buffer_pct)
        logger.debug(
            "Breakeven stop: entry={} dir={} buffer={:.3%} stop={:.4f}",
            entry,
            direction,
            buffer_pct,
            stop,
        )
        return round(max(stop, 0.0), 8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int,
    ) -> float:
        """Calculate Average True Range over *period* candles."""
        if len(closes) < period + 1:
            if len(closes) >= 2:
                trs = [abs(highs[i] - lows[i]) for i in range(len(closes))]
                return sum(trs) / len(trs)
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return sum(trs[-period:]) / period
