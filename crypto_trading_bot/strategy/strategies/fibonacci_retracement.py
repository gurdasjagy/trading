"""Fibonacci Retracement strategy — trade bounces from key Fib levels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

_FIB_LEVELS = (0.236, 0.382, 0.500, 0.618, 0.786)
_BOUNCE_LEVELS = (0.382, 0.500, 0.618)  # levels we actually trade


class FibonacciRetracementStrategy(BaseStrategy):
    """Fibonacci retracement bounce strategy.

    Identifies the most recent swing high and swing low over a rolling
    window, computes 0.382 / 0.5 / 0.618 retracement levels, then trades
    when price touches a level and shows a reversal candle.

    Entry conditions
    ----------------
    * **Long**: price pulls back into a Fib level in an uptrend (swing low →
      swing high) and the last candle closes above the level.
    * **Short**: price rallies into a Fib level in a downtrend and the last
      candle closes below the level.
    """

    _STRATEGY_NAME = "fibonacci_retracement"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        swing_window: int = 50,
        level_tolerance: float = 0.003,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._swing_window = swing_window
        self._level_tolerance = level_tolerance
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _find_swing_points(ohlcv: pd.DataFrame, window: int) -> Tuple[float, float, bool]:
        """Return (swing_low, swing_high, uptrend) for the last *window* bars."""
        recent = ohlcv.iloc[-window:]
        swing_low = float(recent["low"].min())
        swing_high = float(recent["high"].max())
        low_idx = recent["low"].idxmin()
        high_idx = recent["high"].idxmax()
        # Uptrend: swing low came before swing high
        uptrend = low_idx < high_idx
        return swing_low, swing_high, uptrend

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._swing_window + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        swing_low, swing_high, uptrend = self._find_swing_points(ohlcv, self._swing_window)
        price_range = swing_high - swing_low
        if price_range <= 0:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr) or curr_atr == 0:
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        prev_price = float(ohlcv["close"].iloc[-2])

        # Compute retracement levels relative to trend direction
        if uptrend:
            # Retracement: pull back from swing_high toward swing_low
            fib_levels = {
                lvl: swing_high - price_range * lvl for lvl in _BOUNCE_LEVELS
            }
            direction = "long"
        else:
            # Retracement: rally from swing_low toward swing_high
            fib_levels = {
                lvl: swing_low + price_range * lvl for lvl in _BOUNCE_LEVELS
            }
            direction = "short"

        tol = price_range * self._level_tolerance

        hit_level: Optional[float] = None
        hit_ratio: Optional[float] = None
        for ratio, level in fib_levels.items():
            if abs(curr_price - level) <= tol:
                # Check reversal: prev candle was closer to the wrong side
                if uptrend and prev_price < level and curr_price >= level:
                    hit_level = level
                    hit_ratio = ratio
                    break
                if not uptrend and prev_price > level and curr_price <= level:
                    hit_level = level
                    hit_ratio = ratio
                    break

        if hit_level is None:
            return None

        # Higher Fib levels (0.618) give stronger confidence
        confidence_map = {0.382: 0.6, 0.500: 0.7, 0.618: 0.8}
        confidence = confidence_map.get(hit_ratio, 0.6)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "fib_level": hit_ratio,
            "fib_price": hit_level,
            "swing_low": swing_low,
            "swing_high": swing_high,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Fibonacci bounce signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            fib_price = sig["fib_price"]

            if direction == "long":
                stop_loss = fib_price - atr * 1.0
                take_profit = sig["swing_high"]
            else:
                stop_loss = fib_price + atr * 1.0
                take_profit = sig["swing_low"]

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Fib {sig['fib_level']:.3f} bounce {direction}: "
                    f"level={fib_price:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=self._swing_window + 20)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return True  # No longer at a Fib level
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close if price breaks through the next major Fib level
        if side == "long" and curr_price < sig.get("fib_price", curr_price) * 0.99:
            return True
        if side == "short" and curr_price > sig.get("fib_price", curr_price) * 1.01:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.015
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.03, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 3.0),
            "leverage": 2,
        }
