"""Gold Fibonacci Levels strategy — trade bounces off Fibonacci retracements."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# Fibonacci retracement ratios (levels we actively trade)
_FIB_TRADE_LEVELS = (0.382, 0.500, 0.618)
# Confidence map per Fib level: 0.618 ("golden ratio") is strongest
_FIB_CONFIDENCE = {0.382: 0.62, 0.500: 0.72, 0.618: 0.82}


class GoldFibonacciStrategy(BaseStrategy):
    """Gold Fibonacci Retracement strategy.

    Gold respects Fibonacci retracement levels (38.2 %, 50 %, 61.8 %)
    extremely well because institutional traders place limit orders at
    these levels.

    Signal logic
    ------------
    * Identify swing high and swing low over the last ``swing_window`` bars.
    * Compute 38.2 %, 50 %, and 61.8 % retracement levels.
    * In an uptrend: long when price touches a level and closes above it.
    * In a downtrend: short when price touches a level and closes below it.
    * Confidence is higher for 61.8 % (golden ratio).
    """

    _STRATEGY_NAME = "gold_fibonacci"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        swing_window: int = 50,
        level_tolerance: float = 0.003,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._swing_window = swing_window
        self._level_tolerance = level_tolerance
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_swing_points(
        ohlcv: pd.DataFrame, window: int
    ) -> Tuple[float, float, bool]:
        """Return (swing_low, swing_high, is_uptrend) for the last *window* bars."""
        recent = ohlcv.iloc[-window:]
        swing_low = float(recent["low"].min())
        swing_high = float(recent["high"].max())
        low_idx = int(recent["low"].values.argmin())
        high_idx = int(recent["high"].values.argmax())
        uptrend = low_idx < high_idx
        return swing_low, swing_high, uptrend

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._swing_window + self._atr_period + 5
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        swing_low, swing_high, uptrend = self._find_swing_points(ohlcv, self._swing_window)
        price_range = swing_high - swing_low
        if price_range <= 0:
            return None

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        prev_price = float(ohlcv["close"].iloc[-2])

        # Retracement levels
        if uptrend:
            fib_levels = {lvl: swing_high - price_range * lvl for lvl in _FIB_TRADE_LEVELS}
            direction = "long"
        else:
            fib_levels = {lvl: swing_low + price_range * lvl for lvl in _FIB_TRADE_LEVELS}
            direction = "short"

        tol = price_range * self._level_tolerance

        hit_level: Optional[float] = None
        hit_ratio: Optional[float] = None
        for ratio, level in fib_levels.items():
            if abs(curr_price - level) <= tol:
                if uptrend and prev_price < level <= curr_price:
                    hit_level = level
                    hit_ratio = ratio
                    break
                if not uptrend and prev_price > level >= curr_price:
                    hit_level = level
                    hit_ratio = ratio
                    break

        if hit_level is None:
            return None

        confidence = _FIB_CONFIDENCE.get(hit_ratio, 0.60)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
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
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Fibonacci bounce on gold")

            direction = sig["direction"]
            atr = sig["atr"]
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
                    f"Gold Fib {sig['fib_level']:.3f} bounce {direction}: "
                    f"level={fib_price:.4f}, ATR={atr:.4f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
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
            return True
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and curr_price < sig.get("fib_price", curr_price) * 0.99:
            return True
        if side == "short" and curr_price > sig.get("fib_price", curr_price) * 1.01:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.015
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.03, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 3.0),
            "leverage": 2,
        }
