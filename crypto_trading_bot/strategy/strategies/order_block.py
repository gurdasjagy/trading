"""Order Block strategy — smart money concept: bullish/bearish order blocks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class OrderBlockStrategy(BaseStrategy):
    """Smart money Order Block (OB) strategy.

    An order block is the last opposing candle before a strong impulsive move:

    * **Bullish OB**: the last bearish (red) candle before a strong up-move.
      Entry when price returns to this candle's range.
    * **Bearish OB**: the last bullish (green) candle before a strong down-move.
      Entry when price returns to this candle's range.

    The impulse is validated by requiring the move to be ≥ *impulse_factor*
    times the ATR.
    """

    _STRATEGY_NAME = "order_block"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        lookback: int = 50,
        impulse_factor: float = 2.0,
        ob_tolerance: float = 0.002,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._lookback = lookback
        self._impulse_factor = impulse_factor
        self._ob_tolerance = ob_tolerance
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Order block detection
    # ------------------------------------------------------------------

    def _find_order_blocks(
        self, ohlcv: pd.DataFrame, atr: float
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return (bullish_obs, bearish_obs) as (low, high) tuples."""
        bullish_obs: List[Tuple[float, float]] = []
        bearish_obs: List[Tuple[float, float]] = []
        impulse_min = self._impulse_factor * atr

        for i in range(1, len(ohlcv) - 1):
            body_prev = float(ohlcv["close"].iloc[i]) - float(ohlcv["open"].iloc[i])
            body_curr = float(ohlcv["close"].iloc[i + 1]) - float(ohlcv["open"].iloc[i + 1])

            # Bullish OB: previous candle is bearish, next is strong bullish
            if body_prev < 0 and body_curr >= impulse_min:
                ob_low = float(ohlcv["low"].iloc[i])
                ob_high = float(ohlcv["high"].iloc[i])
                bullish_obs.append((ob_low, ob_high))

            # Bearish OB: previous candle is bullish, next is strong bearish
            elif body_prev > 0 and body_curr <= -impulse_min:
                ob_low = float(ohlcv["low"].iloc[i])
                ob_high = float(ohlcv["high"].iloc[i])
                bearish_obs.append((ob_low, ob_high))

        return bullish_obs, bearish_obs

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr) or curr_atr == 0:
            return None

        # Find OBs from historical data (exclude last 2 bars for formation)
        historical = ohlcv.iloc[-(self._lookback + 2):-2]
        bullish_obs, bearish_obs = self._find_order_blocks(historical, curr_atr)

        curr_price = float(ohlcv["close"].iloc[-1])
        tol = curr_price * self._ob_tolerance

        # Check bullish OBs — price returns to the OB range
        for ob_low, ob_high in reversed(bullish_obs):
            if ob_low - tol <= curr_price <= ob_high + tol:
                ob_size = ob_high - ob_low
                confidence = round(min(0.82, 0.55 + ob_size / curr_atr * 0.1), 3)
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "ob_low": ob_low,
                    "ob_high": ob_high,
                    "ob_type": "bullish",
                }

        # Check bearish OBs
        for ob_low, ob_high in reversed(bearish_obs):
            if ob_low - tol <= curr_price <= ob_high + tol:
                ob_size = ob_high - ob_low
                confidence = round(min(0.82, 0.55 + ob_size / curr_atr * 0.1), 3)
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "ob_low": ob_low,
                    "ob_high": ob_high,
                    "ob_type": "bearish",
                }

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No order block interaction")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["ob_low"] - atr * 0.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = sig["ob_high"] + atr * 0.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Order block {sig['ob_type']} {direction}: "
                    f"OB={sig['ob_low']:.4f}–{sig['ob_high']:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and curr_price < sig.get("ob_low", curr_price) * 0.99:
            return True
        if side == "short" and curr_price > sig.get("ob_high", curr_price) * 1.01:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.015
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.03, sl_pct),
            "take_profit_pct": min(0.09, sl_pct * 3.0),
            "leverage": 3,
        }
