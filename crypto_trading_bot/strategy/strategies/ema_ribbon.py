"""EMA Ribbon strategy — full ribbon alignment across 6 EMAs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

_EMA_PERIODS = [8, 13, 21, 34, 55, 89]


class EMARibbonStrategy(BaseStrategy):
    """EMA ribbon strategy using periods 8, 13, 21, 34, 55, 89.

    Entry conditions
    ----------------
    * **Long**: all EMAs are in ascending order (8 > 13 > 21 > 34 > 55 > 89)
      AND price is above EMA-8.
    * **Short**: all EMAs in descending order (8 < 13 < 21 < 34 < 55 < 89)
      AND price is below EMA-8.

    The degree of separation between EMAs provides the confidence score.
    """

    _STRATEGY_NAME = "ema_ribbon"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        ema_periods: Optional[List[int]] = None,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._ema_periods = ema_periods or _EMA_PERIODS
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        max_period = max(self._ema_periods)
        min_rows = max_period + self._atr_period + 10
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        ema_values: List[float] = []

        for period in self._ema_periods:
            ema = ta.ema(closes, length=period)
            if ema is None:
                return None
            val = float(ema.iloc[-1])
            if pd.isna(val):
                return None
            ema_values.append(val)

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(closes.iloc[-1])

        # Check full alignment
        bullish_aligned = all(ema_values[i] > ema_values[i + 1] for i in range(len(ema_values) - 1))
        bearish_aligned = all(ema_values[i] < ema_values[i + 1] for i in range(len(ema_values) - 1))

        direction: Optional[str] = None
        if bullish_aligned and curr_price > ema_values[0]:
            direction = "long"
        elif bearish_aligned and curr_price < ema_values[0]:
            direction = "short"

        if direction is None:
            return None

        # Confidence = average normalised spread between adjacent EMAs
        spreads = [abs(ema_values[i] - ema_values[i + 1]) / ema_values[i + 1]
                   for i in range(len(ema_values) - 1) if ema_values[i + 1] > 0]
        avg_spread = sum(spreads) / len(spreads) if spreads else 0.0
        confidence = round(min(0.9, 0.5 + avg_spread * 500), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "ema_values": ema_values,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "EMA ribbon not fully aligned")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            ema_vals = sig["ema_values"]

            if direction == "long":
                stop_loss = ema_vals[-1] - atr * 0.5  # Below slowest EMA
                take_profit = entry + atr * 3.0
            else:
                stop_loss = ema_vals[-1] + atr * 0.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"EMA ribbon fully {direction}: "
                    f"EMA{self._ema_periods[0]}={ema_vals[0]:.4f} → "
                    f"EMA{self._ema_periods[-1]}={ema_vals[-1]:.4f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=200)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        side = str(getattr(position, "side", "long")).lower()
        # Close if alignment breaks
        if sig is None:
            return True
        if side == "long" and sig["direction"] != "long":
            return True
        if side == "short" and sig["direction"] != "short":
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 3.0),
            "leverage": 3,
        }
