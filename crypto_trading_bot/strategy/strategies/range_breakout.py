"""Range Breakout strategy — consolidation detection + breakout entry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class RangeBreakoutStrategy(BaseStrategy):
    """Range consolidation breakout strategy.

    Detects a consolidation period (low ATR relative to its own average)
    and then fires a signal when price breaks out of the range's high/low.

    Entry conditions
    ----------------
    * **Long**: ATR was compressed for ≥ *consolidation_bars* bars AND price
      breaks above the range high.
    * **Short**: ATR compressed AND price breaks below the range low.
    """

    _STRATEGY_NAME = "range_breakout"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        consolidation_bars: int = 15,
        atr_period: int = 14,
        atr_compression_ratio: float = 0.7,
        breakout_confirmation: float = 0.001,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._consolidation_bars = consolidation_bars
        self._atr_period = atr_period
        self._atr_compression_ratio = atr_compression_ratio
        self._breakout_confirmation = breakout_confirmation

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._consolidation_bars + self._atr_period * 3 + 10
        if len(ohlcv) < min_rows:
            return None

        highs = ohlcv["high"]
        lows = ohlcv["low"]
        closes = ohlcv["close"]

        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)
        if atr_series is None:
            return None

        # ATR baseline: average over long window before consolidation
        baseline_window = self._atr_period * 3
        atr_baseline = float(atr_series.iloc[-self._consolidation_bars - baseline_window:
                                               -self._consolidation_bars].mean())

        if pd.isna(atr_baseline) or atr_baseline == 0:
            return None

        # Check if ATR has been compressed over the consolidation window
        consol_atrs = atr_series.iloc[-self._consolidation_bars - 1:-1]
        if consol_atrs.isna().any():
            return None

        max_consol_atr = float(consol_atrs.max())
        compression_ok = max_consol_atr < atr_baseline * self._atr_compression_ratio

        if not compression_ok:
            return None

        # Range bounds
        range_high = float(highs.iloc[-self._consolidation_bars - 1:-1].max())
        range_low = float(lows.iloc[-self._consolidation_bars - 1:-1].min())
        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])
        curr_atr = float(atr_series.iloc[-1])

        if pd.isna(curr_atr):
            return None

        tol = curr_price * self._breakout_confirmation
        direction: Optional[str] = None

        if curr_price > range_high + tol and prev_price <= range_high + tol:
            direction = "long"
        elif curr_price < range_low - tol and prev_price >= range_low - tol:
            direction = "short"

        if direction is None:
            return None

        range_size = range_high - range_low
        compression_quality = 1.0 - (max_consol_atr / atr_baseline)
        confidence = round(min(0.88, 0.55 + compression_quality * 0.33), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "range_high": range_high,
            "range_low": range_low,
            "range_size": range_size,
            "compression": compression_quality,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No range breakout detected")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["range_high"] - atr * 0.5
                take_profit = entry + sig["range_size"] * 1.5
            else:
                stop_loss = sig["range_low"] + atr * 0.5
                take_profit = entry - sig["range_size"] * 1.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Range breakout {direction}: "
                    f"range={sig['range_low']:.4f}–{sig['range_high']:.4f}, "
                    f"compression={sig['compression']:.2f}, ATR={atr:.6f}"
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
        # Close if price re-enters the range
        if side == "long" and curr_price < sig["range_high"]:
            return True
        if side == "short" and curr_price > sig["range_low"]:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.015
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 3.0),
            "leverage": 3,
        }
