"""Donchian Breakout strategy — Turtle trading system."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout strategy (classic Turtle rules).

    Entry conditions
    ----------------
    * **Long**: current close breaks above the 20-period highest high.
    * **Short**: current close breaks below the 20-period lowest low.

    Exit uses the 10-period Donchian channel (tighter) as trailing stop.
    """

    _STRATEGY_NAME = "donchian_breakout"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        enabled: bool = True,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._entry_period = entry_period
        self._exit_period = exit_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._entry_period + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        highs = ohlcv["high"]
        lows = ohlcv["low"]

        # Donchian high/low computed over previous N bars (exclude current)
        dc_high = float(highs.iloc[-(self._entry_period + 1):-1].max())
        dc_low = float(lows.iloc[-(self._entry_period + 1):-1].min())
        exit_high = float(highs.iloc[-(self._exit_period + 1):-1].max())
        exit_low = float(lows.iloc[-(self._exit_period + 1):-1].min())

        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])

        direction: Optional[str] = None
        if curr_price > dc_high and prev_price <= dc_high:
            direction = "long"
        elif curr_price < dc_low and prev_price >= dc_low:
            direction = "short"

        if direction is None:
            return None

        if direction == "long":
            breakout_pct = (curr_price - dc_high) / dc_high
        else:
            breakout_pct = (dc_low - curr_price) / dc_low

        confidence = round(min(0.85, 0.55 + breakout_pct * 10), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "dc_high": dc_high,
            "dc_low": dc_low,
            "exit_high": exit_high,
            "exit_low": exit_low,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Donchian breakout")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["exit_low"] - atr * 0.5
                take_profit = entry + atr * 4.0
            else:
                stop_loss = sig["exit_high"] + atr * 0.5
                take_profit = entry - atr * 4.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Donchian {self._entry_period}-period breakout {direction}: "
                    f"DC_high={sig['dc_high']:.4f}, DC_low={sig['dc_low']:.4f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        if ohlcv.empty:
            return False
        highs = ohlcv["high"]
        lows = ohlcv["low"]
        exit_high = float(highs.iloc[-(self._exit_period + 1):-1].max())
        exit_low = float(lows.iloc[-(self._exit_period + 1):-1].min())
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and curr_price < exit_low:
            return True
        if side == "short" and curr_price > exit_high:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.06, sl_pct),
            "take_profit_pct": min(0.15, sl_pct * 4.0),
            "leverage": 2,
        }
