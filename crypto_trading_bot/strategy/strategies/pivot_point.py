"""Pivot Point strategy — daily pivot levels as support/resistance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class PivotPointStrategy(BaseStrategy):
    """Daily Pivot Point bounce strategy.

    Computes classic pivot points from the previous day's OHLC:
      PP  = (H + L + C) / 3
      R1  = 2*PP − L,  R2 = PP + (H − L),  R3 = H + 2*(PP − L)
      S1  = 2*PP − H,  S2 = PP − (H − L),  S3 = L − 2*(H − PP)

    Entry conditions
    ----------------
    * **Long**: price is near S1/S2/S3 (within *tolerance*) and the last
      candle shows a bullish close (close > open).
    * **Short**: price is near R1/R2/R3 and the last candle is bearish.

    Uses intraday OHLCV data; pivot points are recomputed from the
    daily candle embedded in the dataset.
    """

    _STRATEGY_NAME = "pivot_point"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        tolerance: float = 0.003,
        daily_bars: int = 24,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._tolerance = tolerance
        self._daily_bars = daily_bars
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Pivot computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pivots(
        prev_high: float, prev_low: float, prev_close: float
    ) -> Dict[str, float]:
        pp = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2 * pp - prev_low
        r2 = pp + (prev_high - prev_low)
        r3 = prev_high + 2 * (pp - prev_low)
        s1 = 2 * pp - prev_high
        s2 = pp - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pp)
        return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._daily_bars + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        # Use previous "day" window (last daily_bars candles before the current bar)
        prev_window = ohlcv.iloc[-(self._daily_bars + 1):-1]
        prev_high = float(prev_window["high"].max())
        prev_low = float(prev_window["low"].min())
        prev_close = float(prev_window["close"].iloc[-1])

        pivots = self._compute_pivots(prev_high, prev_low, prev_close)

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        curr_open = float(ohlcv["open"].iloc[-1])
        tol = curr_price * self._tolerance

        # Check support levels for long
        for key in ("S1", "S2", "S3"):
            level = pivots[key]
            if abs(curr_price - level) <= tol and curr_price > curr_open:
                strength_map = {"S1": 0.6, "S2": 0.7, "S3": 0.75}
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": strength_map[key],
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "pivot_level": key,
                    "pivot_price": level,
                    "pp": pivots["PP"],
                }

        # Check resistance levels for short
        for key in ("R1", "R2", "R3"):
            level = pivots[key]
            if abs(curr_price - level) <= tol and curr_price < curr_open:
                strength_map = {"R1": 0.6, "R2": 0.7, "R3": 0.75}
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": strength_map[key],
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "pivot_level": key,
                    "pivot_price": level,
                    "pp": pivots["PP"],
                }

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "Price not near any pivot level")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            pivot = sig["pivot_price"]
            pp = sig["pp"]

            if direction == "long":
                stop_loss = pivot - atr * 1.0
                take_profit = pp  # Target PP or next resistance
            else:
                stop_loss = pivot + atr * 1.0
                take_profit = pp  # Target PP or next support

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Pivot {sig['pivot_level']} bounce {direction}: "
                    f"level={pivot:.4f}, PP={pp:.4f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=self._daily_bars + 20)
        if ohlcv.empty:
            return False
        prev_window = ohlcv.iloc[-(self._daily_bars + 1):-1]
        prev_high = float(prev_window["high"].max())
        prev_low = float(prev_window["low"].min())
        prev_close = float(prev_window["close"].iloc[-1])
        pivots = self._compute_pivots(prev_high, prev_low, prev_close)
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close when price reaches the PP
        if side == "long" and curr_price >= pivots["PP"]:
            return True
        if side == "short" and curr_price <= pivots["PP"]:
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
            "take_profit_pct": min(0.06, sl_pct * 2.5),
            "leverage": 2,
        }
