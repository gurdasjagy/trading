"""Fair Value Gap strategy — trade price returns into 3-candle imbalances."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class FairValueGapStrategy(BaseStrategy):
    """Fair Value Gap (FVG) strategy.

    An FVG (also called an imbalance) is formed by three consecutive candles
    where there is a gap between candle 1's high/low and candle 3's low/high:

    * **Bullish FVG**: candle[i-2].high < candle[i].low  (upside gap)
    * **Bearish FVG**: candle[i-2].low > candle[i].high  (downside gap)

    Entry when price retraces back into an unfilled FVG:
    * **Long**: price falls back into a bullish FVG zone.
    * **Short**: price rallies back into a bearish FVG zone.
    """

    _STRATEGY_NAME = "fair_value_gap"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        lookback: int = 50,
        min_gap_atr_ratio: float = 0.3,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._lookback = lookback
        self._min_gap_atr_ratio = min_gap_atr_ratio
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # FVG detection
    # ------------------------------------------------------------------

    def _find_fvgs(
        self, ohlcv: pd.DataFrame, atr: float
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return (bullish_fvgs, bearish_fvgs) as (low, high) tuples."""
        bullish: List[Tuple[float, float]] = []
        bearish: List[Tuple[float, float]] = []
        min_gap = atr * self._min_gap_atr_ratio

        for i in range(2, len(ohlcv)):
            c1_high = float(ohlcv["high"].iloc[i - 2])
            c1_low = float(ohlcv["low"].iloc[i - 2])
            c3_high = float(ohlcv["high"].iloc[i])
            c3_low = float(ohlcv["low"].iloc[i])

            # Bullish FVG: gap above c1.high up to c3.low
            if c3_low > c1_high and (c3_low - c1_high) >= min_gap:
                bullish.append((c1_high, c3_low))

            # Bearish FVG: gap below c1.low down to c3.high
            elif c3_high < c1_low and (c1_low - c3_high) >= min_gap:
                bearish.append((c3_high, c1_low))

        return bullish, bearish

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

        recent = ohlcv.iloc[-self._lookback:]
        # Find FVGs from historical data excluding the last 2 candles
        historical = recent.iloc[:-2]
        bullish_fvgs, bearish_fvgs = self._find_fvgs(historical, curr_atr)

        curr_price = float(ohlcv["close"].iloc[-1])

        # Check if price is inside a bullish FVG (potential long)
        for fvg_low, fvg_high in reversed(bullish_fvgs):
            if fvg_low <= curr_price <= fvg_high:
                gap_size = fvg_high - fvg_low
                confidence = round(min(0.8, 0.5 + gap_size / curr_atr * 0.15), 3)
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "fvg_low": fvg_low,
                    "fvg_high": fvg_high,
                    "fvg_type": "bullish",
                }

        # Check if price is inside a bearish FVG (potential short)
        for fvg_low, fvg_high in reversed(bearish_fvgs):
            if fvg_low <= curr_price <= fvg_high:
                gap_size = fvg_high - fvg_low
                confidence = round(min(0.8, 0.5 + gap_size / curr_atr * 0.15), 3)
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "fvg_low": fvg_low,
                    "fvg_high": fvg_high,
                    "fvg_type": "bearish",
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
                return self._neutral_signal(symbol, "No FVG fill signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["fvg_low"] - atr * 0.5
                take_profit = entry + atr * 2.5
            else:
                stop_loss = sig["fvg_high"] + atr * 0.5
                take_profit = entry - atr * 2.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"FVG fill {direction}: zone={sig['fvg_low']:.4f}–{sig['fvg_high']:.4f}, "
                    f"type={sig['fvg_type']}, ATR={atr:.6f}"
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
            return True  # FVG filled — exit
        side = str(getattr(position, "side", "long")).lower()
        curr_price = float(ohlcv["close"].iloc[-1])
        # Close if price exits the FVG on the far side
        if side == "long" and curr_price > sig.get("fvg_high", curr_price):
            return True
        if side == "short" and curr_price < sig.get("fvg_low", curr_price):
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
            "take_profit_pct": min(0.08, sl_pct * 2.5),
            "leverage": 3,
        }
