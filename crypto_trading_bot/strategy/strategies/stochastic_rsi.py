"""Stochastic RSI strategy — oversold/overbought crossover entries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class StochasticRSIStrategy(BaseStrategy):
    """Stochastic RSI strategy.

    Computes StochRSI via pandas_ta and trades crossovers:

    Entry conditions
    ----------------
    * **Long**: StochRSI %K crosses above %D from below 20 (oversold).
    * **Short**: StochRSI %K crosses below %D from above 80 (overbought).

    An EMA trend filter is applied — only take longs above EMA and shorts below.
    """

    _STRATEGY_NAME = "stochastic_rsi"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        ema_period: int = 50,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._rsi_period = rsi_period
        self._stoch_period = stoch_period
        self._smooth_k = smooth_k
        self._smooth_d = smooth_d
        self._oversold = oversold
        self._overbought = overbought
        self._ema_period = ema_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._rsi_period + self._stoch_period + self._ema_period + 10
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        stochrsi_df = ta.stochrsi(
            closes,
            length=self._rsi_period,
            rsi_length=self._rsi_period,
            k=self._smooth_k,
            d=self._smooth_d,
        )
        ema_series = ta.ema(closes, length=self._ema_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if stochrsi_df is None or ema_series is None or atr_series is None:
            return None

        k_cols = [c for c in stochrsi_df.columns if "STOCHRSIk" in c]
        d_cols = [c for c in stochrsi_df.columns if "STOCHRSId" in c]
        if not k_cols or not d_cols:
            return None

        curr_k = float(stochrsi_df[k_cols[0]].iloc[-1])
        prev_k = float(stochrsi_df[k_cols[0]].iloc[-2])
        curr_d = float(stochrsi_df[d_cols[0]].iloc[-1])
        prev_d = float(stochrsi_df[d_cols[0]].iloc[-2])
        curr_ema = float(ema_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        for val in (curr_k, prev_k, curr_d, prev_d, curr_ema, curr_atr):
            if pd.isna(val):
                return None

        cross_up = prev_k <= prev_d and curr_k > curr_d
        cross_down = prev_k >= prev_d and curr_k < curr_d

        direction: Optional[str] = None
        if cross_up and curr_k < self._oversold + 15 and curr_price > curr_ema:
            direction = "long"
        elif cross_down and curr_k > self._overbought - 15 and curr_price < curr_ema:
            direction = "short"

        if direction is None:
            return None

        # Confidence based on distance from extreme
        if direction == "long":
            extreme_dist = (self._oversold - min(curr_k, self._oversold)) / self._oversold
        else:
            extreme_dist = (max(curr_k, self._overbought) - self._overbought) / (100 - self._overbought)

        confidence = round(min(0.85, 0.5 + extreme_dist * 0.35), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "stochrsi_k": curr_k,
            "stochrsi_d": curr_d,
            "ema": curr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No StochRSI crossover signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 2.5
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 2.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"StochRSI {direction}: K={sig['stochrsi_k']:.1f}, "
                    f"D={sig['stochrsi_d']:.1f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        k = sig["stochrsi_k"]
        if side == "long" and k > self._overbought:
            return True
        if side == "short" and k < self._oversold:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2,
        }
