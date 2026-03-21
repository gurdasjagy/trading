"""Momentum Divergence strategy — price vs RSI/MACD divergence combo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MomentumDivergenceStrategy(BaseStrategy):
    """Combined RSI + MACD momentum divergence strategy.

    Fires when both RSI and MACD histogram show divergence against price,
    increasing confidence in a reversal.

    Entry conditions
    ----------------
    * **Long (double bullish divergence)**:
      - Price makes lower low over *lookback* bars.
      - RSI makes higher low (bullish RSI divergence).
      - MACD histogram makes higher low (bullish MACD divergence).
    * **Short (double bearish divergence)**:
      - Price makes higher high.
      - RSI makes lower high.
      - MACD histogram makes lower high.
    """

    _STRATEGY_NAME = "momentum_divergence"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        lookback: int = 30,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._lookback = lookback
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Divergence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_lower_low(series: pd.Series, lookback: int) -> bool:
        """Return True if the last value is the lowest in the window."""
        if len(series) < lookback:
            return False
        return float(series.iloc[-1]) < float(series.iloc[-lookback:-1].min()) * 0.999

    @staticmethod
    def _is_higher_high(series: pd.Series, lookback: int) -> bool:
        if len(series) < lookback:
            return False
        return float(series.iloc[-1]) > float(series.iloc[-lookback:-1].max()) * 1.001

    @staticmethod
    def _is_higher_low(series: pd.Series, lookback: int) -> bool:
        """Return True if the last value is higher than the minimum in the window."""
        if len(series) < lookback:
            return False
        window_min = float(series.iloc[-lookback:-1].min())
        current = float(series.iloc[-1])
        return current > window_min * 1.001

    @staticmethod
    def _is_lower_high(series: pd.Series, lookback: int) -> bool:
        if len(series) < lookback:
            return False
        window_max = float(series.iloc[-lookback:-1].max())
        current = float(series.iloc[-1])
        return current < window_max * 0.999

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._macd_slow + self._macd_signal + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        rsi_series = ta.rsi(closes, length=self._rsi_period)
        macd_df = ta.macd(
            closes, fast=self._macd_fast, slow=self._macd_slow, signal=self._macd_signal
        )
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if rsi_series is None or macd_df is None or atr_series is None:
            return None

        hist_cols = [c for c in macd_df.columns if c.startswith("MACDh")]
        if not hist_cols:
            hist_cols = [macd_df.columns[2]]
        hist_series = macd_df[hist_cols[0]]

        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        if pd.isna(curr_atr):
            return None

        # Bullish double divergence
        price_lower_low = self._is_lower_low(closes, self._lookback)
        rsi_higher_low = self._is_higher_low(rsi_series, self._lookback)
        hist_higher_low = self._is_higher_low(hist_series, self._lookback)

        if price_lower_low and rsi_higher_low and hist_higher_low:
            curr_rsi = float(rsi_series.iloc[-1])
            curr_hist = float(hist_series.iloc[-1])
            if not pd.isna(curr_rsi) and not pd.isna(curr_hist):
                confidence = round(min(0.88, 0.65 + abs(curr_rsi - 50) / 100 * 0.23), 3)
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "rsi": curr_rsi,
                    "macd_hist": curr_hist,
                    "divergence": "double_bullish",
                }

        # Bearish double divergence
        price_higher_high = self._is_higher_high(closes, self._lookback)
        rsi_lower_high = self._is_lower_high(rsi_series, self._lookback)
        hist_lower_high = self._is_lower_high(hist_series, self._lookback)

        if price_higher_high and rsi_lower_high and hist_lower_high:
            curr_rsi = float(rsi_series.iloc[-1])
            curr_hist = float(hist_series.iloc[-1])
            if not pd.isna(curr_rsi) and not pd.isna(curr_hist):
                confidence = round(min(0.88, 0.65 + abs(curr_rsi - 50) / 100 * 0.23), 3)
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "rsi": curr_rsi,
                    "macd_hist": curr_hist,
                    "divergence": "double_bearish",
                }

        # Single divergence (RSI only) as fallback with lower confidence
        if price_lower_low and rsi_higher_low:
            curr_rsi = float(rsi_series.iloc[-1])
            if not pd.isna(curr_rsi):
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": 0.6,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "rsi": curr_rsi,
                    "macd_hist": float(hist_series.iloc[-1]) if not pd.isna(hist_series.iloc[-1]) else 0.0,
                    "divergence": "rsi_bullish",
                }

        if price_higher_high and rsi_lower_high:
            curr_rsi = float(rsi_series.iloc[-1])
            if not pd.isna(curr_rsi):
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": 0.6,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "rsi": curr_rsi,
                    "macd_hist": float(hist_series.iloc[-1]) if not pd.isna(hist_series.iloc[-1]) else 0.0,
                    "divergence": "rsi_bearish",
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
                return self._neutral_signal(symbol, "No momentum divergence")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 2.0
                take_profit = entry + atr * 3.5
            else:
                stop_loss = entry + atr * 2.0
                take_profit = entry - atr * 3.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Momentum divergence {sig['divergence']} {direction}: "
                    f"RSI={sig['rsi']:.1f}, MACD_hist={sig['macd_hist']:.6f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        macd_data = self._calculate_macd(closes, self._macd_fast, self._macd_slow, self._macd_signal)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 70 and macd_data["histogram"] < 0:
            return True
        if side == "short" and rsi < 30 and macd_data["histogram"] > 0:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 3.0),
            "leverage": 2,
        }
