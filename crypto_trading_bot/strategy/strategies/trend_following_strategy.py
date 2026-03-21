"""Trend-following strategy — MACD + ADX(14) with 4h/15m multi-timeframe analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    """MACD histogram direction + ADX(14) trend-strength strategy.

    Entry conditions
    ----------------
    * ADX > 25 (strong trend present).
    * **Long**: MACD histogram is positive AND trending upward.
    * **Short**: MACD histogram is negative AND trending downward.

    Multi-timeframe
    ---------------
    Trend direction is confirmed on the 4-hour chart; entries are timed on
    the 15-minute chart.  In :meth:`analyze` only the primary timeframe
    DataFrame is used; :meth:`generate_signal` fetches both.
    """

    _STRATEGY_NAME = "trend_following"
    _ADX_THRESHOLD = 25.0

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        trend_timeframe: str = "4h",
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._trend_timeframe = trend_timeframe

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """Detect a trend-following opportunity from *ohlcv* data.

        Returns a signal dict or ``None`` when conditions are not met.
        """
        min_rows = self._macd_slow + self._adx_period + 10
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]

        macd_df = ta.macd(
            closes,
            fast=self._macd_fast,
            slow=self._macd_slow,
            signal=self._macd_signal,
        )
        adx_df = ta.adx(ohlcv["high"], ohlcv["low"], closes, length=self._adx_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=14)

        if macd_df is None or adx_df is None:
            return None

        # Locate the histogram column (MACDh_fast_slow_signal)
        hist_col = [c for c in macd_df.columns if c.startswith("MACDh")]
        adx_col = [c for c in adx_df.columns if c.startswith("ADX")]
        if not hist_col or not adx_col:
            return None

        curr_hist = float(macd_df[hist_col[0]].iloc[-1])
        prev_hist = float(macd_df[hist_col[0]].iloc[-2])
        curr_adx = float(adx_df[adx_col[0]].iloc[-1])
        entry_price = float(closes.iloc[-1])
        curr_atr = float(atr_series.iloc[-1]) if atr_series is not None else 0.0

        if pd.isna(curr_hist) or pd.isna(prev_hist) or pd.isna(curr_adx):
            return None

        if curr_adx < self._adx_threshold:
            return None

        # Histogram rising and positive → bullish momentum
        if curr_hist > 0 and curr_hist > prev_hist:
            confidence = self._compute_confidence(curr_adx, curr_hist, "long")
            return {
                "symbol": symbol,
                "direction": "long",
                "entry_price": entry_price,
                "atr": curr_atr,
                "confidence": confidence,
                "strategy": self._STRATEGY_NAME,
                "timeframe": self._timeframe,
            }

        # Histogram falling and negative → bearish momentum
        if curr_hist < 0 and curr_hist < prev_hist:
            confidence = self._compute_confidence(curr_adx, curr_hist, "short")
            return {
                "symbol": symbol,
                "direction": "short",
                "entry_price": entry_price,
                "atr": curr_atr,
                "confidence": confidence,
                "strategy": self._STRATEGY_NAME,
                "timeframe": self._timeframe,
            }

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        """Fetch 15m and 4h OHLCV data, confirm trend on 4h, enter on 15m."""
        try:
            ohlcv_15m = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv_15m, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "MACD/ADX conditions not met on 15m")

            # 4-hour trend confirmation: MACD histogram must agree with direction
            ohlcv_4h = await self._get_ohlcv(symbol, self._trend_timeframe, limit=80)
            if not ohlcv_4h.empty and len(ohlcv_4h) >= self._macd_slow + self._adx_period + 10:
                macd_4h = ta.macd(
                    ohlcv_4h["close"],
                    fast=self._macd_fast,
                    slow=self._macd_slow,
                    signal=self._macd_signal,
                )
                if macd_4h is not None:
                    hist_col = [c for c in macd_4h.columns if c.startswith("MACDh")]
                    if hist_col:
                        hist_4h = float(macd_4h[hist_col[0]].iloc[-1])
                        if not pd.isna(hist_4h):
                            if sig["direction"] == "long" and hist_4h < 0:
                                return self._neutral_signal(symbol, "4h MACD opposes long signal")
                            if sig["direction"] == "short" and hist_4h > 0:
                                return self._neutral_signal(symbol, "4h MACD opposes short signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry_price = sig["entry_price"]
            if direction == "long":
                stop_loss = entry_price - atr * 2.0
                take_profit = entry_price + atr * 5.0
            else:
                stop_loss = entry_price + atr * 2.0
                take_profit = entry_price - atr * 5.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"MACD histogram {direction}, ADX confirmed strong trend, " f"ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when ADX weakens below 20 or MACD histogram reverses."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"]
        macd_df = ta.macd(
            closes, fast=self._macd_fast, slow=self._macd_slow, signal=self._macd_signal
        )
        adx_df = ta.adx(ohlcv["high"], ohlcv["low"], closes, length=self._adx_period)
        if macd_df is None or adx_df is None:
            return False
        hist_col = [c for c in macd_df.columns if c.startswith("MACDh")]
        adx_col = [c for c in adx_df.columns if c.startswith("ADX")]
        if not hist_col or not adx_col:
            return False
        hist = float(macd_df[hist_col[0]].iloc[-1])
        adx = float(adx_df[adx_col[0]].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if pd.isna(hist) or pd.isna(adx):
            return False
        if adx < 20:
            return True
        if side == "long" and hist < 0:
            return True
        if side == "short" and hist > 0:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.025
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(adx: float, histogram: float, direction: str) -> float:
        """Derive a [0, 1] confidence score from ADX strength and histogram magnitude."""
        adx_score = min(1.0, max(0.0, (adx - 25.0) / 50.0))
        hist_score = min(1.0, abs(histogram) * 500)
        confidence = 0.55 + 0.3 * adx_score + 0.15 * hist_score
        return round(min(1.0, max(0.0, confidence)), 3)
