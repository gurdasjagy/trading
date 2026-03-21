"""Mean-reversion strategy — Bollinger Bands(20, 2) + RSI(14) + volume filter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Bollinger-Band mean-reversion strategy with volume confirmation.

    Entry conditions
    ----------------
    * **Long**: price touches (or breaches) the lower Bollinger Band AND
      ``RSI < 30`` (oversold) AND current volume exceeds 1.5× the
      20-bar average (excluding the current bar).
    * **Short**: price touches (or breaches) the upper Bollinger Band AND
      ``RSI > 70`` (overbought) AND volume exceeds 1.5× the average.

    Primary timeframe: 15 m.
    """

    _STRATEGY_NAME = "mean_reversion"
    _VOLUME_MULTIPLIER = 1.5

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        volume_avg_period: int = 20,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._volume_avg_period = volume_avg_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """Detect a mean-reversion opportunity from *ohlcv* data.

        Returns a signal dict or ``None`` when conditions are not met.
        """
        min_rows = max(self._bb_period, self._rsi_period) + self._volume_avg_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]

        bb = ta.bbands(closes, length=self._bb_period, std=self._bb_std)
        rsi_series = ta.rsi(closes, length=self._rsi_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=14)

        if bb is None or rsi_series is None:
            return None

        # pandas_ta names columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        lower_col = [c for c in bb.columns if c.startswith("BBL")]
        upper_col = [c for c in bb.columns if c.startswith("BBU")]
        if not lower_col or not upper_col:
            return None

        lower_bb = float(bb[lower_col[0]].iloc[-1])
        upper_bb = float(bb[upper_col[0]].iloc[-1])
        curr_rsi = float(rsi_series.iloc[-1])
        entry_price = float(closes.iloc[-1])
        curr_atr = float(atr_series.iloc[-1]) if atr_series is not None else 0.0

        if pd.isna(lower_bb) or pd.isna(upper_bb) or pd.isna(curr_rsi):
            return None

        if not self._volume_confirmed(ohlcv):
            return None

        # Long: price at or below lower band + oversold RSI
        if entry_price <= lower_bb and curr_rsi < self._rsi_oversold:
            distance = (lower_bb - entry_price) / lower_bb if lower_bb > 0 else 0.0
            confidence = round(
                min(1.0, 0.55 + distance * 5 + (self._rsi_oversold - curr_rsi) / 60), 3
            )
            return {
                "symbol": symbol,
                "direction": "long",
                "entry_price": entry_price,
                "atr": curr_atr,
                "confidence": confidence,
                "strategy": self._STRATEGY_NAME,
                "timeframe": self._timeframe,
            }

        # Short: price at or above upper band + overbought RSI
        if entry_price >= upper_bb and curr_rsi > self._rsi_overbought:
            distance = (entry_price - upper_bb) / upper_bb if upper_bb > 0 else 0.0
            confidence = round(
                min(1.0, 0.55 + distance * 5 + (curr_rsi - self._rsi_overbought) / 60), 3
            )
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
        """Fetch 15 m OHLCV and return a mean-reversion :class:`Signal`."""
        try:
            limit = self._bb_period + self._volume_avg_period + 30
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=limit)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "BB/RSI/volume conditions not met")

            direction = sig["direction"]
            atr = sig["atr"]
            entry_price = sig["entry_price"]
            if direction == "long":
                stop_loss = entry_price - atr * 2.0
                take_profit = entry_price + atr * 3.0
            else:
                stop_loss = entry_price + atr * 2.0
                take_profit = entry_price - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"BB mean-reversion {direction}: " f"price={entry_price:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when price reverts to the middle band or RSI normalises."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        limit = self._bb_period + self._volume_avg_period + 10
        ohlcv = await self._get_ohlcv(symbol, limit=limit)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"]
        bb = ta.bbands(closes, length=self._bb_period, std=self._bb_std)
        rsi_series = ta.rsi(closes, length=self._rsi_period)
        if bb is None or rsi_series is None:
            return False
        middle_col = [c for c in bb.columns if c.startswith("BBM")]
        if not middle_col:
            return False
        middle = float(bb[middle_col[0]].iloc[-1])
        last_close = float(closes.iloc[-1])
        rsi = float(rsi_series.iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and (last_close >= middle or rsi >= 55):
            return True
        if side == "short" and (last_close <= middle or rsi <= 45):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        limit = self._bb_period + self._volume_avg_period + 30
        ohlcv = await self._get_ohlcv(symbol, limit=limit)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.06, sl_pct * 1.5),
            "leverage": 2,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _volume_confirmed(self, ohlcv: pd.DataFrame) -> bool:
        """Return ``True`` if the latest bar's volume exceeds 1.5× the prior average."""
        volumes = ohlcv["volume"].values
        if len(volumes) < self._volume_avg_period + 1:
            return False
        # Exclude the current bar so we don't self-referentially inflate the average
        avg = float(volumes[-(self._volume_avg_period + 1) : -1].mean())
        return avg > 0 and float(volumes[-1]) >= avg * self._VOLUME_MULTIPLIER
