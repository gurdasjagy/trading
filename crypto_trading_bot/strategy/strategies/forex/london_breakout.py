"""London Session Breakout strategy for Gold — trade Asian-range breakouts at London open."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class LondonBreakoutStrategy(BaseStrategy):
    """London Session Breakout for XAU/USD.

    Gold is most volatile during the London session (08:00–16:00 GMT).
    This strategy identifies the Asian session range (00:00–08:00 GMT),
    then trades breakouts above/below that range when London opens.

    Signal logic
    ------------
    * Identify the high/low of the Asian session candles in the OHLCV data.
    * When a candle close breaks above the Asian high → long.
    * When a candle close breaks below the Asian low → short.
    * Confidence is based on the range width relative to ATR.
    """

    _STRATEGY_NAME = "london_breakout"

    # GMT hours for session boundaries (inclusive start, exclusive end)
    _ASIAN_START_H = 0
    _ASIAN_END_H = 8
    _LONDON_START_H = 8
    _LONDON_END_H = 16

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        atr_period: int = 14,
        min_range_atr_ratio: float = 0.3,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_period = atr_period
        self._min_range_atr_ratio = min_range_atr_ratio

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if ohlcv is None or len(ohlcv) < 50:
            return None

        closes = ohlcv["close"].tolist()

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        # Determine timestamp column
        ts_col = None
        for col in ("timestamp", "time", "date", "datetime"):
            if col in ohlcv.columns:
                ts_col = col
                break

        if ts_col is None:
            # Fall back to index if it is datetime-like
            if isinstance(ohlcv.index, pd.DatetimeIndex):
                timestamps = ohlcv.index
            else:
                return None
        else:
            timestamps = pd.to_datetime(ohlcv[ts_col], utc=True)

        now_hour = datetime.now(timezone.utc).hour
        in_london = self._LONDON_START_H <= now_hour < self._LONDON_END_H

        if not in_london:
            return None

        # Collect Asian session bars
        asian_highs: List[float] = []
        asian_lows: List[float] = []
        for i, ts in enumerate(timestamps):
            try:
                h = ts.hour
            except AttributeError:
                continue
            if self._ASIAN_START_H <= h < self._ASIAN_END_H:
                asian_highs.append(float(ohlcv["high"].iloc[i]))
                asian_lows.append(float(ohlcv["low"].iloc[i]))

        if not asian_highs or not asian_lows:
            return None

        asian_high = max(asian_highs)
        asian_low = min(asian_lows)
        range_size = asian_high - asian_low

        if range_size <= 0:
            return None

        range_atr_ratio = range_size / atr

        curr_price = closes[-1]
        prev_price = closes[-2]

        direction: Optional[str] = None
        # Breakout: previous close was at or below Asian high and current close broke above
        # (uses close prices; price gaps through the level are implicitly handled since
        # curr_price > asian_high will still be True even if the gap skipped prev_price)
        if curr_price > asian_high and prev_price <= asian_high:
            direction = "long"
        elif curr_price < asian_low and prev_price >= asian_low:
            direction = "short"

        if direction is None:
            return None

        # Confidence based on how significant the range is relative to ATR
        confidence = round(min(0.9, 0.5 + min(range_atr_ratio, 1.0) * 0.4), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "asian_high": asian_high,
            "asian_low": asian_low,
            "range_size": range_size,
            "range_atr_ratio": range_atr_ratio,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No London breakout detected")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["asian_low"]
                take_profit = entry + atr * 2.0
            else:
                stop_loss = sig["asian_high"]
                take_profit = entry - atr * 2.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"London breakout {direction}: price={entry:.4f}, "
                    f"Asian range=[{sig['asian_low']:.4f}, {sig['asian_high']:.4f}], "
                    f"range/ATR={sig['range_atr_ratio']:.2f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
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
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close if price reverts inside the Asian range
        if side == "long" and curr_price < sig.get("asian_high", curr_price):
            return True
        if side == "short" and curr_price > sig.get("asian_low", curr_price):
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
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 3,
        }
