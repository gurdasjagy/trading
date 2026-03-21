from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class FootprintChartStrategy(BaseStrategy):
    """Footprint chart absorption signal."""

    _STRATEGY_NAME = "footprint_chart"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.3, "ranging": 0.7}
        self._min_rows = 30

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        buy_vol = volume.where(close >= open_, 0.0)
        sell_vol = volume.where(close < open_, 0.0)
        hl_range = high - low
        avg_range = float(hl_range.rolling(10).mean().iloc[-1])
        avg_vol = float(volume.rolling(10).mean().iloc[-1])
        cur_range = float(hl_range.iloc[-1])
        cur_vol = float(volume.iloc[-1])
        cur_buy = float(buy_vol.iloc[-1])
        cur_sell = float(sell_vol.iloc[-1])
        if np.isnan(avg_range) or avg_range == 0:
            return None
        high_volume = cur_vol > avg_vol * 1.5
        small_range = cur_range < avg_range * 0.5
        if high_volume and small_range:
            direction = "long" if cur_buy > cur_sell else "short"
            confidence = round(min(0.9, 0.5 + (cur_vol / avg_vol - 1.5) * 0.1), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Footprint absorption: cur_vol/avg={cur_vol/avg_vol:.2f}, range ratio={cur_range/avg_range:.2f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=40)
            sig = self.analyze(df, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No signal")
            return Signal(
                symbol=symbol,
                direction=sig["direction"],
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=sig["reasoning"],
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        df = await self._get_ohlcv(symbol, limit=40)
        if df.empty:
            return False
        sig = self.analyze(df, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        return (side == "long" and sig["direction"] == "short") or (
            side == "short" and sig["direction"] == "long"
        )

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        df = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(df)
        last = float(df["close"].iloc[-1]) if not df.empty else 1.0
        sl = (atr / last * 1.5) if last > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl),
            "take_profit_pct": min(0.08, sl * 2),
            "leverage": 3,
        }
