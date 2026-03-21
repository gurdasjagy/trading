from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class WyckoffStrategy(BaseStrategy):
    """Wyckoff accumulation/distribution detection."""

    _STRATEGY_NAME = "wyckoff"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.4, "ranging": 0.6}
        self._min_rows = 40

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        high20 = float(close.rolling(20).max().iloc[-1])
        low20 = float(close.rolling(20).min().iloc[-1])
        cur = float(close.iloc[-1])
        rng = high20 - low20
        if rng == 0:
            return None
        in_range = (cur >= low20) and (cur <= high20)
        if not in_range:
            return None
        down_days = close < close.shift(1)
        up_days = close >= close.shift(1)
        vol_down = (volume * down_days.astype(float)).rolling(10).mean()
        vol_up = (volume * up_days.astype(float)).rolling(10).mean()
        avg_vol_down = float(vol_down.iloc[-1])
        avg_vol_up = float(vol_up.iloc[-1])
        if np.isnan(avg_vol_down) or np.isnan(avg_vol_up):
            return None
        if avg_vol_down < avg_vol_up * 0.8 and cur < low20 + rng * 0.3:
            direction = "long"
            confidence = round(min(0.9, 0.5 + (1 - avg_vol_down / (avg_vol_up + 1e-9)) * 0.3), 3)
        elif avg_vol_up < avg_vol_down * 0.8 and cur > high20 - rng * 0.3:
            direction = "short"
            confidence = round(min(0.9, 0.5 + (1 - avg_vol_up / (avg_vol_down + 1e-9)) * 0.3), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": cur, "atr": atr,
            "reasoning": f"Wyckoff: vol_down={avg_vol_down:.2f}, vol_up={avg_vol_up:.2f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=50)
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
        df = await self._get_ohlcv(symbol, limit=50)
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
