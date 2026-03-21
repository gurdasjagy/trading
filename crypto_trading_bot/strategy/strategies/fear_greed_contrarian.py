from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class FearGreedContrarianStrategy(BaseStrategy):
    """Contrarian strategy based on fear/greed proxy from RSI + volume."""

    _STRATEGY_NAME = "fear_greed_contrarian"

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
        volume = df["volume"]
        close_list = close.tolist()
        rsi = self._calculate_rsi(close_list, 14)
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        cur_vol = float(volume.iloc[-1])
        if np.isnan(rsi) or np.isnan(avg_vol) or avg_vol == 0:
            return None
        high_vol = cur_vol > avg_vol
        if rsi < 25 and high_vol:
            direction = "long"
            confidence = round(min(0.9, 0.5 + (25 - rsi) / 25 * 0.4), 3)
        elif rsi > 75 and high_vol:
            direction = "short"
            confidence = round(min(0.9, 0.5 + (rsi - 75) / 25 * 0.4), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Fear/Greed contrarian: RSI={rsi:.1f}, vol_ratio={cur_vol/avg_vol:.2f}",
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
