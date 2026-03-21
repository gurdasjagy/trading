from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class KagiReversalStrategy(BaseStrategy):
    """Kagi chart reversal at shoulder/waist levels."""

    _STRATEGY_NAME = "kagi_reversal"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.7, "ranging": 0.3}
        self._min_rows = 30

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        atr = self._calculate_atr(df)
        if atr == 0:
            return None
        closes = close.values
        direction_k = 1 if closes[1] >= closes[0] else -1
        turning_point = closes[0]
        shoulders = []
        waists = []
        for c in closes[1:]:
            if direction_k == 1:
                if c > turning_point:
                    turning_point = c
                elif turning_point - c >= atr:
                    shoulders.append(turning_point)
                    direction_k = -1
                    turning_point = c
            else:
                if c < turning_point:
                    turning_point = c
                elif c - turning_point >= atr:
                    waists.append(turning_point)
                    direction_k = 1
                    turning_point = c
        cur = float(close.iloc[-1])
        if shoulders and cur > shoulders[-1]:
            direction = "long"
            confidence = round(min(0.9, 0.5 + (cur - shoulders[-1]) / (atr + 1e-9) * 0.2), 3)
        elif waists and cur < waists[-1]:
            direction = "short"
            confidence = round(min(0.9, 0.5 + (waists[-1] - cur) / (atr + 1e-9) * 0.2), 3)
        else:
            return None
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": cur, "atr": atr,
            "reasoning": f"Kagi breakout: direction={direction}, shoulders={len(shoulders)}, waists={len(waists)}",
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
