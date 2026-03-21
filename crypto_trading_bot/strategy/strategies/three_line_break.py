from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class ThreeLineBreakStrategy(BaseStrategy):
    """Three Line Break chart reversal signal."""

    _STRATEGY_NAME = "three_line_break"

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
        closes = close.values
        lines = [closes[0]]
        line_dirs = [0]
        for c in closes[1:]:
            if not lines:
                lines.append(c)
                line_dirs.append(0)
                continue
            last3_high = max(lines[-3:]) if len(lines) >= 3 else max(lines)
            last3_low = min(lines[-3:]) if len(lines) >= 3 else min(lines)
            if c > last3_high:
                lines.append(c)
                line_dirs.append(1)
            elif c < last3_low:
                lines.append(c)
                line_dirs.append(-1)
        if len(line_dirs) < 3:
            return None
        last3_dirs = line_dirs[-3:]
        if all(d == 1 for d in last3_dirs):
            direction = "long"
        elif all(d == -1 for d in last3_dirs):
            direction = "short"
        else:
            return None
        streak = sum(1 for d in reversed(line_dirs) if d == line_dirs[-1])
        confidence = round(min(0.9, 0.5 + streak * 0.07), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Three Line Break: {streak} consecutive {'white' if direction=='long' else 'black'} lines",
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
