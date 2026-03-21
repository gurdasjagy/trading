from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class PointFigureStrategy(BaseStrategy):
    """Point & Figure chart proxy signal."""

    _STRATEGY_NAME = "point_figure"

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
        box_size = atr / 4.0
        reversal = 3 * box_size
        closes = close.values
        columns = []
        cur_dir = 1 if closes[1] >= closes[0] else -1
        cur_level = closes[0]
        box_count = 0
        for c in closes[1:]:
            if cur_dir == 1:
                if c >= cur_level + box_size:
                    steps = int((c - cur_level) / box_size)
                    box_count += steps
                    cur_level += steps * box_size
                elif c <= cur_level - reversal:
                    columns.append((cur_dir, box_count))
                    cur_dir = -1
                    steps = int((cur_level - c) / box_size)
                    box_count = steps
                    cur_level -= steps * box_size
            else:
                if c <= cur_level - box_size:
                    steps = int((cur_level - c) / box_size)
                    box_count += steps
                    cur_level -= steps * box_size
                elif c >= cur_level + reversal:
                    columns.append((cur_dir, box_count))
                    cur_dir = 1
                    steps = int((c - cur_level) / box_size)
                    box_count = steps
                    cur_level += steps * box_size
        columns.append((cur_dir, box_count))
        if len(columns) < 1:
            return None
        last_dir, last_count = columns[-1]
        if last_count >= 3:
            direction = "long" if last_dir == 1 else "short"
            confidence = round(min(0.9, 0.5 + last_count * 0.05), 3)
        else:
            return None
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"P&F: {last_count} boxes in {'X' if last_dir==1 else 'O'} column",
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
