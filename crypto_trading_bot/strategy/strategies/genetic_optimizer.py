from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GeneticOptimizerStrategy(BaseStrategy):
    """Evolve EMA window selection via rolling performance tracking."""

    _STRATEGY_NAME = "genetic_optimizer"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.7, "ranging": 0.3}
        self._min_rows = 60
        self._windows = [9, 13, 21, 34, 55]

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        close_arr = close.values
        best_window = None
        best_perf = -np.inf
        for w in self._windows:
            if len(close_arr) < w + 20:
                continue
            ema_now = self._calculate_ema(close_arr[-20:].tolist(), w)
            ema_then = self._calculate_ema(close_arr[-40:-20].tolist(), w)
            perf = ema_now - ema_then
            if perf > best_perf:
                best_perf = perf
                best_window = w
        if best_window is None:
            return None
        close_list = close.tolist()
        ema_fast = self._calculate_ema(close_list, best_window)
        ema_slow = self._calculate_ema(close_list, best_window * 2)
        direction = "long" if ema_fast > ema_slow else "short"
        diff = abs(ema_fast - ema_slow) / (ema_slow + 1e-9)
        confidence = round(min(0.9, 0.5 + diff * 5), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Genetic best_window={best_window}, perf={best_perf:.4f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=70)
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
        df = await self._get_ohlcv(symbol, limit=70)
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
