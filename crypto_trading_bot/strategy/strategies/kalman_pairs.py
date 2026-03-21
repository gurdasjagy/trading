from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class KalmanPairsStrategy(BaseStrategy):
    """Kalman-filter-inspired spread mean reversion."""

    _STRATEGY_NAME = "kalman_pairs"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.3, "ranging": 0.7}
        self._min_rows = 60

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        ewm = close.ewm(span=20).mean()
        x = ewm.values
        y = close.values
        cov_xy = float(np.cov(x[-20:], y[-20:])[0, 1])
        var_x = float(np.var(x[-20:]))
        hedge = cov_xy / (var_x + 1e-9)
        spread = y - x * hedge
        s_mean = float(np.mean(spread[-20:]))
        s_std = float(np.std(spread[-20:]))
        s_last = float(spread[-1])
        if s_std == 0:
            return None
        z = (s_last - s_mean) / s_std
        if z < -2.0:
            direction = "long"
        elif z > 2.0:
            direction = "short"
        else:
            return None
        confidence = round(min(0.9, 0.5 + min(1.0, (abs(z) - 2.0) / 2.0) * 0.4), 3)
        atr = self._calculate_atr(df)
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": float(close.iloc[-1]), "atr": atr,
            "reasoning": f"Kalman spread z={z:.2f}",
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
