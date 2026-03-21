from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class IVSurfaceStrategy(BaseStrategy):
    """IV proxy from price range compared to rolling mean."""

    _STRATEGY_NAME = "iv_surface"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.4, "ranging": 0.6}
        self._min_rows = 30

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        iv_proxy = (df["high"] - df["low"]) / (close + 1e-9) * np.sqrt(252)
        mean_iv = iv_proxy.rolling(20).mean()
        std_iv = iv_proxy.rolling(20).std()
        cur_iv = float(iv_proxy.iloc[-1])
        m = float(mean_iv.iloc[-1])
        s = float(std_iv.iloc[-1])
        if np.isnan(cur_iv) or np.isnan(m) or s == 0:
            return None
        z = (cur_iv - m) / s
        if z > 1.5:
            close_chg = float(close.iloc[-1]) - float(close.iloc[-2])
            direction = "short" if close_chg > 0 else "long"
            confidence = round(min(0.9, 0.5 + (z - 1.5) * 0.15), 3)
        elif z < -1.5:
            close_chg = float(close.iloc[-1]) - float(close.iloc[-2])
            direction = "long" if close_chg > 0 else "short"
            confidence = round(min(0.9, 0.5 + (abs(z) - 1.5) * 0.15), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"IV proxy z={z:.2f}, cur_iv={cur_iv:.4f}",
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
