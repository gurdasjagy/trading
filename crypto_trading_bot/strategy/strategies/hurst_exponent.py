from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class HurstExponentStrategy(BaseStrategy):
    """Hurst exponent regime detection for mean reversion vs trend following."""

    _STRATEGY_NAME = "hurst_exponent"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.5, "ranging": 0.5}
        self._min_rows = 80

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"].values.astype(float)
        lags = [8, 16, 32, 64]
        rs_vals = []
        for lag in lags:
            if len(close) < lag:
                continue
            sub = close[-lag:]
            mean_sub = np.mean(sub)
            dev = np.cumsum(sub - mean_sub)
            r = np.max(dev) - np.min(dev)
            s = np.std(sub)
            if s == 0:
                continue
            rs_vals.append(np.log(r / s + 1e-9) / np.log(lag))
        if len(rs_vals) < 2:
            return None
        H = float(np.mean(rs_vals))
        close_list = close.tolist()
        atr = self._calculate_atr(df)
        entry = float(close[-1])
        if H < 0.45:
            rsi = self._calculate_rsi(close_list, 14)
            if rsi < 30:
                direction = "long"
            elif rsi > 70:
                direction = "short"
            else:
                return None
            confidence = round(0.5 + (0.5 - H) * 0.8, 3)
        elif H > 0.55:
            ema9 = self._calculate_ema(close_list, 9)
            ema21 = self._calculate_ema(close_list, 21)
            if ema9 > ema21:
                direction = "long"
            else:
                direction = "short"
            confidence = round(0.5 + (H - 0.5) * 0.8, 3)
        else:
            return None
        confidence = round(min(0.9, confidence), 3)
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Hurst={H:.2f}, direction={direction}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=90)
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
        df = await self._get_ohlcv(symbol, limit=90)
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
