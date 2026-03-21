from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class BayesianStrategy(BaseStrategy):
    """Bayesian posterior update using RSI, MACD, and volume signals."""

    _STRATEGY_NAME = "bayesian_strategy"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.5, "ranging": 0.5}
        self._min_rows = 40

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        close_list = close.tolist()
        rsi = self._calculate_rsi(close_list, 14)
        ema12 = self._calculate_ema(close_list, 12)
        ema26 = self._calculate_ema(close_list, 26)
        macd_val = ema12 - ema26
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        cur_vol = float(volume.iloc[-1])
        if any(np.isnan(v) for v in [rsi, ema12, ema26]):
            return None
        prior = 0.5
        p_up = prior
        p_up = p_up * (0.7 if rsi < 40 else 0.3 if rsi > 60 else 0.5) / (
            p_up * (0.7 if rsi < 40 else 0.3 if rsi > 60 else 0.5)
            + (1 - p_up) * (0.3 if rsi < 40 else 0.7 if rsi > 60 else 0.5)
        )
        lk_macd = 0.65 if macd_val > 0 else 0.35
        p_up = p_up * lk_macd / (p_up * lk_macd + (1 - p_up) * (1 - lk_macd))
        lk_vol = 0.6 if cur_vol > avg_vol else 0.4
        p_up = p_up * lk_vol / (p_up * lk_vol + (1 - p_up) * (1 - lk_vol))
        if p_up > 0.65:
            direction = "long"
            confidence = round(min(0.9, p_up), 3)
        elif p_up < 0.35:
            direction = "short"
            confidence = round(min(0.9, 1 - p_up), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Bayesian P(up)={p_up:.3f}",
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
