from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class EntropyStrategy(BaseStrategy):
    """Shannon entropy of return distribution regime filter."""

    _STRATEGY_NAME = "entropy_strategy"

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
        returns = close.pct_change().dropna().values[-20:]
        if len(returns) < 10:
            return None
        counts, _ = np.histogram(returns, bins=10)
        probs = counts / (counts.sum() + 1e-9)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = float(np.log(10))
        norm_entropy = entropy / max_entropy
        if norm_entropy > 0.85:
            return None
        close_list = close.tolist()
        rsi = self._calculate_rsi(close_list, 14)
        if rsi < 35:
            direction = "long"
        elif rsi > 65:
            direction = "short"
        else:
            return None
        confidence = round(min(0.9, 0.5 + (0.85 - norm_entropy) * 0.5), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Entropy={norm_entropy:.2f}, RSI={rsi:.1f}",
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
