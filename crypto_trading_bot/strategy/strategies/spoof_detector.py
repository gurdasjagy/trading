from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class SpoofDetectorStrategy(BaseStrategy):
    """Detect volume-reversal spoofing patterns, trade normal EMA signal otherwise."""

    _STRATEGY_NAME = "spoof_detector"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.6, "ranging": 0.4}
        self._min_rows = 20

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        avg_vol = float(volume.rolling(10).mean().iloc[-1])
        spoofed = False
        for i in range(-3, 0):
            v = float(volume.iloc[i])
            c_chg = float(close.iloc[i]) - float(close.iloc[i - 1])
            c_next = float(close.iloc[i + 1]) - float(close.iloc[i]) if i < -1 else 0.0
            if v > 2 * avg_vol and abs(c_chg) > 0 and c_chg * c_next < 0:
                spoofed = True
                break
        if spoofed:
            return None
        close_list = close.tolist()
        ema9 = self._calculate_ema(close_list, 9)
        ema21 = self._calculate_ema(close_list, 21)
        direction = "long" if ema9 > ema21 else "short"
        diff = abs(ema9 - ema21) / (ema21 + 1e-9)
        confidence = round(min(0.9, 0.5 + diff * 5), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"No spoof detected, EMA9={ema9:.4f} vs EMA21={ema21:.4f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=30)
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
        df = await self._get_ohlcv(symbol, limit=30)
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
