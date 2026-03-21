"""Gold Mean Reversion strategy — trade deviations from the 200-period MA."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldMeanReversionStrategy(BaseStrategy):
    """Gold Mean Reversion strategy.

    Gold tends to revert to its 200-period moving average.  When price
    deviates more than 2 standard deviations from that MA, enter a mean
    reversion trade.  Works best in ranging/consolidating markets.

    Signal logic
    ------------
    * Compute 200-period SMA and rolling standard deviation.
    * Long when price is below ``MA - 2×σ`` and beginning to recover.
    * Short when price is above ``MA + 2×σ`` and beginning to pull back.
    * Confidence scales with the number of standard deviations from the mean.
    """

    _STRATEGY_NAME = "gold_mean_reversion"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        ma_period: int = 200,
        std_period: int = 20,
        std_multiplier: float = 2.0,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._ma_period = ma_period
        self._std_period = std_period
        self._std_multiplier = std_multiplier
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._ma_period + self._std_period + 5
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        ma = float(closes.rolling(self._ma_period).mean().iloc[-1])
        std = float(closes.rolling(self._std_period).std().iloc[-1])

        if pd.isna(ma) or pd.isna(std) or std <= 0:
            return None

        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])

        upper_band = ma + self._std_multiplier * std
        lower_band = ma - self._std_multiplier * std

        z_score = (curr_price - ma) / std

        direction: Optional[str] = None
        if curr_price < lower_band and prev_price < curr_price:
            direction = "long"
        elif curr_price > upper_band and prev_price > curr_price:
            direction = "short"

        if direction is None:
            return None

        # Confidence based on how far past the band price has moved
        excess = abs(abs(z_score) - self._std_multiplier)
        confidence = round(min(0.9, 0.55 + min(excess, 1.5) * 0.2), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "ma": round(ma, 4),
            "upper_band": round(upper_band, 4),
            "lower_band": round(lower_band, 4),
            "z_score": round(z_score, 3),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=300)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No mean reversion signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            # Target: return to the mean
            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = sig["ma"]
            else:
                stop_loss = entry + atr * 1.5
                take_profit = sig["ma"]

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Mean reversion {direction}: z={sig['z_score']:.2f}, "
                    f"MA={sig['ma']:.4f}, ATR={atr:.4f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=300)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close when price returns to within 0.5σ of mean
        if side == "long" and curr_price >= sig["ma"] * 0.999:
            return True
        if side == "short" and curr_price <= sig["ma"] * 1.001:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=300)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2,
        }
