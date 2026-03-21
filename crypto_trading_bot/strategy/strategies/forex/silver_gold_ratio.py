"""Silver/Gold Ratio strategy — trade reversion of an extreme gold/silver ratio."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# Historical mean of the gold/silver ratio (~80–90 in recent years)
_RATIO_MEAN = 82.0
_RATIO_STD = 10.0  # approximate 1σ band


class SilverGoldRatioStrategy(BaseStrategy):
    """Gold/Silver Ratio Trading strategy.

    The gold/silver ratio (XAU price / XAG price) tends to mean-revert.
    When the ratio is extremely high, silver is undervalued vs gold (long
    silver / short gold).  When the ratio is extremely low, gold is
    undervalued vs silver (long gold / short silver).

    Signal logic
    ------------
    * If only XAU/USD OHLCV is provided, use the current price vs
      ``xag_price`` argument to compute the ratio.
    * If ``xag_ohlcv`` is provided, compute the ratio from both series.
    * Ratio > ``ratio_mean + ratio_std_mult × std`` → short gold (or long silver).
    * Ratio < ``ratio_mean - ratio_std_mult × std`` → long gold (or short silver).
    * Confidence scales with the distance from the mean ratio.
    """

    _STRATEGY_NAME = "silver_gold_ratio"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        ratio_mean: float = _RATIO_MEAN,
        ratio_std: float = _RATIO_STD,
        ratio_std_mult: float = 1.5,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._ratio_mean = ratio_mean
        self._ratio_std = ratio_std
        self._ratio_std_mult = ratio_std_mult
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        symbol: str = "",
        xag_ohlcv: Optional[pd.DataFrame] = None,
        xag_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyse gold OHLCV plus optional silver data.

        Args:
            ohlcv: XAU/USD OHLCV DataFrame.
            symbol: Trading symbol (typically ``"XAU/USD"``).
            xag_ohlcv: XAG/USD OHLCV DataFrame (preferred).
            xag_price: Spot XAG/USD price if DataFrame not available.
        """
        if ohlcv is None or len(ohlcv) < 50:
            return None

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        xau_price = float(ohlcv["close"].iloc[-1])

        if xag_ohlcv is not None and len(xag_ohlcv) >= 2:
            silver_price = float(xag_ohlcv["close"].iloc[-1])
        elif xag_price is not None and xag_price > 0:
            silver_price = xag_price
        else:
            # Cannot compute ratio without silver price — skip
            return None

        if silver_price <= 0:
            return None

        ratio = xau_price / silver_price
        threshold_high = self._ratio_mean + self._ratio_std_mult * self._ratio_std
        threshold_low = self._ratio_mean - self._ratio_std_mult * self._ratio_std

        direction: Optional[str] = None
        if ratio > threshold_high:
            # Gold overvalued vs silver → short gold
            direction = "short"
        elif ratio < threshold_low:
            # Gold undervalued vs silver → long gold
            direction = "long"

        if direction is None:
            return None

        # Distance from mean in σ units
        z = abs(ratio - self._ratio_mean) / self._ratio_std
        confidence = round(min(0.9, 0.50 + min(z - self._ratio_std_mult, 2.0) * 0.2), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": xau_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "ratio": round(ratio, 2),
            "ratio_mean": round(self._ratio_mean, 2),
            "threshold_high": round(threshold_high, 2),
            "threshold_low": round(threshold_low, 2),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            # Without silver data we cannot generate a signal
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No gold/silver ratio signal (need XAG data)")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 2.0
                take_profit = entry + atr * 4.0
            else:
                stop_loss = entry + atr * 2.0
                take_profit = entry - atr * 4.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold/Silver ratio {direction}: ratio={sig['ratio']:.1f} "
                    f"(mean={sig['ratio_mean']:.1f})"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 2.5),
            "leverage": 2,
        }
