"""Gold RSI Divergence strategy — trade hidden and regular RSI divergences."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldRSIDivergenceStrategy(BaseStrategy):
    """Gold RSI Divergence strategy.

    RSI divergence is highly effective on gold.

    * **Bullish divergence**: price makes a lower low while RSI makes a
      higher low → buy signal (price should rise to catch up with RSI).
    * **Bearish divergence**: price makes a higher high while RSI makes a
      lower high → sell signal.

    Signal logic
    ------------
    * Find the two most recent local price lows/highs within the lookback window.
    * Compare RSI values at those pivots.
    * Divergence confirmed when price and RSI direction diverge.
    * Confidence based on RSI divergence magnitude.
    """

    _STRATEGY_NAME = "gold_rsi_divergence"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        rsi_period: int = 14,
        lookback: int = 50,
        pivot_window: int = 5,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._rsi_period = rsi_period
        self._lookback = lookback
        self._pivot_window = pivot_window
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pivots(
        series: List[float], window: int, find_highs: bool
    ) -> List[int]:
        """Return indices of local highs (find_highs=True) or lows."""
        pivots: List[int] = []
        for i in range(window, len(series) - window):
            segment = series[i - window: i + window + 1]
            if find_highs:
                if series[i] == max(segment):
                    pivots.append(i)
            else:
                if series[i] == min(segment):
                    pivots.append(i)
        return pivots

    def _compute_rsi_series(self, closes: List[float]) -> List[float]:
        """Return a list of RSI values (one per bar, starting from index rsi_period)."""
        period = self._rsi_period
        if len(closes) < period + 1:
            return []
        rsi_vals: List[float] = [50.0] * period  # pad with neutral RSI (50) before first real value
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
            rsi_vals.append(100.0 - 100.0 / (1 + rs))
        return rsi_vals

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._rsi_period + self._pivot_window + 5
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        rsi_series = self._compute_rsi_series(closes)
        if len(rsi_series) < self._lookback:
            return None

        # Work on last lookback bars
        price_window = closes[-self._lookback:]
        rsi_window = rsi_series[-self._lookback:]

        lows = self._find_pivots(price_window, self._pivot_window, find_highs=False)
        highs = self._find_pivots(price_window, self._pivot_window, find_highs=True)

        direction: Optional[str] = None
        rsi_div: float = 0.0
        price_div: float = 0.0

        # Bullish divergence: two consecutive lows
        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            price_low1, price_low2 = price_window[i1], price_window[i2]
            rsi_low1, rsi_low2 = rsi_window[i1], rsi_window[i2]
            # price lower low, RSI higher low
            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                direction = "long"
                price_div = price_low1 - price_low2  # positive
                rsi_div = rsi_low2 - rsi_low1        # positive

        # Bearish divergence: two consecutive highs
        if direction is None and len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            price_hi1, price_hi2 = price_window[i1], price_window[i2]
            rsi_hi1, rsi_hi2 = rsi_window[i1], rsi_window[i2]
            # price higher high, RSI lower high
            if price_hi2 > price_hi1 and rsi_hi2 < rsi_hi1:
                direction = "short"
                price_div = price_hi2 - price_hi1   # positive
                rsi_div = rsi_hi1 - rsi_hi2          # positive

        if direction is None:
            return None

        # Confirm we are not too deep in the opposite RSI extreme
        curr_rsi = rsi_series[-1]
        if direction == "long" and curr_rsi > 65:
            return None
        if direction == "short" and curr_rsi < 35:
            return None

        rsi_div_normalized = min(1.0, rsi_div / 10.0)
        confidence = round(min(0.88, 0.55 + rsi_div_normalized * 0.33), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": closes[-1],
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "rsi_divergence": round(rsi_div, 2),
            "price_divergence": round(price_div, 4),
            "current_rsi": round(curr_rsi, 2),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No RSI divergence on gold")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold RSI divergence {direction}: rsi_div={sig['rsi_divergence']:.2f}, "
                    f"RSI={sig['current_rsi']:.1f}, ATR={atr:.4f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 2,
        }
