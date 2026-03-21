"""Correlation Divergence strategy — price vs rolling self-correlation divergence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class CorrelationDivergenceStrategy(BaseStrategy):
    """Correlation divergence strategy.

    Computes the rolling autocorrelation of price returns at a given lag.
    When price momentum (short EMA - long EMA spread) diverges from the
    autocorrelation trend — i.e., momentum is high but autocorrelation is
    falling — it suggests a potential mean-reversion.

    Entry conditions
    ----------------
    * **Long**: price momentum is negative (price below EMA) but rolling
      autocorrelation is positive and rising → trend may resume upward.
    * **Short**: price momentum is positive but autocorrelation is falling → 
      trend exhaustion.
    """

    _STRATEGY_NAME = "correlation_divergence"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        corr_window: int = 20,
        corr_lag: int = 1,
        fast_ema: int = 10,
        slow_ema: int = 30,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._corr_window = corr_window
        self._corr_lag = corr_lag
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Rolling autocorrelation
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_autocorr(returns: pd.Series, window: int, lag: int) -> pd.Series:
        """Compute rolling autocorrelation of *returns* at *lag*."""
        def _autocorr(x: np.ndarray) -> float:
            if len(x) < lag + 2:
                return float("nan")
            s = pd.Series(x)
            return float(s.autocorr(lag=lag))

        return returns.rolling(window).apply(_autocorr, raw=True)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._corr_window + self._slow_ema + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        returns = closes.pct_change().fillna(0)

        autocorr = self._rolling_autocorr(returns, self._corr_window, self._corr_lag)
        fast_ema = ta.ema(closes, length=self._fast_ema)
        slow_ema = ta.ema(closes, length=self._slow_ema)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if fast_ema is None or slow_ema is None or atr_series is None:
            return None

        curr_autocorr = float(autocorr.iloc[-1])
        prev_autocorr = float(autocorr.iloc[-2])
        curr_fast = float(fast_ema.iloc[-1])
        curr_slow = float(slow_ema.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        for val in (curr_autocorr, prev_autocorr, curr_fast, curr_slow, curr_atr):
            if pd.isna(val):
                return None

        momentum_spread = (curr_fast - curr_slow) / curr_slow if curr_slow > 0 else 0.0
        autocorr_rising = curr_autocorr > prev_autocorr
        autocorr_falling = curr_autocorr < prev_autocorr

        direction: Optional[str] = None

        # Bullish divergence: price below slow EMA (bearish momentum) but autocorr rising
        if momentum_spread < -0.002 and curr_autocorr > 0.1 and autocorr_rising:
            direction = "long"

        # Bearish divergence: price above slow EMA (bullish momentum) but autocorr falling
        elif momentum_spread > 0.002 and curr_autocorr < -0.1 and autocorr_falling:
            direction = "short"

        if direction is None:
            return None

        divergence = abs(momentum_spread) + abs(curr_autocorr)
        confidence = round(min(0.80, 0.45 + divergence * 5.0), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "autocorr": curr_autocorr,
            "momentum_spread": momentum_spread,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No correlation divergence")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 2.5
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 2.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Correlation divergence {direction}: autocorr={sig['autocorr']:.3f}, "
                    f"momentum={sig['momentum_spread']:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and sig["direction"] == "short":
            return True
        if side == "short" and sig["direction"] == "long":
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.03,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2,
        }
