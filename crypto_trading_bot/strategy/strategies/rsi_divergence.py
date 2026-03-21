"""RSI Divergence strategy — bullish and bearish divergence detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class RSIDivergenceStrategy(BaseStrategy):
    """RSI divergence strategy.

    Detects classic divergences between price and RSI over a *lookback*
    window by comparing two recent price/RSI swing points.

    Entry conditions
    ----------------
    * **Long (bullish divergence)**: price makes a lower low but RSI makes a
      higher low → bullish divergence → potential reversal upward.
    * **Short (bearish divergence)**: price makes a higher high but RSI makes
      a lower high → bearish divergence → potential reversal downward.
    """

    _STRATEGY_NAME = "rsi_divergence"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        rsi_period: int = 14,
        lookback: int = 40,
        swing_strength: int = 3,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._rsi_period = rsi_period
        self._lookback = lookback
        self._swing_strength = swing_strength
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Swing-point detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_swing_lows(series: pd.Series, strength: int) -> List[Tuple[int, float]]:
        swings: List[Tuple[int, float]] = []
        for i in range(strength, len(series) - strength):
            window = series.iloc[i - strength: i + strength + 1]
            if float(series.iloc[i]) == float(window.min()):
                swings.append((i, float(series.iloc[i])))
        return swings

    @staticmethod
    def _find_swing_highs(series: pd.Series, strength: int) -> List[Tuple[int, float]]:
        swings: List[Tuple[int, float]] = []
        for i in range(strength, len(series) - strength):
            window = series.iloc[i - strength: i + strength + 1]
            if float(series.iloc[i]) == float(window.max()):
                swings.append((i, float(series.iloc[i])))
        return swings

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._rsi_period + self._swing_strength + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        rsi_series = ta.rsi(closes, length=self._rsi_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if rsi_series is None or atr_series is None:
            return None

        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        # Work on the recent lookback window (exclude last few candles to avoid unconfirmed swings)
        window_closes = closes.iloc[-(self._lookback + self._swing_strength):-self._swing_strength]
        window_rsi = rsi_series.iloc[-(self._lookback + self._swing_strength):-self._swing_strength]

        swing_lows_price = self._find_swing_lows(window_closes, self._swing_strength)
        swing_lows_rsi = self._find_swing_lows(window_rsi, self._swing_strength)
        swing_highs_price = self._find_swing_highs(window_closes, self._swing_strength)
        swing_highs_rsi = self._find_swing_highs(window_rsi, self._swing_strength)

        curr_price = float(closes.iloc[-1])
        direction: Optional[str] = None
        divergence_strength: float = 0.0

        # Bullish divergence: last two swing lows of price are lower-low but RSI higher-low
        if len(swing_lows_price) >= 2 and len(swing_lows_rsi) >= 2:
            pl1_idx, pl1 = swing_lows_price[-2]
            pl2_idx, pl2 = swing_lows_price[-1]
            # Find closest RSI swing lows by index
            rl_near1 = min(swing_lows_rsi, key=lambda x: abs(x[0] - pl1_idx), default=None)
            rl_near2 = min(swing_lows_rsi, key=lambda x: abs(x[0] - pl2_idx), default=None)
            if rl_near1 and rl_near2 and rl_near1[0] != rl_near2[0]:
                price_lower_low = pl2 < pl1 * 0.999
                rsi_higher_low = rl_near2[1] > rl_near1[1] * 1.001
                if price_lower_low and rsi_higher_low:
                    direction = "long"
                    divergence_strength = (rl_near2[1] - rl_near1[1]) / max(rl_near1[1], 1.0)

        # Bearish divergence: last two swing highs of price are higher-high but RSI lower-high
        if direction is None and len(swing_highs_price) >= 2 and len(swing_highs_rsi) >= 2:
            ph1_idx, ph1 = swing_highs_price[-2]
            ph2_idx, ph2 = swing_highs_price[-1]
            rh_near1 = min(swing_highs_rsi, key=lambda x: abs(x[0] - ph1_idx), default=None)
            rh_near2 = min(swing_highs_rsi, key=lambda x: abs(x[0] - ph2_idx), default=None)
            if rh_near1 and rh_near2 and rh_near1[0] != rh_near2[0]:
                price_higher_high = ph2 > ph1 * 1.001
                rsi_lower_high = rh_near2[1] < rh_near1[1] * 0.999
                if price_higher_high and rsi_lower_high:
                    direction = "short"
                    divergence_strength = (rh_near1[1] - rh_near2[1]) / max(rh_near1[1], 1.0)

        if direction is None:
            return None

        confidence = round(min(0.85, 0.55 + divergence_strength * 2.0), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "divergence_strength": divergence_strength,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No RSI divergence detected")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 2.0
                take_profit = entry + atr * 3.0
            else:
                stop_loss = entry + atr * 2.0
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"RSI {'bullish' if direction=='long' else 'bearish'} divergence, "
                    f"strength={sig['divergence_strength']:.3f}, ATR={atr:.6f}"
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
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 70:
            return True
        if side == "short" and rsi < 30:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 2,
        }
