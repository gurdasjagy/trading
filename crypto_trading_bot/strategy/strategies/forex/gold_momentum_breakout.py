"""Gold Momentum Breakout strategy — ADX + RSI + volume confirmation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldMomentumBreakoutStrategy(BaseStrategy):
    """Gold Momentum Breakout strategy.

    Trades strong momentum moves in gold using ADX, RSI, and volume
    confirmation.  Gold often trends strongly during geopolitical events.

    Signal logic
    ------------
    * ADX > ``adx_threshold`` confirms a trending market.
    * RSI crossing above 55 → long; RSI crossing below 45 → short.
    * Current volume > ``volume_multiplier`` × average volume confirms the move.
    * Confidence scales with ADX strength and volume ratio.
    """

    _STRATEGY_NAME = "gold_momentum_breakout"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        volume_multiplier: float = 1.5,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._rsi_period = rsi_period
        self._volume_multiplier = volume_multiplier
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_adx(ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Return the most recent ADX value."""
        if len(ohlcv) < period * 2:
            return 0.0
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values

        plus_dm: List[float] = []
        minus_dm: List[float] = []
        trs: List[float] = []

        for i in range(1, len(highs)):
            h_diff = highs[i] - highs[i - 1]
            l_diff = lows[i - 1] - lows[i]
            plus_dm.append(h_diff if h_diff > l_diff and h_diff > 0 else 0.0)
            minus_dm.append(l_diff if l_diff > h_diff and l_diff > 0 else 0.0)
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

        def _smooth(data: List[float], p: int) -> List[float]:
            # Wilder's smoothing: S[n] = S[n-1] - S[n-1]/p + data[n]
            result = [sum(data[:p])]
            for v in data[p:]:
                result.append(result[-1] - result[-1] / p + v)
            return result

        atr_s = _smooth(trs, period)
        plus_s = _smooth(plus_dm, period)
        minus_s = _smooth(minus_dm, period)

        dx_list: List[float] = []
        for a, p, m in zip(atr_s, plus_s, minus_s):
            if a == 0:
                dx_list.append(0.0)
                continue
            pdi = 100 * p / a
            mdi = 100 * m / a
            denom = pdi + mdi
            dx_list.append(100 * abs(pdi - mdi) / denom if denom > 0 else 0.0)

        if len(dx_list) < period:
            return 0.0

        adx = sum(dx_list[:period]) / period
        for dx in dx_list[period:]:
            adx = (adx * (period - 1) + dx) / period
        return adx

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._adx_period * 3 + 10
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"].tolist()
        volumes = ohlcv["volume"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        adx = self._calculate_adx(ohlcv, self._adx_period)
        if adx < self._adx_threshold:
            return None

        rsi = self._calculate_rsi(closes, self._rsi_period)
        prev_rsi = self._calculate_rsi(closes[:-1], self._rsi_period)

        # Volume confirmation
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        curr_volume = volumes[-1]
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1.0

        if volume_ratio < self._volume_multiplier:
            return None

        direction: Optional[str] = None
        if prev_rsi <= 55 < rsi:
            direction = "long"
        elif prev_rsi >= 45 > rsi:
            direction = "short"

        if direction is None:
            return None

        # Reject over-extended RSI
        if direction == "long" and rsi > 80:
            return None
        if direction == "short" and rsi < 20:
            return None

        adx_factor = min(1.0, (adx - self._adx_threshold) / 25.0)
        vol_factor = min(1.0, (volume_ratio - self._volume_multiplier) / 2.0)
        confidence = round(min(0.9, 0.55 + adx_factor * 0.2 + vol_factor * 0.15), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": closes[-1],
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "adx": round(adx, 2),
            "rsi": round(rsi, 2),
            "volume_ratio": round(volume_ratio, 3),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No gold momentum breakout")

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
                    f"Gold momentum breakout {direction}: ADX={sig['adx']:.1f}, "
                    f"RSI={sig['rsi']:.1f}, vol_ratio={sig['volume_ratio']:.2f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if ohlcv.empty:
            return False
        adx = self._calculate_adx(ohlcv, self._adx_period)
        return adx < self._adx_threshold * 0.8  # trend fading

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 2.5),
            "leverage": 3,
        }
