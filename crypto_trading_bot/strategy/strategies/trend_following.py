"""Trend-following strategy — rides established trends using EMA cross + ADX."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    """Trend-following strategy combining dual-EMA crossover with ADX filter.

    Entry conditions
    ----------------
    * **Long**: fast EMA > slow EMA, ADX > 25 (strong trend), MACD histogram > 0.
    * **Short**: fast EMA < slow EMA, ADX > 25, MACD histogram < 0.

    Exit conditions
    ---------------
    * Close on EMA crossover reversal or ADX dropping below 20.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        fast_ema: int = 20,
        slow_ema: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
    ) -> None:
        super().__init__(
            name="trend_following",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=self._slow_ema + 30)
            if len(ohlcv) < self._slow_ema + 5:
                return self._neutral_signal(symbol, "Insufficient data for trend following")

            closes = ohlcv["close"].tolist()
            fast = self._calculate_ema(closes, self._fast_ema)
            slow = self._calculate_ema(closes, self._slow_ema)
            macd = self._calculate_macd(closes)
            histogram = macd["histogram"]
            adx = self._calculate_adx(ohlcv, self._adx_period)

            if adx < self._adx_threshold:
                return self._neutral_signal(
                    symbol,
                    f"ADX={adx:.1f} below threshold ({self._adx_threshold}) — no trend",
                )

            # Bullish trend
            if fast > slow and histogram > 0:
                strength = min(1.0, 0.4 + (adx - self._adx_threshold) / 100 + abs(histogram) * 50)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Uptrend: EMA{self._fast_ema}={fast:.4f} > EMA{self._slow_ema}={slow:.4f}, "
                        f"ADX={adx:.1f}, MACD hist={histogram:.6f}"
                    ),
                    leverage=3,
                )

            # Bearish trend
            if fast < slow and histogram < 0:
                strength = min(1.0, 0.4 + (adx - self._adx_threshold) / 100 + abs(histogram) * 50)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Downtrend: EMA{self._fast_ema}={fast:.4f} < EMA{self._slow_ema}={slow:.4f}, "
                        f"ADX={adx:.1f}, MACD hist={histogram:.6f}"
                    ),
                    leverage=3,
                )

            return self._neutral_signal(
                symbol,
                f"Mixed: fast={fast:.4f}, slow={slow:.4f}, hist={histogram:.6f}, ADX={adx:.1f}",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close on EMA crossover reversal or ADX weakening below 20."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=self._slow_ema + 10)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        fast = self._calculate_ema(closes, self._fast_ema)
        slow = self._calculate_ema(closes, self._slow_ema)
        adx = self._calculate_adx(ohlcv, self._adx_period)
        side = str(getattr(position, "side", "long")).lower()

        if adx < 20:
            return True
        if side == "long" and fast < slow:
            return True
        if side == "short" and fast > slow:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=self._slow_ema + 30)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.025
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }

    @staticmethod
    def _calculate_adx(ohlcv: Any, period: int = 14) -> float:
        """Compute the Average Directional Index (ADX) from OHLCV data."""
        if len(ohlcv) < period + 2:
            return 0.0

        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values

        plus_dm: list[float] = []
        minus_dm: list[float] = []
        tr_list: list[float] = []

        for i in range(1, len(ohlcv)):
            high_diff = float(highs[i] - highs[i - 1])
            low_diff = float(lows[i - 1] - lows[i])
            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0.0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0.0)
            tr = max(
                float(highs[i] - lows[i]),
                abs(float(highs[i] - closes[i - 1])),
                abs(float(lows[i] - closes[i - 1])),
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return 0.0

        # Wilder smoothing
        atr = sum(tr_list[:period]) / period
        smooth_plus = sum(plus_dm[:period]) / period
        smooth_minus = sum(minus_dm[:period]) / period

        dx_values: list[float] = []
        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
            smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
            smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period

            plus_di = (smooth_plus / atr * 100) if atr > 0 else 0.0
            minus_di = (smooth_minus / atr * 100) if atr > 0 else 0.0
            di_sum = plus_di + minus_di
            dx = (abs(plus_di - minus_di) / di_sum * 100) if di_sum > 0 else 0.0
            dx_values.append(dx)

        if not dx_values:
            return 0.0

        # First ADX is the SMA of the first *period* DX values
        if len(dx_values) < period:
            return float(sum(dx_values) / len(dx_values))

        adx = sum(dx_values[:period]) / period
        for dx in dx_values[period:]:
            adx = (adx * (period - 1) + dx) / period
        return adx
