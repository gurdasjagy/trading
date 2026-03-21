"""Sentiment reversal strategy — contrarian trades at extreme sentiment."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class SentimentReversalStrategy(BaseStrategy):
    """Contrarian strategy that fades extreme market sentiment.

    Entry conditions
    ----------------
    * **Long**: sentiment score < -0.8 AND RSI ≤ 30 (oversold).
    * **Short**: sentiment score > 0.8 AND RSI ≥ 70 (overbought).
    """

    BEARISH_THRESHOLD = -0.8
    BULLISH_THRESHOLD = 0.8
    RSI_OVERSOLD = 30.0
    RSI_OVERBOUGHT = 70.0

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="sentiment_reversal",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._sentiment_cache: Dict[str, float] = {}

    def update_sentiment(self, symbol: str, score: float) -> None:
        """Inject a fresh sentiment score for *symbol* (range -1 to 1)."""
        self._sentiment_cache[symbol] = score

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            score = self._sentiment_cache.get(symbol, 0.0)
            ohlcv = await self._get_ohlcv(symbol, limit=50)
            if ohlcv.empty:
                return self._neutral_signal(symbol, "No OHLCV data")

            prices = ohlcv["close"].tolist()
            rsi = self._calculate_rsi(prices)

            if score < self.BEARISH_THRESHOLD and rsi <= self.RSI_OVERSOLD:
                strength = min(1.0, abs(score) * 0.8 + (self.RSI_OVERSOLD - rsi) / 100)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Extreme bearish sentiment ({score:.2f}) + oversold RSI ({rsi:.1f})"
                    ),
                    leverage=3,
                )

            if score > self.BULLISH_THRESHOLD and rsi >= self.RSI_OVERBOUGHT:
                strength = min(1.0, score * 0.8 + (rsi - self.RSI_OVERBOUGHT) / 100)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Extreme bullish sentiment ({score:.2f}) + overbought RSI ({rsi:.1f})"
                    ),
                    leverage=3,
                )

            return self._neutral_signal(
                symbol,
                f"Sentiment {score:.2f} and RSI {rsi:.1f} — no extreme reading",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        score = self._sentiment_cache.get(symbol, 0.0)
        side = str(getattr(position, "side", "long")).lower()
        # Close long when sentiment normalises (> -0.3)
        if side == "long" and score > -0.3:
            return True
        # Close short when sentiment normalises (< 0.3)
        if side == "short" and score < 0.3:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.025
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }
