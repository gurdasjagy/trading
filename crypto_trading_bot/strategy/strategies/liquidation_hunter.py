"""Liquidation hunter strategy — counter-trend after large cascade events."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class LiquidationHunterStrategy(BaseStrategy):
    """Trades the counter-trend bounce after large liquidation cascades.

    Entry conditions
    ----------------
    * Cascade size ≥ $50 M in the last hour.
    * RSI < 20 → long (market exhausted to the downside).
    * RSI > 80 → short (market exhausted to the upside).
    """

    CASCADE_THRESHOLD_USD = 50_000_000  # $50 M
    RSI_LONG_MAX = 20.0
    RSI_SHORT_MIN = 80.0

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="liquidation_hunter",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        # symbol -> total liquidation USD in last hour
        self._liq_cache: Dict[str, float] = {}

    def update_liquidations(self, symbol: str, total_usd: float) -> None:
        """Inject recent hourly liquidation volume for *symbol*."""
        self._liq_cache[symbol] = total_usd

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            liq_usd = self._liq_cache.get(symbol, 0.0)
            if liq_usd < self.CASCADE_THRESHOLD_USD:
                return self._neutral_signal(
                    symbol,
                    f"Liquidations ${liq_usd:,.0f} below ${self.CASCADE_THRESHOLD_USD:,.0f} threshold",
                )

            ohlcv = await self._get_ohlcv(symbol, limit=50)
            if ohlcv.empty:
                return self._neutral_signal(symbol, "No OHLCV data")

            prices = ohlcv["close"].tolist()
            rsi = self._calculate_rsi(prices)

            if rsi < self.RSI_LONG_MAX:
                strength = min(1.0, (self.RSI_LONG_MAX - rsi) / self.RSI_LONG_MAX + 0.5)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.7,
                    strategy_name=self.name,
                    reasoning=(
                        f"Liquidation cascade ${liq_usd/1e6:.1f}M + RSI={rsi:.1f} (oversold)"
                    ),
                    leverage=3,
                )

            if rsi > self.RSI_SHORT_MIN:
                strength = min(1.0, (rsi - self.RSI_SHORT_MIN) / (100 - self.RSI_SHORT_MIN) + 0.5)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.7,
                    strategy_name=self.name,
                    reasoning=(
                        f"Liquidation cascade ${liq_usd/1e6:.1f}M + RSI={rsi:.1f} (overbought)"
                    ),
                    leverage=3,
                )

            return self._neutral_signal(
                symbol,
                f"Cascade detected but RSI {rsi:.1f} not at extreme",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close once RSI returns to a neutral zone."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=20)
        if ohlcv.empty:
            return False
        rsi = self._calculate_rsi(ohlcv["close"].tolist())
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 50:
            return True
        if side == "short" and rsi < 50:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.06, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 3.0),
            "leverage": 3,
        }
