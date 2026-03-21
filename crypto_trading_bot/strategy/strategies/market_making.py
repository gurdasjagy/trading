"""Market making strategy — quotes bid/ask around mid price in ranging markets."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MarketMakingStrategy(BaseStrategy):
    """Spot market-making strategy.

    Places bid and ask orders symmetrically around the mid price.
    Only active in low-volatility, ranging market conditions.

    This strategy emits *neutral* signals since it manages orders directly
    rather than taking directional positions, but will emit a *long* signal
    to tilt inventory when very skewed long, and vice versa.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1m",
        enabled: bool = True,
        spread_pct: float = 0.001,  # 0.1 % total spread
        max_volatility_atr_pct: float = 0.015,  # pause if ATR/price > 1.5 %
    ) -> None:
        super().__init__(
            name="market_making",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._spread_pct = spread_pct
        self._max_vol_atr_pct = max_volatility_atr_pct
        self._inventory: Dict[str, float] = {}  # symbol -> net inventory (positive = long)

    def update_inventory(self, symbol: str, qty: float) -> None:
        """Update the current inventory level for *symbol*."""
        self._inventory[symbol] = qty

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=30)
            if ohlcv.empty:
                return self._neutral_signal(symbol, "No OHLCV data")

            last_price = float(ohlcv["close"].iloc[-1])
            atr = self._calculate_atr(ohlcv)
            atr_pct = atr / last_price if last_price > 0 else 0.0

            if atr_pct > self._max_vol_atr_pct:
                return self._neutral_signal(
                    symbol,
                    f"Volatility too high (ATR/price={atr_pct:.3%}) — pausing market making",
                )

            inv = self._inventory.get(symbol, 0.0)
            # Tilt signal based on inventory imbalance
            if inv > 0.5:  # too long → emit weak short to rebalance
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=0.2,
                    confidence=0.5,
                    strategy_name=self.name,
                    reasoning=f"Inventory rebalance — long inventory={inv:.4f}",
                    leverage=1,
                )
            if inv < -0.5:  # too short → emit weak long
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=0.2,
                    confidence=0.5,
                    strategy_name=self.name,
                    reasoning=f"Inventory rebalance — short inventory={inv:.4f}",
                    leverage=1,
                )

            return self._neutral_signal(
                symbol,
                f"Market making active — spread {self._spread_pct:.3%}, ATR {atr_pct:.3%}",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if volatility spikes above threshold."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=20)
        if ohlcv.empty:
            return False
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1])
        return (atr / last_price) > self._max_vol_atr_pct if last_price > 0 else False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        return {
            "position_size_pct": 0.02,  # small rebalancing trades
            "stop_loss_pct": 0.01,
            "take_profit_pct": self._spread_pct / 2,
            "leverage": 1,
        }
