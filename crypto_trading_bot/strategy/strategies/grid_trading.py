"""Grid trading strategy — systematic buy/sell orders at fixed intervals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GridTradingStrategy(BaseStrategy):
    """Creates a grid of buy and sell orders at fixed price intervals.

    Active only in ranging / sideways markets.
    The signal indicates whether to place the next grid order.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
        grid_levels: int = 10,
        grid_spacing_pct: float = 0.01,  # 1 % between levels
        max_atr_pct: float = 0.02,  # pause in high volatility
    ) -> None:
        super().__init__(
            name="grid_trading",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._grid_levels = grid_levels
        self._grid_spacing_pct = grid_spacing_pct
        self._max_atr_pct = max_atr_pct
        self._grids: Dict[str, Dict[str, Any]] = {}  # symbol -> grid state

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=50)
            if ohlcv.empty:
                return self._neutral_signal(symbol, "No OHLCV data")

            last_price = float(ohlcv["close"].iloc[-1])
            atr = self._calculate_atr(ohlcv)
            atr_pct = atr / last_price if last_price > 0 else 0.0

            if atr_pct > self._max_atr_pct:
                return self._neutral_signal(
                    symbol, f"Volatility too high for grid (ATR {atr_pct:.3%})"
                )

            # Check whether price is near a grid level
            grid = self._get_or_create_grid(symbol, last_price)
            nearest_level, side = self._nearest_grid_level(last_price, grid)

            if nearest_level is None:
                return self._neutral_signal(symbol, "Price outside grid range")

            distance_pct = abs(last_price - nearest_level) / last_price
            if distance_pct > self._grid_spacing_pct * 0.5:
                return self._neutral_signal(
                    symbol, f"Price not close enough to grid level {nearest_level:.4f}"
                )

            return Signal(
                symbol=symbol,
                direction=side,
                strength=0.5,
                confidence=0.6,
                strategy_name=self.name,
                reasoning=(
                    f"Grid order at level {nearest_level:.4f} "
                    f"({side}); spacing={self._grid_spacing_pct:.1%}"
                ),
                leverage=1,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if price exits the grid range."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        grid = self._grids.get(symbol)
        if grid is None:
            return False
        ohlcv = await self._get_ohlcv(symbol, limit=5)
        if ohlcv.empty:
            return False
        last_price = float(ohlcv["close"].iloc[-1])
        return last_price < grid["lower"] or last_price > grid["upper"]

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        return {
            "position_size_pct": 1.0 / self._grid_levels * 0.5,
            "stop_loss_pct": self._grid_spacing_pct * self._grid_levels,
            "take_profit_pct": self._grid_spacing_pct,
            "leverage": 1,
        }

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def _get_or_create_grid(self, symbol: str, anchor_price: float) -> Dict[str, Any]:
        if symbol not in self._grids:
            half = self._grid_levels // 2
            levels = [
                anchor_price * (1 + self._grid_spacing_pct * i) for i in range(-half, half + 1)
            ]
            self._grids[symbol] = {
                "levels": levels,
                "lower": levels[0],
                "upper": levels[-1],
            }
        return self._grids[symbol]

    def _nearest_grid_level(
        self, price: float, grid: Dict[str, Any]
    ) -> tuple[Optional[float], str]:
        levels = grid.get("levels", [])
        if not levels:
            return None, "neutral"
        nearest = min(levels, key=lambda lvl: abs(lvl - price))
        side = "long" if price <= nearest else "short"
        return nearest, side
