"""DCA (Dollar-Cost Averaging) strategy — periodic buys/sells into positions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class DCAStrategy(BaseStrategy):
    """Dollar-cost averaging strategy.

    Buys or sells at regular time intervals and adjusts order size
    based on market conditions (dips increase allocation).
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        enabled: bool = True,
        direction: str = "long",  # always buy or always sell
        interval_hours: float = 24.0,
        base_size_pct: float = 0.02,  # 2 % of portfolio per interval
        dip_multiplier: float = 1.5,  # buy more on dips > 3 %
        dip_threshold_pct: float = 0.03,
    ) -> None:
        super().__init__(
            name="dca",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._direction = direction
        self._interval = timedelta(hours=interval_hours)
        self._base_size_pct = base_size_pct
        self._dip_multiplier = dip_multiplier
        self._dip_threshold_pct = dip_threshold_pct
        self._last_buy: Dict[str, datetime] = {}

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            now = datetime.now(tz=timezone.utc)
            last = self._last_buy.get(symbol)

            if last is not None and (now - last) < self._interval:
                remaining = self._interval - (now - last)
                return self._neutral_signal(
                    symbol,
                    f"Next DCA in {remaining.seconds // 3600}h {(remaining.seconds % 3600) // 60}m",
                )

            ohlcv = await self._get_ohlcv(symbol, limit=10)
            if ohlcv.empty:
                return self._neutral_signal(symbol, "No OHLCV data")

            # Detect dip from previous bar
            prev_close = float(ohlcv["close"].iloc[-2]) if len(ohlcv) >= 2 else 0.0
            last_close = float(ohlcv["close"].iloc[-1])
            dip = (prev_close - last_close) / prev_close if prev_close > 0 else 0.0
            is_dip = dip >= self._dip_threshold_pct and self._direction == "long"

            self._last_buy[symbol] = now

            strength = min(1.0, self._dip_multiplier * dip + 0.4) if is_dip else 0.4
            reason = f"DCA buy on dip ({dip:.2%})" if is_dip else "Scheduled DCA interval"
            return Signal(
                symbol=symbol,
                direction=self._direction,
                strength=round(strength, 3),
                confidence=0.6,
                strategy_name=self.name,
                reasoning=reason,
                leverage=1,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        # DCA positions are typically held long-term; no active close logic
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=10)
        last_close = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        prev_close = float(ohlcv["close"].iloc[-2]) if len(ohlcv) >= 2 else last_close
        dip = (prev_close - last_close) / prev_close if prev_close > 0 else 0.0
        is_dip = dip >= self._dip_threshold_pct
        size_pct = self._base_size_pct * (self._dip_multiplier if is_dip else 1.0)
        return {
            "position_size_pct": min(0.10, size_pct),
            "stop_loss_pct": 0.10,  # wide stop — long-term accumulation
            "take_profit_pct": 0.30,
            "leverage": 1,
        }
