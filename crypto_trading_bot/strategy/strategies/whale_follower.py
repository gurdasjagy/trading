"""Whale follower strategy — follows large on-chain wallet movements."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class WhaleFollowerStrategy(BaseStrategy):
    """Follows large whale wallet movements with a delayed, smaller position.

    Signal logic
    ------------
    * Whale → exchange inflow → bearish (sell signal / short).
    * Whale ← exchange outflow → bullish (accumulation / long).

    Position size is intentionally smaller than the whale move and entry
    is delayed to confirm the trend, not just react to it.
    """

    WHALE_THRESHOLD_USD = 1_000_000  # $1 M minimum

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="whale_follower",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        # symbol -> list of {"direction": "inflow"|"outflow", "amount_usd": float}
        self._whale_events: Dict[str, List[Dict[str, Any]]] = {}

    def update_whale_events(self, symbol: str, events: List[Dict[str, Any]]) -> None:
        """Push fresh whale transfer events for *symbol*."""
        self._whale_events[symbol] = events

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            events = self._whale_events.get(symbol, [])
            large = [e for e in events if e.get("amount_usd", 0) >= self.WHALE_THRESHOLD_USD]
            if not large:
                return self._neutral_signal(symbol, "No large whale movements detected")

            inflows = sum(e["amount_usd"] for e in large if e.get("direction") == "inflow")
            outflows = sum(e["amount_usd"] for e in large if e.get("direction") == "outflow")
            total = inflows + outflows
            if total == 0:
                return self._neutral_signal(symbol, "No qualifying whale events")

            net_bias = (outflows - inflows) / total  # > 0 = bullish (outflow dominant)

            if net_bias > 0.3:
                strength = min(1.0, 0.5 + net_bias * 0.5)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.65,
                    strategy_name=self.name,
                    reasoning=(
                        f"Whale outflows ${outflows/1e6:.1f}M dominate — accumulation signal"
                    ),
                    leverage=2,
                )

            if net_bias < -0.3:
                strength = min(1.0, 0.5 + abs(net_bias) * 0.5)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.65,
                    strategy_name=self.name,
                    reasoning=(f"Whale inflows ${inflows/1e6:.1f}M dominate — distribution signal"),
                    leverage=2,
                )

            return self._neutral_signal(symbol, f"Whale flows mixed (net_bias={net_bias:.2f})")
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if whale flow reverses."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        events = self._whale_events.get(symbol, [])
        large = [e for e in events if e.get("amount_usd", 0) >= self.WHALE_THRESHOLD_USD]
        if not large:
            return False
        inflows = sum(e["amount_usd"] for e in large if e.get("direction") == "inflow")
        outflows = sum(e["amount_usd"] for e in large if e.get("direction") == "outflow")
        total = inflows + outflows
        if total == 0:
            return False
        net_bias = (outflows - inflows) / total
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and net_bias < -0.2:
            return True
        if side == "short" and net_bias > 0.2:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,  # smaller than whale
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2,
        }
