"""Signal generator — unified interface for gathering signals from all strategies."""

from __future__ import annotations

import asyncio
from typing import List, Optional

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class SignalGenerator:
    """Aggregates signals from multiple strategies into a single view.

    Usage
    -----
    .. code-block:: python

        gen = SignalGenerator(strategies)
        signals = await gen.generate_all_signals("BTC/USDT")
        agg = await gen.aggregate_signals(signals)
    """

    def __init__(self, strategies: Optional[List[BaseStrategy]] = None) -> None:
        self._strategies: List[BaseStrategy] = strategies or []

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)

    async def generate_all_signals(self, symbol: str) -> List[Signal]:
        """Gather signals from every enabled strategy for *symbol*."""
        tasks = []
        names = []
        for strategy in self._strategies:
            if not strategy.enabled:
                continue
            if strategy.symbols and symbol not in strategy.symbols:
                continue
            tasks.append(strategy.generate_signal(symbol))
            names.append(strategy.name)

        if not tasks:
            logger.debug(f"No enabled strategies for {symbol}")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        signals: List[Signal] = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {name!r} failed for {symbol}: {result}")
            elif isinstance(result, Signal):
                signals.append(result)
        return signals

    async def aggregate_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """Combine multiple signals into a single consensus signal.

        The aggregation weights each signal by ``strength × confidence``.
        Returns *None* if there are no actionable (non-neutral) signals.
        """
        if not signals:
            return None

        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        reasonings: List[str] = []

        for sig in signals:
            weight = sig.strength * sig.confidence
            if sig.direction == "long":
                long_score += weight
            elif sig.direction == "short":
                short_score += weight
            total_weight += weight
            if sig.reasoning:
                reasonings.append(f"[{sig.strategy_name}] {sig.reasoning}")

        if total_weight == 0:
            return None

        net = long_score - short_score
        if abs(net) < 0.1:
            return None  # no clear consensus

        direction = "long" if net > 0 else "short"
        strength = min(1.0, abs(net) / total_weight)
        confidence = min(1.0, abs(net) / len(signals))

        symbol = signals[0].symbol
        avg_leverage = round(
            sum(s.leverage for s in signals if s.direction == direction) / max(1, len(signals))
        )

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=round(strength, 3),
            confidence=round(confidence, 3),
            strategy_name="aggregated",
            reasoning="; ".join(reasonings[:5]),  # top 5 reasons
            leverage=max(1, int(avg_leverage)),
        )
