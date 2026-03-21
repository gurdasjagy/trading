"""Signal filter — quality-based filtering and conflict removal."""

from __future__ import annotations

from typing import List

from loguru import logger

from strategy.base_strategy import Signal


class SignalFilter:
    """Filters and deduplicates trading signals.

    Methods
    -------
    filter_signals
        Remove signals below a minimum strength or confidence threshold.
    remove_conflicting
        Remove signals where the same symbol has both long and short signals.
    """

    def filter_signals(
        self,
        signals: List[Signal],
        min_strength: float = 0.5,
        min_confidence: float = 0.5,
    ) -> List[Signal]:
        """Return only signals meeting the minimum quality criteria."""
        filtered = [
            s
            for s in signals
            if s.direction != "neutral"
            and s.strength >= min_strength
            and s.confidence >= min_confidence
        ]
        dropped = len(signals) - len(filtered)
        if dropped:
            logger.debug(
                f"SignalFilter: dropped {dropped}/{len(signals)} signals "
                f"(min_strength={min_strength}, min_confidence={min_confidence})"
            )
        return filtered

    def remove_conflicting(self, signals: List[Signal]) -> List[Signal]:
        """Remove all signals for symbols that have conflicting directions.

        A symbol is considered conflicting when it has at least one *long*
        signal **and** at least one *short* signal.  In that case, all
        signals for that symbol are discarded to avoid entering a position
        with unclear direction.
        """
        from collections import defaultdict

        by_symbol: dict[str, list[Signal]] = defaultdict(list)
        for sig in signals:
            by_symbol[sig.symbol].append(sig)

        clean: List[Signal] = []
        for sym, sigs in by_symbol.items():
            directions = {s.direction for s in sigs if s.direction != "neutral"}
            if len(directions) > 1:
                logger.debug(f"SignalFilter: removed {len(sigs)} conflicting signals for {sym}")
                continue
            clean.extend(sigs)
        return clean
