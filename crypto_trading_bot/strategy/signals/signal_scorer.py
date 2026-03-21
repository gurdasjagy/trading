"""Signal scorer — ranks signals by composite quality score."""

from __future__ import annotations

from typing import List

from strategy.base_strategy import Signal


class SignalScorer:
    """Scores and ranks trading signals by composite quality.

    The composite score is::

        score = (strength * 0.5) + (confidence * 0.4) + (leverage_bonus * 0.1)

    where ``leverage_bonus`` rewards signals with conservative leverage
    (lower leverage = more conservative = slightly higher score up to 0.1).
    """

    MAX_LEVERAGE_FOR_BONUS = 5

    def score_signal(self, signal: Signal) -> float:
        """Return a scalar quality score in [0, 1] for *signal*."""
        if signal.direction == "neutral":
            return 0.0
        lev = min(signal.leverage, self.MAX_LEVERAGE_FOR_BONUS)
        leverage_bonus = (self.MAX_LEVERAGE_FOR_BONUS - lev) / self.MAX_LEVERAGE_FOR_BONUS * 0.1
        score = signal.strength * 0.5 + signal.confidence * 0.4 + leverage_bonus
        return round(min(1.0, score), 4)

    def rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Return *signals* sorted from highest to lowest quality score."""
        return sorted(signals, key=self.score_signal, reverse=True)
