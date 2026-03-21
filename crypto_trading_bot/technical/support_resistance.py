"""Support and resistance level detector."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


class SupportResistanceDetector:
    """Identifies significant support and resistance levels from price history.

    Uses a rolling window approach: price levels that have been tested
    multiple times (within a *sensitivity* percentage) are treated as
    significant S/R zones.
    """

    def find_levels(self, ohlcv: pd.DataFrame, sensitivity: float = 0.02) -> Dict[str, List[float]]:
        """Return all support and resistance levels.

        Parameters
        ----------
        ohlcv:
            Standard OHLCV DataFrame.
        sensitivity:
            Two price levels are merged into the same zone if they are
            within ``sensitivity * 100`` % of each other (default 2 %).

        Returns
        -------
        dict with keys ``"support"`` and ``"resistance"``.
        """
        support = self.find_support(ohlcv, sensitivity)
        resistance = self.find_resistance(ohlcv, sensitivity)
        return {"support": support, "resistance": resistance}

    def find_support(self, ohlcv: pd.DataFrame, sensitivity: float = 0.02) -> List[float]:
        """Return a sorted list of support price levels."""
        lows = ohlcv["low"].values.tolist()
        return self._cluster_levels(lows, sensitivity)

    def find_resistance(self, ohlcv: pd.DataFrame, sensitivity: float = 0.02) -> List[float]:
        """Return a sorted list of resistance price levels."""
        highs = ohlcv["high"].values.tolist()
        return self._cluster_levels(highs, sensitivity)

    def is_near_level(self, price: float, level: float, tolerance: float = 0.005) -> bool:
        """Return True if *price* is within *tolerance* of *level*.

        Parameters
        ----------
        tolerance:
            Fractional distance (default 0.5 %).
        """
        if level == 0:
            return False
        return abs(price - level) / level <= tolerance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_levels(prices: List[float], sensitivity: float) -> List[float]:
        """Cluster nearby price levels and return representative values."""
        if not prices:
            return []

        # Find local extremes
        extremes = sorted(set(prices))
        if not extremes:
            return []

        clusters: List[List[float]] = []
        for price in extremes:
            merged = False
            for cluster in clusters:
                rep = cluster[0]
                if rep > 0 and abs(price - rep) / rep <= sensitivity:
                    cluster.append(price)
                    merged = True
                    break
            if not merged:
                clusters.append([price])

        # Use the mean of each cluster as the representative level
        return sorted(sum(c) / len(c) for c in clusters)
