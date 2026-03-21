"""Cross-asset correlation analyzer for crypto markets."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Minimum number of overlapping observations required for a valid correlation
_MIN_OBSERVATIONS: int = 30

# Threshold above which a correlation is considered "high"
_HIGH_CORRELATION_THRESHOLD: float = 0.7

# Minimum rolling-window correlation to track over time (for breakdown detection)
_BREAKDOWN_WINDOW: int = 20


class CorrelationAnalyzer:
    """Computes and tracks cross-asset price correlations.

    Stores a rolling history of price series to enable efficient correlation
    calculations without requiring callers to pass full histories each time.
    """

    def __init__(self) -> None:
        # symbol → pd.Series of log returns
        self._return_cache: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_prices(self, symbol: str, prices: List[float]) -> None:
        """Ingest a price series for *symbol* and compute log returns.

        Args:
            symbol: Asset ticker (e.g. ``"BTC/USDT"``).
            prices: Ordered list of close prices (oldest first).
        """
        if len(prices) < 2:
            return
        s = pd.Series(prices, dtype=float)
        self._return_cache[symbol] = np.log(s / s.shift(1)).dropna()

    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        timeframe: str = "all",
        prices1: Optional[List[float]] = None,
        prices2: Optional[List[float]] = None,
    ) -> float:
        """Return the Pearson correlation between *symbol1* and *symbol2*.

        Args:
            symbol1:  First asset ticker.
            symbol2:  Second asset ticker.
            timeframe: ``"all"`` to use all cached data, or an integer string
                       e.g. ``"100"`` to use the last N observations.
            prices1:  Optional price list for *symbol1* (overrides cache).
            prices2:  Optional price list for *symbol2* (overrides cache).

        Returns:
            Pearson correlation in [-1.0, 1.0], or 0.0 when data is insufficient.
        """
        try:
            r1, r2 = self._get_aligned_returns(symbol1, symbol2, prices1, prices2)
            if r1 is None or len(r1) < _MIN_OBSERVATIONS:
                logger.debug(f"Insufficient data for correlation {symbol1}/{symbol2}")
                return 0.0

            # Apply timeframe slice if numeric
            if timeframe != "all":
                try:
                    n = int(timeframe)
                    r1 = r1.tail(n)
                    r2 = r2.tail(n)
                except ValueError:
                    pass

            if len(r1) < _MIN_OBSERVATIONS:
                return 0.0

            corr = float(r1.corr(r2))
            return corr if not np.isnan(corr) else 0.0
        except Exception as exc:
            logger.warning(f"CorrelationAnalyzer.calculate_correlation error: {exc}")
            return 0.0

    def get_correlated_assets(
        self,
        symbol: str,
        threshold: float = _HIGH_CORRELATION_THRESHOLD,
    ) -> List[Dict]:
        """Return a list of assets highly correlated with *symbol*.

        Args:
            symbol:    Target asset ticker.
            threshold: Minimum absolute correlation to include.

        Returns:
            List of dicts with keys ``"symbol"`` and ``"correlation"``,
            sorted by descending absolute correlation.
        """
        results: List[Dict] = []
        for other in self._return_cache:
            if other == symbol:
                continue
            corr = self.calculate_correlation(symbol, other)
            if abs(corr) >= threshold:
                results.append({"symbol": other, "correlation": round(corr, 4)})
        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results

    def detect_correlation_breakdown(
        self,
        symbol1: str,
        symbol2: str,
        prices1: Optional[List[float]] = None,
        prices2: Optional[List[float]] = None,
    ) -> bool:
        """Detect whether the historical correlation between two assets has broken down.

        Compares the long-run correlation with the recent rolling correlation and
        flags a breakdown when they diverge significantly.

        Args:
            symbol1: First asset ticker.
            symbol2: Second asset ticker.
            prices1: Optional price list for *symbol1*.
            prices2: Optional price list for *symbol2*.

        Returns:
            ``True`` if a correlation breakdown is detected.
        """
        try:
            r1, r2 = self._get_aligned_returns(symbol1, symbol2, prices1, prices2)
            if r1 is None or len(r1) < _BREAKDOWN_WINDOW * 2:
                return False

            long_run_corr = float(r1.corr(r2))
            recent_corr = float(r1.tail(_BREAKDOWN_WINDOW).corr(r2.tail(_BREAKDOWN_WINDOW)))

            if np.isnan(long_run_corr) or np.isnan(recent_corr):
                return False

            divergence = abs(long_run_corr - recent_corr)
            breakdown = divergence > 0.4  # heuristic threshold
            if breakdown:
                logger.debug(
                    f"Correlation breakdown {symbol1}/{symbol2}: "
                    f"long={long_run_corr:.2f} recent={recent_corr:.2f}"
                )
            return breakdown
        except Exception as exc:
            logger.warning(f"CorrelationAnalyzer.detect_correlation_breakdown error: {exc}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_aligned_returns(
        self,
        symbol1: str,
        symbol2: str,
        prices1: Optional[List[float]],
        prices2: Optional[List[float]],
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Return aligned log-return Series for two symbols.

        Prefers caller-supplied *prices* over the internal cache.
        """
        if prices1 is not None:
            s = pd.Series(prices1, dtype=float)
            r1 = np.log(s / s.shift(1)).dropna()
        elif symbol1 in self._return_cache:
            r1 = self._return_cache[symbol1]
        else:
            return None, None

        if prices2 is not None:
            s = pd.Series(prices2, dtype=float)
            r2 = np.log(s / s.shift(1)).dropna()
        elif symbol2 in self._return_cache:
            r2 = self._return_cache[symbol2]
        else:
            return None, None

        # Align on common index length
        min_len = min(len(r1), len(r2))
        return r1.iloc[-min_len:].reset_index(drop=True), r2.iloc[-min_len:].reset_index(drop=True)
