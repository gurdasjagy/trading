"""Portfolio correlation risk management."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# Correlation map for common crypto pairs (simplified representative values).
# Keys are frozensets of two symbols; values are correlation coefficients.
_CORRELATION_MAP: Dict[frozenset, float] = {
    frozenset({"BTC/USDT", "ETH/USDT"}): 0.85,
    frozenset({"BTC/USDT", "BNB/USDT"}): 0.75,
    frozenset({"BTC/USDT", "SOL/USDT"}): 0.80,
    frozenset({"BTC/USDT", "XRP/USDT"}): 0.65,
    frozenset({"ETH/USDT", "BNB/USDT"}): 0.80,
    frozenset({"ETH/USDT", "SOL/USDT"}): 0.82,
    frozenset({"ETH/USDT", "XRP/USDT"}): 0.60,
    frozenset({"BNB/USDT", "SOL/USDT"}): 0.72,
    frozenset({"SOL/USDT", "XRP/USDT"}): 0.58,
}

# Sector classification for L1 tokens and others
_SECTOR_MAP: Dict[str, str] = {
    "BTC/USDT": "L1",
    "ETH/USDT": "L1",
    "SOL/USDT": "L1",
    "BNB/USDT": "L1",
    "AVAX/USDT": "L1",
    "ADA/USDT": "L1",
    "DOT/USDT": "L1",
    "MATIC/USDT": "L2",
    "ARB/USDT": "L2",
    "OP/USDT": "L2",
    "XRP/USDT": "payments",
    "LTC/USDT": "payments",
    "LINK/USDT": "oracle",
    "UNI/USDT": "defi",
    "AAVE/USDT": "defi",
}

# Maximum number of positions in the same sector
_MAX_SECTOR_EXPOSURE: int = 2

# Thresholds for position size reduction or trade rejection
_REDUCE_THRESHOLD: float = 0.70
_REJECT_THRESHOLD: float = 0.85


class CorrelationRiskManager:
    """Manages portfolio correlation to avoid over-exposure to correlated assets."""

    def __init__(self, default_correlation: float = 0.5) -> None:
        self._default_correlation = default_correlation
        # Cache for real-time correlation matrix computed from 24 h returns
        # key: frozenset of two symbols, value: correlation coefficient
        self._realtime_cache: Dict[frozenset, float] = {}

    # ------------------------------------------------------------------
    # Real-time correlation matrix
    # ------------------------------------------------------------------

    def update_realtime_correlation(
        self,
        returns: Dict[str, "pd.Series"],
    ) -> None:
        """Compute and cache pairwise correlations from 24 h return series.

        Args:
            returns: Mapping of symbol → pandas Series of hourly/daily returns.
                     All series should cover approximately the same time window.
        """
        symbols = list(returns.keys())
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                try:
                    r1 = returns[s1].dropna()
                    r2 = returns[s2].dropna()
                    aligned = pd.concat([r1, r2], axis=1).dropna()
                    if len(aligned) < 5:
                        continue
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if not pd.isna(corr):
                        self._realtime_cache[frozenset({s1, s2})] = round(corr, 4)
                except Exception as exc:
                    logger.debug("[CorrelationRisk] Could not compute {}/{}: {}", s1, s2, exc)

    # ------------------------------------------------------------------
    # New position validation
    # ------------------------------------------------------------------

    def validate_new_position(
        self,
        new_symbol: str,
        existing_positions: List[dict],
    ) -> Tuple[bool, float, str]:
        """Check whether a new position is acceptable from a correlation perspective.

        Args:
            new_symbol: Symbol about to be traded.
            existing_positions: Current open positions (each dict must have ``symbol``).

        Returns:
            Tuple of (allowed: bool, size_multiplier: float, reason: str).
            ``size_multiplier`` is 1.0 (full size), 0.5 (reduce by 50 %), or 0.0 (reject).
        """
        for pos in existing_positions:
            sym = pos.get("symbol", "")
            if not sym or sym == new_symbol:
                continue
            corr = self._get_pairwise_correlation(new_symbol, sym)
            if corr >= _REJECT_THRESHOLD:
                reason = (
                    f"{new_symbol} is {corr:.0%} correlated with existing position {sym} "
                    f"(≥{_REJECT_THRESHOLD:.0%}) — trade rejected"
                )
                logger.warning("[CorrelationRisk] {}", reason)
                return False, 0.0, reason
            if corr >= _REDUCE_THRESHOLD:
                reason = (
                    f"{new_symbol} is {corr:.0%} correlated with {sym} "
                    f"(≥{_REDUCE_THRESHOLD:.0%}) — size reduced by 50 %"
                )
                logger.info("[CorrelationRisk] {}", reason)
                return True, 0.5, reason

        # Sector exposure check
        sector_allowed, sector_reason = self._check_sector_exposure(
            new_symbol, existing_positions
        )
        if not sector_allowed:
            return False, 0.0, sector_reason

        return True, 1.0, "OK"

    # ------------------------------------------------------------------
    # Anti-correlation seeking
    # ------------------------------------------------------------------

    def find_least_correlated(
        self,
        candidate_symbols: List[str],
        existing_positions: List[dict],
    ) -> Optional[str]:
        """Return the candidate that *reduces* portfolio correlation the most.

        Args:
            candidate_symbols: Symbols under consideration for the next trade.
            existing_positions: Currently open positions.

        Returns:
            Symbol with the lowest average correlation to existing positions,
            or ``None`` if the candidate list is empty.
        """
        if not candidate_symbols:
            return None
        existing_syms = [p.get("symbol", "") for p in existing_positions if p.get("symbol")]
        if not existing_syms:
            return candidate_symbols[0]

        scores: Dict[str, float] = {}
        for cand in candidate_symbols:
            corrs = [self._get_pairwise_correlation(cand, s) for s in existing_syms]
            scores[cand] = sum(corrs) / len(corrs)

        best = min(scores, key=lambda s: scores[s])
        logger.debug("[CorrelationRisk] Least correlated candidate: {} (avg={:.2f})", best, scores[best])
        return best

    # ------------------------------------------------------------------
    # Original public methods (unchanged signatures)
    # ------------------------------------------------------------------

    def is_correlated(
        self,
        symbol1: str,
        symbol2: str,
        threshold: float = 0.7,
    ) -> bool:
        """Return True if the two symbols are correlated above *threshold*.

        Args:
            symbol1: First trading symbol.
            symbol2: Second trading symbol.
            threshold: Correlation coefficient above which assets are considered correlated.

        Returns:
            ``True`` if the pair is correlated.
        """
        corr = self._get_pairwise_correlation(symbol1, symbol2)
        result = corr >= threshold
        logger.debug(
            "Correlation {} vs {}: {:.2f} (threshold={}) correlated={}",
            symbol1,
            symbol2,
            corr,
            threshold,
            result,
        )
        return result

    def _get_pairwise_correlation(self, symbol1: str, symbol2: str) -> float:
        """Look up the correlation between *symbol1* and *symbol2*.

        Priority: real-time cache → static map → default.
        """
        if symbol1 == symbol2:
            return 1.0
        key = frozenset({symbol1, symbol2})
        # Prefer real-time correlation when available
        if key in self._realtime_cache:
            return self._realtime_cache[key]
        return _CORRELATION_MAP.get(key, self._default_correlation)

    def _check_sector_exposure(
        self,
        new_symbol: str,
        existing_positions: List[dict],
    ) -> Tuple[bool, str]:
        """Return (allowed, reason) based on sector concentration limits."""
        new_sector = _SECTOR_MAP.get(new_symbol, "other")
        if new_sector == "other":
            return True, "OK"

        sector_count = sum(
            1
            for p in existing_positions
            if _SECTOR_MAP.get(p.get("symbol", ""), "other") == new_sector
        )
        if sector_count >= _MAX_SECTOR_EXPOSURE:
            reason = (
                f"Already have {sector_count} open {new_sector!r} positions "
                f"(max {_MAX_SECTOR_EXPOSURE}) — {new_symbol} rejected"
            )
            logger.warning("[CorrelationRisk] Sector limit: {}", reason)
            return False, reason
        return True, "OK"

    def calculate_portfolio_correlation(self, positions: List[dict]) -> float:
        """Return the average pairwise correlation across all open positions.

        Args:
            positions: List of position dicts, each containing at least ``{"symbol": str}``.

        Returns:
            Average pairwise correlation coefficient (0–1).
        """
        symbols = [p.get("symbol", "") for p in positions]
        if len(symbols) < 2:
            return 0.0
        total, count = 0.0, 0
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1 :]:
                total += self._get_pairwise_correlation(s1, s2)
                count += 1
        avg = total / count if count > 0 else 0.0
        logger.debug("Average portfolio correlation: {:.2f} ({} pairs)", avg, count)
        return avg

    def get_effective_positions(
        self,
        positions: List[dict],
        threshold: float = 0.7,
    ) -> int:
        """Count distinct (non-correlated) position groups.

        Correlated positions are merged into a single effective position.

        Args:
            positions: List of position dicts.
            threshold: Correlation threshold above which positions are considered the same.

        Returns:
            Effective number of independent positions.
        """
        symbols = [p.get("symbol", "") for p in positions]
        if not symbols:
            return 0

        # Simple union-find grouping
        parent = {s: s for s in symbols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent[find(a)] = find(b)

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1 :]:
                if self._get_pairwise_correlation(s1, s2) >= threshold:
                    union(s1, s2)

        effective = len({find(s) for s in symbols})
        logger.debug("Effective positions: {} (from {} raw positions)", effective, len(symbols))
        return effective

    def recommend_hedge(self, positions: List[dict]) -> List[dict]:
        """Suggest hedging instruments for highly correlated portfolios.

        Args:
            positions: List of position dicts with ``symbol`` and ``side`` keys.

        Returns:
            List of hedge recommendation dicts with ``symbol`` and ``action`` keys.
        """
        recommendations: List[dict] = []
        avg_corr = self.calculate_portfolio_correlation(positions)
        if avg_corr < 0.7:
            return recommendations

        # Recommend a BTC inverse hedge if the portfolio is long-heavy
        long_count = sum(1 for p in positions if p.get("side", "") == "long")
        if long_count > len(positions) // 2:
            recommendations.append(
                {
                    "symbol": "BTC/USDT",
                    "action": "short",
                    "reason": f"High portfolio correlation ({avg_corr:.2f}) with long bias",
                    "suggested_size_pct": 10.0,
                }
            )
        logger.info("Hedge recommendations: {}", recommendations)
        return recommendations
