"""Multi-timeframe analyzer — aligns signals across 1m, 5m, 15m, 1h, 4h, 1d."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from loguru import logger

from technical.indicators import TechnicalIndicators
from technical.market_structure import MarketStructureAnalyzer

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

_indicators = TechnicalIndicators()
_structure = MarketStructureAnalyzer()


class MultiTimeframeAnalyzer:
    """Performs top-down multi-timeframe analysis.

    The analyzer fetches OHLCV data for each standard timeframe and
    computes a bias (bullish / bearish / ranging) per timeframe.
    Results are combined to produce an overall signal.

    Requires an exchange object to be provided via :meth:`set_exchange`.
    """

    def __init__(self, exchange: Optional[Any] = None) -> None:
        self._exchange = exchange

    def set_exchange(self, exchange: Any) -> None:
        """Inject the exchange client used for data fetching."""
        self._exchange = exchange

    async def analyze_all_timeframes(self, symbol: str) -> Dict[str, Any]:
        """Fetch data for every standard timeframe and return a per-TF analysis.

        Returns a dict keyed by timeframe string, each value being a dict
        with keys: ``bias``, ``rsi``, ``ema_9``, ``ema_21``, ``structure``.
        """
        tasks = {tf: self._analyze_timeframe(symbol, tf) for tf in TIMEFRAMES}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        analyses: Dict[str, Any] = {}
        for tf, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"MTF analysis failed for {symbol} @ {tf}: {result}")
                analyses[tf] = {"bias": "ranging", "error": str(result)}
            else:
                analyses[tf] = result
        return analyses

    async def get_htf_bias(self, symbol: str) -> str:
        """Return the higher-timeframe bias for *symbol*.

        Uses the 4h and 1d timeframes; returns ``"bullish"``,
        ``"bearish"``, or ``"neutral"``.
        """
        tasks = [
            self._analyze_timeframe(symbol, "4h"),
            self._analyze_timeframe(symbol, "1d"),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        biases = []
        for r in results:
            if isinstance(r, dict):
                biases.append(r.get("bias", "ranging"))
        if not biases:
            return "neutral"
        bullish_count = biases.count("bullish")
        bearish_count = biases.count("bearish")
        if bullish_count > bearish_count:
            return "bullish"
        if bearish_count > bullish_count:
            return "bearish"
        return "neutral"

    async def check_timeframe_alignment(self, symbol: str) -> bool:
        """Return True if all standard timeframes agree on direction."""
        analyses = await self.analyze_all_timeframes(symbol)
        biases = [v.get("bias") for v in analyses.values() if isinstance(v, dict)]
        non_ranging = [b for b in biases if b in ("bullish", "bearish")]
        if len(non_ranging) < 3:
            return False
        return len(set(non_ranging)) == 1  # all agree

    async def combine_timeframe_signals(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate per-timeframe analyses into an overall view.

        Returns a dict with:
        * ``"overall_bias"`` — majority direction.
        * ``"aligned"`` — whether most timeframes agree.
        * ``"bullish_tfs"`` / ``"bearish_tfs"`` / ``"ranging_tfs"`` — counts.
        * ``"score"`` — net bias score (positive = bullish).
        """
        weights = {"1m": 0.5, "5m": 0.75, "15m": 1.0, "1h": 1.5, "4h": 2.0, "1d": 2.5}
        score = 0.0
        bullish = 0
        bearish = 0
        ranging = 0

        for tf, data in analyses.items():
            if not isinstance(data, dict):
                continue
            bias = data.get("bias", "ranging")
            w = weights.get(tf, 1.0)
            if bias == "bullish":
                score += w
                bullish += 1
            elif bias == "bearish":
                score -= w
                bearish += 1
            else:
                ranging += 1

        if score > 1.0:
            overall = "bullish"
        elif score < -1.0:
            overall = "bearish"
        else:
            overall = "ranging"

        aligned = bullish >= 4 or bearish >= 4
        return {
            "overall_bias": overall,
            "aligned": aligned,
            "score": round(score, 2),
            "bullish_tfs": bullish,
            "bearish_tfs": bearish,
            "ranging_tfs": ranging,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _analyze_timeframe(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Return a bias dict for a single timeframe."""
        if self._exchange is None:
            return {"bias": "ranging", "error": "No exchange attached"}

        try:
            ohlcv = await self._exchange.get_ohlcv(symbol, timeframe, 100)
            if ohlcv.empty or len(ohlcv) < 21:
                return {"bias": "ranging", "error": "Insufficient data"}

            closes = ohlcv["close"].tolist()
            rsi = _indicators.rsi(closes)
            ema9 = _indicators.ema(closes, 9)
            ema21 = _indicators.ema(closes, 21)
            structure = _structure.classify_structure(ohlcv)

            # Simple bias: EMA cross + structure
            if ema9 > ema21 and structure == "bullish":
                bias = "bullish"
            elif ema9 < ema21 and structure == "bearish":
                bias = "bearish"
            else:
                bias = "ranging"

            return {
                "bias": bias,
                "rsi": round(rsi, 2),
                "ema_9": round(ema9, 6),
                "ema_21": round(ema21, 6),
                "structure": structure,
            }
        except Exception as exc:
            logger.debug(f"_analyze_timeframe({symbol}, {timeframe}) error: {exc}")
            return {"bias": "ranging", "error": str(exc)}
