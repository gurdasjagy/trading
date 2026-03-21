"""Market microstructure analyzer — spread, VPIN, stop hunts, OBV, and market quality."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass
class MicrostructureResult:
    """Result of market microstructure analysis."""

    market_quality_score: float  # 0–100
    should_trade: bool           # True when quality score > 60
    spread_pct: float
    spread_ok: bool              # True when spread < 0.1 %
    stop_hunt_detected: bool
    vpin: float                  # 0–1, higher = more informed trading
    obv_trend: str               # "bullish", "bearish", or "neutral"
    accumulation_distribution: str  # "accumulation", "distribution", or "neutral"
    reasons: List[str] = field(default_factory=list)


class MarketMicrostructureAnalyzer:
    """Analyses market microstructure to assess trade quality and detect anomalies.

    Components:
    * **Spread analysis**: avoid trading when spread > 0.1 %.
    * **VPIN** (Volume-Synchronized Probability of Informed Trading):
      detects informed trading activity (high VPIN = toxic flow).
    * **Stop hunt detection**: rapid price spike that reverses within 5 minutes.
    * **OBV divergence**: detects accumulation and distribution phases.
    * **Market quality score** (0–100): composite score from all factors.
    """

    _SPREAD_THRESHOLD_PCT: float = 0.1   # 0.1 %
    _MIN_QUALITY_SCORE: float = 60.0
    _STOP_HUNT_SPIKE_PCT: float = 0.005  # 0.5 % spike
    _STOP_HUNT_WINDOW_SECONDS: float = 300.0  # 5 minutes
    _VPIN_BUCKETS: int = 50

    def __init__(self) -> None:
        # Recent price ticks for stop-hunt detection: (timestamp, price)
        self._price_ticks: Deque[Tuple[float, float]] = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        orderbook: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
    ) -> MicrostructureResult:
        """Run full microstructure analysis.

        Args:
            ohlcv: OHLCV DataFrame (at least 20 rows recommended).
            orderbook: Current order book snapshot (optional).
            current_price: Latest trade price for stop hunt detection.

        Returns:
            :class:`MicrostructureResult` with composite quality score.
        """
        reasons: List[str] = []

        # 1. Spread analysis
        spread_pct = 0.0
        spread_ok = True
        if orderbook:
            spread_pct = self._calculate_spread_pct(orderbook)
            spread_ok = spread_pct < self._SPREAD_THRESHOLD_PCT
            if not spread_ok:
                reasons.append(f"High spread: {spread_pct:.3f}%")

        # 2. Stop hunt detection
        stop_hunt = False
        if current_price is not None:
            self._price_ticks.append((time.time(), current_price))
            stop_hunt = self._detect_stop_hunt()
            if stop_hunt:
                reasons.append("Stop hunt pattern detected")

        # 3. VPIN
        vpin = self._calculate_vpin(ohlcv) if not ohlcv.empty else 0.5
        if vpin > 0.7:
            reasons.append(f"High VPIN ({vpin:.2f}) — informed trading likely")

        # 4. OBV trend and accumulation/distribution
        obv_trend = "neutral"
        acc_dist = "neutral"
        if not ohlcv.empty and len(ohlcv) >= 10:
            obv_trend = self._calculate_obv_trend(ohlcv)
            acc_dist = self._calculate_acc_dist(ohlcv)

        # 5. Composite quality score (0–100)
        score = self._calculate_quality_score(
            spread_ok=spread_ok,
            stop_hunt=stop_hunt,
            vpin=vpin,
            obv_trend=obv_trend,
        )

        should_trade = score >= self._MIN_QUALITY_SCORE

        if not should_trade:
            reasons.append(f"Quality score {score:.0f} < threshold {self._MIN_QUALITY_SCORE:.0f}")

        logger.debug(
            "[Microstructure] quality={:.0f} spread={:.3f}% stop_hunt={} vpin={:.2f} obv={}",
            score,
            spread_pct,
            stop_hunt,
            vpin,
            obv_trend,
        )
        return MicrostructureResult(
            market_quality_score=round(score, 1),
            should_trade=should_trade,
            spread_pct=round(spread_pct, 4),
            spread_ok=spread_ok,
            stop_hunt_detected=stop_hunt,
            vpin=round(vpin, 3),
            obv_trend=obv_trend,
            accumulation_distribution=acc_dist,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_spread_pct(orderbook: Dict[str, Any]) -> float:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        if mid == 0:
            return 0.0
        return ((best_ask - best_bid) / mid) * 100.0

    # ------------------------------------------------------------------
    # Stop hunt detection
    # ------------------------------------------------------------------

    def _detect_stop_hunt(self) -> bool:
        """Detect a rapid spike-and-reversal pattern in recent price ticks."""
        if len(self._price_ticks) < 3:
            return False
        now = time.time()
        cutoff = now - self._STOP_HUNT_WINDOW_SECONDS
        recent = [(ts, p) for ts, p in self._price_ticks if ts >= cutoff]
        if len(recent) < 3:
            return False

        prices = [p for _, p in recent]
        baseline = prices[0]
        if baseline <= 0:
            return False

        max_p = max(prices)
        min_p = min(prices)
        last_p = prices[-1]

        # Spike up then revert
        if (max_p - baseline) / baseline >= self._STOP_HUNT_SPIKE_PCT:
            if (max_p - last_p) / max_p >= self._STOP_HUNT_SPIKE_PCT * 0.8:
                return True

        # Spike down then revert
        if (baseline - min_p) / baseline >= self._STOP_HUNT_SPIKE_PCT:
            if last_p > 0 and (last_p - min_p) / last_p >= self._STOP_HUNT_SPIKE_PCT * 0.8:
                return True

        return False

    def record_price_tick(self, price: float, timestamp: Optional[float] = None) -> None:
        """Manually record a price tick for stop hunt detection.

        Args:
            price: Latest trade price.
            timestamp: Unix timestamp (defaults to now).
        """
        ts = timestamp or time.time()
        self._price_ticks.append((ts, price))

    # ------------------------------------------------------------------
    # VPIN (simplified)
    # ------------------------------------------------------------------

    def _calculate_vpin(self, ohlcv: pd.DataFrame) -> float:
        """Calculate a simplified VPIN estimate from OHLCV data.

        The full VPIN uses tick data; this version approximates using
        the ratio of directional volume (up candles vs. down candles)
        normalised by total volume.

        Returns:
            VPIN estimate in [0, 1].  Values > 0.5 indicate seller-initiated,
            values < 0.5 indicate buyer-initiated.  Values near 0.5 = balanced.
        """
        if ohlcv.empty or len(ohlcv) < self._VPIN_BUCKETS:
            return 0.5
        try:
            df = ohlcv.tail(self._VPIN_BUCKETS).copy()
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["open"] = df["open"].astype(float)

            buy_vol = df.loc[df["close"] >= df["open"], "volume"].sum()
            sell_vol = df.loc[df["close"] < df["open"], "volume"].sum()
            total = buy_vol + sell_vol
            if total == 0:
                return 0.5
            # |buy - sell| / total — higher = more imbalance (informed trading)
            vpin = abs(buy_vol - sell_vol) / total
            return min(1.0, float(vpin))
        except Exception as exc:
            logger.debug("[Microstructure] VPIN calculation error: {}", exc)
            return 0.5

    # ------------------------------------------------------------------
    # On-Balance Volume trend
    # ------------------------------------------------------------------

    def _calculate_obv_trend(self, ohlcv: pd.DataFrame) -> str:
        """Calculate OBV trend to detect accumulation/distribution.

        Returns:
            ``"bullish"``, ``"bearish"``, or ``"neutral"``.
        """
        try:
            closes = ohlcv["close"].astype(float)
            volumes = ohlcv["volume"].astype(float)
            obv = [0.0]
            for i in range(1, len(closes)):
                if closes.iloc[i] > closes.iloc[i - 1]:
                    obv.append(obv[-1] + volumes.iloc[i])
                elif closes.iloc[i] < closes.iloc[i - 1]:
                    obv.append(obv[-1] - volumes.iloc[i])
                else:
                    obv.append(obv[-1])

            if len(obv) < 5:
                return "neutral"

            # Compare recent OBV trend
            recent = obv[-5:]
            first_half = sum(recent[:2]) / 2
            second_half = sum(recent[3:]) / 2
            if second_half > first_half * 1.01:
                return "bullish"
            elif second_half < first_half * 0.99:
                return "bearish"
            return "neutral"
        except Exception as exc:
            logger.debug("[Microstructure] OBV error: {}", exc)
            return "neutral"

    def _calculate_acc_dist(self, ohlcv: pd.DataFrame) -> str:
        """Determine accumulation or distribution using the A/D line.

        Returns:
            ``"accumulation"``, ``"distribution"``, or ``"neutral"``.
        """
        try:
            high = ohlcv["high"].astype(float)
            low = ohlcv["low"].astype(float)
            close = ohlcv["close"].astype(float)
            volume = ohlcv["volume"].astype(float)

            hl_range = high - low
            clv = ((2 * close - low - high) / hl_range.replace(0, float("nan"))).fillna(0)
            ad = (clv * volume).cumsum()

            if len(ad) < 5:
                return "neutral"
            recent = ad.tail(5)
            slope = float(recent.iloc[-1] - recent.iloc[0])
            if slope > 0:
                return "accumulation"
            elif slope < 0:
                return "distribution"
            return "neutral"
        except Exception as exc:
            logger.debug("[Microstructure] A/D error: {}", exc)
            return "neutral"

    # ------------------------------------------------------------------
    # Composite quality score
    # ------------------------------------------------------------------

    def _calculate_quality_score(
        self,
        spread_ok: bool,
        stop_hunt: bool,
        vpin: float,
        obv_trend: str,
    ) -> float:
        """Compute a 0–100 market quality score.

        Scoring:
        * Spread OK:           +30 pts
        * No stop hunt:        +25 pts
        * Low VPIN (< 0.5):    +25 pts  (scaled)
        * OBV not bearish:     +20 pts
        """
        score = 0.0
        score += 30.0 if spread_ok else 0.0
        score += 25.0 if not stop_hunt else 0.0
        # VPIN: lower = better (balanced flow)
        score += max(0.0, 25.0 * (1.0 - vpin * 2)) if vpin < 0.5 else 0.0
        score += 20.0 if obv_trend != "bearish" else 0.0
        return min(100.0, score)
