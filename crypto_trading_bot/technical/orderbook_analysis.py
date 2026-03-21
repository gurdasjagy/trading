"""Order book analyzer — depth imbalance, walls, spread, direction, and flow signals."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger


class OrderBookAnalyzer:
    """Analyses order book depth to detect imbalance, walls, spoofing, and direction.

    The *orderbook* argument accepted by all methods is a dict with the
    structure returned by CCXT / the exchange clients::

        {
            "bids": [[price, size], ...],  # sorted high → low
            "asks": [[price, size], ...],  # sorted low → high
        }

    Wall spoofing detection requires calling :meth:`update_snapshot` at
    regular intervals so that the history of wall appearances is tracked.
    """

    # Imbalance ratio threshold for strong directional signal
    _STRONG_IMBALANCE_RATIO: float = 3.0
    # History window for spoofing detection (number of snapshots)
    _WALL_HISTORY_SIZE: int = 20

    def __init__(self) -> None:
        # Track wall history: price → deque of (timestamp, size) tuples
        self._bid_wall_history: Dict[float, Deque[Tuple[float, float]]] = {}
        self._ask_wall_history: Dict[float, Deque[Tuple[float, float]]] = {}
        self._last_snapshot_ts: float = 0.0

    def analyze_imbalance(self, orderbook: Dict[str, Any]) -> float:
        """Return the bid/ask volume imbalance in [-1, +1].

        * ``+1.0`` — all volume on the bid side (strong buy pressure).
        * ``-1.0`` — all volume on the ask side (strong sell pressure).
        * ``0.0``  — perfectly balanced.
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        bid_vol = sum(row[1] for row in bids)
        ask_vol = sum(row[1] for row in asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return round((bid_vol - ask_vol) / total, 4)

    def find_bid_walls(
        self, orderbook: Dict[str, Any], wall_multiplier: float = 5.0
    ) -> List[Dict[str, float]]:
        """Return significant bid walls (large passive buy orders).

        A level is a "wall" if its size is *wall_multiplier* × the average
        bid size across all visible levels.
        """
        return self._find_walls(orderbook.get("bids", []), wall_multiplier)

    def find_ask_walls(
        self, orderbook: Dict[str, Any], wall_multiplier: float = 5.0
    ) -> List[Dict[str, float]]:
        """Return significant ask walls (large passive sell orders)."""
        return self._find_walls(orderbook.get("asks", []), wall_multiplier)

    def calculate_spread(self, orderbook: Dict[str, Any]) -> float:
        """Return the absolute bid-ask spread.

        Returns 0.0 if the order book is empty.
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return round(best_ask - best_bid, 8)

    def predict_short_term_direction(self, orderbook: Dict[str, Any]) -> str:
        """Return ``"long"``, ``"short"``, or ``"neutral"`` based on book imbalance.

        Thresholds
        ----------
        * imbalance > +0.2 → bullish (``"long"``).
        * imbalance < -0.2 → bearish (``"short"``).
        * otherwise → ``"neutral"``.
        """
        imbalance = self.analyze_imbalance(orderbook)
        if imbalance > 0.2:
            return "long"
        if imbalance < -0.2:
            return "short"
        return "neutral"

    def calculate_imbalance_ratio(self, orderbook: Dict[str, Any]) -> Tuple[float, str]:
        """Return the bid/ask imbalance ratio and direction signal.

        Unlike :meth:`analyze_imbalance` (which returns [-1, +1]), this method
        returns the raw volume ratio together with a directional label.

        A ratio ≥ :attr:`_STRONG_IMBALANCE_RATIO` indicates strong pressure.

        Args:
            orderbook: Order book snapshot dict.

        Returns:
            Tuple of (ratio: float, signal: str) where signal is one of
            ``"strong_buy"``, ``"strong_sell"``, ``"buy"``, ``"sell"``, or
            ``"neutral"``.
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        bid_vol = sum(float(r[1]) for r in bids)
        ask_vol = sum(float(r[1]) for r in asks)

        if ask_vol == 0 and bid_vol == 0:
            return 1.0, "neutral"

        if ask_vol == 0:
            ratio = float("inf")
            return ratio, "strong_buy"
        if bid_vol == 0:
            ratio = float("inf")
            return ratio, "strong_sell"

        if bid_vol >= ask_vol:
            ratio = bid_vol / ask_vol
            signal = "strong_buy" if ratio >= self._STRONG_IMBALANCE_RATIO else "buy"
        else:
            ratio = ask_vol / bid_vol
            signal = "strong_sell" if ratio >= self._STRONG_IMBALANCE_RATIO else "sell"

        logger.debug("[OrderBook] imbalance_ratio={:.2f} signal={}", ratio, signal)
        return round(ratio, 3), signal

    def generates_signal(self, orderbook: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a trading signal dict when imbalance ratio > 3:1.

        Can be used as a confirmation filter for other strategy signals.

        Args:
            orderbook: Order book snapshot.

        Returns:
            Dict with ``direction``, ``ratio``, ``confidence`` keys when a
            strong signal is detected, else ``None``.
        """
        ratio, signal = self.calculate_imbalance_ratio(orderbook)
        if signal in ("strong_buy", "strong_sell"):
            direction = "long" if signal == "strong_buy" else "short"
            confidence = min(1.0, round((ratio - self._STRONG_IMBALANCE_RATIO + 1) / 4.0, 3))
            return {
                "direction": direction,
                "ratio": ratio,
                "confidence": confidence,
                "signal": signal,
            }
        return None

    def update_snapshot(
        self, orderbook: Dict[str, Any], timestamp: Optional[float] = None
    ) -> None:
        """Record a snapshot of order book walls for spoofing detection.

        Call this method at regular intervals (e.g. every few seconds) to
        build the history required by :meth:`detect_spoofed_walls`.

        Args:
            orderbook: Current order book snapshot.
            timestamp: Unix timestamp (defaults to ``time.time()``).
        """
        ts = timestamp or time.time()
        self._last_snapshot_ts = ts

        bid_walls = self.find_bid_walls(orderbook)
        ask_walls = self.find_ask_walls(orderbook)

        for wall in bid_walls:
            price = wall["price"]
            if price not in self._bid_wall_history:
                self._bid_wall_history[price] = deque(maxlen=self._WALL_HISTORY_SIZE)
            self._bid_wall_history[price].append((ts, wall["size"]))

        for wall in ask_walls:
            price = wall["price"]
            if price not in self._ask_wall_history:
                self._ask_wall_history[price] = deque(maxlen=self._WALL_HISTORY_SIZE)
            self._ask_wall_history[price].append((ts, wall["size"]))

    def detect_spoofed_walls(
        self,
        max_age_seconds: float = 60.0,
        min_appearances: int = 2,
    ) -> List[Dict[str, Any]]:
        """Detect walls that appeared and then disappeared (spoofing pattern).

        A wall is considered potentially spoofed when it was seen in multiple
        recent snapshots but has not appeared in the *most recent* snapshot.

        Args:
            max_age_seconds: Only consider walls seen within this window.
            min_appearances: Minimum number of past appearances to flag.

        Returns:
            List of dicts with ``price``, ``side``, ``appearances``, and
            ``last_seen`` (timestamp) for each suspected spoofed wall.
        """
        now = self._last_snapshot_ts or time.time()
        spoofed: List[Dict[str, Any]] = []

        for side_label, history in (
            ("bid", self._bid_wall_history),
            ("ask", self._ask_wall_history),
        ):
            for price, snapshots in history.items():
                recent = [(ts, sz) for ts, sz in snapshots if now - ts <= max_age_seconds]
                if len(recent) < min_appearances:
                    continue
                last_ts = recent[-1][0]
                # If last appearance was not in the most recent snapshot window
                if now - last_ts > max_age_seconds / min_appearances:
                    spoofed.append({
                        "price": price,
                        "side": side_label,
                        "appearances": len(recent),
                        "last_seen": last_ts,
                    })
                    logger.debug(
                        "[OrderBook] Potential spoof wall: side={} price={} appearances={}",
                        side_label,
                        price,
                        len(recent),
                    )

        return spoofed

    def calculate_spread_pct(self, orderbook: Dict[str, Any]) -> float:
        """Return the bid-ask spread as a percentage of mid price.

        Args:
            orderbook: Order book snapshot.

        Returns:
            Spread percentage (e.g. 0.05 for 0.05 %), or 0.0 if data absent.
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        if mid == 0:
            return 0.0
        spread_pct = ((best_ask - best_bid) / mid) * 100.0
        return round(spread_pct, 6)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_walls(levels: List[List[float]], wall_multiplier: float) -> List[Dict[str, float]]:
        if not levels:
            return []
        sizes = [row[1] for row in levels]
        avg_size = sum(sizes) / len(sizes)
        threshold = avg_size * wall_multiplier
        return [
            {"price": float(row[0]), "size": float(row[1])} for row in levels if row[1] >= threshold
        ]
