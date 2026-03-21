"""Order flow analysis from Level 2 order book data.

.. deprecated::
    This module is superseded by the **Synthetic L3 Microstructure Engine**
    (``rust_engine/src/microstructure.rs`` + Python bridge).

    The Rust engine provides:
    * Synthetic queue-position estimation (``SyntheticQueueTracker``)
    * Enhanced VPIN with Lee-Ready trade classification (``EnhancedVpin``)
    * Kyle's Lambda rolling estimator (``KyleLambdaEstimator``)
    * Book-pressure gradient + spoofing detection (``BookPressureAnalyzer``)
    * Composite microstructure edge score (``MicrostructureEngine``)

    All of the above run at sub-microsecond latency in the Rust hot path.
    This Python module is retained for backward compatibility only and will
    be removed in a future release.  Please migrate to the Rust engine.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "ai.market_analyzer.order_flow_analyzer is deprecated. "
    "Use the Rust MicrostructureEngine (rust_engine/src/microstructure.rs) instead.",
    DeprecationWarning,
    stacklevel=2,
)

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class OrderFlowSignal:
    """Order flow signal output."""
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    confidence: float  # 0-1
    imbalance: float  # bid-ask imbalance ratio
    delta: float  # cumulative volume delta
    absorption_detected: bool
    iceberg_detected: bool
    timestamp: float


class OrderFlowAnalyzer:
    """Analyze Level 2 order book data for microstructure signals.

    Computes:
    - Order flow imbalance (bid vs ask volume)
    - Cumulative delta (aggressive buys vs sells)
    - Delta divergence (price vs delta)
    - Absorption detection (large resting orders)
    - Iceberg order detection
    - Book pressure gradient
    - Depth of Market (DOM) metrics
    """

    def __init__(
        self,
        history_length: int = 100,
        large_order_std_threshold: float = 2.0,
        absorption_threshold: float = 0.7,
    ) -> None:
        """Initialize order flow analyzer.

        Args:
            history_length: Number of recent updates to track.
            large_order_std_threshold: Standard deviations for large order detection.
            absorption_threshold: Threshold for absorption detection (0-1).
        """
        self.history_length = history_length
        self.large_order_std_threshold = large_order_std_threshold
        self.absorption_threshold = absorption_threshold

        # Order book history: deque of (timestamp, bids, asks) tuples
        self._book_history: Deque[Tuple[float, List, List]] = deque(maxlen=history_length)

        # Trade history: deque of (timestamp, price, volume, side) tuples
        self._trade_history: Deque[Tuple[float, float, float, str]] = deque(maxlen=history_length)

        # Cumulative delta
        self._cumulative_delta: float = 0.0

        # Order size statistics
        self._order_sizes: Deque[float] = deque(maxlen=history_length)

        # Price levels with repeated fills (iceberg detection)
        self._price_level_fills: Dict[float, int] = {}

        logger.info("OrderFlowAnalyzer initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_orderbook(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        """Update with new order book snapshot.

        Args:
            bids: List of (price, volume) tuples for bids (descending price).
            asks: List of (price, volume) tuples for asks (ascending price).
        """
        timestamp = time.time()
        self._book_history.append((timestamp, bids, asks))

        # Track order sizes for statistics
        for price, volume in bids + asks:
            self._order_sizes.append(volume)

    def update_trade(
        self,
        price: float,
        volume: float,
        side: str,  # "buy" or "sell"
    ) -> None:
        """Update with new trade (aggressive fill).

        Args:
            price: Trade price.
            volume: Trade volume.
            side: "buy" (aggressive buy) or "sell" (aggressive sell).
        """
        timestamp = time.time()
        self._trade_history.append((timestamp, price, volume, side))

        # Update cumulative delta
        if side == "buy":
            self._cumulative_delta += volume
        elif side == "sell":
            self._cumulative_delta -= volume

        # Track fills at each price level (iceberg detection)
        price_key = round(price, 2)
        self._price_level_fills[price_key] = self._price_level_fills.get(price_key, 0) + 1

    def analyze(self) -> OrderFlowSignal:
        """Analyze current order flow and return signal.

        Returns:
            OrderFlowSignal with direction, strength, and confidence.
        """
        if not self._book_history:
            return self._neutral_signal()

        # Get latest book
        _, bids, asks = self._book_history[-1]

        if not bids or not asks:
            return self._neutral_signal()

        # 1. Order flow imbalance
        imbalance = self._compute_imbalance(bids, asks)

        # 2. Book pressure (bid/ask depth ratio)
        pressure = self._compute_book_pressure(bids, asks)

        # 3. Delta analysis
        delta_signal = self._analyze_delta()

        # 4. Absorption detection
        absorption_detected = self._detect_absorption(bids, asks)

        # 5. Iceberg detection
        iceberg_detected = self._detect_iceberg()

        # 6. Large order detection
        large_orders = self._detect_large_orders(bids, asks)

        # Combine signals
        direction, strength, confidence = self._combine_signals(
            imbalance=imbalance,
            pressure=pressure,
            delta_signal=delta_signal,
            absorption=absorption_detected,
            iceberg=iceberg_detected,
            large_orders=large_orders,
        )

        signal = OrderFlowSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            imbalance=imbalance,
            delta=self._cumulative_delta,
            absorption_detected=absorption_detected,
            iceberg_detected=iceberg_detected,
            timestamp=time.time(),
        )

        logger.debug(
            f"OrderFlow: {direction} (strength={strength:.2f}, conf={confidence:.2f}, "
            f"imbalance={imbalance:.2f})"
        )

        return signal

    # ------------------------------------------------------------------
    # Analysis components
    # ------------------------------------------------------------------

    def _compute_imbalance(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> float:
        """Compute order flow imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).

        Uses top N levels (default 10).
        """
        n_levels = 10
        bid_volume = sum(vol for _, vol in bids[:n_levels])
        ask_volume = sum(vol for _, vol in asks[:n_levels])

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / total
        return imbalance

    def _compute_book_pressure(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> float:
        """Compute book pressure gradient.

        Measures steepness of bid vs ask side.
        """
        if len(bids) < 5 or len(asks) < 5:
            return 0.0

        # Compute volume slope on each side
        bid_volumes = [vol for _, vol in bids[:5]]
        ask_volumes = [vol for _, vol in asks[:5]]

        bid_slope = np.polyfit(range(5), bid_volumes, 1)[0]
        ask_slope = np.polyfit(range(5), ask_volumes, 1)[0]

        # Positive pressure = bids steeper than asks
        pressure = bid_slope - ask_slope
        return pressure

    def _analyze_delta(self) -> str:
        """Analyze cumulative delta for divergence.

        Returns "bullish", "bearish", or "neutral".
        """
        if not self._trade_history or len(self._trade_history) < 10:
            return "neutral"

        # Get recent price trend
        recent_trades = list(self._trade_history)[-10:]
        prices = [price for _, price, _, _ in recent_trades]
        price_change = prices[-1] - prices[0]

        # Check for divergence
        delta = self._cumulative_delta

        if price_change > 0 and delta < 0:
            return "bearish"  # Price up but delta down = bearish divergence
        elif price_change < 0 and delta > 0:
            return "bullish"  # Price down but delta up = bullish divergence
        elif delta > 0:
            return "bullish"
        elif delta < 0:
            return "bearish"
        else:
            return "neutral"

    def _detect_absorption(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> bool:
        """Detect absorption (large resting orders absorbing aggressive flow).

        Returns True if absorption detected.
        """
        if len(self._book_history) < 5:
            return False

        # Compare current book depth to recent average
        current_bid_depth = sum(vol for _, vol in bids[:5])
        current_ask_depth = sum(vol for _, vol in asks[:5])

        # Average depth from recent history
        recent_books = list(self._book_history)[-5:]
        avg_bid_depth = np.mean([
            sum(vol for _, vol in b[:5]) for _, b, _ in recent_books
        ])
        avg_ask_depth = np.mean([
            sum(vol for _, vol in a[:5]) for _, _, a in recent_books
        ])

        # Absorption: current depth is significantly higher than average
        bid_absorption = current_bid_depth > avg_bid_depth * (1 + self.absorption_threshold)
        ask_absorption = current_ask_depth > avg_ask_depth * (1 + self.absorption_threshold)

        return bid_absorption or ask_absorption

    def _detect_iceberg(self) -> bool:
        """Detect iceberg orders (repeated fills at same price level).

        Returns True if iceberg detected.
        """
        if not self._price_level_fills:
            return False

        # Check if any price level has abnormally high fill count
        fill_counts = list(self._price_level_fills.values())
        if len(fill_counts) < 3:
            return False

        mean_fills = np.mean(fill_counts)
        std_fills = np.std(fill_counts)

        if std_fills == 0:
            return False

        max_fills = max(fill_counts)
        z_score = (max_fills - mean_fills) / std_fills

        return z_score > 2.0

    def _detect_large_orders(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> List[Tuple[float, float, str]]:
        """Detect large orders (> N std deviations from mean).

        Returns list of (price, volume, side) tuples.
        """
        if len(self._order_sizes) < 10:
            return []

        mean_size = np.mean(self._order_sizes)
        std_size = np.std(self._order_sizes)

        if std_size == 0:
            return []

        threshold = mean_size + self.large_order_std_threshold * std_size

        large_orders = []
        for price, volume in bids[:10]:
            if volume > threshold:
                large_orders.append((price, volume, "bid"))

        for price, volume in asks[:10]:
            if volume > threshold:
                large_orders.append((price, volume, "ask"))

        return large_orders

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def _combine_signals(
        self,
        imbalance: float,
        pressure: float,
        delta_signal: str,
        absorption: bool,
        iceberg: bool,
        large_orders: List,
    ) -> Tuple[str, float, float]:
        """Combine all signals into final direction, strength, confidence.

        Returns (direction, strength, confidence).
        """
        # Base signal from imbalance
        if imbalance > 0.3:
            direction = "bullish"
            strength = min(1.0, abs(imbalance))
        elif imbalance < -0.3:
            direction = "bearish"
            strength = min(1.0, abs(imbalance))
        else:
            direction = "neutral"
            strength = 0.5

        # Adjust with delta signal
        if delta_signal == "bullish" and direction == "bullish":
            strength += 0.15
        elif delta_signal == "bearish" and direction == "bearish":
            strength += 0.15
        elif delta_signal != "neutral" and delta_signal != direction:
            strength -= 0.1  # Conflicting signals reduce strength

        # Adjust with pressure
        if pressure > 0 and direction == "bullish":
            strength += 0.1
        elif pressure < 0 and direction == "bearish":
            strength += 0.1

        # Absorption and iceberg reduce confidence (hidden liquidity)
        confidence = 0.7
        if absorption:
            confidence -= 0.15
        if iceberg:
            confidence -= 0.1

        # Large orders increase confidence
        if large_orders:
            confidence += 0.1

        # Clamp values
        strength = max(0.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))

        return direction, strength, confidence

    def _neutral_signal(self) -> OrderFlowSignal:
        """Return neutral signal when no data available."""
        return OrderFlowSignal(
            direction="neutral",
            strength=0.5,
            confidence=0.0,
            imbalance=0.0,
            delta=0.0,
            absorption_detected=False,
            iceberg_detected=False,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracking state."""
        self._book_history.clear()
        self._trade_history.clear()
        self._cumulative_delta = 0.0
        self._order_sizes.clear()
        self._price_level_fills.clear()
        logger.info("OrderFlowAnalyzer: reset state")

    def __repr__(self) -> str:
        return (
            f"OrderFlowAnalyzer(book_updates={len(self._book_history)}, "
            f"trades={len(self._trade_history)}, delta={self._cumulative_delta:.2f})"
        )
