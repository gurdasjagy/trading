"""Tick-Level Data Processor.

.. deprecated::
    This module is superseded by the **Synthetic L3 Microstructure Engine**
    (``rust_engine/src/microstructure.rs``).

    The Rust engine provides enhanced equivalents:
    * ``EnhancedVpin`` — VPIN with Lee-Ready trade classification
    * ``KyleLambdaEstimator`` — rolling Kyle's Lambda (price impact)
    * ``MicrostructureEngine.process_tick_with_book()`` — per-tick processing
      with book mid-price for accurate trade classification

    This Python module is retained for backward compatibility only.

Processes raw trade ticks from WebSocket streams into real-time
microstructure signals used by strategies that require tick data:

* Real-time VWAP
* Tick imbalance (buy vs sell aggressor volume)
* Volume-weighted mid-price
* VPIN (Volume-synchronized Probability of Informed Trading)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "data.tick_processor.TickProcessor is deprecated. "
    "Use the Rust MicrostructureEngine (rust_engine/src/microstructure.rs) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from collections import deque
from typing import Dict, List

import numpy as np

try:
    from rust_trading_engine.tick_processor import RustTickProcessor
    _USE_RUST_TICK = True
except ImportError:
    _USE_RUST_TICK = False


class TickProcessor:
    """Process raw trade ticks into real-time microstructure metrics.

    Args:
        window_size: Number of ticks to retain in the rolling window.
        vpin_bucket_size: Target volume per VPIN bucket.
    """

    def __init__(
        self,
        window_size: int = 1000,
        vpin_bucket_size: float = 1000.0,
    ) -> None:
        self._window = window_size
        self._vpin_bucket_size = vpin_bucket_size
        if _USE_RUST_TICK:
            self._rust_engine = RustTickProcessor(window_size, vpin_bucket_size)
        else:
            self._rust_engine = None

        # Per-symbol rolling tick buffers  {symbol: deque of tick dicts}
        self._ticks: Dict[str, deque] = {}
        # VPIN bucket accumulators  {symbol: {"buy_vol": float, "sell_vol": float, "total_vol": float}}
        self._vpin_buckets: Dict[str, List[float]] = {}
        self._vpin_current: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def process_tick(self, symbol: str, tick: dict) -> None:
        """Ingest a single raw trade tick.

        Expected tick keys: ``price`` (or ``last``), ``amount`` (or ``size``),
        ``side`` (``"buy"`` or ``"sell"``).
        """
        if self._rust_engine is not None:
            try:
                price = float(tick.get("price") or tick.get("last") or 0.0)
                amount = float(tick.get("amount") or tick.get("size") or 0.0)
                side = str(tick.get("side") or "").lower()
                if price > 0 and amount > 0:
                    self._rust_engine.process_tick(symbol, price, amount, side)
                    return
            except Exception:
                pass  # Fall through to Python
        # Original Python implementation (unchanged)
        if symbol not in self._ticks:
            self._ticks[symbol] = deque(maxlen=self._window)
        self._ticks[symbol].append(tick)
        self._update_vpin(symbol, tick)

    def process_ticks(self, symbol: str, ticks: List[dict]) -> None:
        """Ingest a batch of ticks for *symbol*."""
        for tick in ticks:
            self.process_tick(symbol, tick)

    # ------------------------------------------------------------------
    # Real-time metrics
    # ------------------------------------------------------------------

    def get_vwap(self, symbol: str) -> float:
        """Return the volume-weighted average price over the rolling window."""
        if self._rust_engine is not None:
            return self._rust_engine.get_vwap(symbol)
        ticks = self._ticks.get(symbol)
        if not ticks:
            return 0.0
        total_vol = 0.0
        total_pv = 0.0
        for t in ticks:
            price = float(t.get("price") or t.get("last") or 0.0)
            vol = float(t.get("amount") or t.get("size") or 0.0)
            if price > 0 and vol > 0:
                total_pv += price * vol
                total_vol += vol
        if total_vol <= 0:
            return 0.0
        return total_pv / total_vol

    def get_tick_imbalance(self, symbol: str) -> float:
        """Return tick imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol).

        Range: [-1, +1].  Positive values indicate aggressive buying.
        """
        if self._rust_engine is not None:
            return self._rust_engine.get_tick_imbalance(symbol)
        ticks = self._ticks.get(symbol)
        if not ticks:
            return 0.0
        buy_vol = 0.0
        sell_vol = 0.0
        for t in ticks:
            side = str(t.get("side") or "").lower()
            vol = float(t.get("amount") or t.get("size") or 0.0)
            if side == "buy":
                buy_vol += vol
            elif side == "sell":
                sell_vol += vol
        total = buy_vol + sell_vol
        if total <= 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def get_vwap_mid_price(self, symbol: str, bid: float, ask: float) -> float:
        """Return volume-weighted mid-price blending VWAP with order-book mid.

        Uses the formula: 0.5 × (VWAP + (bid + ask) / 2).
        """
        if self._rust_engine is not None:
            return self._rust_engine.get_vwap_mid_price(symbol, bid, ask)
        vwap = self.get_vwap(symbol)
        book_mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        if vwap <= 0:
            return book_mid
        if book_mid <= 0:
            return vwap
        return (vwap + book_mid) / 2.0

    def get_vpin(self, symbol: str) -> float:
        """Return the current VPIN estimate.

        VPIN ≈ |buy_vol - sell_vol| / total_vol averaged across completed
        buckets.  Range [0, 1]; higher values indicate higher trade-flow
        toxicity (informed trading).
        """
        if self._rust_engine is not None:
            return self._rust_engine.get_vpin(symbol)
        buckets = self._vpin_buckets.get(symbol)
        if not buckets or len(buckets) < 2:
            return 0.0
        return float(np.mean(buckets[-50:]))  # average last 50 bucket estimates

    def get_metrics(self, symbol: str) -> dict:
        """Return all microstructure metrics for *symbol* as a dict."""
        if self._rust_engine is not None:
            return self._rust_engine.get_metrics(symbol)
        return {
            "vwap": self.get_vwap(symbol),
            "tick_imbalance": self.get_tick_imbalance(symbol),
            "vpin": self.get_vpin(symbol),
            "tick_count": len(self._ticks.get(symbol, [])),
        }

    # ------------------------------------------------------------------
    # VPIN bucket accumulation
    # ------------------------------------------------------------------

    def _update_vpin(self, symbol: str, tick: dict) -> None:
        """Update the current VPIN bucket with the incoming tick."""
        if symbol not in self._vpin_current:
            self._vpin_current[symbol] = {
                "buy_vol": 0.0,
                "sell_vol": 0.0,
                "total_vol": 0.0,
            }
        if symbol not in self._vpin_buckets:
            self._vpin_buckets[symbol] = []

        side = str(tick.get("side") or "").lower()
        vol = float(tick.get("amount") or tick.get("size") or 0.0)

        bucket = self._vpin_current[symbol]
        if side == "buy":
            bucket["buy_vol"] += vol
        elif side == "sell":
            bucket["sell_vol"] += vol
        bucket["total_vol"] += vol

        # Close bucket when target volume is reached
        if bucket["total_vol"] >= self._vpin_bucket_size:
            total = bucket["total_vol"]
            if total > 0:
                estimate = abs(bucket["buy_vol"] - bucket["sell_vol"]) / total
                self._vpin_buckets[symbol].append(estimate)
                # Retain last 200 buckets
                if len(self._vpin_buckets[symbol]) > 200:
                    self._vpin_buckets[symbol] = self._vpin_buckets[symbol][-200:]
            # Reset bucket
            self._vpin_current[symbol] = {
                "buy_vol": 0.0,
                "sell_vol": 0.0,
                "total_vol": 0.0,
            }

