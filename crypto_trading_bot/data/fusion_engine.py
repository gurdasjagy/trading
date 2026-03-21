"""Data Fusion Engine — real-time multi-source alternative data fusion.

Combines signals from all alternative data sources into a single
``FusedSignal`` with:
* Weighted composite sentiment score
* Urgency-ranked DataItem list
* Per-asset and per-source breakdowns
* Staleness detection and automatic source health monitoring

The engine polls all registered sources on a background loop and maintains
a rolling window of the latest N fused signals per symbol.

Usage::

    engine = DataFusionEngine()
    engine.register_source(DeribitOptionsSource())
    engine.register_source(BinanceFuturesSource())
    engine.register_source(TokenUnlocksMonitor())
    await engine.start()

    signal = await engine.get_fused_signal("BTC")
    print(signal.composite_sentiment, signal.urgency_score)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from data.sources.base_source import BaseSource, DataItem

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class FusedSignal:
    """Composite alternative-data signal for a single asset.

    Attributes:
        asset: Base asset ticker (e.g. ``"BTC"``).
        composite_sentiment: Weighted average sentiment in ``[-1, +1]``.
        urgency_score: Maximum urgency across all contributing items.
        confidence: Fraction of active sources contributing signal (0–1).
        top_items: Top-5 DataItems ranked by urgency × relevance.
        source_breakdown: ``{source_name: sentiment_score}`` mapping.
        item_count: Total number of items contributing to this signal.
        timestamp: UTC datetime of fusion computation.
        staleness_seconds: Seconds since the oldest item included.
    """

    asset: str
    composite_sentiment: float
    urgency_score: float
    confidence: float
    top_items: List[DataItem] = field(default_factory=list)
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    item_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    staleness_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DataFusionEngine:
    """Fuses alternative data from multiple registered sources.

    Args:
        max_item_age_seconds: Items older than this are discarded before
            fusion (default 3 600 s = 1 hour).
        fusion_window_seconds: How frequently the background fusion loop
            runs (default 30 s).
        source_weights: Optional ``{source_name: weight}`` override.
            Sources not in the dict receive weight 1.0.
        history_maxlen: Maximum number of fused signals to keep per asset.
    """

    def __init__(
        self,
        max_item_age_seconds: float = 3_600.0,
        fusion_window_seconds: float = 30.0,
        source_weights: Optional[Dict[str, float]] = None,
        history_maxlen: int = 100,
    ) -> None:
        self.max_item_age_seconds = max_item_age_seconds
        self.fusion_window_seconds = fusion_window_seconds
        self._source_weights: Dict[str, float] = source_weights or {}
        self._history_maxlen = history_maxlen

        # Registered data sources
        self._sources: Dict[str, BaseSource] = {}

        # Item buffer: {source_name: deque of DataItems}
        self._item_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

        # Fused signal history per asset
        self._signal_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_maxlen)
        )

        # Latest fused signal per asset (for fast reads)
        self._latest_signals: Dict[str, FusedSignal] = {}

        # Callbacks invoked whenever a new fused signal is computed
        self._on_signal_callbacks: List[Callable[[FusedSignal], None]] = []

        # Background task handle
        self._fusion_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "DataFusionEngine initialised: max_age={}s, fusion_window={}s",
            max_item_age_seconds,
            fusion_window_seconds,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_source(self, source: BaseSource) -> None:
        """Register an alternative data source."""
        self._sources[source.name] = source
        logger.info("DataFusionEngine: registered source {!r}", source.name)

    def set_source_weight(self, source_name: str, weight: float) -> None:
        """Override the fusion weight for a specific source."""
        self._source_weights[source_name] = max(0.0, weight)

    def on_signal(self, callback: Callable[[FusedSignal], None]) -> None:
        """Register a callback invoked each time a new FusedSignal is produced."""
        self._on_signal_callbacks.append(callback)

    async def start(self) -> None:
        """Start all sources and the background fusion loop."""
        # Start individual source monitors
        for source in self._sources.values():
            if source.enabled:
                asyncio.create_task(source.start_monitoring())

        self._running = True
        self._fusion_task = asyncio.create_task(self._fusion_loop())
        logger.info(
            "DataFusionEngine started: {} sources",
            len(self._sources),
        )

    async def stop(self) -> None:
        """Stop the fusion loop and all sources."""
        self._running = False
        if self._fusion_task and not self._fusion_task.done():
            self._fusion_task.cancel()
        for source in self._sources.values():
            try:
                await source.stop_monitoring()
            except Exception as exc:
                logger.debug("Error stopping {}: {}", source.name, exc)
        logger.info("DataFusionEngine stopped")

    # ------------------------------------------------------------------
    # Public signal API
    # ------------------------------------------------------------------

    async def get_fused_signal(
        self,
        asset: str,
        force_refresh: bool = False,
    ) -> FusedSignal:
        """Return the latest fused signal for *asset*.

        Args:
            asset: Base asset ticker (e.g. ``"BTC"``).
            force_refresh: If ``True``, collect and fuse items immediately
                before returning (ignoring cached signal).

        Returns:
            :class:`FusedSignal` for the asset.  Returns a neutral signal
            if no data is available.
        """
        if force_refresh or asset not in self._latest_signals:
            await self._collect_and_fuse()

        return self._latest_signals.get(asset, self._neutral_signal(asset))

    def get_all_latest_signals(self) -> Dict[str, FusedSignal]:
        """Return the most recent fused signal for every asset."""
        return dict(self._latest_signals)

    def get_signal_history(self, asset: str) -> List[FusedSignal]:
        """Return the full history of fused signals for *asset*."""
        return list(self._signal_history.get(asset, []))

    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Return health stats for each registered source."""
        health: Dict[str, Dict[str, Any]] = {}
        for name, source in self._sources.items():
            health[name] = {
                "enabled": source.enabled,
                "items_collected": source._items_collected,
                "errors": source._errors,
                "last_update": source._last_update.isoformat() if source._last_update else None,
                "buffer_size": len(self._item_buffer[name]),
            }
        return health

    # ------------------------------------------------------------------
    # Internal fusion logic
    # ------------------------------------------------------------------

    async def _fusion_loop(self) -> None:
        """Background loop that periodically collects and fuses all sources."""
        while self._running:
            try:
                await self._collect_and_fuse()
            except Exception as exc:
                logger.error("DataFusionEngine fusion loop error: {}", exc)
            await asyncio.sleep(self.fusion_window_seconds)

    async def _collect_and_fuse(self) -> None:
        """Collect items from all sources, prune stale ones, and fuse."""
        # Collect latest items from every source
        for name, source in self._sources.items():
            if not source.enabled:
                continue
            try:
                items = await source.fetch_latest(limit=100)
                for item in items:
                    self._item_buffer[name].append(item)
            except Exception as exc:
                logger.debug("Fusion collect error {}: {}", name, exc)

        # Prune stale items
        self._prune_stale()

        # Build per-asset item lists
        asset_items: Dict[str, List[DataItem]] = defaultdict(list)
        for name, buf in self._item_buffer.items():
            for item in buf:
                for asset in item.mentioned_assets:
                    asset_items[asset.upper()].append(item)
                if not item.mentioned_assets:
                    asset_items["MARKET"].append(item)

        # Fuse per asset
        for asset, items in asset_items.items():
            signal = self._fuse(asset, items)
            self._latest_signals[asset] = signal
            self._signal_history[asset].append(signal)
            for cb in self._on_signal_callbacks:
                try:
                    cb(signal)
                except Exception:
                    pass

    def _fuse(self, asset: str, items: List[DataItem]) -> FusedSignal:
        """Compute a FusedSignal from a list of DataItems for *asset*."""
        if not items:
            return self._neutral_signal(asset)

        # Sort by urgency × relevance descending
        items_sorted = sorted(
            items,
            key=lambda i: float(i.urgency_score) * float(i.relevance_score),
            reverse=True,
        )

        # Compute weighted composite sentiment
        total_weight = 0.0
        weighted_sentiment = 0.0
        source_sentiments: Dict[str, List[float]] = defaultdict(list)
        max_urgency = 0.0
        oldest_ts: Optional[datetime] = None

        for item in items:
            source_weight = self._source_weights.get(item.source_name, 1.0)
            urgency = float(item.urgency_score or 0)
            relevance = float(item.relevance_score or 0)
            sentiment = float(item.metadata.get("sentiment_score", 0.0))

            w = source_weight * urgency * relevance
            weighted_sentiment += w * sentiment
            total_weight += w
            max_urgency = max(max_urgency, urgency)
            source_sentiments[item.source_name].append(sentiment)

            if item.timestamp and (oldest_ts is None or item.timestamp < oldest_ts):
                oldest_ts = item.timestamp

        composite = (weighted_sentiment / total_weight) if total_weight > 0 else 0.0
        composite = max(-1.0, min(1.0, composite))

        # Source breakdown (mean sentiment per source)
        source_breakdown = {
            src: sum(vals) / len(vals)
            for src, vals in source_sentiments.items()
        }

        # Confidence: fraction of active sources that contributed
        contributing_sources = len(source_sentiments)
        active_sources = max(1, sum(1 for s in self._sources.values() if s.enabled))
        confidence = min(1.0, contributing_sources / active_sources)

        # Staleness
        if oldest_ts:
            try:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                ts = oldest_ts.replace(tzinfo=None) if hasattr(oldest_ts, "tzinfo") else oldest_ts
                staleness = (now - ts).total_seconds()
            except Exception:
                staleness = 0.0
        else:
            staleness = 0.0

        return FusedSignal(
            asset=asset,
            composite_sentiment=round(composite, 4),
            urgency_score=round(max_urgency, 4),
            confidence=round(confidence, 4),
            top_items=items_sorted[:5],
            source_breakdown=source_breakdown,
            item_count=len(items),
            timestamp=datetime.now(timezone.utc),
            staleness_seconds=max(0.0, staleness),
        )

    def _prune_stale(self) -> None:
        """Remove items older than ``max_item_age_seconds`` from all buffers."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_seconds = self.max_item_age_seconds
        for name, buf in self._item_buffer.items():
            fresh = deque(maxlen=buf.maxlen)
            for item in buf:
                ts = item.timestamp
                if ts is None:
                    fresh.append(item)
                    continue
                ts_naive = ts.replace(tzinfo=None) if hasattr(ts, "tzinfo") else ts
                age = (now - ts_naive).total_seconds()
                if age <= cutoff_seconds:
                    fresh.append(item)
            self._item_buffer[name] = fresh

    @staticmethod
    def _neutral_signal(asset: str) -> FusedSignal:
        """Return a neutral (zero-sentiment) signal for *asset*."""
        return FusedSignal(
            asset=asset,
            composite_sentiment=0.0,
            urgency_score=0.0,
            confidence=0.0,
            timestamp=datetime.now(timezone.utc),
        )
