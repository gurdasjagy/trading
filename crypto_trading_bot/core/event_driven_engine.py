"""Event-Driven Engine — repurposed as a monitoring-only event consumer.

Strategy evaluation and order submission now happen entirely inside the Rust
``trading_engine`` binary (see ``rust_engine/src/strategy_engine.rs`` and
``execution_gateway.rs``).  This Python engine subscribes to the Rust ZeroMQ
telemetry feed (tcp://127.0.0.1:5555) and updates its internal state cache for
dashboard / logging purposes only.

Events flow through a priority queue (kept for monitoring purposes):
  FILL > POSITION_UPDATE > TICKER > KLINE
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger


class EventPriority(enum.IntEnum):
    """Lower numeric value = higher priority in the queue."""

    FILL = 0
    POSITION_UPDATE = 1
    TICKER = 2
    KLINE = 3
    GENERIC = 9


@dataclass(order=True)
class Event:
    """A single market or trading event placed on the priority queue.

    ``priority`` drives ordering; ``symbol``, ``event_type`` and ``data``
    carry the payload.
    """

    priority: int
    # The fields below must NOT participate in ordering comparisons, so they
    # are assigned a default and excluded from comparison via field().
    symbol: str = field(default="", compare=False)
    event_type: str = field(default="", compare=False)
    data: Any = field(default=None, compare=False)
    timestamp: float = field(default_factory=time.monotonic, compare=False)


class EventDrivenEngine:
    """Monitoring-only event consumer for the Rust-driven trading engine.

    In the upgraded architecture, strategy evaluation and order submission run
    entirely in the standalone Rust binary (``trading_engine``).  This class
    now serves as a **read-only dashboard aggregator**:

    * Subscribes to the Rust ZeroMQ telemetry feed at
      ``tcp://127.0.0.1:5555`` (PUB/SUB) and updates ``_state_cache`` with
      fills, PnL snapshots, and microstructure snapshots.
    * Continues to accept events pushed by Python slow-loop code (balance
      checks, position reconciliation) so that monitoring dashboards still
      work without changes.

    The ``strategy_callback`` parameter is retained for backward compatibility
    but is **no longer invoked** during normal operation -- all real-time
    strategy evaluation happens in Rust.

    Args:
        strategy_callback: *Deprecated* -- kept for backward compatibility.
            Was invoked with ``(symbol, data)``; now unused in the hot path.
        max_queue_size: Maximum number of queued events (0 = unlimited).
        state_cache_ttl: Seconds after which a cached state entry is considered
            stale and should trigger a REST refresh.
        telemetry_addr: ZeroMQ PUB address where the Rust engine publishes
            telemetry (default ``tcp://127.0.0.1:5555``).
    """

    def __init__(
        self,
        strategy_callback: Optional[Callable] = None,
        max_queue_size: int = 10_000,
        state_cache_ttl: float = 60.0,
        telemetry_addr: str = "tcp://127.0.0.1:5555",
    ) -> None:
        # Kept for backward compatibility; not used in the hot path.
        self._strategy_callback = strategy_callback
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._state_ttl = state_cache_ttl
        self._running: bool = False

        # ZeroMQ telemetry subscription address (Rust → Python)
        self._telemetry_addr: str = telemetry_addr
        self._telemetry_task: Optional[asyncio.Task] = None

        # Local state cache updated by WebSocket events and Rust telemetry
        self._state_cache: Dict[str, Dict[str, Any]] = {}
        self._state_timestamps: Dict[str, float] = {}

        # Consumer task handle
        self._consumer_task: Optional[asyncio.Task] = None

        # Registered event handlers keyed by event_type
        self._handlers: Dict[str, List[Callable]] = {}

        # Latency tracking: signal-to-order in seconds
        self._latency_samples: List[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the event consumer coroutine and Rust telemetry subscriber."""
        if self._running:
            logger.debug("EventDrivenEngine already running.")
            return
        self._running = True
        self._consumer_task = asyncio.create_task(
            self._consume(), name="event_driven_engine_consumer"
        )
        # Subscribe to the Rust ZeroMQ telemetry feed for dashboard updates.
        self._telemetry_task = asyncio.create_task(
            self._consume_rust_telemetry(), name="rust_telemetry_subscriber"
        )
        logger.info(
            "EventDrivenEngine started — monitoring mode active, "
            "subscribing to Rust telemetry at {}.",
            self._telemetry_addr,
        )

    async def stop(self) -> None:
        """Stop the event consumer and telemetry subscriber."""
        self._running = False
        # Unblock the consumer with a sentinel event
        await self._queue.put(
            Event(priority=EventPriority.GENERIC, event_type="__stop__")
        )
        for task in (self._consumer_task, self._telemetry_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("EventDrivenEngine stopped.")

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def push_ticker(self, symbol: str, ticker: dict) -> None:
        """Push a ticker update event onto the queue (non-blocking)."""
        self._put_nowait(
            Event(
                priority=EventPriority.TICKER,
                symbol=symbol,
                event_type="ticker",
                data=ticker,
            )
        )

    def push_fill(self, symbol: str, order: dict) -> None:
        """Push an order-fill event onto the queue (highest priority)."""
        self._put_nowait(
            Event(
                priority=EventPriority.FILL,
                symbol=symbol,
                event_type="fill",
                data=order,
            )
        )

    def push_position_update(self, symbol: str, position: dict) -> None:
        """Push a position-change event onto the queue."""
        self._put_nowait(
            Event(
                priority=EventPriority.POSITION_UPDATE,
                symbol=symbol,
                event_type="position_update",
                data=position,
            )
        )

    def push_kline(self, symbol: str, kline: dict) -> None:
        """Push a kline (OHLCV) update event onto the queue."""
        self._put_nowait(
            Event(
                priority=EventPriority.KLINE,
                symbol=symbol,
                event_type="kline",
                data=kline,
            )
        )

    def push_event(
        self,
        event_type: str,
        symbol: str = "",
        data: Any = None,
        priority: int = EventPriority.GENERIC,
    ) -> None:
        """Push a generic event onto the queue."""
        self._put_nowait(
            Event(
                priority=priority,
                symbol=symbol,
                event_type=event_type,
                data=data,
            )
        )

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler called whenever an event of *event_type* is consumed."""
        self._handlers.setdefault(event_type, []).append(handler)

    # ------------------------------------------------------------------
    # State cache
    # ------------------------------------------------------------------

    def update_state(self, symbol: str, state: dict) -> None:
        """Update the local state cache entry for *symbol*."""
        self._state_cache[symbol] = state
        self._state_timestamps[symbol] = time.monotonic()

    def get_state(self, symbol: str) -> Optional[dict]:
        """Return cached state for *symbol* (or None if absent / stale)."""
        ts = self._state_timestamps.get(symbol, 0.0)
        if time.monotonic() - ts > self._state_ttl:
            return None
        return self._state_cache.get(symbol)

    # ------------------------------------------------------------------
    # Latency helpers
    # ------------------------------------------------------------------

    def record_signal_to_order_latency(self, latency_seconds: float) -> None:
        """Record a signal-to-order latency sample."""
        self._latency_samples.append(latency_seconds)
        # Keep only the last 1000 samples
        if len(self._latency_samples) > 1_000:
            self._latency_samples = self._latency_samples[-1_000:]

    def get_latency_stats(self) -> Dict[str, float]:
        """Return p50/p95/p99 latency percentiles in milliseconds."""
        if not self._latency_samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "samples": 0}
        arr = np.array(self._latency_samples) * 1000  # → ms
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "samples": len(arr),
        }

    # ------------------------------------------------------------------
    # Internal consumer
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        """Main event consumer loop — processes events in priority order."""
        while self._running:
            try:
                event: Event = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if event.event_type == "__stop__":
                self._queue.task_done()
                break

            try:
                await self._dispatch(event)
            except Exception as exc:
                logger.error(
                    "EventDrivenEngine dispatch error for {}/{}: {}",
                    event.event_type,
                    event.symbol,
                    exc,
                )
            finally:
                self._queue.task_done()

    async def _dispatch(self, event: Event) -> None:
        """Dispatch *event* to registered handlers (monitoring only).

        Strategy evaluation no longer runs here — it executes entirely in the
        Rust ``trading_engine`` binary.  This method updates the local state
        cache and fires any registered monitoring/dashboard handlers.
        """
        # Update local state cache for ticker and kline events
        if event.event_type in ("ticker", "kline") and event.data and event.symbol:
            cached = self._state_cache.get(event.symbol, {})
            cached.update(event.data if isinstance(event.data, dict) else {})
            self.update_state(event.symbol, cached)

        # Fire registered handlers (dashboard / monitoring callbacks only)
        for handler in self._handlers.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event.symbol, event.data)
                else:
                    handler(event.symbol, event.data)
            except Exception as exc:
                logger.debug(
                    "Event handler error for {}: {}", event.event_type, exc
                )

        # NOTE: Strategy callback intentionally removed from hot path.
        # Strategy evaluation now happens in Rust (strategy_engine.rs).
        # The _strategy_callback field is retained only for backward
        # compatibility with code that may read it.

    async def _consume_rust_telemetry(self) -> None:
        """Subscribe to the Rust engine's ZeroMQ PUB telemetry feed.

        Receives fill confirmations, PnL snapshots, microstructure snapshots,
        and heartbeats published by the Rust binary and updates the internal
        state cache so that dashboard views always reflect live engine state.

        Gracefully handles the case where ``pyzmq`` is not installed or the
        Rust process is not yet running.
        """
        try:
            import zmq
            import zmq.asyncio as azmq
        except ImportError:
            logger.debug(
                "pyzmq not installed — Rust telemetry subscription disabled. "
                "Install with: pip install pyzmq"
            )
            return

        ctx = azmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics
        sock.setsockopt(zmq.RCVTIMEO, 1000)   # 1 s receive timeout

        try:
            sock.connect(self._telemetry_addr)
            logger.info(
                "Rust telemetry subscriber connected to {}",
                self._telemetry_addr,
            )

            while self._running:
                try:
                    raw = await asyncio.wait_for(sock.recv_string(), timeout=1.0)
                    # Messages format: "<topic> <json_payload>"
                    space_idx = raw.find(" ")
                    if space_idx == -1:
                        continue
                    topic = raw[:space_idx]
                    payload_str = raw[space_idx + 1:]

                    try:
                        import json as _json
                        payload = _json.loads(payload_str)
                    except Exception:
                        continue

                    self._handle_rust_telemetry(topic, payload)

                except asyncio.TimeoutError:
                    continue
                except Exception as exc:
                    logger.debug("Rust telemetry recv error: {}", exc)
                    await asyncio.sleep(0.1)
        finally:
            sock.close()

    def _handle_rust_telemetry(self, topic: str, payload: dict) -> None:
        """Process a telemetry message from the Rust engine and update state cache."""
        if topic == "fill":
            symbol = payload.get("symbol", "")
            if symbol:
                self.update_state(symbol, {"last_fill": payload})
                # Fire any registered fill handlers
                for handler in self._handlers.get("fill", []):
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.ensure_future(handler(symbol, payload))
                        else:
                            handler(symbol, payload)
                    except Exception as exc:
                        logger.debug("Fill handler error: {}", exc)

        elif topic == "microstructure":
            book_key = payload.get("book_key", "")
            symbol = book_key.split(":")[-1] if ":" in book_key else book_key
            if symbol:
                self.update_state(symbol, {"microstructure": payload})

        elif topic == "order_intent":
            symbol = payload.get("symbol", "")
            if symbol:
                self.update_state(symbol, {"last_order_intent": payload})

        elif topic == "heartbeat":
            # Keep-alive — no state update needed
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _put_nowait(self, event: Event) -> None:
        """Put *event* onto the queue without blocking."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "EventDrivenEngine queue full — dropping {} event for {}",
                event.event_type,
                event.symbol,
            )
