"""Microstructure Dashboard — real-time monitoring of the Synthetic L3 Engine.

Provides a FastAPI router that can be mounted on the main dashboard app or
run standalone.  Exposes:

* ``GET  /microstructure/snapshot/{symbol}``  — latest microstructure snapshot
* ``GET  /microstructure/regime``             — latest regime state
* ``WS   /microstructure/stream/{symbol}``    — live push of snapshots every second
* ``GET  /microstructure/health``             — service health check

All data is read from:
1. An in-process :class:`MicrostructureStore` that the strategy engine writes to.
2. The regime-service JSON file at ``REGIME_OUTPUT_PATH``.

Usage
-----
Mount in the main dashboard::

    from monitoring.microstructure_dashboard import create_router
    app.include_router(create_router(store), prefix="/api/v1")

Or run standalone::

    python -m monitoring.microstructure_dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

from loguru import logger

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available — microstructure dashboard disabled")


# ---------------------------------------------------------------------------
# MicrostructureSnapshot Python mirror
# ---------------------------------------------------------------------------


@dataclass
class MicrostructureSnapshot:
    """Python-side mirror of the Rust ``MicrostructureSnapshot`` struct.

    Fields match ``rust_engine/src/microstructure.rs::MicrostructureSnapshot``
    exactly so that we can convert from JSON or directly from PyO3 bindings.
    """

    symbol: str = ""
    timestamp_ms: int = 0

    # Basic tick-processor metrics
    vwap: float = 0.0
    tick_imbalance: float = 0.0
    vpin: float = 0.0

    # Synthetic L3: queue position
    estimated_queue_position_bid: float = 1.0
    estimated_queue_position_ask: float = 1.0
    bid_fill_probability: float = 0.0
    ask_fill_probability: float = 0.0

    # Enhanced flow analysis
    flow_toxicity_score: float = 0.0
    kyle_lambda: float = 0.0
    trade_arrival_rate: float = 0.0

    # Book dynamics
    bid_pressure_gradient: float = 0.0
    ask_pressure_gradient: float = 0.0
    spoofing_score: float = 0.0
    absorption_detected: bool = False

    # Composite signal
    microstructure_edge_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MicrostructureSnapshot":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_rust(cls, symbol: str, rust_snapshot: Any) -> "MicrostructureSnapshot":
        """Construct from a PyO3-exposed Rust ``MicrostructureSnapshot``."""
        return cls(
            symbol=symbol,
            timestamp_ms=int(time.time() * 1000),
            vwap=float(getattr(rust_snapshot, "vwap", 0.0)),
            tick_imbalance=float(getattr(rust_snapshot, "tick_imbalance", 0.0)),
            vpin=float(getattr(rust_snapshot, "vpin", 0.0)),
            estimated_queue_position_bid=float(
                getattr(rust_snapshot, "estimated_queue_position_bid", 1.0)
            ),
            estimated_queue_position_ask=float(
                getattr(rust_snapshot, "estimated_queue_position_ask", 1.0)
            ),
            bid_fill_probability=float(getattr(rust_snapshot, "bid_fill_probability", 0.0)),
            ask_fill_probability=float(getattr(rust_snapshot, "ask_fill_probability", 0.0)),
            flow_toxicity_score=float(getattr(rust_snapshot, "flow_toxicity_score", 0.0)),
            kyle_lambda=float(getattr(rust_snapshot, "kyle_lambda", 0.0)),
            trade_arrival_rate=float(getattr(rust_snapshot, "trade_arrival_rate", 0.0)),
            bid_pressure_gradient=float(getattr(rust_snapshot, "bid_pressure_gradient", 0.0)),
            ask_pressure_gradient=float(getattr(rust_snapshot, "ask_pressure_gradient", 0.0)),
            spoofing_score=float(getattr(rust_snapshot, "spoofing_score", 0.0)),
            absorption_detected=bool(getattr(rust_snapshot, "absorption_detected", False)),
            microstructure_edge_score=float(
                getattr(rust_snapshot, "microstructure_edge_score", 0.0)
            ),
        )


# ---------------------------------------------------------------------------
# MicrostructureStore
# ---------------------------------------------------------------------------


class MicrostructureStore:
    """Thread-safe in-process store for the latest microstructure snapshots.

    The Rust strategy engine (or its Python bridge) calls
    :meth:`update_snapshot` after every orderbook delta/trade.

    Parameters
    ----------
    history_per_symbol: Number of historical snapshots to keep per symbol
        (used for the WebSocket stream).
    """

    def __init__(self, history_per_symbol: int = 300) -> None:
        self._history_per_symbol = history_per_symbol
        self._snapshots: Dict[str, MicrostructureSnapshot] = {}
        self._history: Dict[str, Deque[MicrostructureSnapshot]] = {}

    def update_snapshot(self, snapshot: MicrostructureSnapshot) -> None:
        """Store the latest snapshot for *symbol*."""
        symbol = snapshot.symbol
        self._snapshots[symbol] = snapshot
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_per_symbol)
        self._history[symbol].append(snapshot)

    def get_latest(self, symbol: str) -> Optional[MicrostructureSnapshot]:
        """Return the latest snapshot for *symbol*, or ``None``."""
        return self._snapshots.get(symbol)

    def get_history(self, symbol: str, n: int = 60) -> List[MicrostructureSnapshot]:
        """Return the last *n* snapshots for *symbol*."""
        hist = self._history.get(symbol)
        if hist is None:
            return []
        snaps = list(hist)
        return snaps[-n:]

    @property
    def symbols(self) -> List[str]:
        """List of symbols with at least one snapshot."""
        return list(self._snapshots.keys())


# ---------------------------------------------------------------------------
# Regime state reader
# ---------------------------------------------------------------------------


def _read_regime_state(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the latest regime state from the JSON file.

    Returns an empty dict if the file is absent or unreadable.
    """
    regime_path = path or os.environ.get(
        "REGIME_OUTPUT_PATH", "/dev/shm/regime_state.json"
    )
    try:
        with open(regime_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"microstructure_dashboard: regime file not readable: {exc}")
        return {}


# ---------------------------------------------------------------------------
# FastAPI Router factory
# ---------------------------------------------------------------------------


def create_router(store: MicrostructureStore) -> "APIRouter":
    """Create and return the FastAPI router.

    Mount with::

        app.include_router(create_router(store), prefix="/api/v1")
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required to use the microstructure dashboard")

    router = APIRouter(tags=["microstructure"])

    # ── REST endpoints ───────────────────────────────────────────────────

    @router.get("/microstructure/health")
    async def health() -> JSONResponse:
        """Health check: returns 200 if at least one snapshot is available."""
        if store.symbols:
            return JSONResponse(
                {"status": "ok", "symbols": store.symbols, "count": len(store.symbols)}
            )
        return JSONResponse({"status": "no_data", "symbols": []}, status_code=503)

    @router.get("/microstructure/regime")
    async def get_regime() -> JSONResponse:
        """Return the latest regime state written by the Python regime service."""
        data = _read_regime_state()
        if not data:
            return JSONResponse({"error": "regime_state_unavailable"}, status_code=503)
        return JSONResponse(data)

    @router.get("/microstructure/snapshot/{symbol}")
    async def get_snapshot(symbol: str) -> JSONResponse:
        """Return the latest microstructure snapshot for *symbol*."""
        snap = store.get_latest(symbol)
        if snap is None:
            return JSONResponse(
                {"error": f"No snapshot available for {symbol}"}, status_code=404
            )
        return JSONResponse(snap.to_dict())

    @router.get("/microstructure/history/{symbol}")
    async def get_history(symbol: str, n: int = 60) -> JSONResponse:
        """Return the last *n* snapshots for *symbol* (max 300)."""
        n = min(n, 300)
        snaps = store.get_history(symbol, n)
        return JSONResponse([s.to_dict() for s in snaps])

    @router.get("/microstructure/summary")
    async def get_summary() -> JSONResponse:
        """Return a summary of the current microstructure edge across all symbols."""
        summary = []
        for sym in store.symbols:
            snap = store.get_latest(sym)
            if snap:
                summary.append(
                    {
                        "symbol": sym,
                        "edge_score": round(snap.microstructure_edge_score, 4),
                        "tick_imbalance": round(snap.tick_imbalance, 4),
                        "flow_toxicity": round(snap.flow_toxicity_score, 4),
                        "spoofing_score": round(snap.spoofing_score, 4),
                        "vpin": round(snap.vpin, 4),
                        "absorption": snap.absorption_detected,
                        "kyle_lambda": round(snap.kyle_lambda, 8),
                        "trade_arrival_rate": round(snap.trade_arrival_rate, 2),
                        "timestamp_ms": snap.timestamp_ms,
                    }
                )
        # Sort by abs edge score descending
        summary.sort(key=lambda x: abs(x["edge_score"]), reverse=True)
        return JSONResponse({"symbols": summary, "count": len(summary)})

    # ── WebSocket streaming ──────────────────────────────────────────────

    @router.websocket("/microstructure/stream/{symbol}")
    async def stream_snapshot(websocket: WebSocket, symbol: str) -> None:
        """WebSocket: push the latest snapshot every second."""
        await websocket.accept()
        logger.info(f"microstructure_dashboard: WebSocket client connected for {symbol}")
        try:
            while True:
                snap = store.get_latest(symbol)
                if snap is not None:
                    await websocket.send_text(json.dumps(snap.to_dict()))
                else:
                    await websocket.send_text(
                        json.dumps({"error": f"no_data_for_{symbol}"})
                    )
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            logger.info(f"microstructure_dashboard: WebSocket client disconnected ({symbol})")
        except Exception as exc:
            logger.warning(f"microstructure_dashboard: WebSocket error ({symbol}): {exc}")

    @router.websocket("/microstructure/stream_all")
    async def stream_all_snapshots(websocket: WebSocket) -> None:
        """WebSocket: push snapshots for ALL tracked symbols every second."""
        await websocket.accept()
        logger.info("microstructure_dashboard: WebSocket client connected (all symbols)")
        try:
            while True:
                data: Dict[str, Any] = {}
                for sym in store.symbols:
                    snap = store.get_latest(sym)
                    if snap:
                        data[sym] = snap.to_dict()
                await websocket.send_text(json.dumps(data))
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            logger.info("microstructure_dashboard: all-symbols WebSocket client disconnected")
        except Exception as exc:
            logger.warning(f"microstructure_dashboard: all-symbols WebSocket error: {exc}")

    return router


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _run_standalone() -> None:  # pragma: no cover
    """Run the microstructure dashboard as a standalone FastAPI app."""
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        print("uvicorn and fastapi are required to run the dashboard standalone")
        return

    store = MicrostructureStore()
    app = FastAPI(title="Microstructure Dashboard")
    app.include_router(create_router(store))

    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")


if __name__ == "__main__":
    _run_standalone()
