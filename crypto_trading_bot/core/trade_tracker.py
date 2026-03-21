"""Real-time trade tracking — READ-ONLY observer via shared memory.

Directive 1: ALL active trade lifecycle management, peak PnL tracking, and
"intelligent close" (reversal) logic has been moved to the Rust hot path
(position_lifecycle.rs). Python's TradeTracker is now a READ-ONLY observer
that reads position state from shared memory for the web dashboard.

The 2-second REST polling loop has been replaced with shared memory reads.
No exit decisions or order submissions are made from Python.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class TradeState(str, Enum):
    PENDING = "pending"          # Order placed, not yet filled
    OPEN = "open"                # Position confirmed on exchange
    IN_PROFIT = "in_profit"      # Currently profitable
    IN_LOSS = "in_loss"          # Currently at a loss
    PEAK_PROFIT = "peak_profit"  # Hit a new profit high (trailing logic)
    REVERSING = "reversing"      # Was in profit, now declining
    CLOSING = "closing"          # Close order submitted
    CLOSED = "closed"            # Position fully closed


@dataclass
class TradeSnapshot:
    """Point-in-time snapshot of a trade's state."""

    timestamp: float
    price: float
    unrealized_pnl: float
    pnl_pct: float
    state: TradeState


@dataclass
class TrackedTrade:
    """Complete trade tracking record with full history."""

    trade_id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    amount: float
    leverage: int
    strategy: str
    market_type: str  # "futures" or "forex"

    # Current state
    state: TradeState = TradeState.OPEN
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0

    # Peak tracking for intelligent exits
    peak_pnl: float = 0.0
    peak_pnl_pct: float = 0.0
    peak_price: float = 0.0
    peak_time: Optional[datetime] = None

    # Trough tracking
    worst_pnl: float = 0.0
    worst_pnl_pct: float = 0.0

    # Reversal detection
    pnl_from_peak_pct: float = 0.0  # How far we've fallen from peak
    consecutive_declining_ticks: int = 0
    consecutive_improving_ticks: int = 0

    # History (last 100 snapshots for analysis)
    snapshots: List[TradeSnapshot] = field(default_factory=list)

    # Close info
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    close_reason: Optional[str] = None
    realized_pnl: Optional[float] = None

    # Forex-specific
    lot_size: Optional[float] = None
    pip_pnl: Optional[float] = None


class TradeTracker:
    """READ-ONLY trade tracking observer via shared memory.

    Directive 1: ALL exit decisions (reversal, SL/TP, sustained decline)
    are now made by the Rust engine's PositionLifecycleManager in real-time
    (tick-by-tick, not 2-second polling).

    This class reads position state from shared memory and provides it
    to the web dashboard. It does NOT submit any orders.

    For backward compatibility, the same interface is preserved but:
    - _tracking_loop reads from shared memory instead of REST polling
    - _evaluate_close_decisions is a no-op (Rust handles exits)
    - register_trade is still called for dashboard display

    The Rust dashboard HTTP server (port 8080) also provides real-time
    data directly without going through Python.
    """

    POLL_INTERVAL = 2.0  # seconds — read shared memory for dashboard updates
    MAX_SNAPSHOTS = 100
    STATE_FILE = Path("data") / "trade_tracker_state.json"

    # These thresholds are ONLY used for dashboard display coloring.
    # Actual close decisions are made by Rust's PositionLifecycleManager.
    REVERSAL_CLOSE_PCT = 30.0
    MIN_PROFIT_TO_PROTECT = 0.5
    MAX_LOSS_PCT = 2.0
    CONSECUTIVE_DECLINE_THRESHOLD = 10

    def __init__(self, exchange=None, position_manager=None, settings=None):
        self._exchange = exchange
        self._position_manager = position_manager
        self._settings = settings
        self._trades: Dict[str, TrackedTrade] = {}
        self._closed_trades: List[TrackedTrade] = []
        self._running = False
        self._lock = asyncio.Lock()
        self._last_save_time: float = 0.0

        # Directive 1: SharedStateReader for reading position data from Rust
        self._shm_reader = None
        try:
            from crypto_trading_bot.core.shared_state_reader import SharedStateReader
            self._shm_reader = SharedStateReader()
            logger.info("TradeTracker: using SharedStateReader (READ-ONLY mode)")
        except Exception as e:
            logger.warning("TradeTracker: SharedStateReader unavailable: {} — falling back to REST", e)

        # Load persisted state on init
        self._load_state()

    async def start(self) -> None:
        """Start the read-only tracking loop (shared memory reader)."""
        self._running = True
        logger.info("TradeTracker started — READ-ONLY mode (exits handled by Rust)")
        asyncio.create_task(self._tracking_loop())

    async def stop(self) -> None:
        """Stop tracking and persist state."""
        self._running = False
        self._save_state()
        if self._shm_reader:
            self._shm_reader.close()
        logger.info("TradeTracker stopped — state persisted")

    async def register_trade(self, trade: TrackedTrade) -> None:
        """Register a new trade for tracking."""
        async with self._lock:
            self._trades[trade.trade_id] = trade
        logger.info(
            "Trade registered for tracking: {} {} {} @ {}",
            trade.trade_id,
            trade.symbol,
            trade.side,
            trade.entry_price,
        )

    async def _tracking_loop(self) -> None:
        """Main tracking loop — reads from shared memory every 2 seconds.

        Directive 1: No longer polls REST APIs or makes close decisions.
        Rust's PositionLifecycleManager handles exits tick-by-tick.
        """
        while self._running:
            try:
                await self._update_from_shared_memory()

                # Persist state every 30 seconds using elapsed-time check
                now = time.time()
                if now - self._last_save_time >= 30:
                    self._save_state()
                    self._last_save_time = now

            except Exception as exc:
                logger.error("TradeTracker loop error: {}", exc)

            await asyncio.sleep(self.POLL_INTERVAL)

    async def _update_from_shared_memory(self) -> None:
        """Read position state from Rust's shared memory (READ-ONLY).

        Directive 1: Replaces the old REST-polling _update_all_trades().
        All PnL tracking, peak detection, and state transitions are done
        by Rust's PositionLifecycleManager. Python just reads the result.
        """
        if self._shm_reader is None:
            # Fallback: try REST if shared memory not available
            await self._update_all_trades_rest_fallback()
            return

        try:
            snapshot = self._shm_reader.read_consistent()
            if not snapshot.is_consistent:
                return

            async with self._lock:
                # Update engine-level data for dashboard
                self._engine_balance = snapshot.engine.balance
                self._engine_equity = snapshot.engine.equity

                # Sync position data from Rust shared memory to tracked trades
                active_symbol_ids = set()
                for sym_state in snapshot.symbols:
                    if sym_state.position_side == 0:
                        continue  # No position

                    active_symbol_ids.add(sym_state.symbol_id)

                    # Find or create the tracked trade for this symbol
                    trade_key = f"shm_{sym_state.symbol_id}"
                    if trade_key not in self._trades:
                        # New position detected from shared memory
                        side = "long" if sym_state.position_side == 1 else "short"
                        self._trades[trade_key] = TrackedTrade(
                            trade_id=trade_key,
                            symbol=f"SYM_{sym_state.symbol_id}",
                            side=side,
                            entry_price=sym_state.entry_price,
                            entry_time=datetime.now(tz=timezone.utc),
                            amount=float(sym_state.position_size),
                            leverage=1,
                            strategy="rust_engine",
                            market_type="futures",
                        )

                    trade = self._trades[trade_key]
                    # Update from shared memory (read-only — Rust computed these)
                    trade.current_price = sym_state.mid_price
                    trade.unrealized_pnl = sym_state.unrealized_pnl
                    trade.pnl_pct = sym_state.pnl_pct
                    trade.peak_pnl = sym_state.peak_pnl
                    trade.pnl_from_peak_pct = sym_state.pnl_from_peak_pct
                    trade.consecutive_declining_ticks = sym_state.consecutive_declining

                    # Map Rust position state to Python TradeState for dashboard
                    state_map = {
                        0: TradeState.OPEN,
                        1: TradeState.IN_PROFIT,
                        2: TradeState.IN_LOSS,
                        3: TradeState.PEAK_PROFIT,
                        4: TradeState.REVERSING,
                        5: TradeState.CLOSING,
                        6: TradeState.CLOSED,
                    }
                    trade.state = state_map.get(sym_state.position_state, TradeState.OPEN)

                    # Record snapshot for dashboard charting
                    snap = TradeSnapshot(
                        timestamp=time.time(),
                        price=trade.current_price,
                        unrealized_pnl=trade.unrealized_pnl,
                        pnl_pct=trade.pnl_pct,
                        state=trade.state,
                    )
                    trade.snapshots.append(snap)
                    if len(trade.snapshots) > self.MAX_SNAPSHOTS:
                        trade.snapshots = trade.snapshots[-self.MAX_SNAPSHOTS:]

                # Detect closed positions (were tracked but no longer have a position)
                closed_ids = []
                for tid, trade in self._trades.items():
                    if tid.startswith("shm_"):
                        sym_id = int(tid.split("_")[1])
                        if sym_id not in active_symbol_ids:
                            trade.state = TradeState.CLOSED
                            trade.close_time = datetime.now(tz=timezone.utc)
                            trade.close_reason = "rust_lifecycle_close"
                            closed_ids.append(tid)

                for tid in closed_ids:
                    trade = self._trades.pop(tid)
                    self._closed_trades.append(trade)

        except Exception as exc:
            logger.warning("SharedMemory read failed: {} — skipping update", exc)

    async def _update_all_trades_rest_fallback(self) -> None:
        """Fallback: fetch positions via REST if shared memory unavailable."""
        if not self._trades or self._exchange is None:
            return

        try:
            positions = await self._exchange.get_positions()
            position_map = {p.symbol: p for p in positions}
        except Exception as exc:
            logger.warning("REST fallback: Failed to fetch positions: {}", exc)
            return

        async with self._lock:
            for trade_id, trade in self._trades.items():
                pos = position_map.get(trade.symbol)
                if pos is None:
                    continue
                trade.current_price = getattr(pos, 'current_price', 0) or getattr(pos, 'mark_price', 0)
                trade.unrealized_pnl = getattr(pos, 'unrealized_pnl', 0)

    async def _evaluate_close_decisions(self) -> None:
        """NO-OP — Directive 1: All exit decisions handled by Rust.

        Rust's PositionLifecycleManager evaluates close conditions on
        EVERY tick (reversal, max-loss, sustained decline). Python never
        submits close orders. This method is retained for interface
        compatibility only.
        """
        pass

    def _should_close(self, trade: TrackedTrade) -> Optional[str]:
        """NO-OP — Directive 1: All exit decisions handled by Rust.

        Rust's PositionLifecycleManager evaluates these conditions tick-by-tick:
          - Hard stop loss (max_loss_pct)
          - Profit reversal (pnl_from_peak_pct >= reversal_close_pct)
          - Sustained decline (consecutive_declining >= threshold & in_loss)

        This method is retained for interface compatibility and dashboard
        display coloring only. It NEVER triggers order submissions.
        """
        return None

    def _save_state(self) -> None:
        """Persist trade state to disk for crash recovery."""
        try:
            state: dict = {
                "timestamp": time.time(),
                "active_trades": {},
                "closed_trades_count": len(self._closed_trades),
            }
            for tid, trade in self._trades.items():
                state["active_trades"][tid] = {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_price": trade.entry_price,
                    "entry_time": trade.entry_time.isoformat(),
                    "amount": trade.amount,
                    "leverage": trade.leverage,
                    "strategy": trade.strategy,
                    "market_type": trade.market_type,
                    "state": trade.state.value,
                    "peak_pnl": trade.peak_pnl,
                    "peak_pnl_pct": trade.peak_pnl_pct,
                    "worst_pnl": trade.worst_pnl,
                    "lot_size": trade.lot_size,
                }

            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.STATE_FILE, "w") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            logger.warning("Failed to save trade tracker state: {}", exc)

    def _load_state(self) -> None:
        """Load persisted state on startup."""
        try:
            if self.STATE_FILE.exists():
                with open(self.STATE_FILE) as fh:
                    state = json.load(fh)
                for tid, data in state.get("active_trades", {}).items():
                    trade = TrackedTrade(
                        trade_id=data["trade_id"],
                        symbol=data["symbol"],
                        side=data["side"],
                        entry_price=data["entry_price"],
                        entry_time=datetime.fromisoformat(data["entry_time"]),
                        amount=data["amount"],
                        leverage=data["leverage"],
                        strategy=data["strategy"],
                        market_type=data["market_type"],
                        state=TradeState(data["state"]),
                        peak_pnl=data.get("peak_pnl", 0),
                        peak_pnl_pct=data.get("peak_pnl_pct", 0),
                        worst_pnl=data.get("worst_pnl", 0),
                        lot_size=data.get("lot_size"),
                    )
                    self._trades[tid] = trade
                logger.info(
                    "Loaded {} active trades from persisted state", len(self._trades)
                )
        except Exception as exc:
            logger.warning("Failed to load trade tracker state: {}", exc)

    # --- Query methods for dashboard ---

    async def get_all_tracked_trades(self) -> List[dict]:
        """Return all active trades with full tracking data."""
        async with self._lock:
            return [self._trade_to_dict(t) for t in self._trades.values()]

    async def get_trade_history(self, limit: int = 50) -> List[dict]:
        """Return recently closed trades."""
        return [self._trade_to_dict(t) for t in self._closed_trades[-limit:]]

    def _trade_to_dict(self, trade: TrackedTrade) -> dict:
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "current_price": trade.current_price,
            "unrealized_pnl": round(trade.unrealized_pnl, 4),
            "pnl_pct": round(trade.pnl_pct, 2),
            "state": trade.state.value,
            "peak_pnl": round(trade.peak_pnl, 4),
            "peak_pnl_pct": round(trade.peak_pnl_pct, 2),
            "pnl_from_peak_pct": round(trade.pnl_from_peak_pct, 1),
            "worst_pnl": round(trade.worst_pnl, 4),
            "leverage": trade.leverage,
            "strategy": trade.strategy,
            "market_type": trade.market_type,
            "entry_time": trade.entry_time.isoformat(),
            "close_time": trade.close_time.isoformat() if trade.close_time else None,
            "close_reason": trade.close_reason,
            "lot_size": trade.lot_size,
            "consecutive_declining_ticks": trade.consecutive_declining_ticks,
            "time_open_seconds": (
                datetime.now(tz=timezone.utc) - trade.entry_time
            ).total_seconds(),
        }
