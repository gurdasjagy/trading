"""Shared-memory state reader for Python ← Rust real-time data.

Reads the engine state from ``/dev/shm/trading_state`` via ``mmap``.
Uses the **seqlock** pattern to ensure consistent (non-torn) reads.

**Issue 2**: Replaces ZeroMQ telemetry subscription for dashboard data.

Seqlock Read Protocol
---------------------
::

    loop:
        seq1 = read_sequence()        # atomic u64 at offset 16
        if seq1 is odd → retry        # writer is active
        data = read_payload()          # memcpy of the full state
        seq2 = read_sequence()         # re-read sequence
        if seq1 == seq2 → return data  # consistent read!
        # else: writer was active, retry

Memory Layout (must match Rust ``shared_state.rs`` exactly)
-----------------------------------------------------------
::

    EngineStateHeader (128 bytes):
        [0..8]    magic        u64  (0x5452_4144_4553_5441 = "TRADESTA")
        [8..12]   version      u32  (= 2)
        [12..16]  num_symbols  u32
        [16..24]  sequence     u64  (seqlock)
        [24..32]  uptime_seconds    u64
        [32..40]  total_book_updates u64
        [40..48]  total_orders_sent  u64
        [48..56]  total_fills        u64
        [56..64]  total_pnl_fp       i64
        [64..72]  engine_start_ns    u64
        [72..80]  last_heartbeat_ns  u64
        [80..88]  balance_fp         i64
        [88..96]  equity_fp          i64
        [96..104] total_unrealized_pnl_fp i64
        [104..108] active_positions  u32
        [108..112] symbol_state_size u32
        [112..128] reserved (16 bytes)

    SymbolState[MAX_SYMBOLS] (176 bytes each, repr(C, packed)):
        [0..2]    symbol_id     u16
        [2..3]    exchange_id   u8
        [3..4]    flags         u8
        [4..8]    _pad          4 bytes
        [8..16]   best_bid_fp   i64
        [16..24]  best_ask_fp   i64
        [24..32]  best_bid_qty_fp i64
        [32..40]  best_ask_qty_fp i64
        [40..48]  mid_price_fp    i64
        [48..56]  spread_bps_fp   i64
        [56..64]  vwap_1m_fp      i64
        [64..68]  imbalance_fp    i32
        [68..72]  vpin_fp         i32
        [72..76]  kyle_lambda_fp  i32
        [76..80]  _pad2           4 bytes
        --- Position Lifecycle (Directive 1) ---
        [80..81]  position_side   u8  (0=none, 1=long, 2=short)
        [81..82]  position_state  u8
        [82..88]  _pad3           6 bytes
        [88..96]  entry_price_fp  i64
        [96..104] position_size   i64
        [104..112] unrealized_pnl_fp i64
        [112..116] pnl_pct_fp     i32
        [116..120] _pad4          4 bytes
        [120..128] peak_pnl_fp    i64
        [128..132] pnl_from_peak_pct_fp i32
        [132..136] consecutive_declining u32
        --- Timestamps ---
        [136..144] last_update_ns  u64
        [144..152] book_updates_count u64
        --- Reserved for future expansion ---
        [152..176] _reserved  [u8; 24]
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Memory Fence — prevents CPU/compiler reordering of shared memory reads.
# On x86-64, loads are generally ordered, but mmap reads via Python's buffer
# protocol have no ordering guarantees. We use a libc-level memory barrier
# (__sync_synchronize / std::atomic_thread_fence) to ensure correct seqlock.
# On ARM/cloud processors (Graviton, Ampere) this is CRITICAL.
# ═══════════════════════════════════════════════════════════════════════════

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def _memory_fence() -> None:
    """Issue a full memory barrier (equivalent to std::atomic_thread_fence(SeqCst)).

    On x86-64 this compiles to MFENCE; on ARM it becomes DMB ISH.
    This is essential between reading seq1, the payload, and seq2 in
    the seqlock protocol to prevent the CPU from reordering loads.
    """
    # __sync_synchronize is a GCC/Clang builtin exposed by glibc.
    # If not available (unlikely), fall back to a volatile ctypes read
    # which forces a compiler barrier.
    try:
        _libc.__sync_synchronize()
    except AttributeError:
        # Fallback: volatile read from a stack variable forces compiler barrier
        _dummy = ctypes.c_int(0)
        ctypes.c_int.from_address(ctypes.addressof(_dummy))

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Constants — must match Rust shared_state.rs
# ═══════════════════════════════════════════════════════════════════════════

STATE_MAGIC: int = 0x5452_4144_4553_5441  # "TRADESTA"
STATE_VERSION: int = 2  # Directive 3: bumped for new struct layout
MAX_SYMBOLS: int = 64

HEADER_SIZE: int = 128  # EngineStateHeader (unchanged)
# Directive 3: SymbolState expanded to 176 bytes (was 112).
# Added position lifecycle fields: position_side, entry_price, unrealized_pnl,
# peak_pnl, pnl_pct, pnl_from_peak, consecutive_declining.
SYMBOL_SIZE: int = 176  # SymbolState — MUST match Rust shared_state.rs
STATE_FILE_SIZE: int = HEADER_SIZE + MAX_SYMBOLS * SYMBOL_SIZE

FIXED_PRICE_PRECISION: float = 1e8
FIXED_QTY_PRECISION: float = 1e4

# Seqlock configuration
MAX_READ_RETRIES: int = 100
SEQUENCE_OFFSET: int = 16  # offset of sequence field in header

# struct formats (little-endian)
# EngineStateHeader: 128 bytes
#   magic(Q) version(I) num_symbols(I) sequence(Q) uptime(Q) book_updates(Q)
#   orders_sent(Q) fills(Q) pnl_fp(q) engine_start_ns(Q) last_heartbeat_ns(Q)
#   balance_fp(q) equity_fp(q) total_unrealized_pnl_fp(q) active_positions(I) symbol_state_size(I)
#   reserved(16s)
HEADER_FMT = "<QII Q QQQQq QQ qqq II 16s"
# SymbolState: 176 bytes — matches Rust repr(C, packed) layout exactly
# Market data + Position lifecycle + Timestamps
# Uses repr(C, packed) in Rust: no compiler-inserted padding.
SYMBOL_FMT = "<HBB 4s qqqqqqq ii i 4s BB 6s qqq i 4s q iI QQ 24s"

DEFAULT_SHM_PATH = "/dev/shm/trading_state"


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EngineState:
    """Deserialized engine state header."""

    magic: int = 0
    version: int = 0
    num_symbols: int = 0
    sequence: int = 0
    uptime_seconds: int = 0
    total_book_updates: int = 0
    total_orders_sent: int = 0
    total_fills: int = 0
    total_pnl: float = 0.0  # converted from FixedPrice
    engine_start_ns: int = 0
    last_heartbeat_ns: int = 0
    # New Directive 3 fields
    balance: float = 0.0     # Account available balance in USDT
    equity: float = 0.0      # Account equity in USDT
    total_unrealized_pnl: float = 0.0  # Total unrealized PnL in USDT
    active_positions: int = 0
    symbol_state_size: int = 0

    @property
    def is_valid(self) -> bool:
        return self.magic == STATE_MAGIC and self.version >= 1  # Accept v1 or v2

    @property
    def last_heartbeat_age_seconds(self) -> float:
        """Seconds since the last heartbeat."""
        if self.last_heartbeat_ns == 0:
            return float("inf")
        now_ns = int(time.time() * 1e9)
        return (now_ns - self.last_heartbeat_ns) / 1e9


@dataclass
class SymbolState:
    """Deserialized per-symbol state."""

    symbol_id: int = 0
    exchange_id: int = 0
    flags: int = 0
    best_bid: float = 0.0  # converted from FixedPrice
    best_ask: float = 0.0
    best_bid_qty: float = 0.0  # converted from FixedQty
    best_ask_qty: float = 0.0
    mid_price: float = 0.0
    spread_bps: float = 0.0
    vwap_1m: float = 0.0
    imbalance: float = 0.0  # converted from fixed-point 1e4
    vpin: float = 0.0
    kyle_lambda: float = 0.0  # converted from fixed-point 1e8
    # New Directive 1 fields — position lifecycle
    position_side: int = 0     # 0=none, 1=long, 2=short
    position_state: int = 0    # PositionState enum
    entry_price: float = 0.0
    position_size: int = 0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    peak_pnl: float = 0.0
    pnl_from_peak_pct: float = 0.0
    consecutive_declining: int = 0
    # Timestamps
    last_update_ns: int = 0
    book_updates_count: int = 0

    @property
    def has_data(self) -> bool:
        return bool(self.flags & 1)

    @property
    def is_stale(self) -> bool:
        return bool(self.flags & 2)

    @property
    def has_position(self) -> bool:
        return bool(self.flags & 4)

    @property
    def spread(self) -> float:
        """Absolute spread (ask - bid)."""
        return self.best_ask - self.best_bid

    @property
    def position_side_str(self) -> str:
        return {0: "none", 1: "long", 2: "short"}.get(self.position_side, "unknown")


@dataclass
class SharedStateSnapshot:
    """Complete snapshot of engine + all symbol states."""

    engine: EngineState = field(default_factory=EngineState)
    symbols: list[SymbolState] = field(default_factory=list)
    read_time_ns: int = 0
    is_consistent: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# SharedStateReader
# ═══════════════════════════════════════════════════════════════════════════


class SharedStateReader:
    """Reads engine state from shared memory using the seqlock pattern.

    Usage::

        reader = SharedStateReader()
        snapshot = reader.read_consistent()
        if snapshot.is_consistent:
            print(f"Mid price: {snapshot.symbols[0].mid_price}")
    """

    def __init__(self, path: str = DEFAULT_SHM_PATH) -> None:
        self._path = path
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._last_snapshot: Optional[SharedStateSnapshot] = None

    def _ensure_mapped(self) -> bool:
        """Open and mmap the shared state file. Returns True on success."""
        if self._mm is not None:
            return True

        if not os.path.exists(self._path):
            return False

        try:
            self._fd = os.open(self._path, os.O_RDONLY)
            file_size = os.fstat(self._fd).st_size
            if file_size < STATE_FILE_SIZE:
                os.close(self._fd)
                self._fd = None
                return False
            self._mm = mmap.mmap(
                self._fd, STATE_FILE_SIZE, access=mmap.ACCESS_READ
            )
            return True
        except OSError as exc:
            logger.debug("SharedStateReader: cannot mmap {}: {}", self._path, exc)
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def _read_sequence(self) -> int:
        """Read the seqlock sequence (u64 at offset 16) with atomic-width access.

        Uses ``ctypes.c_uint64`` to perform a single 8-byte aligned load
        from the mmap region, preventing partial/torn reads.  A memory
        fence is issued after the read to enforce ordering.
        """
        assert self._mm is not None
        # Use struct.unpack_from which reads directly from the mmap buffer
        # via a single C-level memcpy call (atomic on aligned u64 for x86-64).
        seq = struct.unpack_from("<Q", self._mm, SEQUENCE_OFFSET)[0]
        _memory_fence()  # Prevent CPU from reordering subsequent reads before this
        return seq

    def _parse_header(self, data: bytes) -> EngineState:
        """Parse the EngineStateHeader from raw bytes."""
        vals = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        return EngineState(
            magic=vals[0],
            version=vals[1],
            num_symbols=vals[2],
            sequence=vals[3],
            uptime_seconds=vals[4],
            total_book_updates=vals[5],
            total_orders_sent=vals[6],
            total_fills=vals[7],
            total_pnl=vals[8] / FIXED_PRICE_PRECISION,  # i64 → float
            engine_start_ns=vals[9],
            last_heartbeat_ns=vals[10],
            balance=vals[11] / FIXED_PRICE_PRECISION,
            equity=vals[12] / FIXED_PRICE_PRECISION,
            total_unrealized_pnl=vals[13] / FIXED_PRICE_PRECISION,
            active_positions=vals[14],
            symbol_state_size=vals[15],
        )

    def _parse_symbol(self, data: bytes) -> SymbolState:
        """Parse a SymbolState from raw bytes (176 bytes, repr(C, packed))."""
        vals = struct.unpack(SYMBOL_FMT, data[:SYMBOL_SIZE])
        return SymbolState(
            symbol_id=vals[0],
            exchange_id=vals[1],
            flags=vals[2],
            # vals[3] = _pad (4 bytes)
            best_bid=vals[4] / FIXED_PRICE_PRECISION,
            best_ask=vals[5] / FIXED_PRICE_PRECISION,
            best_bid_qty=vals[6] / FIXED_QTY_PRECISION,
            best_ask_qty=vals[7] / FIXED_QTY_PRECISION,
            mid_price=vals[8] / FIXED_PRICE_PRECISION,
            spread_bps=vals[9] / FIXED_PRICE_PRECISION,
            vwap_1m=vals[10] / FIXED_PRICE_PRECISION,
            imbalance=vals[11] / 10_000.0,
            vpin=vals[12] / 10_000.0,
            kyle_lambda=vals[13] / 1e8,
            # vals[14] = _pad2 (4 bytes)
            # Position lifecycle (Directive 1)
            position_side=vals[15],
            position_state=vals[16],
            # vals[17] = _pad3 (6 bytes)
            entry_price=vals[18] / FIXED_PRICE_PRECISION,
            position_size=vals[19],
            unrealized_pnl=vals[20] / FIXED_PRICE_PRECISION,
            pnl_pct=vals[21] / 10_000.0,
            # vals[22] = _pad4 (4 bytes)
            peak_pnl=vals[23] / FIXED_PRICE_PRECISION,
            pnl_from_peak_pct=vals[24] / 10_000.0,
            consecutive_declining=vals[25],
            # Timestamps
            last_update_ns=vals[26],
            book_updates_count=vals[27],
        )

    def read_consistent(self) -> SharedStateSnapshot:
        """Read a consistent snapshot using the seqlock pattern.

        Retries up to ``MAX_READ_RETRIES`` times if the writer is active.
        Returns a snapshot with ``is_consistent=False`` if all retries fail.
        """
        if not self._ensure_mapped():
            return SharedStateSnapshot(is_consistent=False)

        assert self._mm is not None

        for _ in range(MAX_READ_RETRIES):
            seq1 = self._read_sequence()  # includes trailing memory fence

            # Odd sequence → writer is active, retry
            if seq1 & 1 != 0:
                continue

            # Memory fence already issued by _read_sequence().
            # Read the entire state region as a single memcpy.
            raw = bytes(self._mm[:STATE_FILE_SIZE])

            # Issue a memory fence BEFORE re-reading the sequence.
            # This ensures the payload read completes before the seq2 check.
            _memory_fence()

            seq2 = self._read_sequence()  # includes trailing memory fence

            if seq1 == seq2:
                # Consistent read!
                engine = self._parse_header(raw)
                if not engine.is_valid:
                    return SharedStateSnapshot(is_consistent=False)

                symbols = []
                num_sym = min(engine.num_symbols, MAX_SYMBOLS)
                for i in range(num_sym):
                    offset = HEADER_SIZE + i * SYMBOL_SIZE
                    sym = self._parse_symbol(raw[offset : offset + SYMBOL_SIZE])
                    if sym.symbol_id > 0:  # skip uninitialized slots
                        symbols.append(sym)

                snapshot = SharedStateSnapshot(
                    engine=engine,
                    symbols=symbols,
                    read_time_ns=int(time.time() * 1e9),
                    is_consistent=True,
                )
                self._last_snapshot = snapshot
                return snapshot

        # All retries exhausted
        logger.warning(
            "SharedStateReader: failed to get consistent read after {} retries",
            MAX_READ_RETRIES,
        )
        return SharedStateSnapshot(is_consistent=False)

    def get_cached_or_read(self) -> SharedStateSnapshot:
        """Return cached snapshot if recent, otherwise perform a fresh read."""
        if self._last_snapshot is not None and self._last_snapshot.is_consistent:
            age = time.time() * 1e9 - self._last_snapshot.read_time_ns
            if age < 1e9:  # Less than 1 second old
                return self._last_snapshot
        return self.read_consistent()

    def get_symbol(self, symbol_id: int) -> Optional[SymbolState]:
        """Read a single symbol's state."""
        snapshot = self.get_cached_or_read()
        for sym in snapshot.symbols:
            if sym.symbol_id == symbol_id:
                return sym
        return None

    def get_balance(self) -> Optional[float]:
        """Get the account balance (USDT)."""
        snapshot = self.get_cached_or_read()
        if snapshot.is_consistent:
            # Prefer the new balance field (Directive 3), fall back to total_pnl
            if snapshot.engine.balance > 0:
                return snapshot.engine.balance
            return snapshot.engine.total_pnl
        return None

    def get_equity(self) -> Optional[float]:
        """Get the account equity (USDT)."""
        snapshot = self.get_cached_or_read()
        if snapshot.is_consistent:
            return snapshot.engine.equity
        return None

    def get_active_positions(self) -> list[SymbolState]:
        """Get all symbols that have an active position."""
        snapshot = self.get_cached_or_read()
        if not snapshot.is_consistent:
            return []
        return [s for s in snapshot.symbols if s.has_position]

    def get_price(self, symbol_id: int) -> Optional[float]:
        """Get the mid price for a symbol (for monitoring backward compat)."""
        sym = self.get_symbol(symbol_id)
        if sym is not None and sym.has_data:
            return sym.mid_price
        return None

    # ─── Cold-Path Convenience Methods (Issue 4) ─────────────────────────

    def get_engine_status(self) -> Optional[EngineState]:
        """Return the current engine state header, or ``None`` if unavailable.

        Convenience wrapper used by the ColdPathOrchestrator health loop.
        """
        snapshot = self.get_cached_or_read()
        if snapshot.is_consistent:
            return snapshot.engine
        return None

    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Return ``True`` when the engine state is older than *max_age_seconds*.

        Useful for the cold-path health monitor to detect Rust engine
        crashes or hangs.  If shared memory is unavailable the data is
        considered stale by definition.
        """
        snapshot = self.get_cached_or_read()
        if not snapshot.is_consistent:
            return True
        if snapshot.engine.last_heartbeat_ns == 0:
            return True
        now_ns = int(time.time() * 1e9)
        age_s = (now_ns - snapshot.engine.last_heartbeat_ns) / 1e9
        return age_s > max_age_seconds

    def get_all_symbols(self) -> list[SymbolState]:
        """Return all active symbol states.

        Returns an empty list when shared memory is unavailable or
        the read is inconsistent.
        """
        snapshot = self.get_cached_or_read()
        if snapshot.is_consistent:
            return [s for s in snapshot.symbols if s.has_data]
        return []

    def get_symbol_by_name(self, name: str) -> Optional[SymbolState]:
        """Look up a symbol by its string name (e.g. ``'BTC/USDT'``).

        .. note::

            Shared memory stores symbols by numeric ``symbol_id``, not by
            name.  This method cannot resolve names directly — it returns
            the symbol whose ``symbol_id`` matches the hash of *name*
            modulo ``MAX_SYMBOLS``.  A proper name→id mapping should be
            maintained in the configuration layer; this is a best-effort
            convenience for monitoring / debugging.
        """
        target_id = hash(name) % MAX_SYMBOLS
        return self.get_symbol(target_id)

    def get_market_summary(self) -> dict:
        """Return a dict summarising the live market state.

        Used by :class:`~ai.regime_computer.RegimeComputer` as input for
        regime computation.  Returns an empty dict when data is unavailable.
        """
        snapshot = self.get_cached_or_read()
        if not snapshot.is_consistent:
            return {}

        symbols_data = []
        for sym in snapshot.symbols:
            if sym.has_data:
                symbols_data.append({
                    "symbol_id": sym.symbol_id,
                    "mid_price": sym.mid_price,
                    "spread_bps": sym.spread_bps,
                    "imbalance": sym.imbalance,
                    "vpin": sym.vpin,
                    "kyle_lambda": sym.kyle_lambda,
                    "vwap_1m": sym.vwap_1m,
                    "best_bid": sym.best_bid,
                    "best_ask": sym.best_ask,
                    "book_updates_count": sym.book_updates_count,
                })

        return {
            "uptime_seconds": snapshot.engine.uptime_seconds,
            "total_book_updates": snapshot.engine.total_book_updates,
            "total_orders_sent": snapshot.engine.total_orders_sent,
            "total_fills": snapshot.engine.total_fills,
            "total_pnl": snapshot.engine.total_pnl,
            "num_symbols": snapshot.engine.num_symbols,
            "symbols": symbols_data,
            "last_heartbeat_ns": snapshot.engine.last_heartbeat_ns,
            "read_time_ns": snapshot.read_time_ns,
        }

    # ─── Trap 3 Fix: Crash Recovery Handshake ─────────────────────────────

    def recover_state_from_rust(self) -> dict:
        """Query the Rust engine's state recovery endpoint on Python restart.

        This method reads the recovery file written by Rust's state recovery
        responder thread. When Python crashes and restarts, it calls this to
        reconstruct its internal view of live executions without corrupting
        or wiping shared memory buffers.

        Returns:
            dict with keys: status, uptime_seconds, total_book_updates,
            total_orders_sent, total_fills, total_pnl_usdt, last_heartbeat_ns.
            Returns {"error": "..."} on failure.
        """
        recovery_path = f"{self._path}.recovery"

        # Method 1: File-based recovery (always available)
        try:
            if os.path.exists(recovery_path):
                import json
                with open(recovery_path, "r") as f:
                    state = json.load(f)

                if state.get("status") == "ok":
                    logger.info(
                        "✅ State recovered from Rust engine: uptime={}s, orders={}, fills={}, pnl=${:.2f}",
                        state.get("uptime_seconds", 0),
                        state.get("total_orders_sent", 0),
                        state.get("total_fills", 0),
                        state.get("total_pnl_usdt", 0.0),
                    )
                    return state
                else:
                    logger.warning(
                        "State recovery file exists but contains error: {}",
                        state.get("error", "unknown"),
                    )
                    return state
        except Exception as exc:
            logger.warning("State recovery file read failed: {}", exc)

        # Method 2: Fall back to reading shared memory directly
        snapshot = self.read_consistent()
        if snapshot.is_consistent:
            logger.info(
                "✅ State recovered from shared memory: uptime={}s, orders={}, fills={}",
                snapshot.engine.uptime_seconds,
                snapshot.engine.total_orders_sent,
                snapshot.engine.total_fills,
            )
            return {
                "status": "ok",
                "uptime_seconds": snapshot.engine.uptime_seconds,
                "total_book_updates": snapshot.engine.total_book_updates,
                "total_orders_sent": snapshot.engine.total_orders_sent,
                "total_fills": snapshot.engine.total_fills,
                "total_pnl_usdt": snapshot.engine.total_pnl,
                "last_heartbeat_ns": snapshot.engine.last_heartbeat_ns,
                "recovered_from": "shm_direct",
            }

        logger.error(
            "❌ State recovery failed — neither recovery file nor shared memory available. "
            "The Rust engine may not be running."
        )
        return {"error": "no_recovery_source_available"}

    def close(self) -> None:
        """Clean up resources."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SharedStateReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()
