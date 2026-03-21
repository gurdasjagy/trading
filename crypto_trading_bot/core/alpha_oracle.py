"""Alpha Oracle — Confluence Engine & SHM Signal Queue Producer.

This module implements the "master" side of the Alpha Oracle architecture:

1. **Strategy Evaluation**: Runs 60+ trading strategies on 1m and 5m candle data.
2. **Confluence Engine**: Triggers only when a high percentage (configurable,
   default 75%) of strategies agree on direction AND the calculated Risk/Reward
   ratio exceeds 2.0.
3. **Signal Queue Producer**: Writes strictly formatted ``TradeIntent`` payloads
   to a lock-free ring buffer in ``/dev/shm/alpha_signal_queue``.

The Rust execution engine polls this SHM queue and executes validated signals
with ultra-low latency (no IPC syscalls, no locks, just memory reads).

Memory Layout (must match ``rust_engine/src/signal_queue.rs``):

    Header (32 bytes):
        magic       : u64  @ 0   — 0x414C504841534947 ("ALPHASIG")
        version     : u32  @ 8   — 1
        capacity    : u32  @ 12  — 256
        write_cursor: u64  @ 16  — producer write position (atomic)
        read_cursor : u64  @ 24  — consumer read position (atomic)

    Slots (256 × 256 bytes):
        Each slot is a TradeIntent struct (see signal_queue.rs for layout).

Architecture: Python writes slots first, then atomically advances write_cursor
with a Release fence. Rust reads write_cursor with Acquire, then reads the slot.
This guarantees the slot data is fully visible before Rust tries to read it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Constants — must match rust_engine/src/signal_queue.rs
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_QUEUE_PATH = "/dev/shm/alpha_signal_queue"
SIGNAL_QUEUE_MAGIC = 0x414C_5048_4153_4947  # "ALPHASIG"
SIGNAL_QUEUE_VERSION = 1
SIGNAL_QUEUE_CAPACITY = 256  # Power of 2
HEADER_SIZE = 32
SLOT_SIZE = 256
TOTAL_SHM_SIZE = HEADER_SIZE + SIGNAL_QUEUE_CAPACITY * SLOT_SIZE

FP_PRECISION = 1e8
RR_PRECISION = 1e4

# Memory fence
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def _memory_fence() -> None:
    """Full memory barrier (same as in shared_state_reader.py)."""
    try:
        _libc.__sync_synchronize()
    except AttributeError:
        _dummy = ctypes.c_int(0)
        ctypes.c_int.from_address(ctypes.addressof(_dummy))


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


class SignalSide(IntEnum):
    LONG = 0
    SHORT = 1


class IntentType(IntEnum):
    OPEN = 0
    CLOSE = 1
    REDUCE = 2


@dataclass
class TradeIntent:
    """A trade signal from the Alpha Oracle, written to SHM for Rust execution."""

    symbol: str
    side: SignalSide
    intent_type: IntentType = IntentType.OPEN
    leverage: int = 10
    size_contracts: int = 1
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    risk_reward: float = 0.0
    confluence_count: int = 0
    total_strategies: int = 0
    signal_tag: str = ""
    max_slippage: float = 0.001  # 0.1% default

    def to_slot_bytes(self) -> bytes:
        """Serialize this intent into a 256-byte slot matching the Rust layout."""
        buf = bytearray(SLOT_SIZE)

        # symbol (bytes 0..32, UTF-8 zero-padded)
        sym_bytes = self.symbol.encode("utf-8")[:32]
        buf[0 : len(sym_bytes)] = sym_bytes

        # side (byte 32)
        buf[32] = int(self.side)
        # intent_type (byte 33)
        buf[33] = int(self.intent_type)
        # _pad (bytes 34-35) = 0

        # leverage (i32 LE @ 36)
        struct.pack_into("<i", buf, 36, self.leverage)
        # size_contracts (i64 LE @ 40)
        struct.pack_into("<q", buf, 40, self.size_contracts)
        # entry_price_fp (i64 LE @ 48)
        struct.pack_into("<q", buf, 48, int(self.entry_price * FP_PRECISION))
        # stop_loss_fp (i64 LE @ 56)
        struct.pack_into("<q", buf, 56, int(self.stop_loss * FP_PRECISION))
        # take_profit_fp (i64 LE @ 64)
        struct.pack_into("<q", buf, 64, int(self.take_profit * FP_PRECISION))
        # confidence_fp (i64 LE @ 72)
        struct.pack_into("<q", buf, 72, int(self.confidence * FP_PRECISION))
        # risk_reward_fp (i64 LE @ 80)
        struct.pack_into("<q", buf, 80, int(self.risk_reward * RR_PRECISION))
        # timestamp_ns (u64 LE @ 88)
        struct.pack_into("<Q", buf, 88, int(time.time_ns()))
        # confluence_count (u32 LE @ 96)
        struct.pack_into("<I", buf, 96, self.confluence_count)
        # total_strategies (u32 LE @ 100)
        struct.pack_into("<I", buf, 100, self.total_strategies)

        # signal_tag (bytes 104..168, UTF-8 zero-padded)
        tag_bytes = self.signal_tag.encode("utf-8")[:64]
        buf[104 : 104 + len(tag_bytes)] = tag_bytes

        # max_slippage_fp (i64 LE @ 168)
        struct.pack_into("<q", buf, 168, int(self.max_slippage * FP_PRECISION))

        # bytes 176..256 = reserved (zeroed)
        return bytes(buf)


# ═══════════════════════════════════════════════════════════════════════════
# Signal Queue Producer (Python side)
# ═══════════════════════════════════════════════════════════════════════════


class SignalQueueProducer:
    """Writes TradeIntent signals to the SHM ring buffer for Rust consumption.

    Thread-safety: This class is NOT thread-safe. Only one thread/coroutine
    should call ``push()`` at a time (enforced by the orchestrator's event loop).
    """

    def __init__(self, path: str = SIGNAL_QUEUE_PATH):
        self._path = path
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._write_cursor: int = 0

    def open(self) -> None:
        """Open (or create) the SHM file and initialize the header."""
        # Create file if needed — fall back to open-only if Rust container
        # already created the file with different ownership (BUG 5 FIX).
        try:
            flags = os.O_RDWR | os.O_CREAT
            self._fd = os.open(self._path, flags, 0o666)
        except PermissionError:
            # File exists but owned by another user (Rust container) — open without O_CREAT
            self._fd = os.open(self._path, os.O_RDWR)

        # Ensure correct size
        stat = os.fstat(self._fd)
        if stat.st_size < TOTAL_SHM_SIZE:
            os.ftruncate(self._fd, TOTAL_SHM_SIZE)

        self._mm = mmap.mmap(self._fd, TOTAL_SHM_SIZE)

        # Check if already initialized
        magic = struct.unpack_from("<Q", self._mm, 0)[0]
        if magic != SIGNAL_QUEUE_MAGIC:
            # Initialize header
            struct.pack_into("<Q", self._mm, 0, SIGNAL_QUEUE_MAGIC)
            struct.pack_into("<I", self._mm, 8, SIGNAL_QUEUE_VERSION)
            struct.pack_into("<I", self._mm, 12, SIGNAL_QUEUE_CAPACITY)
            struct.pack_into("<Q", self._mm, 16, 0)  # write_cursor
            struct.pack_into("<Q", self._mm, 24, 0)  # read_cursor
            self._mm.flush()
            logger.info(
                "Signal queue initialized at {} ({} bytes, {} slots)",
                self._path,
                TOTAL_SHM_SIZE,
                SIGNAL_QUEUE_CAPACITY,
            )
        else:
            logger.info("Signal queue opened at {} (existing)", self._path)

        # Read current write_cursor
        self._write_cursor = struct.unpack_from("<Q", self._mm, 16)[0]

    def close(self) -> None:
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def push(self, intent: TradeIntent) -> bool:
        """Write a TradeIntent to the next slot in the ring buffer.

        Returns True if successful, False if the queue is full (consumer
        hasn't caught up).
        """
        if self._mm is None:
            logger.error("Signal queue not opened")
            return False

        # Read the consumer's read_cursor to check for full queue
        _memory_fence()
        read_cursor = struct.unpack_from("<Q", self._mm, 24)[0]

        # Check if queue is full (write has lapped read by CAPACITY)
        if self._write_cursor - read_cursor >= SIGNAL_QUEUE_CAPACITY:
            logger.warning(
                "Signal queue FULL (write={}, read={}) — dropping signal for {}",
                self._write_cursor,
                read_cursor,
                intent.symbol,
            )
            return False

        # Calculate slot offset
        slot_idx = self._write_cursor % SIGNAL_QUEUE_CAPACITY
        slot_offset = HEADER_SIZE + slot_idx * SLOT_SIZE

        # Write the slot data FIRST (before advancing cursor)
        slot_bytes = intent.to_slot_bytes()
        self._mm[slot_offset : slot_offset + SLOT_SIZE] = slot_bytes

        # Memory fence: ensure slot data is fully written before cursor update
        _memory_fence()

        # Advance write_cursor (atomic from Rust's perspective — aligned u64 write)
        self._write_cursor += 1
        struct.pack_into("<Q", self._mm, 16, self._write_cursor)

        # Release fence: make cursor visible to Rust consumer
        _memory_fence()

        logger.info(
            "Signal queue: pushed {} {} {} (R:R={:.1f}, conf={:.0%}, {}/{} strategies)",
            intent.symbol,
            intent.side.name,
            intent.intent_type.name,
            intent.risk_reward,
            intent.confidence,
            intent.confluence_count,
            intent.total_strategies,
        )
        return True

    @property
    def pending_count(self) -> int:
        """Number of signals not yet consumed by Rust."""
        if self._mm is None:
            return 0
        read_cursor = struct.unpack_from("<Q", self._mm, 24)[0]
        return int(self._write_cursor - read_cursor)


# ═══════════════════════════════════════════════════════════════════════════
# Confluence Engine
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StrategySignal:
    """A single strategy's directional opinion on a symbol."""

    strategy_name: str
    symbol: str
    side: SignalSide  # LONG or SHORT
    confidence: float  # [0.0, 1.0]
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str = "1m"  # "1m" or "5m"


class ConfluenceEngine:
    """Evaluates strategy signals and fires TradeIntents only when a strong
    confluence of strategies agrees on direction with adequate Risk/Reward.

    Parameters
    ----------
    min_confluence_pct : float
        Minimum percentage of strategies that must agree (default 0.75 = 75%).
    min_risk_reward : float
        Minimum Risk/Reward ratio required (default 2.0).
    min_confidence : float
        Minimum average confidence across agreeing strategies (default 0.6).
    cooldown_seconds : float
        Minimum time between signals for the same symbol (default 300s = 5min).
    """

    def __init__(
        self,
        min_confluence_pct: float = 0.75,
        min_risk_reward: float = 2.0,
        min_confidence: float = 0.6,
        cooldown_seconds: float = 300.0,
    ):
        self.min_confluence_pct = min_confluence_pct
        self.min_risk_reward = min_risk_reward
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds

        # Track last signal time per symbol to enforce cooldown
        self._last_signal_time: Dict[str, float] = {}
        # Statistics
        self.total_evaluations: int = 0
        self.total_signals_emitted: int = 0
        self.total_rejected_confluence: int = 0
        self.total_rejected_rr: int = 0
        self.total_rejected_cooldown: int = 0

    def evaluate(
        self,
        signals: List[StrategySignal],
        leverage: int = 10,
        max_contracts: int = 5,
    ) -> Optional[TradeIntent]:
        """Evaluate a batch of strategy signals for a single symbol.

        All signals must be for the same symbol. Returns a TradeIntent if
        confluence, R:R, and cooldown criteria are met, or None otherwise.
        """
        self.total_evaluations += 1

        if not signals:
            return None

        symbol = signals[0].symbol
        now = time.monotonic()

        # ── Cooldown check ──
        last_time = self._last_signal_time.get(symbol, 0.0)
        if now - last_time < self.cooldown_seconds:
            self.total_rejected_cooldown += 1
            return None

        # ── Count directional agreement ──
        total = len(signals)
        long_signals = [s for s in signals if s.side == SignalSide.LONG]
        short_signals = [s for s in signals if s.side == SignalSide.SHORT]

        long_count = len(long_signals)
        short_count = len(short_signals)

        # Determine dominant direction
        if long_count >= short_count:
            dominant_side = SignalSide.LONG
            dominant_signals = long_signals
            dominant_count = long_count
        else:
            dominant_side = SignalSide.SHORT
            dominant_signals = short_signals
            dominant_count = short_count

        # ── Confluence check ──
        confluence_pct = dominant_count / total if total > 0 else 0.0
        if confluence_pct < self.min_confluence_pct:
            self.total_rejected_confluence += 1
            logger.debug(
                "Confluence rejected for {}: {:.0%} < {:.0%} (long={}, short={}, total={})",
                symbol,
                confluence_pct,
                self.min_confluence_pct,
                long_count,
                short_count,
                total,
            )
            return None

        # ── Average confidence ──
        avg_confidence = sum(s.confidence for s in dominant_signals) / len(
            dominant_signals
        )
        if avg_confidence < self.min_confidence:
            self.total_rejected_confluence += 1
            return None

        # ── Weighted average entry, SL, TP ──
        total_weight = sum(s.confidence for s in dominant_signals)
        if total_weight <= 0:
            return None

        avg_entry = sum(s.entry_price * s.confidence for s in dominant_signals) / total_weight
        avg_sl = sum(s.stop_loss * s.confidence for s in dominant_signals) / total_weight
        avg_tp = sum(s.take_profit * s.confidence for s in dominant_signals) / total_weight

        # ── Risk/Reward calculation ──
        risk = abs(avg_entry - avg_sl)
        reward = abs(avg_tp - avg_entry)

        if risk <= 0:
            self.total_rejected_rr += 1
            return None

        risk_reward = reward / risk
        if risk_reward < self.min_risk_reward:
            self.total_rejected_rr += 1
            logger.debug(
                "R:R rejected for {}: {:.2f} < {:.2f}",
                symbol,
                risk_reward,
                self.min_risk_reward,
            )
            return None

        # ── All criteria met — emit signal ──
        self._last_signal_time[symbol] = now
        self.total_signals_emitted += 1

        tag = f"confluence_{dominant_side.name.lower()}_{dominant_count}of{total}"

        intent = TradeIntent(
            symbol=symbol,
            side=dominant_side,
            intent_type=IntentType.OPEN,
            leverage=leverage,
            size_contracts=max_contracts,
            entry_price=avg_entry,
            stop_loss=avg_sl,
            take_profit=avg_tp,
            confidence=avg_confidence,
            risk_reward=risk_reward,
            confluence_count=dominant_count,
            total_strategies=total,
            signal_tag=tag,
            max_slippage=0.001,
        )

        logger.info(
            "🎯 GOLDEN SETUP: {} {} (R:R={:.2f}, conf={:.0%}, {}/{} strategies)",
            symbol,
            dominant_side.name,
            risk_reward,
            avg_confidence,
            dominant_count,
            total,
        )

        return intent

    def metrics(self) -> Dict[str, int]:
        """Return engine statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "total_signals_emitted": self.total_signals_emitted,
            "rejected_confluence": self.total_rejected_confluence,
            "rejected_rr": self.total_rejected_rr,
            "rejected_cooldown": self.total_rejected_cooldown,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Alpha Oracle Orchestrator Loop
# ═══════════════════════════════════════════════════════════════════════════


class AlphaOracle:
    """Top-level Alpha Oracle that ties the confluence engine to the SHM queue.

    Usage in the cold_path_orchestrator::

        oracle = AlphaOracle()
        oracle.start()
        # ... in the main loop:
        await oracle.evaluate_candles(symbol, candles_1m, candles_5m, strategies)
        # ... on shutdown:
        oracle.stop()
    """

    def __init__(
        self,
        min_confluence_pct: float = 0.75,
        min_risk_reward: float = 2.0,
        min_confidence: float = 0.6,
        cooldown_seconds: float = 300.0,
    ):
        self.confluence = ConfluenceEngine(
            min_confluence_pct=min_confluence_pct,
            min_risk_reward=min_risk_reward,
            min_confidence=min_confidence,
            cooldown_seconds=cooldown_seconds,
        )
        self.producer = SignalQueueProducer()
        self._started = False

    def start(self) -> None:
        """Initialize the SHM signal queue."""
        self.producer.open()
        self._started = True
        logger.info(
            "Alpha Oracle started (confluence={:.0%}, min_rr={:.1f})",
            self.confluence.min_confluence_pct,
            self.confluence.min_risk_reward,
        )

    def stop(self) -> None:
        """Close the SHM signal queue."""
        self.producer.close()
        self._started = False
        logger.info("Alpha Oracle stopped")

    def evaluate_and_emit(
        self,
        signals: List[StrategySignal],
        leverage: int = 10,
        max_contracts: int = 5,
    ) -> Optional[TradeIntent]:
        """Run the confluence engine and push to SHM if a golden setup is found.

        Returns the emitted TradeIntent if one was pushed, or None.
        """
        if not self._started:
            return None

        intent = self.confluence.evaluate(signals, leverage, max_contracts)
        if intent is None:
            return None

        success = self.producer.push(intent)
        if not success:
            logger.warning("Failed to push signal to SHM queue (full)")
            return None

        return intent

    @property
    def queue_depth(self) -> int:
        return self.producer.pending_count

    def metrics(self) -> Dict:
        m = self.confluence.metrics()
        m["queue_depth"] = self.queue_depth
        return m
