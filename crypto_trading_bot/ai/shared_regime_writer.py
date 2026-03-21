"""Shared-memory regime writer for Python → Rust regime weight updates.

Writes regime weights to ``/dev/shm/regime_weights`` via ``mmap`` using
the **seqlock** pattern so that the Rust engine can read them lock-free.

**Issue 2**: Replaces ZeroMQ config push to Rust.

Seqlock Write Protocol
----------------------
::

    1. Increment sequence to odd  (writing)
    2. Write the payload
    3. Increment sequence to even (consistent)

Memory Layout (must match Rust ``regime_shm.rs`` exactly)
---------------------------------------------------------
::

    RegimeWeights (128 bytes):
        [0..8]     magic                    u64  (0x5245_4749_4D45_5754 = "REGIMEWT")
        [8..12]    version                  u32
        [12..16]   _pad0                    u32
        [16..24]   sequence                 u64  (seqlock)
        [24..32]   timestamp_ms             i64
        [32..33]   overall_regime           u8
        [33..34]   volatility_regime        u8
        [34..36]   _pad1                    2 bytes
        [36..40]   sentiment_score_fp       i32  (fixed-point 1e4)
        [40..44]   sentiment_confidence_fp  i32  (fixed-point 1e4)
        [44..48]   fear_greed_index         i32
        [48..49]   btc_dominance_trend      u8   (0=flat, 1=rising, 2=falling)
        [49..50]   funding_rate_bias        u8   (0=neutral, 1=long_crowded, 2=short_crowded)
        [50..52]   _pad2                    2 bytes
        [52..56]   cross_asset_correlation_fp i32 (fixed-point 1e4)
        [56..60]   news_impact_score_fp     i32  (fixed-point 1e4)
        [60..64]   position_scale_fp        i32  (fixed-point 1e4)
        [64..68]   max_leverage_override    i32
        [68..72]   ttl_seconds              i32
        [72..80]   allowed_strategies_mask  u64
        [80..88]   blocked_strategies_mask  u64
        [88..112]  _reserved                24 bytes
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Constants — must match Rust regime_shm.rs
# ═══════════════════════════════════════════════════════════════════════════

REGIME_MAGIC: int = 0x5245_4749_4D45_5754  # "REGIMEWT"
REGIME_VERSION: int = 1
REGIME_FILE_SIZE: int = 128  # Total size of RegimeWeights struct

SEQUENCE_OFFSET: int = 16  # offset of sequence field
DEFAULT_SHM_PATH = "/dev/shm/regime_weights"

# Regime type constants
REGIME_UNKNOWN: int = 0
REGIME_TRENDING_BULLISH: int = 1
REGIME_TRENDING_BEARISH: int = 2
REGIME_RANGING: int = 3
REGIME_HIGH_VOLATILITY: int = 4
REGIME_CHOPPY: int = 5

# Volatility type constants
VOLATILITY_LOW: int = 0
VOLATILITY_MODERATE: int = 1
VOLATILITY_HIGH: int = 2
VOLATILITY_EXTREME: int = 3

# Dominance trend constants
DOMINANCE_FLAT: int = 0
DOMINANCE_RISING: int = 1
DOMINANCE_FALLING: int = 2

# Funding bias constants
FUNDING_NEUTRAL: int = 0
FUNDING_LONG_CROWDED: int = 1
FUNDING_SHORT_CROWDED: int = 2

# struct format (little-endian) — must match Rust #[repr(C, packed)]
REGIME_FMT = "<Q I I Q q BB 2s iii BB 2s iii ii QQ 24s"

# String-to-enum mappings
REGIME_MAP = {
    "unknown": REGIME_UNKNOWN,
    "trending_bullish": REGIME_TRENDING_BULLISH,
    "trending_bearish": REGIME_TRENDING_BEARISH,
    "ranging": REGIME_RANGING,
    "high_volatility": REGIME_HIGH_VOLATILITY,
    "choppy": REGIME_CHOPPY,
}

VOLATILITY_MAP = {
    "low": VOLATILITY_LOW,
    "moderate": VOLATILITY_MODERATE,
    "high": VOLATILITY_HIGH,
    "extreme": VOLATILITY_EXTREME,
}

DOMINANCE_MAP = {
    "flat": DOMINANCE_FLAT,
    "rising": DOMINANCE_RISING,
    "falling": DOMINANCE_FALLING,
}

FUNDING_MAP = {
    "neutral": FUNDING_NEUTRAL,
    "long_crowded": FUNDING_LONG_CROWDED,
    "short_crowded": FUNDING_SHORT_CROWDED,
}


# ═══════════════════════════════════════════════════════════════════════════
# RegimeData — high-level Python representation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RegimeData:
    """Python-friendly regime data (converted to binary on write)."""

    overall_regime: str = "unknown"
    volatility_regime: str = "high"
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    fear_greed_index: int = 50
    btc_dominance_trend: str = "flat"
    funding_rate_bias: str = "neutral"
    cross_asset_correlation: float = 0.0
    news_impact_score: float = 0.0
    recommended_position_scale: float = 0.5
    max_leverage_override: int = 0
    ttl_seconds: int = 600
    allowed_strategies_mask: int = 0xFFFFFFFFFFFFFFFF
    blocked_strategies_mask: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# SharedRegimeWriter
# ═══════════════════════════════════════════════════════════════════════════


class SharedRegimeWriter:
    """Writes regime weights to shared memory using the seqlock pattern.

    Usage::

        writer = SharedRegimeWriter()
        data = RegimeData(
            overall_regime="trending_bullish",
            sentiment_score=0.75,
            recommended_position_scale=1.2,
        )
        writer.update(data)
    """

    def __init__(self, path: str = DEFAULT_SHM_PATH) -> None:
        self._path = path
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._sequence: int = 0  # current seqlock sequence (always even when idle)
        self._ensure_mapped()

    def _ensure_mapped(self) -> bool:
        """Create/open the shared memory file and mmap it."""
        if self._mm is not None:
            return True

        try:
            # Ensure parent directory exists
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            # Create the file if it doesn't exist, or open existing
            self._fd = os.open(
                self._path,
                os.O_RDWR | os.O_CREAT,
                0o666,
            )
            # Ensure file is the right size
            os.ftruncate(self._fd, REGIME_FILE_SIZE)

            self._mm = mmap.mmap(
                self._fd, REGIME_FILE_SIZE, access=mmap.ACCESS_WRITE
            )

            # Read existing sequence if file was already populated
            raw_seq = self._mm[SEQUENCE_OFFSET : SEQUENCE_OFFSET + 8]
            existing_seq = struct.unpack("<Q", raw_seq)[0]
            # Ensure sequence is even (consistent state)
            self._sequence = existing_seq if existing_seq % 2 == 0 else existing_seq + 1

            logger.info("SharedRegimeWriter: mapped {} (seq={})", self._path, self._sequence)
            return True

        except OSError as exc:
            logger.error("SharedRegimeWriter: cannot create/open {}: {}", self._path, exc)
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def _write_sequence(self, seq: int) -> None:
        """Write the seqlock sequence atomically."""
        assert self._mm is not None
        self._mm[SEQUENCE_OFFSET : SEQUENCE_OFFSET + 8] = struct.pack("<Q", seq)

    def validate_regime_data(self, data: RegimeData) -> bool:
        """Validate regime data before writing to shared memory.

        Returns True if the data is valid, False otherwise.
        Logs a warning for each invalid field.
        """
        valid = True

        if data.overall_regime not in REGIME_MAP:
            logger.warning(
                "SharedRegimeWriter: invalid overall_regime '{}', expected one of {}",
                data.overall_regime,
                list(REGIME_MAP.keys()),
            )
            valid = False

        if data.volatility_regime not in VOLATILITY_MAP:
            logger.warning(
                "SharedRegimeWriter: invalid volatility_regime '{}', expected one of {}",
                data.volatility_regime,
                list(VOLATILITY_MAP.keys()),
            )
            valid = False

        if data.btc_dominance_trend not in DOMINANCE_MAP:
            logger.warning(
                "SharedRegimeWriter: invalid btc_dominance_trend '{}', expected one of {}",
                data.btc_dominance_trend,
                list(DOMINANCE_MAP.keys()),
            )
            valid = False

        if data.funding_rate_bias not in FUNDING_MAP:
            logger.warning(
                "SharedRegimeWriter: invalid funding_rate_bias '{}', expected one of {}",
                data.funding_rate_bias,
                list(FUNDING_MAP.keys()),
            )
            valid = False

        if not (-1.0 <= data.sentiment_score <= 1.0):
            logger.warning(
                "SharedRegimeWriter: sentiment_score {} out of range [-1.0, 1.0]",
                data.sentiment_score,
            )
            valid = False

        if not (0.0 <= data.sentiment_confidence <= 1.0):
            logger.warning(
                "SharedRegimeWriter: sentiment_confidence {} out of range [0.0, 1.0]",
                data.sentiment_confidence,
            )
            valid = False

        if not (0 <= data.fear_greed_index <= 100):
            logger.warning(
                "SharedRegimeWriter: fear_greed_index {} out of range [0, 100]",
                data.fear_greed_index,
            )
            valid = False

        if not (0.0 <= data.recommended_position_scale <= 5.0):
            logger.warning(
                "SharedRegimeWriter: recommended_position_scale {} out of range [0.0, 5.0]",
                data.recommended_position_scale,
            )
            valid = False

        if data.ttl_seconds <= 0:
            logger.warning(
                "SharedRegimeWriter: ttl_seconds {} must be > 0",
                data.ttl_seconds,
            )
            valid = False

        return valid

    def write_safe_default(self) -> bool:
        """Write conservative safe-default regime data to shared memory.

        Used when Python cold-path services fail or at startup before
        the first regime computation completes.  Ensures the Rust engine
        always has a valid (if conservative) regime to read.

        Returns True on success, False on failure.
        """
        safe = RegimeData(
            overall_regime="unknown",
            volatility_regime="high",
            sentiment_score=0.0,
            sentiment_confidence=0.0,
            fear_greed_index=50,
            btc_dominance_trend="flat",
            funding_rate_bias="neutral",
            cross_asset_correlation=0.0,
            news_impact_score=0.0,
            recommended_position_scale=0.5,
            max_leverage_override=0,
            ttl_seconds=600,
            allowed_strategies_mask=0xFFFFFFFFFFFFFFFF,
            blocked_strategies_mask=0,
        )
        logger.info("SharedRegimeWriter: writing safe defaults to shared memory")
        return self.update(safe)

    def update(self, data: RegimeData) -> bool:
        """Write regime data to shared memory using seqlock protocol.

        Returns True on success, False on failure.
        Validates the data before writing and logs every write.
        """
        if not self._ensure_mapped():
            return False

        # Validate before writing
        if not self.validate_regime_data(data):
            logger.warning(
                "SharedRegimeWriter: validation failed, writing safe defaults instead"
            )
            return self.write_safe_default() if data.overall_regime != "unknown" else False

        assert self._mm is not None

        try:
            # 1. Increment sequence to odd (writing)
            self._sequence += 1
            self._write_sequence(self._sequence)

            # 2. Pack and write the full struct
            timestamp_ms = int(time.time() * 1000)

            packed = struct.pack(
                REGIME_FMT,
                REGIME_MAGIC,                                          # magic
                REGIME_VERSION,                                        # version
                0,                                                     # _pad0
                self._sequence,                                        # sequence (odd = writing)
                timestamp_ms,                                          # timestamp_ms
                REGIME_MAP.get(data.overall_regime, REGIME_UNKNOWN),   # overall_regime
                VOLATILITY_MAP.get(data.volatility_regime, VOLATILITY_HIGH),  # volatility_regime
                b"\x00\x00",                                           # _pad1
                int(data.sentiment_score * 10_000),                    # sentiment_score_fp
                int(data.sentiment_confidence * 10_000),               # sentiment_confidence_fp
                data.fear_greed_index,                                 # fear_greed_index
                DOMINANCE_MAP.get(data.btc_dominance_trend, DOMINANCE_FLAT),  # btc_dominance_trend
                FUNDING_MAP.get(data.funding_rate_bias, FUNDING_NEUTRAL),  # funding_rate_bias
                b"\x00\x00",                                           # _pad2
                int(data.cross_asset_correlation * 10_000),            # cross_asset_correlation_fp
                int(data.news_impact_score * 10_000),                  # news_impact_score_fp
                int(data.recommended_position_scale * 10_000),         # position_scale_fp
                data.max_leverage_override,                            # max_leverage_override
                data.ttl_seconds,                                      # ttl_seconds
                data.allowed_strategies_mask,                          # allowed_strategies_mask
                data.blocked_strategies_mask,                          # blocked_strategies_mask
                b"\x00" * 24,                                          # _reserved
            )

            # Write everything except the sequence field
            # (sequence is at offset 16..24, we write before and after)
            self._mm[0:SEQUENCE_OFFSET] = packed[0:SEQUENCE_OFFSET]
            self._mm[SEQUENCE_OFFSET + 8 :] = packed[SEQUENCE_OFFSET + 8 :]

            # 3. Increment sequence to even (consistent)
            self._sequence += 1
            self._write_sequence(self._sequence)

            # Flush to ensure visibility
            self._mm.flush()

            logger.info(
                "SharedRegimeWriter: wrote regime={} vol={} sentiment={:.2f} "
                "scale={:.2f} ttl={}s seq={}",
                data.overall_regime,
                data.volatility_regime,
                data.sentiment_score,
                data.recommended_position_scale,
                data.ttl_seconds,
                self._sequence,
            )

            return True

        except Exception as exc:
            logger.error("SharedRegimeWriter: write failed: {}", exc)
            # Try to restore even sequence so readers don't spin forever
            self._sequence = (self._sequence | 1) + 1  # round up to even
            try:
                self._write_sequence(self._sequence)
            except Exception:
                pass
            return False

    def close(self) -> None:
        """Clean up resources."""
        if self._mm is not None:
            try:
                self._mm.flush()
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SharedRegimeWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
