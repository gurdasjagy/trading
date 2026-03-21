"""Crash detection and auto-protection system."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from loguru import logger


class CrashLevel(str, Enum):
    """Crash protection alert levels."""

    NORMAL = "normal"
    YELLOW = "yellow"   # BTC -3 % in 1 h
    ORANGE = "orange"   # BTC -5 % in 1 h
    RED = "red"         # BTC -8 % in 1 h
    BLACK_SWAN = "black_swan"  # BTC -15 % in 4 h


@dataclass
class CrashState:
    """Current crash protection state."""

    level: CrashLevel = CrashLevel.NORMAL
    circuit_breaker_active: bool = False
    circuit_breaker_until: float = 0.0  # Unix timestamp
    reentry_size_pct: float = 1.0       # fraction of normal size (0–1)
    last_crash_ts: float = 0.0
    recovery_phase: bool = False


@dataclass
class PriceSnapshot:
    """BTC price at a given time."""

    price: float
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────────
# Threshold configuration
# ──────────────────────────────────────────────────────────────────────

_LEVELS: List[Tuple[CrashLevel, float, float]] = [
    # (level, drop_pct, lookback_hours)
    (CrashLevel.BLACK_SWAN, 0.15, 4.0),
    (CrashLevel.RED, 0.08, 1.0),
    (CrashLevel.ORANGE, 0.05, 1.0),
    (CrashLevel.YELLOW, 0.03, 1.0),
]

_CIRCUIT_BREAKER_HOURS: Dict[CrashLevel, float] = {
    CrashLevel.RED: 2.0,
    CrashLevel.BLACK_SWAN: 24.0,
}

# How long to wait before increasing re-entry size after recovery
_REENTRY_STEP_HOURS: float = 2.0
_REENTRY_STEP_SIZE: float = 0.25   # increase by 25 % of normal each step
_INITIAL_REENTRY_SIZE: float = 0.25  # start at 25 % of normal size


class CrashProtector:
    """Monitors BTC price and activates protective measures during crashes.

    Integration
    -----------
    Call :meth:`update_price` with the latest BTC price at regular intervals
    (e.g. every 10 seconds).  Before each trading cycle, call
    :meth:`get_current_level` to check whether the engine should proceed.

    Level actions:
    * **YELLOW** : Reduce all new position sizes by 50 %, tighten stops by 30 %.
    * **ORANGE** : Close all profitable positions, tighten losers' stops to 1 %.
    * **RED**    : Close ALL positions immediately, circuit breaker for 2 hours.
    * **BLACK_SWAN**: Close everything, circuit breaker for 24 hours, send emergency alerts.
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[str, CrashLevel], None]] = None,
    ) -> None:
        """
        Args:
            alert_callback: Optional function called with (message, level) on
                level changes.  Use this to send Telegram/email alerts.
        """
        self._state = CrashState()
        self._price_history: Deque[PriceSnapshot] = deque()
        self._alert_callback = alert_callback
        # Track consecutive stable checks for recovery detection
        self._stable_checks: int = 0

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def update_price(self, btc_price: float) -> CrashLevel:
        """Record a new BTC price and re-evaluate the crash level.

        Args:
            btc_price: Latest BTC/USDT price.

        Returns:
            Current :class:`CrashLevel` after evaluation.
        """
        now = time.time()
        self._price_history.append(PriceSnapshot(price=btc_price, timestamp=now))

        # Prune history older than 5 hours (covers BLACK_SWAN lookback)
        cutoff = now - 5.0 * 3600
        while self._price_history and self._price_history[0].timestamp < cutoff:
            self._price_history.popleft()

        new_level = self._evaluate_level(btc_price, now)
        self._update_state(new_level, now)
        return self._state.level

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_current_level(self) -> CrashLevel:
        """Return the current crash protection level."""
        self._check_circuit_breaker_expiry()
        return self._state.level

    def is_circuit_breaker_active(self) -> bool:
        """Return True if trading is halted by the circuit breaker."""
        self._check_circuit_breaker_expiry()
        return self._state.circuit_breaker_active

    def get_position_size_multiplier(self) -> float:
        """Return the factor by which new position sizes should be scaled.

        * NORMAL: 1.0 (no change)
        * YELLOW: 0.5 (50 % reduction)
        * ORANGE+: 0.0 (no new positions)
        * Recovery: starts at 0.25, grows by 0.25 every 2 hours
        """
        self._check_circuit_breaker_expiry()
        if self._state.circuit_breaker_active:
            return 0.0
        level = self._state.level
        if level == CrashLevel.NORMAL:
            if self._state.recovery_phase:
                return self._state.reentry_size_pct
            return 1.0
        if level == CrashLevel.YELLOW:
            return 0.5
        return 0.0

    def get_stop_tightening_pct(self) -> float:
        """Return the fraction by which stop distances should be tightened.

        * NORMAL: 0.0 (no change)
        * YELLOW: 0.30 (tighten by 30 %)
        * ORANGE: use 1 % hard stop (caller's responsibility)
        """
        level = self._state.level
        if level == CrashLevel.YELLOW:
            return 0.30
        return 0.0

    def get_state(self) -> CrashState:
        """Return a copy of the current crash state."""
        return CrashState(
            level=self._state.level,
            circuit_breaker_active=self._state.circuit_breaker_active,
            circuit_breaker_until=self._state.circuit_breaker_until,
            reentry_size_pct=self._state.reentry_size_pct,
            last_crash_ts=self._state.last_crash_ts,
            recovery_phase=self._state.recovery_phase,
        )

    def advance_recovery(self) -> float:
        """Advance the post-crash recovery phase (call every 2 hours).

        Increases the re-entry size fraction by 25 % each step until
        it reaches 100 %.

        Returns:
            Updated reentry_size_pct.
        """
        if not self._state.recovery_phase:
            return 1.0
        new_size = min(1.0, self._state.reentry_size_pct + _REENTRY_STEP_SIZE)
        self._state.reentry_size_pct = new_size
        if new_size >= 1.0:
            self._state.recovery_phase = False
            logger.info("[CrashProtector] Recovery complete — full size restored")
        else:
            logger.info("[CrashProtector] Recovery: reentry size now {:.0%}", new_size)
        return new_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_level(self, current_price: float, now: float) -> CrashLevel:
        """Determine the appropriate crash level from price history."""
        for level, drop_threshold, lookback_hours in _LEVELS:
            lookback_ts = now - lookback_hours * 3600
            # Find the highest price in the lookback window
            window = [s for s in self._price_history if s.timestamp >= lookback_ts]
            if not window:
                continue
            high_price = max(s.price for s in window)
            if high_price <= 0:
                continue
            drop_pct = (high_price - current_price) / high_price
            if drop_pct >= drop_threshold:
                logger.warning(
                    "[CrashProtector] Level {} triggered: drop={:.1%} in {:.0f}h",
                    level.value,
                    drop_pct,
                    lookback_hours,
                )
                return level

        return CrashLevel.NORMAL

    def _update_state(self, new_level: CrashLevel, now: float) -> None:
        """Update internal state when the crash level changes."""
        old_level = self._state.level
        if new_level == old_level:
            # Check for V-shaped recovery
            if old_level == CrashLevel.NORMAL and self._state.recovery_phase:
                self._stable_checks += 1
            return

        self._state.level = new_level
        logger.info(
            "[CrashProtector] Level change: {} → {}", old_level.value, new_level.value
        )

        if new_level in _CIRCUIT_BREAKER_HOURS:
            hours = _CIRCUIT_BREAKER_HOURS[new_level]
            self._state.circuit_breaker_active = True
            self._state.circuit_breaker_until = now + hours * 3600
            self._state.last_crash_ts = now
            logger.warning(
                "[CrashProtector] Circuit breaker active for {:.0f} hours", hours
            )

        if self._alert_callback:
            msg = (
                f"CRASH PROTECTION [{new_level.value.upper()}]: "
                f"Market drop threshold exceeded."
            )
            try:
                self._alert_callback(msg, new_level)
            except Exception as exc:
                logger.error("[CrashProtector] Alert callback failed: {}", exc)

        # Entering recovery phase when returning to NORMAL from a crash
        if new_level == CrashLevel.NORMAL and old_level != CrashLevel.NORMAL:
            self._state.recovery_phase = True
            self._state.reentry_size_pct = _INITIAL_REENTRY_SIZE
            self._stable_checks = 0
            logger.info(
                "[CrashProtector] Recovery phase started — reentry at {:.0%}",
                _INITIAL_REENTRY_SIZE,
            )

    def _check_circuit_breaker_expiry(self) -> None:
        """Deactivate the circuit breaker if its time has elapsed."""
        if (
            self._state.circuit_breaker_active
            and time.time() >= self._state.circuit_breaker_until
        ):
            self._state.circuit_breaker_active = False
            logger.info("[CrashProtector] Circuit breaker expired — trading may resume")
