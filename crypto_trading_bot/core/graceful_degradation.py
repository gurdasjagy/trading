"""Graceful Degradation Manager — Adaptive Resource Management.

Monitors system resources (CPU, memory, I/O) and dynamically adjusts
the bot's operational mode to prevent crashes and maintain core trading
functionality under resource pressure.

Degradation Levels:
  NORMAL      → All systems operational
  CAUTION     → Reduce non-critical services (sentiment polling frequency)
  WARNING     → Disable AI features, reduce strategy count
  CRITICAL    → Close-only mode, minimal monitoring
  EMERGENCY   → Halt all trading, close all positions

Recovery is automatic when resource pressure subsides.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class DegradationLevel(IntEnum):
    """Operational degradation levels, ordered by severity."""
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class ResourceSnapshot:
    """Point-in-time snapshot of system resource usage."""
    cpu_percent: float  # 0-100
    memory_percent: float  # 0-100
    memory_available_mb: float
    disk_percent: float  # 0-100
    shm_usage_mb: float  # /dev/shm usage
    timestamp: float


@dataclass
class DegradationThresholds:
    """Thresholds for each degradation level."""
    # CAUTION thresholds
    caution_cpu_pct: float = 75.0
    caution_mem_pct: float = 70.0

    # WARNING thresholds
    warning_cpu_pct: float = 85.0
    warning_mem_pct: float = 80.0

    # CRITICAL thresholds
    critical_cpu_pct: float = 92.0
    critical_mem_pct: float = 88.0

    # EMERGENCY thresholds
    emergency_cpu_pct: float = 97.0
    emergency_mem_pct: float = 95.0

    # Recovery hysteresis: must drop this far below threshold to recover
    recovery_margin_pct: float = 5.0

    # Minimum time at a level before escalating (prevents flapping)
    escalation_cooldown_seconds: float = 30.0

    # Minimum time at a level before recovering (prevents flapping)
    recovery_cooldown_seconds: float = 60.0


def _read_cpu_percent() -> float:
    """Read CPU usage from /proc/stat (Linux-specific, lightweight)."""
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        idle = int(parts[4])
        total = sum(int(p) for p in parts[1:])
        if not hasattr(_read_cpu_percent, "_prev"):
            _read_cpu_percent._prev = (idle, total)
            return 0.0
        prev_idle, prev_total = _read_cpu_percent._prev
        idle_delta = idle - prev_idle
        total_delta = total - prev_total
        _read_cpu_percent._prev = (idle, total)
        if total_delta <= 0:
            return 0.0
        return (1.0 - idle_delta / total_delta) * 100.0
    except Exception:
        return 0.0


def _read_memory_info() -> tuple:
    """Read memory info from /proc/meminfo. Returns (percent_used, available_mb)."""
    try:
        mem_info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    mem_info[key] = int(parts[1])  # in kB

        total = mem_info.get("MemTotal", 1)
        available = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
        used_pct = (1.0 - available / total) * 100.0
        available_mb = available / 1024.0
        return used_pct, available_mb
    except Exception:
        return 0.0, 8192.0


def _read_shm_usage_mb() -> float:
    """Read /dev/shm usage in MB."""
    try:
        stat = os.statvfs("/dev/shm")
        used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
        return used / (1024 * 1024)
    except Exception:
        return 0.0


class GracefulDegradationManager:
    """Monitors resources and manages operational degradation levels.

    Usage::

        manager = GracefulDegradationManager()
        await manager.start()

        # Check current level
        if manager.level >= DegradationLevel.WARNING:
            # Reduce operations

        # Register callbacks
        manager.on_level_change(my_callback)
    """

    def __init__(
        self,
        thresholds: Optional[DegradationThresholds] = None,
        check_interval_seconds: float = 5.0,
    ) -> None:
        self._thresholds = thresholds or DegradationThresholds()
        self._check_interval = check_interval_seconds
        self._current_level = DegradationLevel.NORMAL
        self._level_entered_at: float = time.time()
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._callbacks: List[Callable] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Action history for audit
        self._action_log: List[Dict[str, Any]] = []
        self._max_action_log = 500

        logger.info(
            "GracefulDegradationManager initialized: check_interval={}s",
            self._check_interval,
        )

    @property
    def level(self) -> DegradationLevel:
        return self._current_level

    @property
    def last_snapshot(self) -> Optional[ResourceSnapshot]:
        return self._last_snapshot

    def on_level_change(self, callback: Callable) -> None:
        """Register a callback for degradation level changes.

        Callback signature: callback(old_level, new_level, snapshot)
        """
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the resource monitoring loop."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("GracefulDegradationManager started")

    async def stop(self) -> None:
        """Stop the resource monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                snapshot = self._collect_snapshot()
                self._last_snapshot = snapshot
                new_level = self._evaluate_level(snapshot)

                if new_level != self._current_level:
                    elapsed = time.time() - self._level_entered_at

                    # Escalation: require cooldown before escalating
                    if new_level > self._current_level:
                        if elapsed >= self._thresholds.escalation_cooldown_seconds:
                            await self._change_level(new_level, snapshot)
                    # Recovery: require longer cooldown before recovering
                    elif new_level < self._current_level:
                        if elapsed >= self._thresholds.recovery_cooldown_seconds:
                            await self._change_level(new_level, snapshot)

            except Exception as e:
                logger.error("GracefulDegradationManager error: {}", e)

            await asyncio.sleep(self._check_interval)

    def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect current resource metrics."""
        cpu = _read_cpu_percent()
        mem_pct, mem_avail = _read_memory_info()
        shm = _read_shm_usage_mb()

        try:
            stat = os.statvfs("/")
            disk_pct = (1.0 - stat.f_bfree / stat.f_blocks) * 100.0
        except Exception:
            disk_pct = 0.0

        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=mem_pct,
            memory_available_mb=mem_avail,
            disk_percent=disk_pct,
            shm_usage_mb=shm,
            timestamp=time.time(),
        )

    def _evaluate_level(self, snap: ResourceSnapshot) -> DegradationLevel:
        """Determine the appropriate degradation level from resource snapshot."""
        t = self._thresholds
        margin = t.recovery_margin_pct

        # Emergency check first
        if snap.cpu_percent >= t.emergency_cpu_pct or snap.memory_percent >= t.emergency_mem_pct:
            return DegradationLevel.EMERGENCY

        # Critical
        if snap.cpu_percent >= t.critical_cpu_pct or snap.memory_percent >= t.critical_mem_pct:
            return DegradationLevel.CRITICAL

        # Warning
        if snap.cpu_percent >= t.warning_cpu_pct or snap.memory_percent >= t.warning_mem_pct:
            return DegradationLevel.WARNING

        # Caution
        if snap.cpu_percent >= t.caution_cpu_pct or snap.memory_percent >= t.caution_mem_pct:
            return DegradationLevel.CAUTION

        # Check recovery with hysteresis
        if self._current_level > DegradationLevel.NORMAL:
            # Only recover if we're sufficiently below the threshold
            if self._current_level == DegradationLevel.CAUTION:
                if (snap.cpu_percent < t.caution_cpu_pct - margin and
                        snap.memory_percent < t.caution_mem_pct - margin):
                    return DegradationLevel.NORMAL
                return DegradationLevel.CAUTION

        return DegradationLevel.NORMAL

    async def _change_level(
        self, new_level: DegradationLevel, snapshot: ResourceSnapshot
    ) -> None:
        """Execute a degradation level change."""
        old_level = self._current_level
        self._current_level = new_level
        self._level_entered_at = time.time()

        direction = "ESCALATED" if new_level > old_level else "RECOVERED"
        log_entry = {
            "time": time.time(),
            "direction": direction,
            "from": old_level.name,
            "to": new_level.name,
            "cpu": round(snapshot.cpu_percent, 1),
            "mem": round(snapshot.memory_percent, 1),
            "mem_avail_mb": round(snapshot.memory_available_mb, 0),
        }
        self._action_log.append(log_entry)
        if len(self._action_log) > self._max_action_log:
            self._action_log = self._action_log[-self._max_action_log:]

        if new_level >= DegradationLevel.WARNING:
            logger.warning(
                "🔶 Degradation {}: {} → {} (CPU={:.1f}% MEM={:.1f}% avail={:.0f}MB)",
                direction, old_level.name, new_level.name,
                snapshot.cpu_percent, snapshot.memory_percent,
                snapshot.memory_available_mb,
            )
        elif new_level >= DegradationLevel.CRITICAL:
            logger.critical(
                "🔴 Degradation {}: {} → {} (CPU={:.1f}% MEM={:.1f}%)",
                direction, old_level.name, new_level.name,
                snapshot.cpu_percent, snapshot.memory_percent,
            )
        else:
            logger.info(
                "🟢 Degradation {}: {} → {} (CPU={:.1f}% MEM={:.1f}%)",
                direction, old_level.name, new_level.name,
                snapshot.cpu_percent, snapshot.memory_percent,
            )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_level, new_level, snapshot)
                else:
                    callback(old_level, new_level, snapshot)
            except Exception as e:
                logger.error("Degradation callback error: {}", e)

    def get_recommended_actions(self) -> Dict[str, Any]:
        """Get recommended operational adjustments for the current level."""
        level = self._current_level
        return {
            "level": level.name,
            "ai_enabled": level <= DegradationLevel.CAUTION,
            "sentiment_polling": level <= DegradationLevel.CAUTION,
            "max_strategies": {
                DegradationLevel.NORMAL: 50,
                DegradationLevel.CAUTION: 20,
                DegradationLevel.WARNING: 10,
                DegradationLevel.CRITICAL: 3,
                DegradationLevel.EMERGENCY: 0,
            }[level],
            "regime_interval_seconds": {
                DegradationLevel.NORMAL: 300,
                DegradationLevel.CAUTION: 600,
                DegradationLevel.WARNING: 900,
                DegradationLevel.CRITICAL: 1800,
                DegradationLevel.EMERGENCY: 0,
            }[level],
            "allow_new_positions": level <= DegradationLevel.WARNING,
            "close_only_mode": level >= DegradationLevel.CRITICAL,
            "halt_trading": level >= DegradationLevel.EMERGENCY,
            "dashboard_refresh_seconds": {
                DegradationLevel.NORMAL: 2,
                DegradationLevel.CAUTION: 5,
                DegradationLevel.WARNING: 10,
                DegradationLevel.CRITICAL: 30,
                DegradationLevel.EMERGENCY: 60,
            }[level],
        }

    def get_status(self) -> Dict[str, Any]:
        """Get full status report."""
        snap = self._last_snapshot
        return {
            "level": self._current_level.name,
            "level_value": int(self._current_level),
            "time_at_level_seconds": round(time.time() - self._level_entered_at, 1),
            "resource_snapshot": {
                "cpu_percent": round(snap.cpu_percent, 1) if snap else 0,
                "memory_percent": round(snap.memory_percent, 1) if snap else 0,
                "memory_available_mb": round(snap.memory_available_mb, 0) if snap else 0,
                "shm_usage_mb": round(snap.shm_usage_mb, 1) if snap else 0,
            },
            "recommendations": self.get_recommended_actions(),
            "recent_actions": self._action_log[-10:],
        }
