"""Rust engine supervisor — Python slow-loop process manager.

.. deprecated:: Issue 4
    The Python supervisor that manages the Rust process is no longer needed.
    Rust is now the master — it runs independently.
    Docker manages process lifecycle.

    Replaced by: core/cold_path_orchestrator.py (health monitoring via SHM)
    Kept for reference and legacy deployments.

Launches the standalone Rust ``trading_engine`` binary as a subprocess,
monitors its health via ZeroMQ heartbeat, pushes config/model parameter
updates via ZeroMQ PUSH, and restarts the binary on crash with exponential
backoff.

Architecture:
    Python (this module) ←→ Rust binary via two ZeroMQ channels:
        • Rust → Python: PUB/SUB  tcp://127.0.0.1:5555  (telemetry)
        • Python → Rust: PUSH/PULL tcp://127.0.0.1:5556  (config updates)

Usage::

    supervisor = RustSupervisor(binary_path="/path/to/trading_engine")
    await supervisor.start()

    # Push updated strategy config
    await supervisor.push_config({"imbalance_threshold": 0.35, "enabled": True})

    # Graceful shutdown
    await supervisor.stop()
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class RustSupervisor:
    """Supervisor for the standalone Rust trading engine binary.

    Responsibilities:
    - Launch ``trading_engine`` as a child process.
    - Monitor it via ZeroMQ heartbeat (expects a message every 500 ms).
    - Push configuration blobs (strategy params, risk limits) via ZeroMQ PUSH.
    - Restart the Rust process on crash with exponential backoff.

    Args:
        binary_path: Path to the compiled ``trading_engine`` binary.
            Defaults to looking for it relative to the repo root.
        telemetry_addr: ZeroMQ PUB address published by Rust.
        config_push_addr: ZeroMQ PUSH address where Python sends config.
        heartbeat_timeout_secs: Seconds without a heartbeat before the
            process is considered dead and restarted.
        env: Optional extra environment variables for the Rust process.
    """

    _INITIAL_BACKOFF: float = 1.0
    _MAX_BACKOFF: float = 60.0
    _HEARTBEAT_TIMEOUT: float = 5.0  # seconds

    def __init__(
        self,
        binary_path: Optional[str] = None,
        telemetry_addr: str = "tcp://127.0.0.1:5555",
        config_push_addr: str = "tcp://127.0.0.1:5556",
        heartbeat_timeout_secs: float = 5.0,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.binary_path = binary_path or self._find_binary()
        self.telemetry_addr = telemetry_addr
        self.config_push_addr = config_push_addr
        self.heartbeat_timeout_secs = heartbeat_timeout_secs
        self._extra_env = env or {}

        self._process: Optional[asyncio.subprocess.Process] = None
        self._running: bool = False
        self._supervisor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._zmq_push_sock: Optional[Any] = None
        self._zmq_ctx: Optional[Any] = None
        self._last_heartbeat_ts: float = 0.0
        self._backoff: float = self._INITIAL_BACKOFF

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the Rust binary and start supervision tasks."""
        if self._running:
            logger.debug("RustSupervisor already running.")
            return

        self._running = True
        await self._init_zmq()

        self._supervisor_task = asyncio.create_task(
            self._supervision_loop(), name="rust_supervisor"
        )
        logger.info(
            "RustSupervisor started — binary: {}, telemetry: {}, config: {}",
            self.binary_path,
            self.telemetry_addr,
            self.config_push_addr,
        )

    async def stop(self) -> None:
        """Gracefully stop the Rust binary and all supervision tasks."""
        self._running = False

        # Stop supervision and heartbeat tasks
        for task in (self._supervisor_task, self._heartbeat_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Terminate the child process
        await self._terminate_process()

        # Close ZMQ sockets
        if self._zmq_push_sock is not None:
            self._zmq_push_sock.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()

        logger.info("RustSupervisor stopped.")

    # ------------------------------------------------------------------
    # Config push
    # ------------------------------------------------------------------

    async def push_config(self, config: Dict[str, Any]) -> bool:
        """Push a strategy/risk config blob to the Rust engine via ZeroMQ.

        The Rust engine polls its PULL socket every 100 ms and applies
        the new config within ~200 ms.

        Args:
            config: Dict matching ``StrategyConfig`` fields in
                ``rust_engine/src/strategy_engine.rs``.

        Returns:
            True if the message was sent, False if ZMQ is unavailable.
        """
        if self._zmq_push_sock is None:
            logger.warning("ZMQ push socket not available — config not sent.")
            return False

        try:
            import zmq
            payload = json.dumps(config).encode()
            self._zmq_push_sock.send(payload, flags=zmq.NOBLOCK)
            logger.debug("Config pushed to Rust engine: {}", config)
            return True
        except ImportError:
            logger.warning("pyzmq not installed — config not sent.")
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _init_zmq(self) -> None:
        """Initialize the ZeroMQ PUSH socket for config delivery."""
        try:
            import zmq

            self._zmq_ctx = zmq.Context()
            self._zmq_push_sock = self._zmq_ctx.socket(zmq.PUSH)
            self._zmq_push_sock.connect(self.config_push_addr)
            logger.debug("ZMQ PUSH socket connected to {}", self.config_push_addr)
        except ImportError:
            logger.warning(
                "pyzmq not installed — config push disabled. "
                "Install with: pip install pyzmq"
            )

    async def _supervision_loop(self) -> None:
        """Main supervision loop: launch, monitor, restart."""
        while self._running:
            try:
                await self._launch_process()
                # Start heartbeat monitor
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_monitor(), name="rust_heartbeat_monitor"
                )
                # Wait for process to exit
                if self._process is not None:
                    retcode = await self._process.wait()
                    logger.warning(
                        "Rust engine exited with code {}. Restarting in {:.1f}s.",
                        retcode,
                        self._backoff,
                    )
            except Exception as exc:
                logger.error("Supervision error: {}", exc)

            if not self._running:
                break

            # Cancel heartbeat monitor before restart
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            await asyncio.sleep(self._backoff)
            # Exponential backoff with cap
            self._backoff = min(self._backoff * 2.0, self._MAX_BACKOFF)

    async def _launch_process(self) -> None:
        """Launch the Rust binary as an async subprocess."""
        if not Path(self.binary_path).exists():
            logger.error("Rust binary not found: {}", self.binary_path)
            raise FileNotFoundError(f"Rust binary not found: {self.binary_path}")

        env = {**os.environ, **self._extra_env}

        logger.info("Launching Rust engine: {}", self.binary_path)
        self._process = await asyncio.create_subprocess_exec(
            self.binary_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._last_heartbeat_ts = time.monotonic()
        self._backoff = self._INITIAL_BACKOFF  # Reset backoff on successful launch

        # Stream process output
        asyncio.create_task(self._stream_output(self._process.stdout, "stdout"))
        asyncio.create_task(self._stream_output(self._process.stderr, "stderr"))

        logger.info("Rust engine launched (PID {})", self._process.pid)

        # Wait a moment for the process to bind its ZMQ sockets
        await asyncio.sleep(0.5)

    async def _heartbeat_monitor(self) -> None:
        """Monitor ZeroMQ heartbeat from the Rust engine.

        If no heartbeat is received within ``heartbeat_timeout_secs``, the
        process is considered crashed and is killed so the supervision loop
        can restart it.
        """
        try:
            import zmq
            import zmq.asyncio as azmq
        except ImportError:
            logger.debug("pyzmq not available — heartbeat monitor disabled.")
            return

        ctx = azmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"heartbeat")
        sock.connect(self.telemetry_addr)

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(sock.recv_string(), timeout=1.0)
                    self._last_heartbeat_ts = time.monotonic()
                except asyncio.TimeoutError:
                    pass

                elapsed = time.monotonic() - self._last_heartbeat_ts
                if elapsed > self.heartbeat_timeout_secs and self._last_heartbeat_ts > 0:
                    logger.warning(
                        "Rust engine heartbeat timeout ({:.1f}s). Killing process.",
                        elapsed,
                    )
                    await self._terminate_process()
                    return
        finally:
            sock.close()

    async def _terminate_process(self) -> None:
        """Send SIGTERM to the Rust process and wait for it to exit."""
        if self._process is None:
            return
        proc = self._process
        self._process = None

        if proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                logger.debug("Rust engine process terminated.")
            except Exception as exc:
                logger.debug("Error terminating Rust process: {}", exc)

    @staticmethod
    async def _stream_output(
        stream: Optional[asyncio.StreamReader], name: str
    ) -> None:
        """Pipe Rust process stdout/stderr through loguru."""
        if stream is None:
            return
        try:
            async for line in stream:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    logger.debug("[rust_engine {}] {}", name, decoded)
        except Exception:
            pass

    @staticmethod
    def _find_binary() -> str:
        """Locate the compiled ``trading_engine`` binary relative to repo root."""
        # Try common Cargo output locations
        candidates = [
            Path(__file__).parents[3] / "rust_engine" / "target" / "release" / "trading_engine",
            Path(__file__).parents[3] / "rust_engine" / "target" / "debug" / "trading_engine",
            Path("/usr/local/bin/trading_engine"),
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        # Default: assume it's on PATH or in the repo's target/release
        return str(
            Path(__file__).parents[3] / "rust_engine" / "target" / "release" / "trading_engine"
        )

    @property
    def is_running(self) -> bool:
        """True if the Rust process is currently alive."""
        return (
            self._process is not None
            and self._process.returncode is None
        )

    @property
    def pid(self) -> Optional[int]:
        """PID of the running Rust process, or None if not running."""
        if self._process is not None and self._process.returncode is None:
            return self._process.pid
        return None
