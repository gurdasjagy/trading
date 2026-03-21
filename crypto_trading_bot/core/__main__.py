"""Entry point for the Python cold-path services.

**Issue 4**: Replaces the legacy ZMQ-based Rust supervisor with the
:class:`~core.cold_path_orchestrator.ColdPathOrchestrator`.

Usage::

    python -m core

This starts the ColdPathOrchestrator which manages:
  - Regime detection (every 5 minutes)
  - AI sentiment analysis (every 15 minutes)
  - Health monitoring (every 10 seconds)

The Rust engine handles ALL hot-path trading decisions.

Legacy mode
-----------
Set ``USE_LEGACY_SUPERVISOR=1`` to fall back to the old ZMQ-based
supervisor for backward compatibility during migration.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger


def _use_legacy() -> bool:
    """Check if the legacy ZMQ supervisor should be used."""
    val = os.environ.get("USE_LEGACY_SUPERVISOR", "").strip().lower()
    return val in ("1", "true", "yes", "on")


async def _main_cold_path() -> None:
    """Start the ColdPathOrchestrator (new default)."""
    from config.settings import Settings
    from core.cold_path_orchestrator import ColdPathOrchestrator

    settings = Settings.get_settings()
    orchestrator = ColdPathOrchestrator(settings)
    await orchestrator.start()


async def _main_legacy() -> None:
    """Legacy ZMQ-based supervisor (for backward compatibility).

    This is the original implementation that monitored the Rust engine
    via ZeroMQ PUB/SUB.  It is kept for deployments that have not yet
    migrated to the shared-memory-based cold-path architecture.
    """
    import signal
    import time
    from typing import Any, Optional

    telemetry_addr = os.getenv("ZMQ_TELEMETRY_URL", "tcp://127.0.0.1:5555")
    config_addr = os.getenv("ZMQ_CONFIG_URL", "tcp://127.0.0.1:5556")
    heartbeat_timeout = float(os.getenv("ZMQ_HEARTBEAT_TIMEOUT", "10"))

    logger.info(
        "Rust Supervisor starting (LEGACY mode) — "
        "telemetry={}, config={}, heartbeat_timeout={}s",
        telemetry_addr,
        config_addr,
        heartbeat_timeout,
    )

    zmq: Optional[Any] = None
    zmq_ctx: Optional[Any] = None
    sub_sock: Optional[Any] = None
    push_sock: Optional[Any] = None

    try:
        import zmq as _zmq
        import zmq.asyncio as azmq

        zmq = _zmq
        zmq_ctx = azmq.Context.instance()

        sub_sock = zmq_ctx.socket(zmq.SUB)
        sub_sock.connect(telemetry_addr)
        sub_sock.setsockopt(zmq.SUBSCRIBE, b"")
        logger.info("ZMQ SUB connected to {}", telemetry_addr)

        push_sock = zmq_ctx.socket(zmq.PUSH)
        push_sock.connect(config_addr)
        logger.info("ZMQ PUSH connected to {}", config_addr)
    except ImportError:
        logger.error(
            "pyzmq is not installed — supervisor cannot communicate with the "
            "Rust engine.  Install with:  pip install pyzmq"
        )
        return

    _push_initial_config(push_sock, zmq)

    running = True

    def _handle_signal() -> None:
        nonlocal running
        logger.info("Received shutdown signal — stopping supervisor…")
        running = False

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    last_heartbeat = time.monotonic()
    engine_alive = False

    logger.info("Supervisor entering monitoring loop…")

    while running:
        try:
            msg = await asyncio.wait_for(sub_sock.recv_string(), timeout=1.0)

            if msg.startswith("heartbeat"):
                last_heartbeat = time.monotonic()
                if not engine_alive:
                    logger.info("Rust engine heartbeat detected — engine is alive")
                    engine_alive = True
            else:
                logger.debug("[rust_telemetry] {}", msg[:200])

        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("ZMQ recv error: {}", exc)
            await asyncio.sleep(1)

        elapsed = time.monotonic() - last_heartbeat
        if engine_alive and elapsed > heartbeat_timeout:
            logger.warning(
                "Rust engine heartbeat lost ({:.1f}s since last). "
                "Engine may have crashed — waiting for it to restart…",
                elapsed,
            )
            engine_alive = False

    logger.info("Supervisor shutting down…")
    if sub_sock:
        sub_sock.close()
    if push_sock:
        push_sock.close()
    logger.info("Supervisor stopped.")


def _push_initial_config(push_sock: Any, zmq: Any) -> None:
    """Read the engine config file and push strategy settings to the Rust engine."""
    config_path = "/etc/trading/engine_config.toml"
    try:
        import toml  # type: ignore[import]
    except ImportError:
        try:
            import tomllib as toml  # type: ignore[import]
        except ImportError:
            logger.warning(
                "Neither 'toml' nor 'tomllib' is available — "
                "skipping initial strategy config push"
            )
            return

    try:
        with open(config_path, "rb") as fh:
            raw = fh.read()
        import re

        def _expand(s: str) -> str:
            return re.sub(
                r"\$\{([^}]+)\}",
                lambda m: os.getenv(m.group(1), m.group(0)),
                s,
            )

        cfg = toml.loads(_expand(raw.decode()))
        strategy_cfg = cfg.get("strategy")
        if strategy_cfg is None:
            logger.debug("No [strategy] section in {}; skipping push", config_path)
            return

        import json

        payload = json.dumps(strategy_cfg).encode()
        push_sock.send(payload, zmq.NOBLOCK)
        logger.info(
            "Initial strategy config pushed to Rust engine (enabled={})",
            strategy_cfg.get("enabled", "?"),
        )
    except FileNotFoundError:
        logger.debug("Config file {} not found; skipping initial strategy push", config_path)
    except Exception as exc:
        logger.warning("Failed to push initial strategy config: {}", exc)


async def _main() -> None:
    """Route to the appropriate entrypoint."""
    if _use_legacy():
        logger.info("Using LEGACY ZMQ-based supervisor (USE_LEGACY_SUPERVISOR=1)")
        await _main_legacy()
    else:
        logger.info("Using ColdPathOrchestrator (Issue 4)")
        await _main_cold_path()


if __name__ == "__main__":
    asyncio.run(_main())

