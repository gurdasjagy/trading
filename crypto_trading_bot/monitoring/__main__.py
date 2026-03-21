"""Monitoring module entrypoint — launches the web dashboard.

**Issue 4**: Updated to read from :class:`~core.shared_state_reader.SharedStateReader`
as the primary data source, with ZMQ telemetry as fallback.

Usage (Docker Compose)::

    command: python -m monitoring

Environment variables (set by ``docker-compose.yml``)::

    DASHBOARD_PORT        8080  (HTTP bind port)
    REDIS_URL             redis://127.0.0.1:6379
    ZMQ_TELEMETRY_URL     tcp://127.0.0.1:5555  (legacy fallback)
    STATE_SHM_PATH        /dev/shm/trading_state  (shared memory primary)
    GATEIO_API_KEY        (optional — enables live balance display)
    GATEIO_SECRET_KEY     (optional — enables live balance display)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

from loguru import logger


def _is_valid_env_key(value: str) -> bool:
    """Return True when *value* is a non-empty, fully-resolved credential.

    Credentials in ``engine_config.toml`` are written as ``${VAR_NAME}``
    placeholders.  When the environment variable is not set the placeholder
    is left as-is, so we must reject such strings rather than treating them
    as real keys.
    """
    return bool(value) and not value.startswith("${")


# ---------------------------------------------------------------------------
# Shared memory state cache — reads from /dev/shm/trading_state (Issue 4)
# ---------------------------------------------------------------------------

class _ShmStateCache:
    """In-memory cache populated by polling SharedStateReader.

    Primary data source for the dashboard (Issue 4).  Polls shared memory
    at ~1 Hz for near-real-time data without any network dependency.
    """

    def __init__(self, shm_path: str = "/dev/shm/trading_state") -> None:
        self._reader: Optional[Any] = None
        self._shm_path = shm_path
        self._balance: Optional[float] = None
        self._prices: dict[str, float] = {}
        self._engine_status: Optional[dict] = None

        try:
            from core.shared_state_reader import SharedStateReader

            self._reader = SharedStateReader(shm_path)
            logger.info("ShmStateCache: SharedStateReader initialized at {}", shm_path)
        except Exception as exc:
            logger.debug("ShmStateCache: SharedStateReader not available: {}", exc)

    def poll(self) -> None:
        """Read the latest state from shared memory and update the cache.

        Called at ~1 Hz by the background poller task.
        """
        if self._reader is None:
            return

        try:
            snapshot = self._reader.read_consistent()
            if not snapshot.is_consistent:
                return

            self._balance = snapshot.engine.total_pnl

            for sym in snapshot.symbols:
                if sym.has_data and sym.mid_price > 0:
                    # Use symbol_id as key (numeric)
                    key = str(sym.symbol_id)
                    self._prices[key] = sym.mid_price

            self._engine_status = {
                "uptime_seconds": snapshot.engine.uptime_seconds,
                "total_book_updates": snapshot.engine.total_book_updates,
                "total_orders_sent": snapshot.engine.total_orders_sent,
                "total_fills": snapshot.engine.total_fills,
                "total_pnl": snapshot.engine.total_pnl,
                "num_symbols": snapshot.engine.num_symbols,
                "is_valid": snapshot.engine.is_valid,
            }
        except Exception as exc:
            logger.debug("ShmStateCache: poll error: {}", exc)

    def get_balance(self) -> Optional[float]:
        return self._balance

    def get_price(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol)

    def get_engine_status(self) -> Optional[dict]:
        return self._engine_status


# ---------------------------------------------------------------------------
# ZMQ telemetry cache — populated by the background subscriber (legacy)
# ---------------------------------------------------------------------------

class _ZmqTelemetryCache:
    """Thread-safe in-memory cache fed by the Rust engine ZMQ telemetry stream.

    Legacy data source — used as fallback when shared memory is unavailable.
    """

    def __init__(self) -> None:
        self._balance: Optional[float] = None
        self._prices: dict[str, float] = {}  # symbol → mid_price

    def update(self, event: str, payload: dict) -> None:
        if event == "account_snapshot":
            self._balance = payload.get("balance_usdt")
        elif event == "microstructure":
            key = payload.get("book_key", "")
            price = payload.get("mid_price", 0.0)
            if key and price:
                symbol = key.split(":", 1)[-1] if ":" in key else key
                self._prices[symbol] = float(price)

    def get_balance(self) -> Optional[float]:
        return self._balance

    def get_price(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol)


# ---------------------------------------------------------------------------
# Minimal exchange adapter used by TradingDashboard in monitoring-only mode
# ---------------------------------------------------------------------------

class _MonitoringExchange:
    """Thin exchange adapter for the dashboard.

    **Issue 4**: Uses SharedStateReader (via _ShmStateCache) as the primary
    data source.  Falls back to ZMQ cache and then REST API.

    * Public market data (OHLCV, tickers) is fetched directly from the
      Gate.io REST API via ccxt — no API key required.
    * Account data (balance) is pulled from shared memory first, then ZMQ,
      then REST API as a last resort.
    """

    def __init__(
        self,
        shm_cache: _ShmStateCache,
        zmq_cache: _ZmqTelemetryCache,
    ) -> None:
        self._shm = shm_cache
        self._zmq = zmq_cache
        self._ccxt: Optional[Any] = None

    async def _ensure_ccxt(self) -> Optional[Any]:
        if self._ccxt is not None:
            return self._ccxt
        try:
            import ccxt.async_support as ccxt  # type: ignore[import]

            api_key = os.getenv("GATEIO_API_KEY", "")
            secret = os.getenv("GATEIO_SECRET_KEY", "")
            kwargs: dict[str, Any] = {"options": {"defaultType": "swap"}}
            if _is_valid_env_key(api_key) and _is_valid_env_key(secret):
                kwargs["apiKey"] = api_key
                kwargs["secret"] = secret
            self._ccxt = ccxt.gateio(**kwargs)
        except Exception as exc:
            logger.debug("ccxt not available for monitoring exchange: {}", exc)
        return self._ccxt

    async def get_balance(self):
        from exchange.base_exchange import Balance  # local import avoids circular deps

        # 1. Prefer shared memory (Issue 4)
        shm_balance = self._shm.get_balance()
        if shm_balance is not None:
            return Balance(usdt_free=shm_balance, usdt_total=shm_balance)

        # 2. Fallback: ZMQ telemetry cache
        cached = self._zmq.get_balance()
        if cached is not None:
            return Balance(usdt_free=cached, usdt_total=cached)

        # 3. Last resort: authenticated REST call
        try:
            ex = await self._ensure_ccxt()
            if ex and ex.apiKey:
                data = await ex.fetch_balance({"type": "swap"})
                usdt = data.get("USDT", {})
                return Balance(
                    usdt_free=float(usdt.get("free", 0.0)),
                    usdt_total=float(usdt.get("total", 0.0)),
                )
        except Exception as exc:
            logger.debug("Monitoring balance fetch error: {}", exc)

        return Balance(usdt_free=0.0, usdt_total=0.0)

    async def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        import pandas as pd

        try:
            ex = await self._ensure_ccxt()
            if ex is None:
                return pd.DataFrame()

            # Normalise Gate.io symbol format: "BTC_USDT" → "BTC/USDT:USDT"
            if "_" in symbol and not symbol.endswith(":USDT"):
                ccxt_sym = symbol.replace("_", "/") + ":USDT"
            else:
                ccxt_sym = symbol

            ohlcv = await ex.fetch_ohlcv(ccxt_sym, timeframe, limit=limit)
            if ohlcv:
                df = pd.DataFrame(
                    ohlcv, columns=["time", "open", "high", "low", "close", "volume"]
                )
                df["time"] = df["time"] / 1000  # milliseconds → seconds
                return df
        except Exception as exc:
            logger.debug("Monitoring OHLCV fetch error for {}: {}", symbol, exc)

        return pd.DataFrame()

    async def get_ticker(self, symbol: str):
        return None

    async def get_positions(self) -> list:
        return []

    async def get_open_orders(self, symbol: str | None = None) -> list:
        return []

    async def fetch_balance(self, params: dict | None = None) -> dict:
        return {}

    async def disconnect(self) -> None:
        if self._ccxt is not None:
            try:
                await self._ccxt.close()
            except Exception:
                pass
            self._ccxt = None


class _MonitoringEngine:
    """Minimal engine shim passed to TradingDashboard in monitoring-only mode.

    Holds an *exchange* attribute so the dashboard's REST endpoints can serve
    live balance and market data.  All trading-related subsystems are ``None``.
    """

    def __init__(self, exchange: _MonitoringExchange) -> None:
        self.exchange = exchange
        # Subsystems not needed for read-only dashboard:
        self.state_manager = None
        self.position_manager = None
        self.trade_journal = None
        self.risk_manager = None
        self.ai_brain = None
        self.strategy_manager = None
        self.realtime_hub = None


# ---------------------------------------------------------------------------
# Background shared memory poller (Issue 4)
# ---------------------------------------------------------------------------

async def _shm_poller(cache: _ShmStateCache) -> None:
    """Poll shared memory at ~1 Hz to keep the dashboard data fresh."""
    logger.info("Dashboard shared memory poller started")
    while True:
        try:
            cache.poll()
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("SHM poller error: {}", exc)
            await asyncio.sleep(5.0)


# ---------------------------------------------------------------------------
# Background ZMQ subscriber (legacy fallback)
# ---------------------------------------------------------------------------

async def _zmq_subscriber(cache: _ZmqTelemetryCache) -> None:
    """Subscribe to the Rust engine's ZMQ PUB socket and update the cache."""
    telemetry_url = os.getenv("ZMQ_TELEMETRY_URL", "tcp://127.0.0.1:5555")
    try:
        import zmq  # type: ignore[import]
        import zmq.asyncio as azmq  # type: ignore[import]

        ctx = azmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(telemetry_url)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        logger.info("Dashboard ZMQ subscriber connected to {}", telemetry_url)

        while True:
            msg: str = await sock.recv_string()
            parts = msg.split(" ", 1)
            if len(parts) == 2:
                event, payload_str = parts
                try:
                    payload = json.loads(payload_str)
                    cache.update(event, payload)
                except Exception:
                    pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Dashboard ZMQ subscriber error: {}", exc)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def _main() -> None:
    logger.info("Dashboard starting…")

    from config.settings import Settings
    from monitoring.dashboard import TradingDashboard

    settings = Settings.get_settings()

    # Initialize data sources
    shm_path = os.getenv("STATE_SHM_PATH", "/dev/shm/trading_state")
    shm_cache = _ShmStateCache(shm_path)
    zmq_cache = _ZmqTelemetryCache()
    exchange = _MonitoringExchange(shm_cache, zmq_cache)
    engine = _MonitoringEngine(exchange)

    # Start background data fetchers
    shm_task = asyncio.create_task(_shm_poller(shm_cache))
    zmq_task = asyncio.create_task(_zmq_subscriber(zmq_cache))

    try:
        dashboard = TradingDashboard(settings=settings, engine=engine)
        await dashboard.run(host="0.0.0.0", port=settings.monitoring.dashboard_port)
    finally:
        shm_task.cancel()
        zmq_task.cancel()
        try:
            await shm_task
        except asyncio.CancelledError:
            pass
        try:
            await zmq_task
        except asyncio.CancelledError:
            pass
        await exchange.disconnect()


if __name__ == "__main__":
    asyncio.run(_main())

