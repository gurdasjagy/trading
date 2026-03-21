"""Shared pytest fixtures for the crypto-trading-bot test suite."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from risk.risk_manager import RiskManager
from strategy.base_strategy import Signal

# ── GateIOClient fixture ──────────────────────────────────────────────────


@pytest.fixture
def mock_gateio_client():
    """Create a mock GateIOClient with pending fills support."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    client._ws_orders = {}
    client._ws_positions = {}
    client._ws_last_update = 0.0
    client._event_bus = None
    client._rate_limiter = AsyncMock()
    client._client = AsyncMock()
    return client


# ── TradeExecutor fixture ─────────────────────────────────────────────────


@pytest.fixture
def mock_trade_executor():
    """Create a mock TradeExecutor for testing."""
    from execution.trade_executor import TradeExecutor

    executor = TradeExecutor.__new__(TradeExecutor)
    executor._exchange = AsyncMock()
    executor._order_manager = AsyncMock()
    executor._position_manager = AsyncMock()
    executor._optimizer = MagicMock()
    executor._fee_calc = MagicMock()
    executor._slippage = MagicMock()
    executor._smart_entry = MagicMock()
    executor._enable_advanced_execution = False
    executor._latency_monitor = None
    executor._execution_quality_analyzer = None
    executor._adaptive_engine = None
    executor._smart_exit_engine = None
    executor._order_flow_engine = None
    executor._anti_gaming = None
    executor._telegram_alerter = None
    executor._local_orderbook_manager = None
    return executor

# ── Settings fixture ──────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance with safe test defaults (paper mode, fake keys)."""
    return Settings(
        TRADING_MODE="paper",
        MEXC_API_KEY="test-key",
        MEXC_SECRET_KEY="test-secret",
        ENCRYPTION_KEY=None,
        SECRET_KEY="test-secret-key-32-chars-padding!",
        DATABASE_URL="sqlite+aiosqlite:///./test.db",
        REDIS_URL="redis://localhost:6379/15",
    )


# ── Event loop fixture ────────────────────────────────────────────────────


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide a fresh asyncio event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── OHLCV fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Return a 100-row DataFrame of realistic fake OHLCV data."""
    rng = np.random.default_rng(42)
    n = 100
    base_price = 50_000.0
    returns = rng.normal(0.0, 0.01, n)
    closes = base_price * np.cumprod(1 + returns)
    highs = closes * (1 + rng.uniform(0.001, 0.02, n))
    lows = closes * (1 - rng.uniform(0.001, 0.02, n))
    opens = closes * (1 + rng.normal(0.0, 0.005, n))
    volumes = rng.uniform(100, 10_000, n)

    start = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# ── Sample trade fixture ──────────────────────────────────────────────────


@pytest.fixture
def sample_trade() -> dict:
    """Return a sample completed trade dictionary."""
    entry_time = datetime(2024, 3, 15, 10, 0, 0)
    exit_time = entry_time + timedelta(hours=4, minutes=30)
    entry_price = 68_500.0
    exit_price = 70_123.0
    size = 0.1
    pnl = (exit_price - entry_price) * size
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        "id": "trade-001",
        "symbol": "BTC/USDT",
        "direction": "long",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "size": size,
        "pnl": round(pnl, 4),
        "pnl_pct": round(pnl_pct, 4),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "strategy": "technical_breakout",
        "status": "closed",
        "leverage": 5,
    }


# ── Sample signal fixture ─────────────────────────────────────────────────


@pytest.fixture
def sample_signal() -> Signal:
    """Return a sample long Signal produced by a strategy."""
    return Signal(
        symbol="ETH/USDT",
        direction="long",
        strength=0.75,
        confidence=0.80,
        strategy_name="technical_breakout",
        reasoning="RSI oversold + MACD crossover + breakout above resistance",
        stop_loss=3_200.0,
        take_profit=3_800.0,
        leverage=3,
    )


# ── Mock exchange fixture ─────────────────────────────────────────────────


@pytest.fixture
def mock_exchange() -> MagicMock:
    """Return a MagicMock exchange client with async OHLCV stub."""
    exchange = MagicMock()
    exchange.get_ohlcv = AsyncMock(
        return_value=pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    )
    exchange.fetch_balance = AsyncMock(return_value={"USDT": {"free": 10_000.0}})
    exchange.create_order = AsyncMock(return_value={"id": "order-123", "status": "open"})
    exchange.cancel_order = AsyncMock(return_value={"id": "order-123", "status": "canceled"})
    return exchange


# ── RiskManager fixture ────────────────────────────────────────────────────


@pytest.fixture
def risk_manager(settings: Settings) -> RiskManager:
    """Return a RiskManager instance initialised with test settings."""
    return RiskManager(settings=settings)
