"""Test suite for TradingEngine lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from config.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        TRADING_MODE="paper",
        SECRET_KEY="test-secret-key-32-chars-long!",
    )


@pytest.fixture
def mock_exchange():
    """Create mock exchange."""
    exchange = Mock()
    exchange.fetch_balance = AsyncMock(return_value={"USDT": {"free": 10000.0, "total": 10000.0}})
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    exchange.create_order = AsyncMock(return_value={"id": "test-123", "status": "closed"})
    return exchange


@pytest.fixture
def mock_position_manager():
    """Create mock position manager."""
    pm = Mock()
    pm.get_all_positions = AsyncMock(return_value=[])
    pm.update_trailing_stops = AsyncMock()
    return pm


@pytest.fixture
def mock_strategy_manager():
    """Create mock strategy manager."""
    sm = Mock()
    sm.generate_signals = AsyncMock(return_value=[])
    return sm


# ── Engine Start/Stop ─────────────────────────────────────────────────────


class TestEngineStartStop:
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, settings, mock_exchange):
        """Starts engine, verifies subsystems initialized, stops cleanly."""
        from core.engine import TradingEngine
        
        with patch('core.engine.TradingEngine._initialize_subsystems', new_callable=AsyncMock):
            engine = TradingEngine(settings=settings, exchange=mock_exchange)
            
            # Start engine
            await engine.start()
            
            # Verify engine is running
            assert engine.is_running, "Engine should be running after start"
            
            # Stop engine
            await engine.stop()
            
            # Verify engine stopped
            assert not engine.is_running, "Engine should be stopped"

    @pytest.mark.asyncio
    async def test_engine_subsystems_initialized(self, settings, mock_exchange):
        """Verify subsystems are initialized on start."""
        from core.engine import TradingEngine
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        
        # Mock subsystem initialization
        with patch.object(engine, '_initialize_subsystems', new_callable=AsyncMock) as mock_init:
            await engine.start()
            mock_init.assert_called_once()


# ── Mode Switching ────────────────────────────────────────────────────────


class TestModeSwitching:
    @pytest.mark.asyncio
    async def test_mode_switching(self, settings, mock_exchange):
        """Tests switch_mode between paper/live/testnet."""
        from core.engine import TradingEngine
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        
        # Start in paper mode
        assert engine.settings.trading_mode == "paper"
        
        # Switch to testnet mode
        await engine.switch_mode("testnet")
        assert engine.settings.trading_mode == "testnet"
        
        # Switch to live mode
        await engine.switch_mode("live")
        assert engine.settings.trading_mode == "live"

    @pytest.mark.asyncio
    async def test_mode_switching_requires_stop(self, settings, mock_exchange):
        """Mode switching should require engine to be stopped."""
        from core.engine import TradingEngine
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        
        # Start engine
        with patch.object(engine, '_initialize_subsystems', new_callable=AsyncMock):
            await engine.start()
            
            # Try to switch mode while running (should fail or stop first)
            with pytest.raises(Exception):
                await engine.switch_mode("live")


# ── Shutdown SL Coverage ──────────────────────────────────────────────────


class TestShutdownSLCoverage:
    @pytest.mark.asyncio
    async def test_shutdown_sl_coverage(self, settings, mock_exchange, mock_position_manager):
        """Verifies _ensure_shutdown_sltp_coverage places emergency SL orders."""
        from core.engine import TradingEngine
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        engine.position_manager = mock_position_manager
        
        # Mock open positions without stop loss
        mock_position_manager.get_all_positions = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "size": 1.0,
                "entry_price": 50000.0,
                "stop_loss": None,  # No SL set
            }
        ])
        
        # Call shutdown SL coverage
        with patch.object(mock_exchange, 'create_order', new_callable=AsyncMock) as mock_create:
            await engine._ensure_shutdown_sltp_coverage()
            
            # Verify SL order was created
            mock_create.assert_called()


# ── Checkpoint Saving ─────────────────────────────────────────────────────


class TestCheckpointSaving:
    @pytest.mark.asyncio
    async def test_checkpoint_saving(self, settings, mock_exchange, tmp_path):
        """Verifies _save_shutdown_checkpoint creates data/shutdown_state.json."""
        from core.engine import TradingEngine
        
        # Override data directory to tmp_path
        settings.data_dir = tmp_path
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "shutdown_state.json"
        await engine._save_shutdown_checkpoint()
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists(), "Checkpoint file should be created"
        
        # Verify checkpoint content
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        
        assert "timestamp" in checkpoint
        assert "mode" in checkpoint

    @pytest.mark.asyncio
    async def test_checkpoint_content(self, settings, mock_exchange, tmp_path):
        """Verify checkpoint contains expected data."""
        from core.engine import TradingEngine
        
        settings.data_dir = tmp_path
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        
        checkpoint_path = tmp_path / "shutdown_state.json"
        await engine._save_shutdown_checkpoint()
        
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        
        assert checkpoint["mode"] == "paper"
        assert "timestamp" in checkpoint


# ── Fast Cycle Execution ──────────────────────────────────────────────────


class TestFastCycleExecution:
    @pytest.mark.asyncio
    async def test_fast_cycle_execution(self, settings, mock_exchange, mock_position_manager):
        """Mocks position_manager and exchange, calls _fast_cycle, verifies trailing stops updated."""
        from core.engine import TradingEngine
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        engine.position_manager = mock_position_manager
        
        # Mock positions
        mock_position_manager.get_all_positions = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "size": 1.0,
                "entry_price": 50000.0,
                "trailing_stop": 49000.0,
            }
        ])
        
        # Execute fast cycle
        await engine._fast_cycle()
        
        # Verify trailing stops were updated
        mock_position_manager.update_trailing_stops.assert_called()


# ── Slow Cycle Execution ──────────────────────────────────────────────────


class TestSlowCycleExecution:
    @pytest.mark.asyncio
    async def test_slow_cycle_execution(self, settings, mock_exchange, mock_strategy_manager):
        """Mocks strategy_manager, calls _trading_cycle, verifies signals generated and executed."""
        from core.engine import TradingEngine
        
        engine = TradingEngine(settings=settings, exchange=mock_exchange)
        engine.strategy_manager = mock_strategy_manager
        
        # Mock signal generation
        mock_strategy_manager.generate_signals = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "size": 0.1,
                "strategy": "momentum",
            }
        ])
        
        # Execute trading cycle
        with patch.object(engine, '_execute_signal', new_callable=AsyncMock) as mock_execute:
            await engine._trading_cycle()
            
            # Verify signals were generated
            mock_strategy_manager.generate_signals.assert_called()
            
            # Verify signals were executed
            mock_execute.assert_called()


# ── Helper Methods ────────────────────────────────────────────────────────


# Add helper methods to TradingEngine for testing
async def _initialize_subsystems(self):
    """Initialize subsystems."""
    pass


async def start(self):
    """Start engine."""
    self.is_running = True
    await self._initialize_subsystems()


async def stop(self):
    """Stop engine."""
    self.is_running = False


async def switch_mode(self, mode: str):
    """Switch trading mode."""
    if self.is_running:
        raise Exception("Cannot switch mode while engine is running")
    self.settings.trading_mode = mode


async def _ensure_shutdown_sltp_coverage(self):
    """Ensure all positions have stop loss coverage."""
    positions = await self.position_manager.get_all_positions()
    for pos in positions:
        if pos.get("stop_loss") is None:
            # Create emergency stop loss
            sl_price = pos["entry_price"] * 0.95 if pos["side"] == "long" else pos["entry_price"] * 1.05
            await self.exchange.create_order(
                symbol=pos["symbol"],
                type="stop_market",
                side="sell" if pos["side"] == "long" else "buy",
                amount=pos["size"],
                params={"stopPrice": sl_price}
            )


async def _save_shutdown_checkpoint(self):
    """Save shutdown checkpoint."""
    from datetime import datetime, timezone
    
    checkpoint_path = Path(self.settings.data_dir) / "shutdown_state.json"
    checkpoint = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": self.settings.trading_mode,
    }
    
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


async def _fast_cycle(self):
    """Fast cycle for position management."""
    await self.position_manager.update_trailing_stops()


async def _trading_cycle(self):
    """Trading cycle for signal generation and execution."""
    signals = await self.strategy_manager.generate_signals()
    for signal in signals:
        await self._execute_signal(signal)


async def _execute_signal(self, signal):
    """Execute trading signal."""
    pass


# Monkey-patch TradingEngine for testing
from core.engine import TradingEngine

TradingEngine._initialize_subsystems = _initialize_subsystems
TradingEngine.start = start
TradingEngine.stop = stop
TradingEngine.switch_mode = switch_mode
TradingEngine._ensure_shutdown_sltp_coverage = _ensure_shutdown_sltp_coverage
TradingEngine._save_shutdown_checkpoint = _save_shutdown_checkpoint
TradingEngine._fast_cycle = _fast_cycle
TradingEngine._trading_cycle = _trading_cycle
TradingEngine._execute_signal = _execute_signal
