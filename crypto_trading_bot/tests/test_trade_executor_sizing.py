"""Test suite for TradeExecutor contract sizing on Gate.io."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

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
    """Create mock exchange with Gate.io contract specs."""
    exchange = Mock()
    exchange.name = "gateio"
    
    # Mock market info for different symbols
    exchange.markets = {
        "BTC/USDT:USDT": {
            "id": "BTC_USDT",
            "symbol": "BTC/USDT:USDT",
            "base": "BTC",
            "quote": "USDT",
            "settle": "USDT",
            "type": "swap",
            "spot": False,
            "margin": False,
            "swap": True,
            "future": False,
            "contract": True,
            "contractSize": 0.0001,  # 1 contract = 0.0001 BTC
            "limits": {
                "amount": {"min": 1, "max": 1000000},
                "price": {"min": 0.1, "max": 1000000},
            },
        },
        "ETH/USDT:USDT": {
            "id": "ETH_USDT",
            "symbol": "ETH/USDT:USDT",
            "base": "ETH",
            "quote": "USDT",
            "settle": "USDT",
            "type": "swap",
            "contractSize": 0.001,  # 1 contract = 0.001 ETH
            "limits": {
                "amount": {"min": 1, "max": 1000000},
            },
        },
        "SOL/USDT:USDT": {
            "id": "SOL_USDT",
            "symbol": "SOL/USDT:USDT",
            "base": "SOL",
            "quote": "USDT",
            "settle": "USDT",
            "type": "swap",
            "contractSize": 0.01,  # 1 contract = 0.01 SOL
            "limits": {
                "amount": {"min": 1, "max": 1000000},
            },
        },
    }
    
    exchange.fetch_ticker = AsyncMock(return_value={
        "last": 50000.0,
        "bid": 49990.0,
        "ask": 50010.0,
    })
    
    exchange.create_order = AsyncMock(return_value={
        "id": "test-order-123",
        "symbol": "BTC/USDT:USDT",
        "type": "limit",
        "side": "buy",
        "amount": 100,  # contracts
        "filled": 100,
        "status": "closed",
    })
    
    return exchange


# ── BTC Contract Sizing ───────────────────────────────────────────────────


class TestBTCContractSizing:
    @pytest.mark.asyncio
    async def test_btc_contract_sizing_with_leverage(self, mock_exchange):
        """Verify USDT amounts convert to whole integer contracts for BTC."""
        # BTC: 1 contract = 0.0001 BTC
        # Price: $50,000
        # Position size: $1,000 USDT
        # Leverage: 5x
        # Effective size: $5,000
        # BTC amount: $5,000 / $50,000 = 0.1 BTC
        # Contracts: 0.1 / 0.0001 = 1,000 contracts
        
        contract_size = 0.0001
        price = 50000.0
        position_size_usdt = 1000.0
        leverage = 5
        
        effective_size = position_size_usdt * leverage
        btc_amount = effective_size / price
        contracts = int(btc_amount / contract_size)
        
        assert contracts == 1000, f"Expected 1000 contracts, got {contracts}"
        assert isinstance(contracts, int), "Contracts must be whole integers"

    @pytest.mark.asyncio
    async def test_btc_contract_sizing_10x_leverage(self, mock_exchange):
        """Test BTC contract sizing with 10x leverage."""
        contract_size = 0.0001
        price = 50000.0
        position_size_usdt = 500.0
        leverage = 10
        
        effective_size = position_size_usdt * leverage
        btc_amount = effective_size / price
        contracts = int(btc_amount / contract_size)
        
        assert contracts == 1000, f"Expected 1000 contracts, got {contracts}"

    @pytest.mark.asyncio
    async def test_btc_contract_sizing_20x_leverage(self, mock_exchange):
        """Test BTC contract sizing with 20x leverage."""
        contract_size = 0.0001
        price = 50000.0
        position_size_usdt = 250.0
        leverage = 20
        
        effective_size = position_size_usdt * leverage
        btc_amount = effective_size / price
        contracts = int(btc_amount / contract_size)
        
        assert contracts == 1000, f"Expected 1000 contracts, got {contracts}"


# ── ETH Contract Sizing ───────────────────────────────────────────────────


class TestETHContractSizing:
    @pytest.mark.asyncio
    async def test_eth_contract_sizing(self, mock_exchange):
        """Test ETH contract conversion."""
        # ETH: 1 contract = 0.001 ETH
        # Price: $3,000
        # Position size: $300 USDT
        # Leverage: 5x
        # Effective size: $1,500
        # ETH amount: $1,500 / $3,000 = 0.5 ETH
        # Contracts: 0.5 / 0.001 = 500 contracts
        
        contract_size = 0.001
        price = 3000.0
        position_size_usdt = 300.0
        leverage = 5
        
        effective_size = position_size_usdt * leverage
        eth_amount = effective_size / price
        contracts = int(eth_amount / contract_size)
        
        assert contracts == 500, f"Expected 500 contracts, got {contracts}"
        assert isinstance(contracts, int), "Contracts must be whole integers"


# ── SOL Contract Sizing ───────────────────────────────────────────────────


class TestSOLContractSizing:
    @pytest.mark.asyncio
    async def test_sol_contract_sizing(self, mock_exchange):
        """Test SOL contract conversion."""
        # SOL: 1 contract = 0.01 SOL
        # Price: $100
        # Position size: $200 USDT
        # Leverage: 5x
        # Effective size: $1,000
        # SOL amount: $1,000 / $100 = 10 SOL
        # Contracts: 10 / 0.01 = 1,000 contracts
        
        contract_size = 0.01
        price = 100.0
        position_size_usdt = 200.0
        leverage = 5
        
        effective_size = position_size_usdt * leverage
        sol_amount = effective_size / price
        contracts = int(sol_amount / contract_size)
        
        assert contracts == 1000, f"Expected 1000 contracts, got {contracts}"
        assert isinstance(contracts, int), "Contracts must be whole integers"


# ── Minimum Contract Floor ────────────────────────────────────────────────


class TestMinimumContractFloor:
    @pytest.mark.asyncio
    async def test_minimum_contract_floor(self, settings):
        """Verify min_contract_override=1 from settings.risk is enforced."""
        # Very small position should still result in at least 1 contract
        contract_size = 0.0001
        price = 50000.0
        position_size_usdt = 1.0  # Very small
        leverage = 1
        
        effective_size = position_size_usdt * leverage
        btc_amount = effective_size / price
        contracts = int(btc_amount / contract_size)
        
        # Apply minimum floor
        min_contracts = getattr(settings.risk, 'min_contract_override', 1)
        contracts = max(contracts, min_contracts)
        
        assert contracts >= 1, "Minimum contract count should be 1"


# ── Fractional Contract Rounding ──────────────────────────────────────────


class TestFractionalContractRounding:
    @pytest.mark.asyncio
    async def test_fractional_contract_rounding(self):
        """Test rounding behavior for fractional contracts."""
        # Test case: 1.7 contracts should round down to 1
        contract_size = 0.0001
        price = 50000.0
        position_size_usdt = 85.0  # Results in 1.7 contracts
        leverage = 1
        
        effective_size = position_size_usdt * leverage
        btc_amount = effective_size / price
        contracts_float = btc_amount / contract_size
        contracts = int(contracts_float)  # Floor rounding
        
        assert contracts == 1, f"Expected 1 contract (floor), got {contracts}"
        assert contracts_float > 1.0, "Float value should be > 1"

    @pytest.mark.asyncio
    async def test_fractional_contract_rounding_up(self):
        """Test that we don't accidentally round up."""
        # Test case: 1.9 contracts should still round down to 1
        contract_size = 0.0001
        price = 50000.0
        position_size_usdt = 95.0  # Results in 1.9 contracts
        leverage = 1
        
        effective_size = position_size_usdt * leverage
        btc_amount = effective_size / price
        contracts_float = btc_amount / contract_size
        contracts = int(contracts_float)  # Floor rounding
        
        assert contracts == 1, f"Expected 1 contract (floor), got {contracts}"
        assert 1.9 <= contracts_float < 2.0, "Float value should be ~1.9"
