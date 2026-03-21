"""Unit tests for Gate.io TradFi MT5 symbol mapping."""

import pytest
from crypto_trading_bot.exchange.gateio_tradfi_mt5_client import GateIOTradFiMT5Client


class TestGateIOTradFiMT5SymbolMapping:
    """Test symbol mapping for Gate.io TradFi MT5 client."""

    @pytest.fixture
    def client(self):
        """Create a GateIOTradFiMT5Client instance for testing."""
        # Note: This creates an instance without connecting
        # MT5 connection is not required for symbol mapping tests
        return GateIOTradFiMT5Client(
            login="12345678",
            password="test_password",
            server="GateIO-TradFi-Test",
            testnet=True
        )

    def test_xauusd_mapping(self, client):
        """Test XAUUSD symbol mapping."""
        assert client._resolve_mt5_symbol("XAUUSD") == "XAUUSD"
        assert client._resolve_mt5_symbol("XAU/USD") == "XAUUSD"
        assert client._resolve_mt5_symbol("XAU_USDT") == "XAUUSD"
        assert client._resolve_mt5_symbol("XAU/USDT") == "XAUUSD"

    def test_xagusd_mapping(self, client):
        """Test XAGUSD symbol mapping."""
        assert client._resolve_mt5_symbol("XAGUSD") == "XAGUSD"
        assert client._resolve_mt5_symbol("XAG/USD") == "XAGUSD"
        assert client._resolve_mt5_symbol("XAG_USDT") == "XAGUSD"
        assert client._resolve_mt5_symbol("XAG/USDT") == "XAGUSD"

    def test_eurusd_mapping(self, client):
        """Test EURUSD symbol mapping."""
        assert client._resolve_mt5_symbol("EURUSD") == "EURUSD"
        assert client._resolve_mt5_symbol("EUR/USD") == "EURUSD"
        assert client._resolve_mt5_symbol("EUR_USDT") == "EURUSD"

    def test_gbpusd_mapping(self, client):
        """Test GBPUSD symbol mapping."""
        assert client._resolve_mt5_symbol("GBPUSD") == "GBPUSD"
        assert client._resolve_mt5_symbol("GBP/USD") == "GBPUSD"
        assert client._resolve_mt5_symbol("GBP_USDT") == "GBPUSD"

    def test_usdjpy_mapping(self, client):
        """Test USDJPY symbol mapping."""
        assert client._resolve_mt5_symbol("USDJPY") == "USDJPY"
        assert client._resolve_mt5_symbol("USD/JPY") == "USDJPY"
        assert client._resolve_mt5_symbol("JPY_USDT") == "USDJPY"

    def test_case_insensitive(self, client):
        """Test that symbol mapping is case-insensitive."""
        assert client._resolve_mt5_symbol("xauusd") == "XAUUSD"
        assert client._resolve_mt5_symbol("XaUuSd") == "XAUUSD"
        assert client._resolve_mt5_symbol("eurusd") == "EURUSD"

    def test_normalize_symbol(self, client):
        """Test symbol normalization."""
        assert client._normalize_symbol("XAUUSD") == "XAUUSD"
        assert client._normalize_symbol("XAU/USD") == "XAUUSD"
        assert client._normalize_symbol("XAU_USDT") == "XAU"
        assert client._normalize_symbol("EURUSD") == "EURUSD"
        assert client._normalize_symbol("EUR/USD") == "EURUSD"

    def test_unsupported_symbol(self, client):
        """Test that unsupported symbols return None."""
        assert client._resolve_mt5_symbol("INVALID") is None
        assert client._resolve_mt5_symbol("BTC/USD") is None

    def test_lot_size_rounding(self, client):
        """Test lot size rounding."""
        config = {
            "min_lot": 0.01,
            "max_lot": 500.0,
            "lot_step": 0.01,
        }

        assert client._round_lot_size(0.015, config) == 0.02
        assert client._round_lot_size(0.014, config) == 0.01
        assert client._round_lot_size(1.555, config) == 1.56
        assert client._round_lot_size(0.005, config) == 0.01  # Below min
        assert client._round_lot_size(600.0, config) == 500.0  # Above max

    def test_get_current_session(self, client):
        """Test session detection."""
        # Note: This test depends on current UTC time
        session = client.get_current_session()
        assert isinstance(session, str)
        # Should return one of: sydney, tokyo, london, new_york, or closed
        valid_sessions = ["sydney", "tokyo", "london", "new_york", "closed"]
        for sess in session.split(","):
            assert sess in valid_sessions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
