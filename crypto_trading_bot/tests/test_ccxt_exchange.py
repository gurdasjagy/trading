"""Tests for the new exchange layer: CcxtExchange, PaperExchange, and ExchangeFactory."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from exchange.base_exchange import (
    Balance,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    PositionSide,
    Ticker,
)
from exchange.bybit_forex_client import BybitForexClient
from exchange.ccxt_exchange import SUPPORTED_EXCHANGES, CcxtExchange, ExchangeNotSupportedError
from exchange.exchange_factory import create_exchange
from exchange.paper_exchange import PaperExchange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ticker(symbol: str = "BTC/USDT", last: float = 50_000.0) -> Ticker:
    return Ticker(
        symbol=symbol,
        bid=last - 1,
        ask=last + 1,
        last=last,
        high=last * 1.01,
        low=last * 0.99,
        volume=1_000.0,
        timestamp=int(time.time() * 1000),
    )


def _mock_ccxt_client(price: float = 50_000.0) -> MagicMock:
    """Return a MagicMock that mimics ccxt.async_support exchange methods."""
    client = MagicMock()
    client.load_markets = AsyncMock()
    client.close = AsyncMock()

    raw_balance = {
        "total": {"USDT": 10_000.0},
        "free": {"USDT": 9_000.0},
        "used": {"USDT": 1_000.0},
    }
    client.fetch_balance = AsyncMock(return_value=raw_balance)
    client.fetch_ticker = AsyncMock(
        return_value={
            "symbol": "BTC/USDT",
            "bid": price - 1,
            "ask": price + 1,
            "last": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "baseVolume": 1_000.0,
            "timestamp": int(time.time() * 1000),
            "info": {},
        }
    )
    client.fetch_order_book = AsyncMock(
        return_value={"bids": [[price - 5, 1.0]], "asks": [[price + 5, 1.0]]}
    )
    raw_ohlcv = [[int(time.time() * 1000), price, price * 1.01, price * 0.99, price, 100.0]]
    client.fetch_ohlcv = AsyncMock(return_value=raw_ohlcv)

    raw_order = {
        "id": "order-001",
        "symbol": "BTC/USDT",
        "type": "market",
        "side": "buy",
        "amount": 0.1,
        "price": price,
        "filled": 0.1,
        "remaining": 0.0,
        "status": "closed",
        "timestamp": int(time.time() * 1000),
        "fee": {"cost": 0.5},
        "info": {},
    }
    client.create_market_order = AsyncMock(return_value=raw_order)
    client.create_limit_order = AsyncMock(
        return_value={**raw_order, "type": "limit", "status": "open"}
    )
    client.create_order = AsyncMock(return_value={**raw_order, "type": "stop_market"})
    client.cancel_order = AsyncMock(return_value={"id": "order-001", "status": "canceled"})
    client.cancel_all_orders = AsyncMock(return_value=[])
    client.fetch_order = AsyncMock(return_value=raw_order)
    client.fetch_open_orders = AsyncMock(return_value=[raw_order])
    client.fetch_my_trades = AsyncMock(return_value=[])
    client.set_leverage = AsyncMock(return_value={"leverage": 10})
    client.set_margin_mode = AsyncMock(return_value={"marginMode": "cross"})
    raw_position = {
        "symbol": "BTC/USDT",
        "side": "long",
        "contracts": 0.1,
        "entryPrice": price,
        "markPrice": price * 1.02,
        "unrealizedPnl": price * 0.02 * 0.1,
        "leverage": 5,
        "initialMargin": price * 0.1 / 5,
        "liquidationPrice": price * 0.5,
        "timestamp": int(time.time() * 1000),
    }
    client.fetch_positions = AsyncMock(return_value=[raw_position])
    client.fetch_funding_rate = AsyncMock(return_value={"fundingRate": 0.0001})
    client.fetch_open_interest = AsyncMock(return_value={"openInterest": 1_000_000.0})
    return client


def _make_settings(mode: str = "paper", exchange_id: str = "mexc") -> MagicMock:
    settings = MagicMock()
    settings.trading_mode = mode
    settings.primary_exchange = exchange_id
    settings.exchange_api_key = "test-api-key"
    settings.exchange_secret = "test-secret"
    settings.exchange_passphrase = ""
    settings.paper_trading_balance = 10_000.0
    exchange_cfg = MagicMock()
    exchange_cfg.primary_exchange = exchange_id
    exchange_cfg.use_testnet = False
    settings.exchange = exchange_cfg
    return settings


# ---------------------------------------------------------------------------
# CcxtExchange tests
# ---------------------------------------------------------------------------


class TestCcxtExchangeInit:
    def test_supported_exchanges_list(self):
        assert "mexc" in SUPPORTED_EXCHANGES
        assert "gateio" in SUPPORTED_EXCHANGES
        assert "bingx" in SUPPORTED_EXCHANGES
        assert "bitget" in SUPPORTED_EXCHANGES
        assert "bybit" in SUPPORTED_EXCHANGES

    def test_init_valid_exchange(self):
        ex = CcxtExchange("mexc", "key", "secret")
        assert ex.name == "mexc"

    def test_init_invalid_exchange_raises(self):
        with pytest.raises(ExchangeNotSupportedError, match="Unsupported exchange"):
            CcxtExchange("binance", "key", "secret")

    def test_name_property(self):
        ex = CcxtExchange("gateio", "key", "secret")
        assert ex.name == "gateio"


class TestCcxtExchangeConnect:
    @pytest.mark.asyncio
    async def test_connect_loads_markets(self):
        ex = CcxtExchange("mexc", "key", "secret")
        mock_client = _mock_ccxt_client()
        # Override the exchange class so no real connection is attempted
        ex._exchange_class = MagicMock(return_value=mock_client)
        await ex.connect()
        mock_client.load_markets.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self):
        ex = CcxtExchange("mexc", "key", "secret")
        mock_client = _mock_ccxt_client()
        ex._exchange_class = MagicMock(return_value=mock_client)
        await ex.connect()
        await ex.disconnect()
        mock_client.close.assert_awaited_once()


class TestCcxtExchangeMarketData:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.ex = CcxtExchange("mexc", "key", "secret")
        self.mock_client = _mock_ccxt_client(price=50_000.0)
        self.ex._client = self.mock_client
        self.ex._rate_limiter = MagicMock()
        self.ex._rate_limiter.acquire = AsyncMock()

    @pytest.mark.asyncio
    async def test_get_balance(self):
        balance = await self.ex.get_balance()
        assert isinstance(balance, Balance)
        assert balance.usdt_total == 10_000.0
        assert balance.usdt_free == 9_000.0

    @pytest.mark.asyncio
    async def test_get_ticker(self):
        ticker = await self.ex.get_ticker("BTC/USDT")
        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTC/USDT"
        assert ticker.last == 50_000.0

    @pytest.mark.asyncio
    async def test_get_orderbook(self):
        book = await self.ex.get_orderbook("BTC/USDT")
        assert "bids" in book
        assert "asks" in book

    @pytest.mark.asyncio
    async def test_get_ohlcv_returns_dataframe(self):
        df = await self.ex.get_ohlcv("BTC/USDT", "1h", 1)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


class TestCcxtExchangeOrders:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.ex = CcxtExchange("mexc", "key", "secret")
        self.mock_client = _mock_ccxt_client(price=50_000.0)
        self.ex._client = self.mock_client
        self.ex._rate_limiter = MagicMock()
        self.ex._rate_limiter.acquire = AsyncMock()

    @pytest.mark.asyncio
    async def test_create_market_order(self):
        order = await self.ex.create_market_order("BTC/USDT", OrderSide.BUY, 0.1)
        assert isinstance(order, Order)
        assert order.id == "order-001"
        assert order.status == OrderStatus.CLOSED

    @pytest.mark.asyncio
    async def test_create_limit_order(self):
        order = await self.ex.create_limit_order("BTC/USDT", OrderSide.BUY, 0.1, 49_000.0)
        assert isinstance(order, Order)

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        result = await self.ex.cancel_order("order-001", "BTC/USDT")
        assert result["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_get_open_orders(self):
        orders = await self.ex.get_open_orders("BTC/USDT")
        assert isinstance(orders, list)
        assert len(orders) == 1
        assert isinstance(orders[0], Order)


class TestCcxtExchangePositions:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.ex = CcxtExchange("mexc", "key", "secret")
        self.mock_client = _mock_ccxt_client(price=50_000.0)
        self.ex._client = self.mock_client
        self.ex._rate_limiter = MagicMock()
        self.ex._rate_limiter.acquire = AsyncMock()

    @pytest.mark.asyncio
    async def test_get_positions(self):
        positions = await self.ex.get_positions()
        assert isinstance(positions, list)
        assert len(positions) == 1
        pos = positions[0]
        assert isinstance(pos, Position)
        assert pos.side == PositionSide.LONG

    @pytest.mark.asyncio
    async def test_get_position_returns_matching(self):
        pos = await self.ex.get_position("BTC/USDT")
        assert pos is not None
        assert pos.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_position_returns_none_for_unknown(self):
        self.mock_client.fetch_positions = AsyncMock(return_value=[])
        pos = await self.ex.get_position("ETH/USDT")
        assert pos is None

    @pytest.mark.asyncio
    async def test_set_leverage(self):
        await self.ex.set_leverage("BTC/USDT", 10)
        self.mock_client.set_leverage.assert_awaited_once_with(10, "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_get_funding_rate(self):
        rate = await self.ex.get_funding_rate("BTC/USDT")
        assert rate == 0.0001

    @pytest.mark.asyncio
    async def test_get_open_interest(self):
        oi = await self.ex.get_open_interest("BTC/USDT")
        assert oi == 1_000_000.0


# ---------------------------------------------------------------------------
# PaperExchange tests
# ---------------------------------------------------------------------------


class TestPaperExchangeInit:
    def test_initializes_with_default_balance(self, tmp_path):
        ex = PaperExchange(starting_balance=5_000.0, state_file=str(tmp_path / "state.json"))
        assert ex._usdt_balance == 5_000.0

    def test_name_property(self, tmp_path):
        ex = PaperExchange(state_file=str(tmp_path / "state.json"))
        assert ex.name == "paper"


class TestPaperExchangeStatePersistence:
    @pytest.mark.asyncio
    async def test_saves_and_loads_state(self, tmp_path):
        state_file = tmp_path / "paper_state.json"
        ex = PaperExchange(starting_balance=1_000.0, state_file=str(state_file))
        await ex.connect()

        # Manually modify balance
        ex._usdt_balance = 8_500.0
        ex._save_state()

        # Load fresh instance
        ex2 = PaperExchange(starting_balance=1_000.0, state_file=str(state_file))
        ex2._load_state()
        assert ex2._usdt_balance == 8_500.0

    @pytest.mark.asyncio
    async def test_starts_fresh_without_state_file(self, tmp_path):
        ex = PaperExchange(starting_balance=7_500.0, state_file=str(tmp_path / "missing.json"))
        await ex.connect()
        assert ex._usdt_balance == 7_500.0


class TestPaperExchangeBalance:
    @pytest.mark.asyncio
    async def test_get_balance_returns_correct_usdt(self, tmp_path):
        ex = PaperExchange(starting_balance=10_000.0, state_file=str(tmp_path / "s.json"))
        await ex.connect()
        balance = await ex.get_balance()
        assert isinstance(balance, Balance)
        assert balance.usdt_total == 10_000.0
        assert balance.usdt_free == 10_000.0


class TestPaperExchangeOrders:
    def _make_price_exchange(self, price: float = 50_000.0) -> MagicMock:
        px = MagicMock()
        px.name = "mock"
        px.connect = AsyncMock()
        px.disconnect = AsyncMock()
        px.get_ticker = AsyncMock(return_value=_ticker("BTC/USDT", price))
        return px

    @pytest.mark.asyncio
    async def test_market_buy_reduces_balance(self, tmp_path):
        price = 50_000.0
        px = self._make_price_exchange(price)
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()

        order = await ex.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        assert isinstance(order, Order)
        assert order.status == OrderStatus.CLOSED
        # Balance should be reduced by fill cost + slippage + fee
        assert ex._usdt_balance < 100_000.0

    @pytest.mark.asyncio
    async def test_market_sell_closes_position(self, tmp_path):
        price = 50_000.0
        px = self._make_price_exchange(price)
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()

        # Open a long
        await ex.create_market_order("BTC/USDT", OrderSide.BUY, 0.5)
        assert "BTC/USDT" in ex._positions

        # Close it
        await ex.create_market_order("BTC/USDT", OrderSide.SELL, 0.5, {"reduceOnly": True})
        assert "BTC/USDT" not in ex._positions

    @pytest.mark.asyncio
    async def test_get_positions_returns_list(self, tmp_path):
        price = 50_000.0
        px = self._make_price_exchange(price)
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()
        await ex.create_market_order("BTC/USDT", OrderSide.BUY, 0.1)
        positions = await ex.get_positions()
        assert isinstance(positions, list)
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_cancel_order(self, tmp_path):
        px = self._make_price_exchange()
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()

        # Register a trigger order to cancel
        order = await ex.create_stop_loss_order("BTC/USDT", OrderSide.SELL, 0.1, 45_000.0)
        assert order.id in ex._open_orders

        result = await ex.cancel_order(order.id, "BTC/USDT")
        assert result["status"] == "canceled"
        assert order.id not in ex._open_orders

    @pytest.mark.asyncio
    async def test_open_orders_returns_pending(self, tmp_path):
        px = self._make_price_exchange()
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()

        await ex.create_stop_loss_order("BTC/USDT", OrderSide.SELL, 0.1, 45_000.0)
        open_orders = await ex.get_open_orders("BTC/USDT")
        assert len(open_orders) == 1

    @pytest.mark.asyncio
    async def test_set_leverage(self, tmp_path):
        ex = PaperExchange(state_file=str(tmp_path / "s.json"))
        await ex.connect()
        result = await ex.set_leverage("BTC/USDT", 10)
        assert result["leverage"] == 10

    @pytest.mark.asyncio
    async def test_trade_history_records_fills(self, tmp_path):
        price = 50_000.0
        px = self._make_price_exchange(price)
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()
        await ex.create_market_order("BTC/USDT", OrderSide.BUY, 0.1)
        history = await ex.get_trade_history()
        assert len(history) == 1
        assert history[0]["type"] == "open"

    @pytest.mark.asyncio
    async def test_pnl_calculated_correctly(self, tmp_path):
        price = 50_000.0
        px = self._make_price_exchange(price)
        ex = PaperExchange(
            starting_balance=100_000.0,
            state_file=str(tmp_path / "s.json"),
            price_exchange=px,
        )
        await ex.connect()

        # Open long at ~50000
        await ex.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        balance_after_open = ex._usdt_balance

        # Close at a higher price (simulate profit)
        new_price = 55_000.0
        px.get_ticker = AsyncMock(return_value=_ticker("BTC/USDT", new_price))
        await ex.create_market_order("BTC/USDT", OrderSide.SELL, 1.0, {"reduceOnly": True})

        # Balance should be higher than after opening (profit)
        assert ex._usdt_balance > balance_after_open


# ---------------------------------------------------------------------------
# ExchangeFactory tests
# ---------------------------------------------------------------------------


class TestExchangeFactory:
    def test_creates_paper_exchange_for_paper_mode(self):
        settings = _make_settings(mode="paper")
        with patch("exchange.exchange_factory._build_price_feed") as mock_feed:
            mock_feed.return_value = None
            exchange = create_exchange(settings)
        assert isinstance(exchange, PaperExchange)

    def test_creates_ccxt_exchange_for_live_mode(self):
        settings = _make_settings(mode="live")
        exchange = create_exchange(settings)
        assert isinstance(exchange, CcxtExchange)
        assert exchange.name == "mexc"

    def test_live_mode_raises_without_api_key(self):
        settings = _make_settings(mode="live")
        settings.exchange_api_key = None
        settings.mexc_api_key = None
        with pytest.raises(ValueError, match="API key"):
            create_exchange(settings)

    def test_live_mode_raises_without_secret(self):
        settings = _make_settings(mode="live")
        settings.exchange_secret = None
        settings.mexc_secret_key = None
        with pytest.raises(ValueError, match="API secret"):
            create_exchange(settings)

    def test_raises_for_unsupported_mode(self):
        settings = _make_settings(mode="backtest")
        with pytest.raises(ValueError, match="Unsupported trading mode"):
            create_exchange(settings)

    def test_paper_trading_balance_passed_through(self):
        settings = _make_settings(mode="paper")
        settings.paper_trading_balance = 25_000.0
        with patch("exchange.exchange_factory._build_price_feed", return_value=None):
            exchange = create_exchange(settings)
        assert isinstance(exchange, PaperExchange)
        assert exchange._starting_balance == 25_000.0

    def test_uses_generic_credentials_over_specific(self):
        settings = _make_settings(mode="live")
        settings.exchange_api_key = "generic-key"
        settings.exchange_secret = "generic-secret"
        settings.mexc_api_key = "specific-key"
        exchange = create_exchange(settings)
        assert isinstance(exchange, CcxtExchange)
        assert exchange.api_key == "generic-key"

    def test_falls_back_to_exchange_specific_credentials(self):
        settings = _make_settings(mode="live")
        settings.exchange_api_key = None
        settings.exchange_secret = None
        settings.mexc_api_key = "mexc-key"
        settings.mexc_secret_key = "mexc-secret"
        exchange = create_exchange(settings)
        assert isinstance(exchange, CcxtExchange)
        assert exchange.api_key == "mexc-key"


# ---------------------------------------------------------------------------
# BybitForexClient tests
# ---------------------------------------------------------------------------


class TestBybitForexClient:
    """Unit tests for BybitForexClient forex helpers."""

    def _make_client(self) -> BybitForexClient:
        return BybitForexClient(api_key="key", secret_key="secret")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_exchange_name(self):
        client = self._make_client()
        assert client.name == "bybit"

    def test_default_options_include_linear_subtype(self):
        client = self._make_client()
        assert client._default_options.get("defaultSubType") == "linear"
        assert client._default_options.get("defaultType") == "swap"

    # ------------------------------------------------------------------
    # calculate_lot_size
    # ------------------------------------------------------------------

    def test_calculate_lot_size_xau(self):
        client = self._make_client()
        # margin_per_lot = 1 * 2000 / 20 = 100 → lots = 500 / 100 = 5
        lots = client.calculate_lot_size("XAU/USD", usdt_amount=500, leverage=20, current_price=2000)
        assert lots == pytest.approx(5.0)

    def test_calculate_lot_size_clamps_to_min(self):
        client = self._make_client()
        # Very small amount → must be at least min_lot = 0.01
        lots = client.calculate_lot_size("XAU/USD", usdt_amount=0.001, leverage=10, current_price=2000)
        assert lots == pytest.approx(0.01)

    def test_calculate_lot_size_clamps_to_max(self):
        client = self._make_client()
        # Huge amount → must be capped at max_lot = 100
        lots = client.calculate_lot_size("XAU/USD", usdt_amount=10_000_000, leverage=500, current_price=2000)
        assert lots == pytest.approx(100.0)

    def test_calculate_lot_size_unknown_symbol_raises(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="Unknown forex pair"):
            client.calculate_lot_size("BTC/USD", usdt_amount=100, leverage=10, current_price=50000)

    # ------------------------------------------------------------------
    # calculate_pip_value
    # ------------------------------------------------------------------

    def test_calculate_pip_value_xau(self):
        client = self._make_client()
        # pip_value = lot_size * pip_value_per_lot * contract_size = 2 * 0.01 * 1 = 0.02
        assert client.calculate_pip_value("XAU/USD", lot_size=2.0) == pytest.approx(0.02)

    def test_calculate_pip_value_xag(self):
        client = self._make_client()
        # pip_value = 1 * 0.05 * 5000 = 250
        assert client.calculate_pip_value("XAG/USD", lot_size=1.0) == pytest.approx(250.0)

    def test_calculate_pip_value_unknown_symbol_raises(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="Unknown forex pair"):
            client.calculate_pip_value("ETH/USD", lot_size=1.0)

    # ------------------------------------------------------------------
    # calculate_margin_required
    # ------------------------------------------------------------------

    def test_calculate_margin_required_xau(self):
        client = self._make_client()
        # margin = (1 * 1 * 2000) / 20 = 100
        margin = client.calculate_margin_required("XAU/USD", lot_size=1.0, price=2000, leverage=20)
        assert margin == pytest.approx(100.0)

    def test_calculate_margin_required_xag(self):
        client = self._make_client()
        # margin = (0.5 * 5000 * 30) / 50 = 1500
        margin = client.calculate_margin_required("XAG/USD", lot_size=0.5, price=30, leverage=50)
        assert margin == pytest.approx(1500.0)

    def test_calculate_margin_required_unknown_symbol_raises(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="Unknown forex pair"):
            client.calculate_margin_required("EUR/USD", lot_size=1.0, price=1.1, leverage=10)

    # ------------------------------------------------------------------
    # get_spread  (async)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_spread_returns_pips(self):
        client = self._make_client()
        mock_ticker = Ticker(
            symbol="XAU/USD",
            bid=1999.5,
            ask=2000.5,
            last=2000.0,
            volume=100.0,
            timestamp=int(time.time() * 1000),
        )
        client.get_ticker = AsyncMock(return_value=mock_ticker)
        result = await client.get_spread("XAU/USD")
        # spread = (2000.5 - 1999.5) / 0.01 = 100 pips
        assert result["spread_pips"] == pytest.approx(100.0)
        assert result["bid"] == pytest.approx(1999.5)
        assert result["ask"] == pytest.approx(2000.5)

    @pytest.mark.asyncio
    async def test_get_spread_unknown_symbol_raises(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="Unknown forex pair"):
            await client.get_spread("GBP/USD")

    # ------------------------------------------------------------------
    # create_forex_order  (async)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_forex_order_buy_with_sl_tp(self):
        client = self._make_client()
        mock_ticker = Ticker(
            symbol="XAU/USD",
            bid=1999.0,
            ask=2001.0,
            last=2000.0,
            volume=100.0,
            timestamp=int(time.time() * 1000),
        )
        client.get_ticker = AsyncMock(return_value=mock_ticker)
        client.set_leverage = AsyncMock(return_value={})
        mock_order = MagicMock()
        client.create_market_order = AsyncMock(return_value=mock_order)

        await client.create_forex_order(
            symbol="XAU/USD",
            side=OrderSide.BUY,
            lot_size=1.0,
            leverage=20,
            sl_pips=50,
            tp_pips=100,
        )

        client.set_leverage.assert_awaited_once_with("XAU/USD", 20)
        call_kwargs = client.create_market_order.call_args
        params = call_kwargs.args[3] if len(call_kwargs.args) > 3 else call_kwargs.kwargs.get("params", {})
        # SL = 2000 - 50 * 0.01 = 1999.5; TP = 2000 + 100 * 0.01 = 2001.0
        assert params["stopLoss"] == pytest.approx(1999.5)
        assert params["takeProfit"] == pytest.approx(2001.0)

    @pytest.mark.asyncio
    async def test_create_forex_order_sell_with_sl_tp(self):
        client = self._make_client()
        mock_ticker = Ticker(
            symbol="XAU/USD",
            bid=1999.0,
            ask=2001.0,
            last=2000.0,
            volume=100.0,
            timestamp=int(time.time() * 1000),
        )
        client.get_ticker = AsyncMock(return_value=mock_ticker)
        client.set_leverage = AsyncMock(return_value={})
        client.create_market_order = AsyncMock(return_value=MagicMock())

        await client.create_forex_order(
            symbol="XAU/USD",
            side=OrderSide.SELL,
            lot_size=1.0,
            leverage=20,
            sl_pips=50,
            tp_pips=100,
        )

        call_kwargs = client.create_market_order.call_args
        params = call_kwargs.args[3] if len(call_kwargs.args) > 3 else call_kwargs.kwargs.get("params", {})
        # SL = 2000 + 50 * 0.01 = 2000.5; TP = 2000 - 100 * 0.01 = 1999.0
        assert params["stopLoss"] == pytest.approx(2000.5)
        assert params["takeProfit"] == pytest.approx(1999.0)

    @pytest.mark.asyncio
    async def test_create_forex_order_no_sl_tp(self):
        client = self._make_client()
        mock_ticker = Ticker(
            symbol="XAU/USD",
            bid=1999.0,
            ask=2001.0,
            last=2000.0,
            volume=100.0,
            timestamp=int(time.time() * 1000),
        )
        client.get_ticker = AsyncMock(return_value=mock_ticker)
        client.set_leverage = AsyncMock(return_value={})
        client.create_market_order = AsyncMock(return_value=MagicMock())

        await client.create_forex_order(
            symbol="XAU/USD",
            side=OrderSide.BUY,
            lot_size=0.5,
            leverage=10,
        )

        call_kwargs = client.create_market_order.call_args
        params = call_kwargs.args[3] if len(call_kwargs.args) > 3 else call_kwargs.kwargs.get("params", {})
        assert "stopLoss" not in params
        assert "takeProfit" not in params
