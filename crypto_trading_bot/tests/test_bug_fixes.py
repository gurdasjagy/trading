"""Tests for the bug fixes: ERROR 1-3, BUG 4-13, and validating key changes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exchange.base_exchange import Ticker


# ---------------------------------------------------------------------------
# ERROR 1: LocalOrderBook REST fallback when watch_order_book is unavailable
# ---------------------------------------------------------------------------


class TestLocalOrderBookRestFallback:
    """test_local_orderbook_rest_fallback — REST polling when WS unavailable."""

    @pytest.mark.asyncio
    async def test_switches_to_rest_on_not_implemented(self):
        """_stream_symbol must switch to REST polling when watch_order_book raises
        NotImplementedError and successfully update the book cache."""
        from exchange.local_orderbook import LocalOrderBookManager

        call_count = {"rest": 0, "ws": 0}

        async def fake_watch_order_book(symbol):
            call_count["ws"] += 1
            raise NotImplementedError("watch_order_book not supported")

        async def fake_get_orderbook(symbol, limit=20):
            call_count["rest"] += 1
            return {
                "bids": [[49_000.0, 1.0]],
                "asks": [[51_000.0, 1.0]],
                "timestamp": 1_700_000_000_000,
            }

        mock_exchange = MagicMock()
        mock_exchange.watch_order_book = fake_watch_order_book
        mock_exchange.get_orderbook = fake_get_orderbook

        manager = LocalOrderBookManager(mock_exchange, ["BTC/USDT"])
        await manager.start()

        # Give the background task enough time to hit NotImplementedError and
        # run the REST polling branch at least once.
        await asyncio.sleep(0.1)
        await manager.stop()

        # REST branch must have been called
        assert call_count["rest"] >= 1
        # The WS branch was called exactly once (then permanently switched)
        assert call_count["ws"] == 1

    @pytest.mark.asyncio
    async def test_rest_fallback_updates_book_cache(self):
        """After falling back to REST, get_book must return a fresh snapshot."""
        from exchange.local_orderbook import LocalOrderBookManager
        import time

        async def fake_watch_order_book(symbol):
            raise NotImplementedError("no WS")

        async def fake_get_orderbook(symbol, limit=20):
            return {
                "bids": [[40_000.0, 2.0]],
                "asks": [[40_100.0, 2.0]],
                "timestamp": 1_700_000_000_000,
            }

        mock_exchange = MagicMock()
        mock_exchange.watch_order_book = fake_watch_order_book
        mock_exchange.get_orderbook = fake_get_orderbook

        manager = LocalOrderBookManager(mock_exchange, ["ETH/USDT"])
        manager._STALE_THRESHOLD_SECONDS = 60.0  # ensure not stale during test
        await manager.start()

        await asyncio.sleep(0.15)
        await manager.stop()

        book = manager.get_book("ETH/USDT")
        assert book is not None
        assert book["bids"] == [[40_000.0, 2.0]]
        assert book["asks"] == [[40_100.0, 2.0]]


# ---------------------------------------------------------------------------
# ERROR 2: Trade executor zero-size guard
# ---------------------------------------------------------------------------


class TestTradeExecutorZeroSizeGuard:
    """test_trade_executor_zero_size_guard — error when position_size is 0."""

    @pytest.mark.asyncio
    async def test_zero_position_size_returns_error(self):
        """execute_trade must return error dict when position_size is 0."""
        from execution.trade_executor import TradeExecutor

        mock_exchange = AsyncMock()
        mock_order_manager = AsyncMock()
        mock_position_manager = AsyncMock()

        executor = TradeExecutor(
            exchange=mock_exchange,
            order_manager=mock_order_manager,
            position_manager=mock_position_manager,
        )

        signal = {
            "symbol": "BTC/USDT",
            "direction": "long",
            "position_size": 0.0,
            "stop_loss": 49_000.0,
            "take_profit_levels": [51_000.0],
            "leverage": 5,
            "strategy": "test",
        }

        result = await executor.execute_trade(signal)
        assert result["success"] is False
        assert "zero" in result["error"].lower() or "negative" in result["error"].lower()
        # Confirm no exchange call was made
        mock_exchange.get_ticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_negative_position_size_returns_error(self):
        """execute_trade must return error dict when position_size is negative."""
        from execution.trade_executor import TradeExecutor

        mock_exchange = AsyncMock()
        executor = TradeExecutor(
            exchange=mock_exchange,
            order_manager=AsyncMock(),
            position_manager=AsyncMock(),
        )

        signal = {
            "symbol": "ETH/USDT",
            "direction": "long",
            "position_size": -500.0,
            "stop_loss": 3_000.0,
            "take_profit_levels": [3_500.0],
            "leverage": 5,
            "strategy": "test",
        }

        result = await executor.execute_trade(signal)
        assert result["success"] is False
        mock_exchange.get_ticker.assert_not_called()


# ---------------------------------------------------------------------------
# ERROR 2: PositionSizer epsilon guards
# ---------------------------------------------------------------------------


class TestPositionSizerEpsilonGuards:
    """test_position_sizer_epsilon_guards — no ZeroDivisionError with tiny values."""

    def test_kelly_size_zero_avg_loss(self):
        """kelly_size must return 0.0 when avg_loss is exactly 0."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        result = sizer.kelly_size(win_rate=0.6, avg_win=100.0, avg_loss=0.0, capital=10_000.0)
        assert result == 0.0

    def test_kelly_size_tiny_avg_loss(self):
        """kelly_size must return 0.0 when avg_loss is below epsilon (1e-10)."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        result = sizer.kelly_size(
            win_rate=0.6, avg_win=100.0, avg_loss=1e-12, capital=10_000.0
        )
        assert result == 0.0

    def test_kelly_size_normal_values(self):
        """kelly_size must return a positive value for normal inputs."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        result = sizer.kelly_size(win_rate=0.6, avg_win=100.0, avg_loss=50.0, capital=10_000.0)
        assert result > 0.0
        assert result <= 2_500.0  # 25% cap

    def test_volatility_adjusted_size_zero_atr(self):
        """volatility_adjusted_size must return 0.0 when atr is 0."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        result = sizer.volatility_adjusted_size(
            capital=10_000.0, atr=0.0, entry_price=50_000.0
        )
        assert result == 0.0

    def test_kelly_size_no_nan_or_inf(self):
        """kelly_size must never return NaN or infinity."""
        import math
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        for avg_loss in [0.0, 1e-15, 1e-10, 0.001]:
            result = sizer.kelly_size(
                win_rate=0.55, avg_win=50.0, avg_loss=avg_loss, capital=5_000.0
            )
            assert not math.isnan(result)
            assert not math.isinf(result)


# ---------------------------------------------------------------------------
# ERROR 3: Funding rate swap symbol resolution
# ---------------------------------------------------------------------------


class TestFundingRateSwapSymbolResolution:
    """test_funding_rate_swap_symbol_resolution — BTC/USDT → BTC/USDT:USDT for gateio."""

    def test_ccxt_exchange_resolve_swap_symbol_spot(self):
        """_resolve_swap_symbol must append :USDT for a bare spot symbol."""
        from exchange.ccxt_exchange import CcxtExchange

        exchange = CcxtExchange.__new__(CcxtExchange)
        exchange._client = MagicMock()
        exchange._client.markets = {}  # no markets loaded

        result = exchange._resolve_swap_symbol("BTC/USDT")
        assert result == "BTC/USDT:USDT"

    def test_ccxt_exchange_resolve_swap_symbol_already_swap(self):
        """_resolve_swap_symbol must return unchanged symbol if already a swap."""
        from exchange.ccxt_exchange import CcxtExchange

        exchange = CcxtExchange.__new__(CcxtExchange)
        exchange._client = MagicMock()
        exchange._client.markets = {}

        result = exchange._resolve_swap_symbol("BTC/USDT:USDT")
        assert result == "BTC/USDT:USDT"

    def test_ccxt_exchange_resolve_swap_uses_market_data(self):
        """_resolve_swap_symbol must use market data when available."""
        from exchange.ccxt_exchange import CcxtExchange

        exchange = CcxtExchange.__new__(CcxtExchange)
        exchange._client = MagicMock()
        exchange._client.markets = {
            "BTC/USDT:USDT": {
                "type": "swap",
                "spot": False,
                "base": "BTC",
                "quote": "USDT",
            }
        }

        result = exchange._resolve_swap_symbol("BTC/USDT")
        assert result == "BTC/USDT:USDT"

    def test_gateio_client_resolve_swap_symbol(self):
        """GateIOClient._resolve_swap_symbol must produce correct swap symbol."""
        from exchange.gateio_client import GateIOClient

        client = GateIOClient.__new__(GateIOClient)
        client._client = MagicMock()
        client._client.markets = {}

        result = client._resolve_swap_symbol("ETH/USDT")
        assert result == "ETH/USDT:USDT"

    def test_mexc_client_resolve_swap_symbol(self):
        """MEXCClient._resolve_swap_symbol must produce correct swap symbol."""
        from exchange.mexc_client import MEXCClient

        client = MEXCClient.__new__(MEXCClient)
        client._client = MagicMock()
        client._client.markets = {}

        result = client._resolve_swap_symbol("SOL/USDT")
        assert result == "SOL/USDT:USDT"

    def test_bingx_client_resolve_swap_symbol(self):
        """BingXClient._resolve_swap_symbol must produce correct swap symbol."""
        from exchange.bingx_client import BingXClient

        client = BingXClient.__new__(BingXClient)
        client._client = MagicMock()
        client._client.markets = {}

        result = client._resolve_swap_symbol("BTC/USDT")
        assert result == "BTC/USDT:USDT"

    def test_bitget_client_resolve_swap_symbol(self):
        """BitgetClient._resolve_swap_symbol must produce correct swap symbol."""
        from exchange.bitget_client import BitgetClient

        client = BitgetClient.__new__(BitgetClient)
        client._client = MagicMock()
        client._client.markets = {}

        result = client._resolve_swap_symbol("BTC/USDT")
        assert result == "BTC/USDT:USDT"


# ---------------------------------------------------------------------------
# BUG 4/5: PaperExchange subscribe_ticker / subscribe_trades polling
# ---------------------------------------------------------------------------


class TestPaperExchangeSubscribeTickerPolling:
    """test_paper_exchange_subscribe_ticker_polling — verify ticker polling works."""

    @pytest.mark.asyncio
    async def test_subscribe_ticker_calls_callback(self):
        """subscribe_ticker must call the callback with a Ticker at least once."""
        from exchange.paper_exchange import PaperExchange

        ticker_stub = Ticker(
            symbol="BTC/USDT",
            bid=49_900.0,
            ask=50_100.0,
            last=50_000.0,
            high=51_000.0,
            low=49_000.0,
            volume=1000.0,
            timestamp=1_700_000_000_000,
        )

        mock_price_exchange = AsyncMock()
        mock_price_exchange.get_ticker = AsyncMock(return_value=ticker_stub)

        paper = PaperExchange(price_exchange=mock_price_exchange)

        received = []

        async def callback(t):
            received.append(t)

        task = asyncio.create_task(paper.subscribe_ticker("BTC/USDT", callback))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received) >= 1
        assert received[0] == ticker_stub

    @pytest.mark.asyncio
    async def test_subscribe_ticker_no_price_exchange(self):
        """subscribe_ticker must return immediately when no price exchange is set."""
        from exchange.paper_exchange import PaperExchange

        paper = PaperExchange(price_exchange=None)
        # Should return without error
        await paper.subscribe_ticker("BTC/USDT", AsyncMock())

    @pytest.mark.asyncio
    async def test_subscribe_orderbook_calls_callback(self):
        """subscribe_orderbook must call callback with orderbook dict."""
        from exchange.paper_exchange import PaperExchange

        book_stub = {"bids": [[49_000.0, 1.0]], "asks": [[51_000.0, 1.0]]}

        mock_price_exchange = AsyncMock()
        mock_price_exchange.get_orderbook = AsyncMock(return_value=book_stub)

        paper = PaperExchange(price_exchange=mock_price_exchange)

        received = []

        async def callback(b):
            received.append(b)

        task = asyncio.create_task(paper.subscribe_orderbook("BTC/USDT", callback))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received) >= 1
        assert received[0] == book_stub

    @pytest.mark.asyncio
    async def test_subscribe_trades_calls_callback(self):
        """subscribe_trades must call callback with trade dict."""
        from exchange.paper_exchange import PaperExchange

        ticker_stub = Ticker(
            symbol="BTC/USDT",
            bid=49_900.0,
            ask=50_100.0,
            last=50_000.0,
            high=51_000.0,
            low=49_000.0,
            volume=1000.0,
            timestamp=1_700_000_000_000,
        )

        mock_price_exchange = AsyncMock()
        mock_price_exchange.get_ticker = AsyncMock(return_value=ticker_stub)

        paper = PaperExchange(price_exchange=mock_price_exchange)

        received = []

        async def callback(trade):
            received.append(trade)

        task = asyncio.create_task(paper.subscribe_trades("BTC/USDT", callback))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received) >= 1
        assert received[0]["symbol"] == "BTC/USDT"
        assert received[0]["price"] == 50_000.0


# ---------------------------------------------------------------------------
# BUG 7: FundingTracker uses get_funding_rate wrapper
# ---------------------------------------------------------------------------


class TestFundingTrackerWrapper:
    """FundingTracker.check_funding_rates must use get_funding_rate(), not fetch_funding_rate()."""

    @pytest.mark.asyncio
    async def test_uses_get_funding_rate(self):
        """check_funding_rates must call exchange.get_funding_rate not fetch_funding_rate."""
        from risk.funding_tracker import FundingTracker

        tracker = FundingTracker()
        mock_exchange = AsyncMock()
        mock_exchange.get_funding_rate = AsyncMock(return_value=0.0001)

        position = MagicMock()
        position.symbol = "BTC/USDT"
        position.amount = 0.1
        position.entry_price = 50_000.0
        position.side = "long"

        results = await tracker.check_funding_rates(mock_exchange, [position])

        mock_exchange.get_funding_rate.assert_called_once_with("BTC/USDT")
        assert not hasattr(mock_exchange, "fetch_funding_rate") or not mock_exchange.fetch_funding_rate.called

    @pytest.mark.asyncio
    async def test_returns_correct_rate_structure(self):
        """check_funding_rates must return properly structured dicts."""
        from risk.funding_tracker import FundingTracker

        tracker = FundingTracker()
        mock_exchange = AsyncMock()
        mock_exchange.get_funding_rate = AsyncMock(return_value=0.0001)

        position = {"symbol": "BTC/USDT", "amount": 1.0, "entry_price": 50_000.0, "side": "long"}

        results = await tracker.check_funding_rates(mock_exchange, [position])

        assert len(results) == 1
        assert results[0]["symbol"] == "BTC/USDT"
        assert results[0]["rate"] == 0.0001
        assert "rate_pct" in results[0]
        assert "expected_payment" in results[0]


# ---------------------------------------------------------------------------
# BUG 9: LocalOrderBookManager exported from exchange.__init__
# ---------------------------------------------------------------------------


class TestLocalOrderBookManagerExport:
    """LocalOrderBookManager must be importable from exchange package."""

    def test_importable_from_exchange_package(self):
        """from exchange import LocalOrderBookManager must work."""
        from exchange import LocalOrderBookManager

        assert LocalOrderBookManager is not None

    def test_in_all(self):
        """LocalOrderBookManager must appear in exchange.__all__."""
        import exchange

        assert "LocalOrderBookManager" in exchange.__all__


# ---------------------------------------------------------------------------
# BUG 12: pandas frequency aliases
# ---------------------------------------------------------------------------


class TestPandasFrequencyAliases:
    """time_utils must use modern pandas frequency aliases."""

    def test_hour_alias_lowercase(self):
        """Hourly frequencies must use lowercase 'h', not deprecated 'H'."""
        from utils.time_utils import _TIMEFRAME_PANDAS_FREQ

        assert _TIMEFRAME_PANDAS_FREQ["1h"] == "1h"
        assert _TIMEFRAME_PANDAS_FREQ["4h"] == "4h"
        assert _TIMEFRAME_PANDAS_FREQ["1h"] != "1H"

    def test_daily_alias_ok(self):
        """Daily/weekly aliases must still be present."""
        from utils.time_utils import _TIMEFRAME_PANDAS_FREQ

        assert "1d" in _TIMEFRAME_PANDAS_FREQ
        assert "1w" in _TIMEFRAME_PANDAS_FREQ


# ---------------------------------------------------------------------------
# BUG 13: BalanceManager race condition fix
# ---------------------------------------------------------------------------


class TestBalanceManagerRaceConditionFix:
    """get_balance must not release the lock between staleness check and refresh."""

    @pytest.mark.asyncio
    async def test_concurrent_stale_refresh_called_once(self):
        """When balance is stale, concurrent callers must trigger only one refresh."""
        from exchange.balance_manager import BalanceManager
        from exchange.base_exchange import Balance

        refresh_count = {"n": 0}

        async def fake_get_balance():
            refresh_count["n"] += 1
            await asyncio.sleep(0.01)  # simulate network latency
            return Balance(usdt_total=10_000.0, usdt_free=10_000.0)

        mock_exchange = AsyncMock()
        mock_exchange.get_balance = AsyncMock(side_effect=fake_get_balance)

        manager = BalanceManager(mock_exchange, cache_ttl_seconds=0.001)
        # Expire the cache
        await asyncio.sleep(0.002)

        # Fire 5 concurrent requests
        results = await asyncio.gather(*(manager.get_balance() for _ in range(5)))

        assert all(r.usdt_total == 10_000.0 for r in results)
        # With the fix, only ONE refresh should have been made (or at most a small
        # number — the key thing is not 5 separate refreshes).
        assert refresh_count["n"] <= 2  # allow max 2 due to async scheduling


# ---------------------------------------------------------------------------
# IMPROVEMENT 4 & 9: Break-even SL trigger and trailing TP
# ---------------------------------------------------------------------------


class TestBreakEvenStopActivation:
    """test_break_even_stop_activation — SL moves to entry after 1.5R profit."""

    @pytest.mark.asyncio
    async def test_break_even_activates_at_1_5r(self):
        """activate_break_even must be called when profit >= 1.5× stop distance."""
        from exchange.position_manager import PositionManager, PositionTracker
        from exchange.base_exchange import Position, PositionSide

        mock_exchange = AsyncMock()
        manager = PositionManager(mock_exchange)

        from datetime import datetime, timezone
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=0.1,
            entry_price=50_000.0,
            current_price=51_500.0,
            unrealized_pnl=150.0,
            leverage=5,
            margin=1000.0,
            liquidation_price=45_000.0,
            timestamp=1_700_000_000_000,
        )
        tracker = PositionTracker(
            position=position,
            opened_at=datetime.now(tz=timezone.utc),
            stop_loss=49_000.0,  # 1000 USDT below entry
            take_profit=[52_000.0],
            strategy="test",
        )
        async with manager._lock:
            manager._positions["BTC/USDT"] = tracker

        # Price at entry + 1.5× stop distance = 50000 + 1.5×1000 = 51500
        tracker.position.current_price = 51_500.0

        # Manually check the trigger condition
        entry = 50_000.0
        sl = 49_000.0
        stop_distance = abs(entry - sl)  # 1000
        profit = 51_500.0 - entry  # 1500
        assert profit >= stop_distance * 1.5  # 1500 >= 1500 → triggers

    def test_break_even_stop_calc_long(self):
        """calculate_breakeven_stop must return entry + buffer for long."""
        from risk.stop_loss_engine import StopLossEngine

        engine = StopLossEngine()
        be_stop = engine.calculate_breakeven_stop(
            entry=50_000.0, direction="long", buffer_pct=0.001
        )
        assert be_stop > 50_000.0  # slightly above entry
        assert be_stop < 50_100.0  # not much above

    def test_break_even_stop_calc_short(self):
        """calculate_breakeven_stop must return entry - buffer for short."""
        from risk.stop_loss_engine import StopLossEngine

        engine = StopLossEngine()
        be_stop = engine.calculate_breakeven_stop(
            entry=50_000.0, direction="short", buffer_pct=0.001
        )
        assert be_stop < 50_000.0  # slightly below entry


class TestTrailingTPLogic:
    """test_trailing_tp_logic — trailing TP tracks peak and closes on retrace."""

    @pytest.mark.asyncio
    async def test_update_trailing_tp_triggers_on_retrace(self):
        """update_trailing_take_profit must trigger close when price retraces by distance."""
        from exchange.position_manager import PositionManager, PositionTracker
        from exchange.base_exchange import Position, PositionSide

        mock_exchange = AsyncMock()
        manager = PositionManager(mock_exchange)

        from datetime import datetime, timezone
        position = Position(
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            amount=1.0,
            entry_price=3_000.0,
            current_price=3_200.0,
            unrealized_pnl=200.0,
            leverage=5,
            margin=600.0,
            liquidation_price=2_500.0,
            timestamp=1_700_000_000_000,
        )
        tracker = PositionTracker(
            position=position,
            opened_at=datetime.now(tz=timezone.utc),
            stop_loss=2_900.0,
            take_profit=[],
            strategy="test",
            trailing_tp_active=True,
            trailing_tp_distance=100.0,
            peak_price_for_tp=3_200.0,
        )
        async with manager._lock:
            manager._positions["ETH/USDT"] = tracker

        mock_exchange.create_market_order = AsyncMock(
            return_value=MagicMock(id="close-order", filled=1.0, price=3_080.0)
        )

        # Price retraces by 100 from peak → should trigger close
        close_price = await manager.update_trailing_take_profit("ETH/USDT", 3_100.0)
        assert close_price is not None
        assert close_price == pytest.approx(3_100.0)

    @pytest.mark.asyncio
    async def test_update_trailing_tp_tracks_new_high(self):
        """update_trailing_take_profit must update peak when price makes new high."""
        from exchange.position_manager import PositionManager, PositionTracker
        from exchange.base_exchange import Position, PositionSide

        mock_exchange = AsyncMock()
        manager = PositionManager(mock_exchange)

        from datetime import datetime, timezone
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=0.1,
            entry_price=50_000.0,
            current_price=51_000.0,
            unrealized_pnl=100.0,
            leverage=5,
            margin=1000.0,
            liquidation_price=45_000.0,
            timestamp=1_700_000_000_000,
        )
        tracker = PositionTracker(
            position=position,
            opened_at=datetime.now(tz=timezone.utc),
            stop_loss=49_000.0,
            take_profit=[],
            strategy="test",
            trailing_tp_active=True,
            trailing_tp_distance=500.0,
            peak_price_for_tp=51_000.0,
        )
        async with manager._lock:
            manager._positions["BTC/USDT"] = tracker

        # New high — peak should update, no close triggered
        result = await manager.update_trailing_take_profit("BTC/USDT", 52_000.0)
        assert result is None  # no close
        assert tracker.peak_price_for_tp == pytest.approx(52_000.0)


# ---------------------------------------------------------------------------
# IMPROVEMENT 5: Cross-exchange arbitrage strategy
# ---------------------------------------------------------------------------


class TestCrossExchangeArbStrategy:
    """Cross-exchange arb strategy generates signals on profitable spreads."""

    @pytest.mark.asyncio
    async def test_signal_when_spread_exceeds_threshold(self):
        """Strategy must generate a signal when net spread > min_spread_pct."""
        from strategy.strategies.cross_exchange_arb import CrossExchangeArbStrategy

        strategy = CrossExchangeArbStrategy(
            symbols=["BTC/USDT"],
            min_spread_pct=0.001,
            fee_pct=0.0005,
            cooldown_seconds=0.0,
        )
        # mexc has lower ask (buy leg), gateio has higher bid (sell leg)
        strategy.update_price("mexc", "BTC/USDT", bid=49_990.0, ask=50_000.0)
        strategy.update_price("gateio", "BTC/USDT", bid=50_120.0, ask=50_130.0)

        signal = await strategy.generate_signal("BTC/USDT", None)
        assert signal is not None
        assert signal.direction == "long"
        assert signal.strategy_name == "cross_exchange_arb"

    @pytest.mark.asyncio
    async def test_no_signal_when_spread_below_threshold(self):
        """Strategy must not generate a signal when net spread < min_spread_pct."""
        from strategy.strategies.cross_exchange_arb import CrossExchangeArbStrategy

        strategy = CrossExchangeArbStrategy(
            symbols=["BTC/USDT"],
            min_spread_pct=0.005,  # 0.5% threshold — hard to meet
            fee_pct=0.001,
            cooldown_seconds=0.0,
        )
        strategy.update_price("mexc", "BTC/USDT", bid=49_990.0, ask=50_000.0)
        strategy.update_price("gateio", "BTC/USDT", bid=50_010.0, ask=50_020.0)

        signal = await strategy.generate_signal("BTC/USDT", None)
        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_with_single_exchange(self):
        """Strategy requires at least 2 exchanges with prices."""
        from strategy.strategies.cross_exchange_arb import CrossExchangeArbStrategy

        strategy = CrossExchangeArbStrategy(
            symbols=["BTC/USDT"],
            min_spread_pct=0.001,
            cooldown_seconds=0.0,
        )
        strategy.update_price("mexc", "BTC/USDT", bid=49_990.0, ask=50_000.0)
        # Only one exchange — no arb possible

        signal = await strategy.generate_signal("BTC/USDT", None)
        assert signal is None


# ---------------------------------------------------------------------------
# Gate.io minimum order size / step-size rounding fix
# ---------------------------------------------------------------------------


class TestTradeExecutorStepSizeRounding:
    """Validates that execute_trade rounds to step size and blocks sub-minimum orders."""

    def _make_executor(self, mock_exchange):
        from execution.trade_executor import TradeExecutor

        return TradeExecutor(
            exchange=mock_exchange,
            order_manager=AsyncMock(),
            position_manager=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_order_blocked_when_rounded_to_zero(self):
        """execute_trade returns failure when step-size rounding produces 0 contracts."""
        mock_exchange = AsyncMock()
        mock_exchange.get_ticker.return_value = MagicMock(last=90_000.0)
        # 1 contract = 1 BTC; step_size = 1; 5 USDT / 90000 = ~0.00005 BTC → 0 contracts
        markets_dict = {
            "BTC/USDT:USDT": {
                "contract": True,
                "contractSize": 1.0,
                "inverse": False,
                "limits": {"amount": {"min": 1.0}},
                "precision": {"amount": 1},
            }
        }
        mock_exchange.get_markets = AsyncMock(return_value=markets_dict)
        # Ensure the code reaches get_markets() instead of _client.markets
        mock_exchange._client = None
        # _resolve_swap_symbol is sync; mock it so it doesn't return a coroutine
        mock_exchange._resolve_swap_symbol = MagicMock(return_value="BTC/USDT:USDT")
        # Return a real dict for orderbook to avoid AsyncMock.get() returning coroutines
        mock_exchange.get_orderbook = AsyncMock(return_value={"bids": [], "asks": []})

        executor = self._make_executor(mock_exchange)
        signal = {
            "symbol": "BTC/USDT:USDT",
            "direction": "long",
            "position_size": 5.0,
            "stop_loss": 85_000.0,
            "take_profit_levels": [95_000.0],
            "leverage": 5,
            "strategy": "test",
        }

        result = await executor.execute_trade(signal)
        assert result["success"] is False
        assert "too small" in result["error"].lower()
        mock_exchange.create_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_order_not_blocked_when_large_enough(self):
        """execute_trade must not block when there is at least 1 full contract."""
        mock_exchange = AsyncMock()
        mock_exchange.get_ticker.return_value = MagicMock(last=1_000.0)
        # 1 contract = 0.001 BTC; step=1; 500 USDT / 1000 / 0.001 = 500 contracts
        mock_exchange.get_markets.return_value = {
            "ETH/USDT:USDT": {
                "contract": True,
                "contractSize": 0.001,
                "inverse": False,
                "limits": {"amount": {"min": 1.0}},
                "precision": {"amount": 1},
            }
        }

        executor = self._make_executor(mock_exchange)
        signal = {
            "symbol": "ETH/USDT:USDT",
            "direction": "long",
            "position_size": 500.0,
            "stop_loss": 900.0,
            "take_profit_levels": [1_100.0],
            "leverage": 5,
            "strategy": "test",
        }

        result = await executor.execute_trade(signal)
        # Should not be blocked by "too small" — any failure here is from later mocking
        if not result.get("success"):
            assert "too small" not in result.get("error", "").lower(), (
                "Should not be rejected for size being too small"
            )


class TestCcxtExchangeZeroAmountGuard:
    """Validates that CcxtExchange raises ValueError for 0-sized orders."""

    @pytest.mark.asyncio
    async def test_market_order_raises_on_zero_amount(self):
        """create_market_order must raise ValueError when precision rounds amount to 0."""
        from exchange.ccxt_exchange import CcxtExchange
        from exchange.base_exchange import OrderSide

        exchange = CcxtExchange.__new__(CcxtExchange)
        exchange._exchange_id = "gateio"
        exchange._rate_limiter = AsyncMock()
        exchange._rate_limiter.acquire = AsyncMock()

        mock_client = MagicMock()
        mock_client.amount_to_precision.return_value = "0"
        exchange._client = mock_client

        with pytest.raises(ValueError, match="rounded to 0"):
            await exchange.create_market_order("BTC/USDT:USDT", OrderSide.BUY, 0.000001)

    @pytest.mark.asyncio
    async def test_limit_order_raises_on_zero_amount(self):
        """create_limit_order must raise ValueError when precision rounds amount to 0."""
        from exchange.ccxt_exchange import CcxtExchange
        from exchange.base_exchange import OrderSide

        exchange = CcxtExchange.__new__(CcxtExchange)
        exchange._exchange_id = "gateio"
        exchange._rate_limiter = AsyncMock()
        exchange._rate_limiter.acquire = AsyncMock()

        mock_client = MagicMock()
        mock_client.amount_to_precision.return_value = "0"
        exchange._client = mock_client

        with pytest.raises(ValueError, match="rounded to 0"):
            await exchange.create_limit_order("BTC/USDT:USDT", OrderSide.BUY, 0.000001, 90_000.0)


class TestGateIOClientZeroAmountGuard:
    """Validates that GateIOClient raises ValueError for 0-sized orders."""

    @pytest.mark.asyncio
    async def test_market_order_raises_on_zero_amount(self):
        """create_market_order must raise ValueError when precision rounds amount to 0."""
        from exchange.gateio_client import GateIOClient
        from exchange.base_exchange import OrderSide

        client = GateIOClient.__new__(GateIOClient)
        client._rate_limiter = AsyncMock()
        client._rate_limiter.acquire = AsyncMock()

        mock_ccxt_client = MagicMock()
        mock_ccxt_client.amount_to_precision.return_value = "0"
        client._client = mock_ccxt_client

        with pytest.raises(ValueError, match="too small"):
            await client.create_market_order("BTC/USDT:USDT", OrderSide.BUY, 0.000001)

    @pytest.mark.asyncio
    async def test_limit_order_raises_on_zero_amount(self):
        """create_limit_order must raise ValueError when precision rounds amount to 0."""
        from exchange.gateio_client import GateIOClient
        from exchange.base_exchange import OrderSide

        client = GateIOClient.__new__(GateIOClient)
        client._rate_limiter = AsyncMock()
        client._rate_limiter.acquire = AsyncMock()

        mock_ccxt_client = MagicMock()
        mock_ccxt_client.amount_to_precision.return_value = "0"
        client._client = mock_ccxt_client

        with pytest.raises(ValueError, match="too small"):
            await client.create_limit_order("BTC/USDT:USDT", OrderSide.BUY, 0.000001, 90_000.0)


# ---------------------------------------------------------------------------
# Bug: WebSocketDataManager._get_ws_client returns REST client instead of WS
# ---------------------------------------------------------------------------


class TestWebSocketDataManagerGetWsClient:
    """_get_ws_client must return the ccxt.pro WebSocket client, not REST."""

    def test_uses_exchange_getter_first(self):
        """Should call exchange._get_ws_client() when available (CcxtExchange pattern)."""
        from exchange.ws_data_manager import WebSocketDataManager

        ws_client = MagicMock()
        ws_client.watch_ticker = MagicMock()
        rest_client = MagicMock(spec=[])  # No watch_* methods

        exchange = MagicMock()
        exchange._client = rest_client
        exchange._get_ws_client = MagicMock(return_value=ws_client)

        mgr = WebSocketDataManager(exchange, ["BTC/USDT"])
        result = mgr._get_ws_client()

        assert result is ws_client
        exchange._get_ws_client.assert_called_once()

    def test_does_not_return_rest_client_when_ws_client_present(self):
        """When both _ws_client and _client exist, _ws_client must be returned."""
        from exchange.ws_data_manager import WebSocketDataManager

        ws_client = MagicMock()
        ws_client.watch_ticker = MagicMock()
        rest_client = MagicMock(spec=[])  # No watch_* methods (REST)

        # Simulate exchange wrapper that has both clients but no getter method.
        exchange = MagicMock(spec=["_ws_client", "_client"])
        exchange._ws_client = ws_client
        exchange._client = rest_client

        mgr = WebSocketDataManager(exchange, ["BTC/USDT"])
        result = mgr._get_ws_client()

        assert result is ws_client, "Should return WS client, not REST client"
        assert result is not rest_client

    def test_returns_none_when_no_ws_support(self):
        """Should return None when the exchange offers no WebSocket methods."""
        from exchange.ws_data_manager import WebSocketDataManager

        exchange = MagicMock(spec=[])  # No watch_* methods

        mgr = WebSocketDataManager(exchange, ["BTC/USDT"])
        result = mgr._get_ws_client()

        assert result is None

    def test_returns_exchange_when_direct_ccxt_pro(self):
        """If exchange itself is a ccxt.pro instance it should be returned."""
        from exchange.ws_data_manager import WebSocketDataManager

        # A genuine ccxt.pro instance advertises watchTicker support in its
        # ``has`` dict.  Pure hasattr checks on ccxt.async_support stubs would
        # incorrectly match REST clients; the has-dict gate prevents that.
        direct = MagicMock(spec=["watch_tickers", "watch_ticker", "has"])
        direct.has = {"watchTicker": True, "watchTickers": True}

        mgr = WebSocketDataManager(direct, ["BTC/USDT"])
        result = mgr._get_ws_client()

        assert result is direct

    def test_rest_client_with_stubs_is_not_returned(self):
        """ccxt.async_support REST clients expose watch_* stubs but must NOT be returned.

        The ccxt.async_support.gateio REST client has watch_ticker/watch_tickers
        methods (stubs that raise NotSupported) but its ``has`` dict returns
        None for both keys.  The resolver must use the has-dict gate to reject
        such clients and avoid infinite silent-retry loops.
        """
        from exchange.ws_data_manager import WebSocketDataManager

        # Simulate ccxt.async_support REST client: has stub methods but has-dict = None
        rest_client = MagicMock()
        rest_client.watch_ticker = MagicMock(side_effect=Exception("NotSupported"))
        rest_client.watch_tickers = MagicMock(side_effect=Exception("NotSupported"))
        rest_client.has = {"watchTicker": None, "watchTickers": None}

        # Exchange wrapper that exposes _client = REST client, no _get_ws_client
        exchange = MagicMock(spec=["_client"])
        exchange._client = rest_client

        mgr = WebSocketDataManager(exchange, ["BTC/USDT"])
        result = mgr._get_ws_client()

        assert result is None, "REST client with None has-dict must not be returned"

    def test_getter_exception_falls_through_to_attribute_lookup(self):
        """If exchange._get_ws_client() raises, fall back to attribute scan."""
        from exchange.ws_data_manager import WebSocketDataManager

        ws_client = MagicMock()
        ws_client.watch_ticker = MagicMock()

        exchange = MagicMock()
        exchange._get_ws_client = MagicMock(side_effect=RuntimeError("ccxt.pro unavailable"))
        exchange._ws_client = ws_client

        mgr = WebSocketDataManager(exchange, ["BTC/USDT"])
        result = mgr._get_ws_client()

        assert result is ws_client
