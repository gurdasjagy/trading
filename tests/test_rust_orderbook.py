"""Tests for the Rust order book engine (Phase 2).

Exercises ``RustOrderBook``, ``RustBookAnalyzer``, and the Python-side
``LocalOrderBookManager._update_rust_book`` / ``get_rust_book`` integration.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("rust_trading_engine")

from rust_trading_engine.orderbook import RustBookAnalyzer, RustOrderBook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_book(
    bids: list | None = None,
    asks: list | None = None,
    symbol: str = "BTC/USDT",
) -> RustOrderBook:
    book = RustOrderBook(symbol)
    bids = bids or [(41999.0, 1.5), (41998.0, 2.0), (41997.0, 3.0)]
    asks = asks or [(42001.0, 1.0), (42002.0, 3.0), (42003.0, 2.0)]
    book.update_snapshot(bids, asks)
    return book


# ---------------------------------------------------------------------------
# RustOrderBook
# ---------------------------------------------------------------------------


class TestRustOrderBook:
    def test_best_bid_ask(self):
        book = _make_book()
        bid = book.get_best_bid()
        ask = book.get_best_ask()
        assert bid is not None and ask is not None
        price_b, size_b = bid
        price_a, size_a = ask
        assert price_b == pytest.approx(41999.0)
        assert size_b == pytest.approx(1.5)
        assert price_a == pytest.approx(42001.0)
        assert size_a == pytest.approx(1.0)

    def test_mid_price(self):
        book = _make_book()
        assert book.get_mid_price() == pytest.approx((41999.0 + 42001.0) / 2)

    def test_mid_price_empty(self):
        book = RustOrderBook("EMPTY/USDT")
        assert book.get_mid_price() == pytest.approx(0.0)

    def test_spread_bps(self):
        book = _make_book()
        spread = book.get_spread_bps()
        expected = (42001.0 - 41999.0) / 42000.0 * 10000.0
        assert spread == pytest.approx(expected, rel=1e-4)

    def test_spread_bps_empty(self):
        book = RustOrderBook("EMPTY/USDT")
        assert book.get_spread_bps() == pytest.approx(0.0)

    def test_symbol(self):
        book = RustOrderBook("ETH/USDT")
        assert book.get_symbol() == "ETH/USDT"

    def test_snapshot_bids_sorted_descending(self):
        """Snapshot bids must come out highest-price-first."""
        book = _make_book()
        snap = book.get_snapshot()
        prices = [lvl[0] for lvl in snap["bids"]]
        assert prices == sorted(prices, reverse=True)

    def test_snapshot_asks_sorted_ascending(self):
        """Snapshot asks must come out lowest-price-first."""
        book = _make_book()
        snap = book.get_snapshot()
        prices = [lvl[0] for lvl in snap["asks"]]
        assert prices == sorted(prices)

    def test_snapshot_has_timestamp(self):
        book = _make_book()
        snap = book.get_snapshot()
        assert "timestamp" in snap
        assert snap["timestamp"] >= 0

    def test_update_snapshot_replaces_levels(self):
        book = _make_book()
        # Replace with entirely new levels
        book.update_snapshot([(50000.0, 1.0)], [(50001.0, 1.0)])
        bid = book.get_best_bid()
        ask = book.get_best_ask()
        assert bid[0] == pytest.approx(50000.0)
        assert ask[0] == pytest.approx(50001.0)
        # Old levels should be gone
        snap = book.get_snapshot()
        assert len(snap["bids"]) == 1
        assert len(snap["asks"]) == 1

    def test_apply_delta_update(self):
        """Delta updates should add/remove/update levels."""
        book = _make_book()
        # Add a new bid level and remove an existing one (size=0 means remove)
        book.apply_delta(
            bids=[(42000.0, 2.5), (41999.0, 0.0)],  # 42000 new, 41999 removed
            asks=[],
        )
        snap = book.get_snapshot()
        bid_prices = {lvl[0] for lvl in snap["bids"]}
        assert 42000.0 in bid_prices
        assert 41999.0 not in bid_prices

    def test_get_bids_depth(self):
        book = _make_book()
        bids = book.get_bids(2)
        assert len(bids) == 2
        assert bids[0][0] == pytest.approx(41999.0)

    def test_get_asks_depth(self):
        book = _make_book()
        asks = book.get_asks(2)
        assert len(asks) == 2
        assert asks[0][0] == pytest.approx(42001.0)

    def test_get_bids_depth_larger_than_book(self):
        book = _make_book()
        bids = book.get_bids(100)
        assert len(bids) == 3

    def test_is_stale_fresh_book(self):
        book = _make_book()
        # A freshly updated book should not be stale with a generous threshold
        assert book.is_stale(60_000.0) is False

    def test_is_stale_without_update(self):
        book = RustOrderBook("X/USDT")
        # Book never updated — should be considered stale
        assert book.is_stale(0.0) is True

    def test_get_age_ms(self):
        book = _make_book()
        age = book.get_age_ms()
        assert age >= 0.0
        assert age < 1_000.0  # Should be less than 1 second for a freshly updated book


# ---------------------------------------------------------------------------
# RustBookAnalyzer
# ---------------------------------------------------------------------------


class TestRustBookAnalyzer:
    def test_analyze_returns_required_keys(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        required = {
            "imbalance",
            "spread_bps",
            "bid_depth_usdt",
            "ask_depth_usdt",
            "large_bid_levels",
            "large_ask_levels",
            "optimal_buy_price",
            "optimal_sell_price",
            "best_bid",
            "best_ask",
            "mid_price",
        }
        assert required.issubset(set(result.keys()))

    def test_analyze_best_prices(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        assert result["best_bid"] == pytest.approx(41999.0)
        assert result["best_ask"] == pytest.approx(42001.0)
        assert result["mid_price"] == pytest.approx(42000.0)

    def test_analyze_spread_bps(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        expected_bps = (42001.0 - 41999.0) / 42000.0 * 10_000.0
        assert result["spread_bps"] == pytest.approx(expected_bps, rel=1e-3)

    def test_analyze_imbalance_range(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        assert -1.0 <= result["imbalance"] <= 1.0

    def test_analyze_optimal_prices_inside_spread(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        assert result["optimal_buy_price"] > result["best_bid"]
        assert result["optimal_buy_price"] < result["best_ask"]
        assert result["optimal_sell_price"] < result["best_ask"]
        assert result["optimal_sell_price"] > result["best_bid"]

    def test_analyze_depth_usdt_positive(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        assert result["bid_depth_usdt"] > 0.0
        assert result["ask_depth_usdt"] > 0.0

    def test_analyze_large_levels_list_type(self):
        book = _make_book()
        result = RustBookAnalyzer.analyze(book, 20)
        assert isinstance(result["large_bid_levels"], list)
        assert isinstance(result["large_ask_levels"], list)

    def test_analyze_large_levels_schema(self):
        """Each large level dict should have price/size/notional_usdt."""
        bids = [(42000.0, 100.0), (41999.0, 1.0), (41998.0, 1.0)]
        asks = [(42001.0, 1.0), (42002.0, 1.0), (42003.0, 1.0)]
        book = _make_book(bids=bids, asks=asks)
        result = RustBookAnalyzer.analyze(book, 20)
        for lvl in result["large_bid_levels"]:
            assert "price" in lvl
            assert "size" in lvl
            assert "notional_usdt" in lvl

    def test_empty_result_static(self):
        result = RustBookAnalyzer.empty_result()
        assert result["imbalance"] == 0.0
        assert result["spread_bps"] == 0.0
        assert isinstance(result["large_bid_levels"], list)

    def test_calculate_market_impact_buy(self):
        book = _make_book()
        vwap = RustBookAnalyzer.calculate_market_impact(book, "buy", 1.5)
        # Should be >= best_ask (walking up the book)
        assert vwap >= 42001.0

    def test_calculate_market_impact_sell(self):
        book = _make_book()
        vwap = RustBookAnalyzer.calculate_market_impact(book, "sell", 1.5)
        # Should be <= best_bid (walking down the book)
        assert vwap <= 41999.0

    def test_calculate_market_impact_zero_size(self):
        book = _make_book()
        vwap = RustBookAnalyzer.calculate_market_impact(book, "buy", 0.0)
        assert vwap == pytest.approx(0.0)

    def test_calculate_market_impact_large_order(self):
        """Large order walking entire book should return a finite price."""
        book = _make_book()
        vwap = RustBookAnalyzer.calculate_market_impact(book, "buy", 1_000_000.0)
        assert vwap > 0.0


# ---------------------------------------------------------------------------
# LocalOrderBookManager integration
# ---------------------------------------------------------------------------


class TestLocalOrderBookManagerRustIntegration:
    """Verify that LocalOrderBookManager correctly maintains RustOrderBook."""

    def _make_manager(self, symbols=None):
        from crypto_trading_bot.exchange.local_orderbook import LocalOrderBookManager, _USE_RUST_BOOK

        mock_exchange = MagicMock()
        symbols = symbols or ["BTC/USDT"]
        manager = LocalOrderBookManager(mock_exchange, symbols)
        return manager, _USE_RUST_BOOK

    def test_update_rust_book_creates_entry(self):
        manager, use_rust = self._make_manager()
        if not use_rust:
            pytest.skip("Rust book not available")

        book_data = {
            "bids": [[41999.0, 1.5], [41998.0, 2.0]],
            "asks": [[42001.0, 1.0], [42002.0, 3.0]],
        }
        manager._update_rust_book("BTC/USDT", book_data)
        rust_book = manager.get_rust_book("BTC/USDT")
        assert rust_book is not None
        bid = rust_book.get_best_bid()
        assert bid is not None
        assert bid[0] == pytest.approx(41999.0)

    def test_get_book_uses_rust_path(self):
        manager, use_rust = self._make_manager()
        if not use_rust:
            pytest.skip("Rust book not available")

        book_data = {
            "bids": [[41999.0, 1.5]],
            "asks": [[42001.0, 1.0]],
        }
        manager._update_rust_book("BTC/USDT", book_data)
        result = manager.get_book("BTC/USDT")
        assert result is not None
        assert "bids" in result
        assert "asks" in result
        assert "timestamp" in result

    def test_get_rust_book_returns_none_when_not_available(self):
        from crypto_trading_bot.exchange.local_orderbook import LocalOrderBookManager
        manager, use_rust = self._make_manager()
        if not use_rust:
            # When Rust is not available, get_rust_book always returns None
            assert manager.get_rust_book("BTC/USDT") is None
        else:
            # When Rust is available but book was never populated
            assert manager.get_rust_book("NEVER/SET") is None

    def test_update_rust_book_handles_bad_data_gracefully(self):
        manager, use_rust = self._make_manager()
        if not use_rust:
            pytest.skip("Rust book not available")

        # Bad data should not raise; should log a debug message
        manager._update_rust_book("BTC/USDT", {"bids": "INVALID", "asks": None})
        # No exception means pass

    def test_update_rust_book_filters_short_pairs(self):
        """Levels with fewer than 2 elements should be silently skipped."""
        manager, use_rust = self._make_manager()
        if not use_rust:
            pytest.skip("Rust book not available")

        book_data = {
            "bids": [[41999.0, 1.5], [41998.0]],  # second entry is malformed
            "asks": [[42001.0, 1.0]],
        }
        manager._update_rust_book("BTC/USDT", book_data)
        rust_book = manager.get_rust_book("BTC/USDT")
        assert rust_book is not None
        snap = rust_book.get_snapshot()
        # Only the valid bid level should be present
        assert len(snap["bids"]) == 1


# ---------------------------------------------------------------------------
# GateioBookAnalyzer integration
# ---------------------------------------------------------------------------


class TestGateioBookAnalyzerRustIntegration:
    """Verify that GateioBookAnalyzer correctly uses the Rust book when
    a RustOrderBook is passed as the rust_book parameter."""

    def test_analyze_with_rust_book_skips_rest(self):
        from crypto_trading_bot.exchange.gateio_book_analyzer import (
            GateioBookAnalyzer,
            _USE_RUST_ANALYZER,
        )
        if not _USE_RUST_ANALYZER:
            pytest.skip("Rust analyzer not available")

        mock_exchange = MagicMock()
        # Exchange REST should NOT be called when rust_book is passed
        mock_exchange.get_orderbook = AsyncMock(return_value={"bids": [], "asks": []})

        analyzer = GateioBookAnalyzer(mock_exchange)
        rust_book = _make_book()

        result = asyncio.get_event_loop().run_until_complete(
            analyzer.analyze_book("BTC/USDT", depth=20, rust_book=rust_book)
        )

        assert result["best_bid"] == pytest.approx(41999.0)
        assert result["best_ask"] == pytest.approx(42001.0)
        # REST should NOT have been called
        mock_exchange.get_orderbook.assert_not_called()

    def test_analyze_without_rust_book_uses_rest(self):
        from crypto_trading_bot.exchange.gateio_book_analyzer import GateioBookAnalyzer

        mock_exchange = MagicMock()
        mock_exchange.get_orderbook = AsyncMock(
            return_value={
                "bids": [[41999.0, 1.5], [41998.0, 2.0]],
                "asks": [[42001.0, 1.0], [42002.0, 3.0]],
            }
        )

        analyzer = GateioBookAnalyzer(mock_exchange)
        result = asyncio.get_event_loop().run_until_complete(
            analyzer.analyze_book("BTC/USDT")
        )

        mock_exchange.get_orderbook.assert_called_once()
        assert result["best_bid"] == pytest.approx(41999.0)

    def test_calculate_market_impact_with_rust_book(self):
        from crypto_trading_bot.exchange.gateio_book_analyzer import (
            GateioBookAnalyzer,
            _USE_RUST_ANALYZER,
        )
        if not _USE_RUST_ANALYZER:
            pytest.skip("Rust analyzer not available")

        mock_exchange = MagicMock()
        analyzer = GateioBookAnalyzer(mock_exchange)
        rust_book = _make_book()

        vwap = analyzer.calculate_market_impact(
            "BTC/USDT", "buy", 1.0, rust_book=rust_book
        )
        assert vwap >= 42001.0

    def test_calculate_market_impact_python_fallback(self):
        from crypto_trading_bot.exchange.gateio_book_analyzer import GateioBookAnalyzer

        mock_exchange = MagicMock()
        analyzer = GateioBookAnalyzer(mock_exchange)
        book = {
            "bids": [[41999.0, 5.0]],
            "asks": [[42001.0, 5.0]],
        }
        vwap = analyzer.calculate_market_impact("BTC/USDT", "buy", 1.0, book=book)
        assert vwap == pytest.approx(42001.0)
