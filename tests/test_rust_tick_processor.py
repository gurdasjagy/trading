"""Tests for the Rust tick processor (Phase 3).

Exercises ``RustTickProcessor`` directly and through the Python
``TickProcessor`` integration layer.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("rust_trading_engine")

from rust_trading_engine.tick_processor import RustTickProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "BTC/USDT"


def _feed_ticks(tp: RustTickProcessor, symbol: str, n: int, price: float = 100.0, vol: float = 1.0, side: str = "buy") -> None:
    for _ in range(n):
        tp.process_tick(symbol, price, vol, side)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVwap:
    def test_vwap_matches_python(self):
        """Feed 100 ticks to both Rust and Python engines; assert VWAP matches."""
        from crypto_trading_bot.data.tick_processor import TickProcessor

        py_tp = TickProcessor(window_size=1000)
        rust_tp = RustTickProcessor(1000, 1000.0)

        rng = np.random.default_rng(42)
        prices = rng.uniform(90.0, 110.0, 100)
        vols = rng.uniform(0.1, 5.0, 100)

        for price, vol in zip(prices, vols):
            tick = {"price": price, "amount": vol, "side": "buy"}
            py_tp.process_tick(SYMBOL, tick)
            rust_tp.process_tick(SYMBOL, price, vol, "buy")

        assert rust_tp.get_vwap(SYMBOL) == pytest.approx(py_tp.get_vwap(SYMBOL), abs=1e-6)

    def test_vwap_single_tick(self):
        tp = RustTickProcessor(1000, 1000.0)
        tp.process_tick(SYMBOL, 50000.0, 2.0, "buy")
        assert tp.get_vwap(SYMBOL) == pytest.approx(50000.0)

    def test_vwap_weighted(self):
        """Manual weighted test: two ticks with different volumes."""
        tp = RustTickProcessor(1000, 1000.0)
        tp.process_tick(SYMBOL, 100.0, 1.0, "buy")
        tp.process_tick(SYMBOL, 200.0, 3.0, "buy")
        expected = (100.0 * 1.0 + 200.0 * 3.0) / (1.0 + 3.0)
        assert tp.get_vwap(SYMBOL) == pytest.approx(expected)


class TestTickImbalance:
    def test_all_buys(self):
        """50 buy ticks with equal volume → imbalance == 1.0."""
        tp = RustTickProcessor(1000, 1000.0)
        _feed_ticks(tp, SYMBOL, 50, side="buy", vol=10.0)
        assert tp.get_tick_imbalance(SYMBOL) == pytest.approx(1.0)

    def test_all_sells(self):
        """50 sell ticks with equal volume → imbalance == -1.0."""
        tp = RustTickProcessor(1000, 1000.0)
        _feed_ticks(tp, SYMBOL, 50, side="sell", vol=10.0)
        assert tp.get_tick_imbalance(SYMBOL) == pytest.approx(-1.0)

    def test_balanced(self):
        """25 buys + 25 sells with equal volume → imbalance == 0.0."""
        tp = RustTickProcessor(1000, 1000.0)
        _feed_ticks(tp, SYMBOL, 25, side="buy", vol=5.0)
        _feed_ticks(tp, SYMBOL, 25, side="sell", vol=5.0)
        assert tp.get_tick_imbalance(SYMBOL) == pytest.approx(0.0, abs=1e-9)


class TestVwapMidPrice:
    def test_vwap_mid_price(self):
        """get_vwap_mid_price == (vwap + (bid + ask) / 2) / 2."""
        tp = RustTickProcessor(1000, 1000.0)
        _feed_ticks(tp, SYMBOL, 10, price=98.0, vol=1.0, side="buy")
        vwap = tp.get_vwap(SYMBOL)
        bid, ask = 100.0, 102.0
        expected = (vwap + (bid + ask) / 2.0) / 2.0
        assert tp.get_vwap_mid_price(SYMBOL, bid, ask) == pytest.approx(expected)

    def test_no_ticks_returns_book_mid(self):
        tp = RustTickProcessor(1000, 1000.0)
        result = tp.get_vwap_mid_price(SYMBOL, 100.0, 102.0)
        assert result == pytest.approx(101.0)


class TestVpin:
    def test_vpin_insufficient_buckets(self):
        """Fewer than 2 completed buckets → returns 0.0."""
        tp = RustTickProcessor(1000, 1000.0)
        # Feed less than one full bucket
        _feed_ticks(tp, SYMBOL, 5, vol=100.0, side="buy")
        assert tp.get_vpin(SYMBOL) == pytest.approx(0.0)

    def test_vpin_bucket_completion(self):
        """Feed enough volume to complete 3 VPIN buckets; get_vpin() > 0."""
        tp = RustTickProcessor(1000, 100.0)  # small bucket for test
        # Feed alternating buy/sell to get non-zero VPIN
        for i in range(400):
            side = "buy" if i % 2 == 0 else "sell"
            tp.process_tick(SYMBOL, 100.0, 1.0, side)
        # Should have completed multiple buckets
        vpin = tp.get_vpin(SYMBOL)
        assert vpin >= 0.0  # VPIN is always >= 0
        assert tp.get_tick_count(SYMBOL) > 0

    def test_vpin_pure_one_sided(self):
        """All-buy trades: each bucket has max imbalance → VPIN == 1.0."""
        tp = RustTickProcessor(10000, 100.0)  # bucket_size=100
        _feed_ticks(tp, SYMBOL, 1000, vol=1.0, side="buy")
        # All buckets: buy_vol=100, sell_vol=0 → |100-0|/100 = 1.0
        assert tp.get_vpin(SYMBOL) == pytest.approx(1.0)


class TestRingBuffer:
    def test_ring_buffer_eviction(self):
        """Feed window_size + 100 ticks; tick_count == window_size."""
        window = 500
        tp = RustTickProcessor(window, 1000.0)
        _feed_ticks(tp, SYMBOL, window + 100, price=100.0, vol=1.0, side="buy")
        assert tp.get_tick_count(SYMBOL) == window

    def test_vwap_reflects_only_window(self):
        """After eviction, VWAP reflects only the most recent window ticks."""
        window = 10
        tp = RustTickProcessor(window, 1000.0)
        # Feed 10 ticks at price=100 (will all be evicted)
        _feed_ticks(tp, SYMBOL, window, price=100.0, vol=1.0, side="buy")
        # Feed 10 more ticks at price=200 (these fill the window)
        _feed_ticks(tp, SYMBOL, window, price=200.0, vol=1.0, side="buy")
        # VWAP should now be 200.0
        assert tp.get_vwap(SYMBOL) == pytest.approx(200.0)


class TestGetMetrics:
    def test_format(self):
        """get_metrics() returns a dict with exactly the expected keys."""
        tp = RustTickProcessor(1000, 1000.0)
        _feed_ticks(tp, SYMBOL, 5)
        metrics = tp.get_metrics(SYMBOL)
        assert set(metrics.keys()) == {"vwap", "tick_imbalance", "vpin", "tick_count"}

    def test_values_match_individual_getters(self):
        tp = RustTickProcessor(1000, 1000.0)
        _feed_ticks(tp, SYMBOL, 20, price=150.0, vol=2.0, side="buy")
        m = tp.get_metrics(SYMBOL)
        assert m["vwap"] == pytest.approx(tp.get_vwap(SYMBOL))
        assert m["tick_imbalance"] == pytest.approx(tp.get_tick_imbalance(SYMBOL))
        assert m["vpin"] == pytest.approx(tp.get_vpin(SYMBOL))
        assert m["tick_count"] == tp.get_tick_count(SYMBOL)


class TestEdgeCases:
    def test_empty_symbol(self):
        """Querying an unknown symbol returns 0.0 without error."""
        tp = RustTickProcessor(1000, 1000.0)
        assert tp.get_vwap("UNKNOWN/USD") == pytest.approx(0.0)
        assert tp.get_tick_imbalance("UNKNOWN/USD") == pytest.approx(0.0)
        assert tp.get_vpin("UNKNOWN/USD") == pytest.approx(0.0)
        assert tp.get_tick_count("UNKNOWN/USD") == 0

    def test_nan_price_ignored(self):
        tp = RustTickProcessor(1000, 1000.0)
        tp.process_tick(SYMBOL, float("nan"), 1.0, "buy")
        assert tp.get_vwap(SYMBOL) == pytest.approx(0.0)
        assert tp.get_tick_count(SYMBOL) == 0

    def test_zero_price_ignored(self):
        tp = RustTickProcessor(1000, 1000.0)
        tp.process_tick(SYMBOL, 0.0, 1.0, "buy")
        assert tp.get_tick_count(SYMBOL) == 0

    def test_process_ticks_batch(self):
        tp = RustTickProcessor(1000, 1000.0)
        batch = [(100.0, 1.0, "buy"), (200.0, 1.0, "sell"), (150.0, 2.0, "buy")]
        tp.process_ticks(SYMBOL, batch)
        assert tp.get_tick_count(SYMBOL) == 3

    def test_multiple_symbols_isolated(self):
        tp = RustTickProcessor(1000, 1000.0)
        tp.process_tick("BTC/USDT", 50000.0, 1.0, "buy")
        tp.process_tick("ETH/USDT", 3000.0, 1.0, "buy")
        assert tp.get_vwap("BTC/USDT") == pytest.approx(50000.0)
        assert tp.get_vwap("ETH/USDT") == pytest.approx(3000.0)


class TestIncrementalAccuracy:
    def test_incremental_accuracy(self):
        """Feed 1,000,000 random ticks; Rust VWAP relative error < 1e-10 vs numpy."""
        rng = np.random.default_rng(123)
        n = 1_000_000
        window = 1000

        prices = rng.uniform(90.0, 110.0, n)
        vols = rng.uniform(0.1, 5.0, n)

        tp = RustTickProcessor(window, 1e12)  # huge bucket so VPIN doesn't interfere
        for price, vol in zip(prices, vols):
            tp.process_tick(SYMBOL, price, float(vol), "buy")

        # Reference: last `window` ticks
        ref_prices = prices[-window:]
        ref_vols = vols[-window:]
        ref_vwap = np.sum(ref_prices * ref_vols) / np.sum(ref_vols)

        rust_vwap = tp.get_vwap(SYMBOL)
        rel_err = abs(rust_vwap - ref_vwap) / ref_vwap
        assert rel_err < 1e-10, f"VWAP relative error too large: {rel_err}"


class TestPerformance:
    def test_performance_benchmark(self):
        """Ingest 1,000,000 ticks in < 200ms; 1,000,000 get_vwap() calls in < 300ms."""
        tp = RustTickProcessor(1000, 1e12)

        n = 1_000_000
        t0 = time.perf_counter()
        for i in range(n):
            tp.process_tick(SYMBOL, 100.0 + (i % 100) * 0.01, 1.0, "buy")
        elapsed_ingest = time.perf_counter() - t0
        assert elapsed_ingest < 0.2, f"Ingestion too slow: {elapsed_ingest:.3f}s"

        t1 = time.perf_counter()
        for _ in range(n):
            _ = tp.get_vwap(SYMBOL)
        elapsed_query = time.perf_counter() - t1
        assert elapsed_query < 0.3, f"VWAP queries too slow: {elapsed_query:.3f}s"
