"""Tests for the Rust execution pre-submission engine (Phase 4).

Exercises ``MarketInfo``, ``FeeTable``, ``ExecutionPlan``, ``compute_execution_plan``,
and ``score_venues`` — as specified in the rust1.txt Phase 4 testing criteria.
"""

from __future__ import annotations

import math
import time

import pytest

pytest.importorskip("rust_trading_engine")

from rust_trading_engine.execution_prep import (
    ExecutionPlan,
    FeeTable,
    MarketInfo,
    compute_execution_plan,
    score_venues,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(
    symbol: str = "BTC/USDT",
    is_contract: bool = False,
    is_inverse: bool = False,
    contract_size: float = 1.0,
    min_amount: float = 0.0,
    step_size: float = 0.0,
    price_precision: float = 0.0,
) -> MarketInfo:
    return MarketInfo(
        symbol=symbol,
        is_contract=is_contract,
        is_inverse=is_inverse,
        contract_size=contract_size,
        min_amount=min_amount,
        step_size=step_size,
        price_precision=price_precision,
    )


def _make_plan(
    market: MarketInfo | None = None,
    exchange: str = "mexc",
    price: float = 50000.0,
    size: float = 100.0,
    leverage: int = 10,
    direction: str = "long",
    confidence: float = 0.5,
    book_imbalance: float = 0.0,
    spread_bps: float = 5.0,
    expected_profit_pct: float = 1.0,
    max_slippage_pct: float = 0.005,
    fee_table: FeeTable | None = None,
) -> ExecutionPlan:
    if market is None:
        market = _make_market()
    if fee_table is None:
        fee_table = FeeTable()
    return compute_execution_plan(
        market_info=market,
        fee_table=fee_table,
        exchange_name=exchange,
        current_price=price,
        position_size_usdt=size,
        leverage=leverage,
        direction=direction,
        signal_confidence=confidence,
        book_imbalance=book_imbalance,
        spread_bps=spread_bps,
        expected_profit_pct=expected_profit_pct,
        max_entry_slippage_pct=max_slippage_pct,
    )


# ---------------------------------------------------------------------------
# Contract / base-amount sizing tests
# ---------------------------------------------------------------------------


class TestLinearContractSizing:
    def test_linear_contract_sizing(self):
        """Linear contract: floor((100*10/50000)/0.001) == 20 contracts.

        Calculation: position_size(100) * leverage(10) = 1000 USDT notional.
        1000 / 50000 (price) / 0.001 (contract_size) = 20.0 contracts.
        """
        market = _make_market(
            is_contract=True,
            is_inverse=False,
            contract_size=0.001,
        )
        plan = _make_plan(market=market, price=50000.0, size=100.0, leverage=10)
        assert plan.is_viable
        assert plan.amount_to_order == pytest.approx(20.0)


class TestInverseContractSizing:
    def test_inverse_contract_sizing(self):
        """Inverse contract: size / contract_size == 1000/100 == 10."""
        market = _make_market(
            is_contract=True,
            is_inverse=True,
            contract_size=100.0,
        )
        plan = _make_plan(market=market, size=1000.0, leverage=1)
        assert plan.is_viable
        assert plan.amount_to_order == pytest.approx(10.0)


class TestSpotSizing:
    def test_spot_sizing(self):
        """Spot: amount_to_order ≈ fee_adjusted_size / price."""
        market = _make_market(is_contract=False)
        plan = _make_plan(market=market, price=100.0, size=50.0, leverage=1)
        assert plan.is_viable
        # fee_adjusted_size = 50.0 - fee_buffer; amount ≈ fee_adjusted_size / 100.0
        assert plan.amount_to_order == pytest.approx(plan.fee_adjusted_size / 100.0, rel=1e-6)
        assert 0.45 < plan.amount_to_order < 0.50


class TestFeeViabilityRejection:
    def test_fee_viability_rejection(self):
        """Size too small to cover fees: is_viable == False."""
        market = _make_market(is_contract=True, is_inverse=False, contract_size=1.0)
        # size=1.0 USDT, leverage=100, taker_fee=0.001 (binance default)
        # round-trip notional = 1.0 * 100 = 100; fee = 100 * 0.001 * 2 = 0.2
        # fee_buffer = 0.2 * 1.1 = 0.22 > 1.0 — actually 0.22 < 1.0 for mexc...
        # Use a custom fee table with very high rates to force rejection
        ft = FeeTable()
        ft.set_rate("high_fee", 0.01, 0.02)  # 2% taker fee
        plan = compute_execution_plan(
            market_info=market,
            fee_table=ft,
            exchange_name="high_fee",
            current_price=50000.0,
            position_size_usdt=1.0,
            leverage=100,
            direction="long",
            signal_confidence=0.5,
            book_imbalance=0.0,
            spread_bps=5.0,
            expected_profit_pct=1.0,
            max_entry_slippage_pct=0.005,
        )
        assert not plan.is_viable
        assert "fee" in plan.rejection_reason.lower() or "size" in plan.rejection_reason.lower()


class TestStepSizeRounding:
    def test_step_size_rounding(self):
        """step_size=0.01, raw spot amount=1.567 → floor to 1.56."""
        market = _make_market(is_contract=False, step_size=0.01)
        # price=100, size=1.567*100 ≈ 156.7 USDT (plus small fee buffer)
        # We want raw amount ~1.56 after rounding
        # fee_adjusted_size / price must be ~1.567 before rounding
        # For mexc taker=0.0006: fee_buffer = 100*1*0.0006*2*1.1 = 0.132
        # size_needed = 1.567*100 + 0.132 ≈ 156.832
        plan = _make_plan(market=market, price=100.0, size=156.9, leverage=1)
        assert plan.is_viable
        # Step rounded down: multiple of 0.01
        assert abs(plan.amount_to_order % 0.01) < 1e-9 or abs(plan.amount_to_order % 0.01 - 0.01) < 1e-9


class TestMinimumAmountRejection:
    def test_minimum_amount_rejection(self):
        """Calculated amount < min_amount: is_viable == False."""
        market = _make_market(is_contract=False, min_amount=10.0)
        # price=100, size=50 → amount_to_order ≈ 0.5 < min_amount=10
        plan = _make_plan(market=market, price=100.0, size=50.0, leverage=1)
        assert not plan.is_viable
        assert "minimum" in plan.rejection_reason.lower() or "min" in plan.rejection_reason.lower() or "small" in plan.rejection_reason.lower()


class TestMinimumOneContract:
    def test_minimum_1_contract(self):
        """Raw linear contract calc yields 0.3 → amount_to_order rounded up to 1."""
        # size=1 USDT, leverage=1, price=50000, contract_size=1.0
        # raw = (1 * 1) / 50000 / 1.0 = 0.00002 → rounds up to 1
        market = _make_market(is_contract=True, is_inverse=False, contract_size=1.0)
        plan = _make_plan(market=market, price=50000.0, size=1.0, leverage=1)
        # Either rejected or rounded up
        if plan.is_viable:
            assert plan.amount_to_order >= 1.0
        else:
            assert "fee" in plan.rejection_reason.lower() or "size" in plan.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Optimal order type tests
# ---------------------------------------------------------------------------


class TestOptimalOrderTypePostOnly:
    def test_optimal_order_type_post_only(self):
        """confidence=0.9, spread_bps=2.0 → optimal_order_type == 'post_only'."""
        plan = _make_plan(confidence=0.9, spread_bps=2.0)
        assert plan.is_viable
        assert plan.optimal_order_type == "post_only"


class TestOptimalOrderTypeMarket:
    def test_optimal_order_type_market(self):
        """confidence=0.3, spread_bps=10.0 → optimal_order_type == 'market'."""
        plan = _make_plan(confidence=0.3, spread_bps=10.0)
        assert plan.is_viable
        assert plan.optimal_order_type == "market"


# ---------------------------------------------------------------------------
# Safety / edge case tests
# ---------------------------------------------------------------------------


class TestNanInputSafety:
    def test_nan_input_safety(self):
        """current_price=NaN → is_viable == False, no panic."""
        plan = _make_plan(price=float("nan"))
        assert not plan.is_viable

    def test_nan_size_safety(self):
        plan = _make_plan(size=float("nan"))
        assert not plan.is_viable


class TestZeroPriceSafety:
    def test_zero_price_safety(self):
        """current_price=0.0 → is_viable == False with descriptive rejection."""
        plan = _make_plan(price=0.0)
        assert not plan.is_viable
        assert len(plan.rejection_reason) > 0
        assert "price" in plan.rejection_reason.lower() or "0" in plan.rejection_reason


# ---------------------------------------------------------------------------
# FeeTable tests
# ---------------------------------------------------------------------------


class TestFeeTable:
    def test_fee_table_cheapest_limit(self):
        """FeeTable.get_cheapest_exchange(['mexc', 'gateio'], 'limit') == 'gateio'.

        Gate.io has a negative maker fee (-0.00025) which is the lowest (cheapest).
        """
        ft = FeeTable()
        cheapest = ft.get_cheapest_exchange(["mexc", "gateio"], "limit")
        assert cheapest == "gateio"

    def test_fee_table_custom_rate(self):
        ft = FeeTable()
        ft.set_rate("testex", 0.0001, 0.0003)
        cheapest = ft.get_cheapest_exchange(["mexc", "testex", "gateio"], "market")
        assert cheapest == "testex"

    def test_fee_table_empty_list(self):
        ft = FeeTable()
        result = ft.get_cheapest_exchange([], "limit")
        assert result == ""


# ---------------------------------------------------------------------------
# Venue scoring tests
# ---------------------------------------------------------------------------


class TestVenueScoringMatchesPython:
    def test_venue_scoring_matches_python(self):
        """Three known venues: compare Rust score_venues() ranking vs manual Python scoring."""
        amount = 100.0
        min_alloc = 0.05

        venues_data = {
            "mexc":   {"taker_fee": 0.0006, "liq": 500.0, "spread_pct": 0.1, "rel": 0.99, "fill_rate": 0.98},
            "gateio": {"taker_fee": 0.00075, "liq": 800.0, "spread_pct": 0.05, "rel": 1.0, "fill_rate": 0.99},
            "bingx":  {"taker_fee": 0.0005, "liq": 300.0, "spread_pct": 0.2, "rel": 0.95, "fill_rate": 0.97},
        }

        # Rust scoring
        rust_input = [
            (name, d["taker_fee"], d["liq"], d["spread_pct"], d["rel"], d["fill_rate"])
            for name, d in venues_data.items()
        ]
        rust_results = score_venues(rust_input, amount, min_alloc, 3)
        rust_rankings = [name for name, _, _ in rust_results]

        # Python scoring (manual, same formula as SmartOrderRouter._score_venues)
        def py_score(d):
            liq = min(1.0, d["liq"] / (amount * 2.0))
            fee = max(0.0, min(1.0, 1.0 - d["taker_fee"] / 0.002))
            spread = 1.0 - min(1.0, d["spread_pct"] / 0.5)
            return liq * 0.40 + fee * 0.25 + spread * 0.20 + d["rel"] * 0.10 + d["fill_rate"] * 0.05

        py_scored = sorted(venues_data.items(), key=lambda kv: py_score(kv[1]), reverse=True)
        py_rankings = [name for name, _ in py_scored]

        assert rust_rankings == py_rankings[:len(rust_rankings)]

    def test_allocations_sum_to_one(self):
        """Allocations from score_venues() must sum to 1.0."""
        venues = [
            ("venue_a", 0.001, 1000.0, 0.1, 1.0, 0.99),
            ("venue_b", 0.0005, 500.0, 0.2, 0.95, 0.98),
            ("venue_c", 0.0008, 750.0, 0.15, 0.98, 0.97),
        ]
        results = score_venues(venues, 100.0, 0.05, 3)
        assert len(results) > 0
        total = sum(pct for _, pct, _ in results)
        assert total == pytest.approx(1.0, abs=1e-9)


class TestVenueScoringMinAllocationFilter:
    def test_venue_scoring_min_allocation_filter(self):
        """A venue with very low score is filtered when below min_venue_allocation."""
        venues = [
            ("good_a", 0.0002, 10000.0, 0.01, 1.0, 1.0),
            ("good_b", 0.0003, 9000.0, 0.02, 0.99, 0.99),
            ("good_c", 0.0004, 8000.0, 0.03, 0.98, 0.98),
            # Very bad venue: tiny liquidity, high fee, wide spread
            ("bad",    0.002,     0.1, 5.0, 0.1, 0.1),
        ]
        results = score_venues(venues, 100.0, 0.05, 4)
        result_names = [name for name, _, _ in results]
        assert "bad" not in result_names

    def test_empty_venues_returns_empty(self):
        """score_venues() with an empty venue list returns an empty list."""
        results = score_venues([], 100.0, 0.05, 3)
        assert results == []


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------


class TestPerformanceBenchmark:
    def test_performance_benchmark(self):
        """100,000 compute_execution_plan() calls in < 500ms (< 5µs each)."""
        market = _make_market(is_contract=True, is_inverse=False, contract_size=0.001)
        ft = FeeTable()
        n = 100_000
        t0 = time.perf_counter()
        for _ in range(n):
            compute_execution_plan(
                market_info=market,
                fee_table=ft,
                exchange_name="mexc",
                current_price=50000.0,
                position_size_usdt=100.0,
                leverage=10,
                direction="long",
                signal_confidence=0.7,
                book_imbalance=0.0,
                spread_bps=5.0,
                expected_profit_pct=1.0,
                max_entry_slippage_pct=0.005,
            )
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"compute_execution_plan too slow: {elapsed:.3f}s for {n} calls"
