"""Tests for utils.circuit_breaker and utils.symbol_resolver.

These tests validate:
  1. CircuitBreaker opens after *failure_threshold* consecutive failures.
  2. CircuitBreaker recovers to HALF_OPEN / CLOSED after recovery_timeout.
  3. with_circuit_breaker decorator blocks calls when circuit is OPEN.
  4. resolve_symbols decorator resolves precious metals + swap symbols.
  5. resolve_symbols restores original symbol in Ticker return values.
  6. resolve_symbols handles cancel_order (where symbol is the 2nd positional arg).
  7. original_symbol_ctx is set correctly inside decorated coroutines.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exchange.base_exchange import Ticker
from exchange.gateio_client import GateIOClient
from utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    get_circuit_breaker,
    reset_circuit_breaker,
    with_circuit_breaker,
)
from utils.symbol_resolver import original_symbol_ctx, resolve_symbols


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
        timestamp=1_000_000,
    )


class _FakeExchange:
    """Minimal stand-in that implements the two resolution helpers."""

    # Note: XAG/USDT maps to XAUT/USDT because there is no dedicated silver
    # token on Gate.io's standard futures API.  This matches the production
    # PRECIOUS_METALS_MAPPING in both GateIOClient and CcxtExchange.
    PRECIOUS_METALS_MAPPING = {"XAU/USDT": "XAUT/USDT", "XAG/USDT": "XAUT/USDT"}

    def _resolve_precious_metals_symbol(self, symbol: str) -> str:
        return self.PRECIOUS_METALS_MAPPING.get(symbol, symbol)

    def _resolve_swap_symbol(self, symbol: str) -> str:
        # Mirrors the updated production behaviour: precious metals first, then
        # spot → swap conversion.
        symbol = self._resolve_precious_metals_symbol(symbol)
        if ":" not in symbol and "/" in symbol:
            quote = symbol.split("/")[-1]
            return f"{symbol}:{quote}"
        return symbol

    @resolve_symbols
    async def get_ticker(self, symbol: str) -> Ticker:
        return _ticker(symbol=symbol)

    @resolve_symbols
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h") -> str:
        return symbol  # just return the resolved symbol for assertion

    @resolve_symbols
    async def cancel_order(self, order_id: str, symbol: str) -> str:
        return symbol  # just return the resolved symbol for assertion

    @resolve_symbols
    async def subscribe_ticker(self, symbol: str, callback: Any) -> None:
        # For testing: capture the ContextVar value and call callback once
        ctx_symbol = original_symbol_ctx.get()
        await callback({"resolved": symbol, "original": ctx_symbol})


# ===========================================================================
# CircuitBreaker unit tests
# ===========================================================================


class TestCircuitBreaker:
    """Unit tests for the CircuitBreaker state machine."""

    def setup_method(self) -> None:
        self.cb = CircuitBreaker("TEST/USDT", failure_threshold=3, recovery_timeout=1.0)

    def test_starts_closed(self) -> None:
        assert self.cb.state == CircuitState.CLOSED
        assert self.cb.allow_request() is True

    def test_opens_after_threshold_failures(self) -> None:
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == CircuitState.OPEN
        assert self.cb.allow_request() is False

    def test_does_not_open_before_threshold(self) -> None:
        for _ in range(2):
            self.cb.record_failure()
        assert self.cb.state == CircuitState.CLOSED
        assert self.cb.allow_request() is True

    def test_success_resets_failures(self) -> None:
        self.cb.record_failure()
        self.cb.record_failure()
        self.cb.record_success()
        # Counter reset — two more failures should NOT open circuit
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state == CircuitState.CLOSED

    def test_enters_half_open_after_recovery_timeout(self) -> None:
        for _ in range(3):
            self.cb.record_failure()
        # Force the opened_at timestamp into the past
        self.cb._opened_at = time.monotonic() - 2.0  # > recovery_timeout of 1.0
        assert self.cb.allow_request() is True
        assert self.cb.state == CircuitState.HALF_OPEN

    def test_closes_from_half_open_on_success(self) -> None:
        for _ in range(3):
            self.cb.record_failure()
        self.cb._opened_at = time.monotonic() - 2.0
        self.cb.allow_request()  # transitions to HALF_OPEN
        self.cb.record_success()
        assert self.cb.state == CircuitState.CLOSED

    def test_reopens_from_half_open_on_failure(self) -> None:
        for _ in range(3):
            self.cb.record_failure()
        self.cb._opened_at = time.monotonic() - 2.0
        self.cb.allow_request()  # transitions to HALF_OPEN
        self.cb.record_failure()
        assert self.cb.state == CircuitState.OPEN

    def test_manual_reset(self) -> None:
        for _ in range(3):
            self.cb.record_failure()
        self.cb.reset()
        assert self.cb.state == CircuitState.CLOSED
        assert self.cb._consecutive_failures == 0
        assert self.cb.allow_request() is True


# ===========================================================================
# with_circuit_breaker decorator tests
# ===========================================================================


class TestWithCircuitBreakerDecorator:
    """Tests for the @with_circuit_breaker decorator."""

    def setup_method(self) -> None:
        reset_circuit_breaker("CB/USDT")

    @pytest.mark.asyncio
    async def test_allows_requests_when_closed(self) -> None:
        call_count = 0

        class _Ex:
            @with_circuit_breaker(failure_threshold=5)
            async def action(self, symbol: str) -> str:
                nonlocal call_count
                call_count += 1
                return "ok"

        ex = _Ex()
        result = await ex.action("CB/USDT")
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_blocks_requests_when_open(self) -> None:
        cb = get_circuit_breaker("CB/USDT", failure_threshold=2)
        # Open the circuit manually
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        class _Ex:
            @with_circuit_breaker(failure_threshold=2)
            async def action(self, symbol: str) -> str:
                return "should not reach"

        ex = _Ex()
        with pytest.raises(CircuitBreakerOpenError):
            await ex.action("CB/USDT")

    @pytest.mark.asyncio
    async def test_records_failure_on_exception(self) -> None:
        reset_circuit_breaker("FAIL/USDT")

        class _Ex:
            @with_circuit_breaker(failure_threshold=3)
            async def action(self, symbol: str) -> None:
                raise RuntimeError("exchange down")

        ex = _Ex()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await ex.action("FAIL/USDT")

        cb = get_circuit_breaker("FAIL/USDT")
        assert cb._consecutive_failures == 2
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self) -> None:
        reset_circuit_breaker("OPEN/USDT")

        class _Ex:
            @with_circuit_breaker(failure_threshold=3)
            async def action(self, symbol: str) -> None:
                raise RuntimeError("bad symbol")

        ex = _Ex()
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await ex.action("OPEN/USDT")

        cb = get_circuit_breaker("OPEN/USDT")
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_records_success_on_ok(self) -> None:
        reset_circuit_breaker("OK/USDT")
        cb = get_circuit_breaker("OK/USDT", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()

        class _Ex:
            @with_circuit_breaker(failure_threshold=3)
            async def action(self, symbol: str) -> str:
                return "ok"

        ex = _Ex()
        await ex.action("OK/USDT")
        assert cb._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_works_with_second_positional_symbol(self) -> None:
        """symbol as the 2nd positional arg (like cancel_order)."""
        reset_circuit_breaker("SYM/USDT")

        class _Ex:
            @with_circuit_breaker(failure_threshold=3)
            async def cancel_order(self, order_id: str, symbol: str) -> str:
                return symbol

        ex = _Ex()
        result = await ex.cancel_order("ord1", "SYM/USDT")
        assert result == "SYM/USDT"


# ===========================================================================
# resolve_symbols decorator tests
# ===========================================================================


class TestResolveSymbolsDecorator:
    """Tests for the @resolve_symbols decorator."""

    @pytest.mark.asyncio
    async def test_precious_metals_resolved_for_get_ticker(self) -> None:
        """XAU/USDT should be resolved to XAUT/USDT:USDT before the API call."""
        ex = _FakeExchange()
        ticker = await ex.get_ticker("XAU/USDT")
        # @resolve_symbols should post-process the Ticker to restore original
        assert ticker.symbol == "XAU/USDT"

    @pytest.mark.asyncio
    async def test_regular_symbol_resolved_to_swap(self) -> None:
        """BTC/USDT should be resolved to BTC/USDT:USDT inside the function."""
        ex = _FakeExchange()
        result = await ex.get_ohlcv("BTC/USDT")
        assert result == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_already_resolved_symbol_unchanged(self) -> None:
        """BTC/USDT:USDT should pass through unchanged."""
        ex = _FakeExchange()
        result = await ex.get_ohlcv("BTC/USDT:USDT")
        assert result == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_cancel_order_symbol_resolved(self) -> None:
        """symbol as the 2nd positional arg should be resolved correctly."""
        ex = _FakeExchange()
        result = await ex.cancel_order("order-1", "XAU/USDT")
        assert result == "XAUT/USDT:USDT"

    @pytest.mark.asyncio
    async def test_original_symbol_in_context(self) -> None:
        """original_symbol_ctx should hold the pre-resolution symbol inside the body."""
        ex = _FakeExchange()
        received: list[dict] = []

        async def cb(data: dict) -> None:
            received.append(data)

        await ex.subscribe_ticker("XAU/USDT", cb)
        assert len(received) == 1
        assert received[0]["resolved"] == "XAUT/USDT:USDT"
        assert received[0]["original"] == "XAU/USDT"

    @pytest.mark.asyncio
    async def test_ticker_symbol_restored_to_original(self) -> None:
        """@resolve_symbols post-processes Ticker to use the original symbol."""
        ex = _FakeExchange()
        ticker = await ex.get_ticker("XAG/USDT")  # XAG → XAUT
        assert ticker.symbol == "XAG/USDT"

    @pytest.mark.asyncio
    async def test_silver_maps_to_xaut(self) -> None:
        """XAG/USDT should map to XAUT/USDT then to XAUT/USDT:USDT."""
        ex = _FakeExchange()
        result = await ex.get_ohlcv("XAG/USDT")
        assert result == "XAUT/USDT:USDT"


# ===========================================================================
# GateIOClient integration: _resolve_swap_symbol now handles precious metals
# ===========================================================================


class TestGateIOResolveSwapSymbolWithMetals:
    """Verify that GateIOClient._resolve_swap_symbol applies precious metals mapping."""

    def test_xau_resolves_to_xaut_swap(self) -> None:
        client = GateIOClient("key", "secret")
        assert client._resolve_swap_symbol("XAU/USDT") == "XAUT/USDT:USDT"

    def test_xag_resolves_to_xaut_swap(self) -> None:
        client = GateIOClient("key", "secret")
        assert client._resolve_swap_symbol("XAG/USDT") == "XAUT/USDT:USDT"

    def test_xaut_usdt_passthrough(self) -> None:
        client = GateIOClient("key", "secret")
        assert client._resolve_swap_symbol("XAUT/USDT") == "XAUT/USDT:USDT"

    def test_btc_resolves_to_swap(self) -> None:
        client = GateIOClient("key", "secret")
        assert client._resolve_swap_symbol("BTC/USDT") == "BTC/USDT:USDT"

    def test_already_swap_passthrough(self) -> None:
        client = GateIOClient("key", "secret")
        assert client._resolve_swap_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"
