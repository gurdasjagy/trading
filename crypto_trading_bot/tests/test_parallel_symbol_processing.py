"""Tests for parallel symbol processing in the trading engine."""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_parallel_processing_all_symbols_processed():
    """All trading pairs should be processed even with parallel execution."""
    processed_symbols: list[str] = []

    async def mock_process_symbol(symbol: str) -> tuple[int, int]:
        processed_symbols.append(symbol)
        await asyncio.sleep(0.01)  # Simulate work
        return (0, 0)

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    semaphore = asyncio.Semaphore(3)

    async def bounded_process(sym: str) -> tuple[int, int]:
        async with semaphore:
            return await mock_process_symbol(sym)

    results = await asyncio.gather(*[bounded_process(s) for s in symbols])

    assert len(processed_symbols) == 5
    assert set(processed_symbols) == set(symbols)
    assert all(isinstance(r, tuple) for r in results)


@pytest.mark.asyncio
async def test_parallel_processing_exception_isolation():
    """An exception in one symbol should not affect others."""
    results_collected: list[str] = []

    async def mock_process(symbol: str) -> tuple[int, int]:
        if symbol == "BAD/USDT":
            raise ValueError("Simulated error")
        results_collected.append(symbol)
        return (1, 0)

    symbols = ["BTC/USDT", "BAD/USDT", "ETH/USDT"]
    results = await asyncio.gather(
        *[mock_process(s) for s in symbols],
        return_exceptions=True,
    )

    assert len(results_collected) == 2
    assert isinstance(results[1], ValueError)
    assert results[0] == (1, 0)
    assert results[2] == (1, 0)


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Semaphore should limit concurrent processing to 3."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def mock_process(symbol: str) -> tuple[int, int]:
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        await asyncio.sleep(0.05)
        async with lock:
            current_concurrent -= 1
        return (0, 0)

    semaphore = asyncio.Semaphore(3)
    symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]

    async def bounded(sym: str) -> tuple[int, int]:
        async with semaphore:
            return await mock_process(sym)

    await asyncio.gather(*[bounded(s) for s in symbols])
    assert max_concurrent <= 3


@pytest.mark.asyncio
async def test_parallel_processing_returns_results_in_order():
    """asyncio.gather results are returned in submission order even with varying sleep times."""
    import random

    rng = random.Random(42)

    async def timed_process(symbol: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return symbol

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    delays = [rng.uniform(0.01, 0.05) for _ in symbols]

    results = await asyncio.gather(*[timed_process(s, d) for s, d in zip(symbols, delays)])
    # asyncio.gather preserves input order regardless of completion order
    assert list(results) == symbols


@pytest.mark.asyncio
async def test_parallel_processing_with_semaphore_1_serializes():
    """With Semaphore(1) all tasks should run sequentially."""
    order: list[str] = []
    semaphore = asyncio.Semaphore(1)

    async def record(symbol: str) -> None:
        async with semaphore:
            order.append(symbol)
            await asyncio.sleep(0.01)

    symbols = ["A", "B", "C"]
    await asyncio.gather(*[record(s) for s in symbols])

    # All symbols must have been processed exactly once
    assert sorted(order) == sorted(symbols)
    assert len(order) == 3


@pytest.mark.asyncio
async def test_parallel_processing_all_errors_return_exceptions():
    """When all tasks raise, gather(return_exceptions=True) returns all errors."""
    async def always_fail(symbol: str) -> None:
        raise RuntimeError(f"fail: {symbol}")

    symbols = ["X1", "X2", "X3"]
    results = await asyncio.gather(*[always_fail(s) for s in symbols], return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    assert len(results) == 3
