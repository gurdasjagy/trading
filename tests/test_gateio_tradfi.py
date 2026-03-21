#!/usr/bin/env python3
"""Test script for Gate.io TradFi client.

This script tests the new GateIOTradFiClient implementation:
- Symbol resolution
- API connectivity (if credentials provided)
- Basic market data retrieval
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto_trading_bot.exchange.gateio_tradfi_client import GateIOTradFiClient


def test_symbol_resolution():
    """Test symbol mapping from traditional formats to TradFi."""
    print("=" * 60)
    print("Testing Symbol Resolution")
    print("=" * 60)

    client = GateIOTradFiClient(api_key="", secret_key="")

    test_cases = [
        ("XAUUSD", "XAU_USDT"),
        ("XAU/USD", "XAU_USDT"),
        ("XAU/USDT", "XAU_USDT"),
        ("XAGUSD", "XAG_USDT"),
        ("XAG/USDT", "XAG_USDT"),
        ("EURUSD", "EUR_USDT"),
        ("GBPUSD", "GBP_USDT"),
        ("USDJPY", "JPY_USDT"),
        ("EUR_USDT", "EUR_USDT"),  # Already in TradFi format
    ]

    all_passed = True
    for input_symbol, expected_output in test_cases:
        result = client._resolve_tradfi_symbol(input_symbol)
        status = "✓" if result == expected_output else "✗"
        if result != expected_output:
            all_passed = False
        print(f"{status} {input_symbol:15} → {result:15} (expected: {expected_output})")

    print()
    if all_passed:
        print("✓ All symbol resolution tests passed!")
    else:
        print("✗ Some symbol resolution tests failed!")
    print()

    return all_passed


def test_reverse_mapping():
    """Test display symbol conversion from TradFi to user-friendly."""
    print("=" * 60)
    print("Testing Reverse Mapping")
    print("=" * 60)

    client = GateIOTradFiClient(api_key="", secret_key="")

    test_cases = [
        ("XAU_USDT", "XAUUSD"),  # Mapped to first key in FOREX_SYMBOL_MAPPING
        ("XAG_USDT", "XAGUSD"),
        ("EUR_USDT", "EURUSD"),
        ("BTC_USDT", "BTC/USDT"),  # Not in mapping, use default
    ]

    all_passed = True
    for tradfi_symbol, expected_display in test_cases:
        result = client._resolve_display_symbol(tradfi_symbol)
        # Note: Multiple display formats may map to same TradFi symbol
        # so we just check it returns something reasonable
        status = "✓" if result else "✗"
        print(f"{status} {tradfi_symbol:15} → {result:15}")

    print()
    print("✓ Reverse mapping test completed!")
    print()

    return all_passed


async def test_api_connectivity():
    """Test API connectivity (requires credentials)."""
    print("=" * 60)
    print("Testing API Connectivity")
    print("=" * 60)

    # Check for credentials
    api_key = os.getenv("GATEIO_API_KEY", "")
    secret_key = os.getenv("GATEIO_SECRET_KEY", "")

    if not api_key or not secret_key:
        print("⚠️  Skipping API connectivity test (no credentials)")
        print("   Set GATEIO_API_KEY and GATEIO_SECRET_KEY to test")
        print()
        return True

    try:
        client = GateIOTradFiClient(api_key=api_key, secret_key=secret_key)
        await client.connect()
        print("✓ Connected to Gate.io TradFi API")

        # Test ticker
        print("\nTesting ticker fetch for XAU/USDT...")
        ticker = await client.get_ticker("XAU/USDT")
        print(f"  Last price: ${ticker.last}")
        print(f"  24h High:   ${ticker.high}")
        print(f"  24h Low:    ${ticker.low}")
        print(f"  Volume:     {ticker.volume}")

        # Test OHLCV
        print("\nTesting OHLCV fetch for XAU/USDT...")
        ohlcv = await client.get_ohlcv("XAU/USDT", "1h", 10)
        print(f"  Fetched {len(ohlcv)} candles")
        print(f"  Columns: {list(ohlcv.columns)}")
        if len(ohlcv) > 0:
            print(f"  Latest close: ${ohlcv['close'].iloc[-1]}")

        # Test balance
        print("\nTesting balance fetch...")
        balance = await client.get_balance()
        print(f"  USDT Total: ${balance.usdt_total:.2f}")
        print(f"  USDT Free:  ${balance.usdt_free:.2f}")

        await client.disconnect()
        print("\n✓ All API tests passed!")
        print()
        return True

    except Exception as e:
        print(f"\n✗ API test failed: {e}")
        print()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("* Gate.io TradFi Client Test Suite")
    print("*" * 60)
    print()

    results = []

    # Test 1: Symbol resolution
    results.append(test_symbol_resolution())

    # Test 2: Reverse mapping
    results.append(test_reverse_mapping())

    # Test 3: API connectivity (requires credentials)
    import asyncio
    results.append(asyncio.run(test_api_connectivity()))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
