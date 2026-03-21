#!/usr/bin/env python3
"""
Test script to check what gold symbols Gate.io actually supports for futures trading.
This will help us understand if Gate.io supports gold futures on their regular futures API.
"""

import asyncio
import ccxt.async_support as ccxt


async def test_gateio_gold_symbols():
    """Test various gold symbol formats on Gate.io to see what works."""

    # Initialize Gate.io client
    exchange = ccxt.gateio({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},  # Use swap/futures market
    })

    try:
        # Load all markets
        print("Loading Gate.io markets...")
        markets = await exchange.load_markets()

        # Search for gold-related symbols
        gold_symbols = []
        for symbol in markets.keys():
            if any(term in symbol.upper() for term in ['XAU', 'GOLD', 'XAUT', 'PAXG']):
                gold_symbols.append(symbol)

        print(f"\n✓ Found {len(gold_symbols)} gold-related symbols on Gate.io:")
        for sym in sorted(gold_symbols):
            market_info = markets[sym]
            print(f"  • {sym}")
            print(f"    - Type: {market_info.get('type', 'N/A')}")
            print(f"    - Contract: {market_info.get('contract', False)}")
            print(f"    - Linear: {market_info.get('linear', False)}")
            print(f"    - Settle: {market_info.get('settle', 'N/A')}")
            print(f"    - Base: {market_info.get('base', 'N/A')} / Quote: {market_info.get('quote', 'N/A')}")

        # Try to fetch ticker for common gold formats
        test_symbols = [
            'XAU/USDT',
            'XAU/USDT:USDT',
            'XAUT/USDT',
            'XAUT/USDT:USDT',
            'PAXG/USDT',
            'PAXG/USDT:USDT',
        ]

        print("\n\nTesting ticker fetches for various gold symbol formats:")
        for symbol in test_symbols:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                print(f"  ✓ {symbol}: SUCCESS - Last price: ${ticker.get('last', 0):.2f}")
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"  ✗ {symbol}: FAILED - {error_msg}")

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(test_gateio_gold_symbols())
