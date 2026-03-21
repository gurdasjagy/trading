#!/usr/bin/env python3
"""backfill_data.py — Fetch and cache historical OHLCV data for configured trading pairs.

Usage:
    python scripts/backfill_data.py --symbol BTC/USDT --timeframe 1h --days 30
    python scripts/backfill_data.py --timeframe 4h --days 90   # all configured pairs
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical OHLCV data for crypto trading pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Trading pair (e.g. BTC/USDT). If omitted, all configured pairs are used.",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="OHLCV timeframe (e.g. 1m, 5m, 15m, 1h, 4h, 1d).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of historical days to fetch.",
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange ID to fetch from (used by CCXT).",
    )
    return parser.parse_args()


async def _backfill_symbol(
    loader,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Download and cache data for a single *symbol*.

    Args:
        loader: HistoricalDataLoader instance.
        symbol: Trading pair string.
        timeframe: OHLCV timeframe.
        start_date: Start of the backfill window.
        end_date: End of the backfill window.
    """
    try:
        print(
            f"  Fetching {symbol} [{timeframe}] {start_date.date()} → {end_date.date()} …",
            end=" ",
            flush=True,
        )
        df = await loader.load(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"{len(df)} bars ✓")
    except Exception as exc:
        print(f"FAILED — {exc}")


async def main() -> None:
    """Main entry point for the backfill script."""
    args = _parse_args()

    try:
        from backtest.data_loader import HistoricalDataLoader
        from config.settings import Settings
    except ImportError as exc:
        print(f"ERROR: Cannot import project modules — {exc}", file=sys.stderr)
        print("Run this script from the project root directory.", file=sys.stderr)
        sys.exit(1)

    settings = Settings()
    symbols: list[str]
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = settings.exchange.trading_pairs

    end_date = datetime.now(timezone.utc).replace(tzinfo=None)
    start_date = end_date - timedelta(days=args.days)

    loader = HistoricalDataLoader(exchange_id=args.exchange)

    print()
    print("=" * 60)
    print("  CryptoBot — Historical Data Backfill")
    print("=" * 60)
    print(f"  Exchange  : {args.exchange}")
    print(f"  Timeframe : {args.timeframe}")
    print(f"  Days      : {args.days}")
    print(f"  Symbols   : {len(symbols)}")
    print("=" * 60)
    print()

    try:
        from tqdm.asyncio import tqdm  # type: ignore

        async def _with_progress() -> None:
            for symbol in tqdm(symbols, desc="Symbols", unit="pair"):
                await _backfill_symbol(loader, symbol, args.timeframe, start_date, end_date)

        await _with_progress()
    except ImportError:
        # tqdm not available — fall back to plain loop
        for symbol in symbols:
            await _backfill_symbol(loader, symbol, args.timeframe, start_date, end_date)

    print()
    print(f"✅  Backfill complete for {len(symbols)} pair(s).")
    print()


if __name__ == "__main__":
    asyncio.run(main())
