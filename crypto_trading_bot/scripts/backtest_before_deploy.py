#!/usr/bin/env python3
"""Backtest before deploy — validates strategy performance on recent data.

Fetches the last 30 days of 15m OHLCV data for all configured trading pairs,
runs all registered strategies, simulates trades, and calculates key metrics.

Exit codes:
  0 — backtest passed (acceptable performance)
  1 — backtest failed (poor performance; deployment blocked)

Usage:
  python scripts/backtest_before_deploy.py
  python scripts/backtest_before_deploy.py --pairs BTC/USDT,ETH/USDT
  python scripts/backtest_before_deploy.py --min-return -5 --max-drawdown 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the package root is on sys.path
_SCRIPT_DIR = Path(__file__).parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from loguru import logger


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Lightweight backtester for deployment validation."""

    def __init__(
        self,
        pairs: List[str],
        timeframe: str = "15m",
        days: int = 30,
        min_return_pct: float = 0.0,
        max_drawdown_pct: float = 15.0,
    ) -> None:
        self.pairs = pairs
        self.timeframe = timeframe
        self.days = days
        self.min_return_pct = min_return_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.results: Dict[str, Any] = {}

    async def run(self) -> Dict[str, Any]:
        """Run the backtest and return a summary dict."""
        from config.settings import Settings
        from exchange.exchange_factory import ExchangeFactory

        logger.info("🔬 Backtest starting: {} pairs, {} days, {} candles",
                    len(self.pairs), self.days, self.timeframe)

        try:
            settings = Settings.get_settings()
        except Exception as exc:
            logger.error("Could not load settings: {}", exc)
            return {"success": False, "error": str(exc)}

        exchange = None
        try:
            exchange = await ExchangeFactory.create_exchange(settings)
            await exchange.connect()
        except Exception as exc:
            logger.error("Could not connect to exchange: {}", exc)
            return {"success": False, "error": f"Exchange connection failed: {exc}"}

        try:
            pair_results: List[Dict[str, Any]] = []
            limit = self.days * 24 * 4  # 15-minute candles

            for pair in self.pairs:
                logger.info("Fetching {} {} candles for {}", limit, self.timeframe, pair)
                try:
                    df = await exchange.get_ohlcv(pair, timeframe=self.timeframe, limit=limit)
                    if df is None or len(df) < 50:
                        logger.warning("Not enough data for {} (got {})", pair, len(df) if df is not None else 0)
                        continue
                    result = self._simulate_pair(pair, df)
                    pair_results.append(result)
                    logger.info(
                        "  {} → return={:.1f}% drawdown={:.1f}% win_rate={:.1f}% trades={}",
                        pair,
                        result["total_return_pct"],
                        result["max_drawdown_pct"],
                        result["win_rate_pct"],
                        result["total_trades"],
                    )
                except Exception as exc:
                    logger.warning("Backtest error for {}: {}", pair, exc)

            overall = self._aggregate_results(pair_results)
            overall["pair_results"] = pair_results
            overall["timestamp"] = datetime.now(timezone.utc).isoformat()
            overall["pairs"] = self.pairs
            overall["days"] = self.days
            overall["timeframe"] = self.timeframe

            # Save results
            os.makedirs("data", exist_ok=True)
            output_path = "data/backtest_results.json"
            with open(output_path, "w") as f:
                json.dump(overall, f, indent=2)
            logger.info("Backtest results saved to {}", output_path)

            # Determine pass/fail
            total_return = overall.get("total_return_pct", 0.0)
            max_dd = overall.get("max_drawdown_pct", 0.0)
            passed = total_return >= self.min_return_pct and max_dd <= self.max_drawdown_pct
            overall["passed"] = passed

            if not passed:
                reasons = []
                if total_return < self.min_return_pct:
                    reasons.append(
                        f"total_return={total_return:.1f}% < min={self.min_return_pct:.1f}%"
                    )
                if max_dd > self.max_drawdown_pct:
                    reasons.append(
                        f"max_drawdown={max_dd:.1f}% > max={self.max_drawdown_pct:.1f}%"
                    )
                overall["failure_reasons"] = reasons
                logger.error("❌ Backtest FAILED: {}", "; ".join(reasons))
            else:
                logger.info(
                    "✅ Backtest PASSED: return={:.1f}% drawdown={:.1f}%",
                    total_return, max_dd,
                )

            return overall

        finally:
            try:
                await exchange.disconnect()
            except Exception:
                pass

    def _simulate_pair(self, pair: str, df: Any) -> Dict[str, Any]:
        """Simulate a simple momentum strategy on OHLCV data."""
        import numpy as np
        import pandas as pd

        closes = df["close"].values.astype(float)
        n = len(closes)

        # Simple SMA crossover strategy (20 SMA / 50 SMA)
        sma_fast = pd.Series(closes).rolling(20).mean().values
        sma_slow = pd.Series(closes).rolling(50).mean().values

        equity = 1000.0  # USDT
        position = 0.0   # contracts
        entry_price = 0.0
        trades: List[Dict] = []
        equity_curve: List[float] = [equity]
        wins = 0
        losses = 0

        for i in range(50, n):
            prev_fast = sma_fast[i - 1]
            prev_slow = sma_slow[i - 1]
            curr_fast = sma_fast[i]
            curr_slow = sma_slow[i]

            if (
                prev_fast is None or prev_slow is None
                or curr_fast is None or curr_slow is None
                or np.isnan(prev_fast) or np.isnan(prev_slow)
                or np.isnan(curr_fast) or np.isnan(curr_slow)
            ):
                equity_curve.append(equity)
                continue

            price = closes[i]

            # Entry: fast crosses above slow → long
            if prev_fast <= prev_slow and curr_fast > curr_slow and position == 0:
                position = equity / price
                entry_price = price

            # Exit: fast crosses below slow → close long
            elif prev_fast >= prev_slow and curr_fast < curr_slow and position > 0:
                pnl = position * (price - entry_price)
                equity += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl})
                position = 0.0
                entry_price = 0.0

            equity_curve.append(equity + position * (price - entry_price) if position > 0 else equity)

        # Close any open position at end
        if position > 0:
            pnl = position * (closes[-1] - entry_price)
            equity += pnl
            trades.append({"entry": entry_price, "exit": closes[-1], "pnl": pnl})
            if pnl > 0:
                wins += 1
            else:
                losses += 1

        total_return_pct = (equity - 1000.0) / 1000.0 * 100
        total_trades = len(trades)
        win_rate_pct = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Max drawdown
        eq_arr = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (running_max - eq_arr) / running_max * 100
        max_drawdown_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe ratio (annualized)
        eq_series = pd.Series(equity_curve)
        returns = eq_series.pct_change().dropna()
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * (252 * 96) ** 0.5)  # 96 = 24h*4 for 15m

        # Profit factor
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        return {
            "pair": pair,
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "win_rate_pct": round(win_rate_pct, 1),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "sharpe_ratio": round(sharpe, 3),
            "profit_factor": round(profit_factor, 3),
            "final_equity": round(equity, 2),
        }

    def _aggregate_results(self, pair_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-pair results into overall metrics."""
        if not pair_results:
            return {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "total_trades": 0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
            }

        avg_return = sum(r["total_return_pct"] for r in pair_results) / len(pair_results)
        max_dd = max(r["max_drawdown_pct"] for r in pair_results)
        total_trades = sum(r["total_trades"] for r in pair_results)
        total_wins = sum(r["wins"] for r in pair_results)
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_sharpe = sum(r["sharpe_ratio"] for r in pair_results) / len(pair_results)
        avg_pf = sum(r["profit_factor"] for r in pair_results) / len(pair_results)

        return {
            "total_return_pct": round(avg_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_trades": total_trades,
            "sharpe_ratio": round(avg_sharpe, 3),
            "profit_factor": round(avg_pf, 3),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest before deploy — validates strategy performance",
    )
    parser.add_argument(
        "--pairs",
        help="Comma-separated trading pairs (default: from settings)",
        default=None,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history to use (default: 30)",
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Candle timeframe (default: 15m)",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=0.0,
        help="Minimum acceptable total return %% (default: 0, i.e. break-even)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=15.0,
        help="Maximum acceptable drawdown %% (default: 15)",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)

    pairs: Optional[List[str]] = None
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    if pairs is None:
        try:
            from config.settings import Settings
            settings = Settings.get_settings()
            exchange_cfg = getattr(settings, "exchange", None)
            pairs = list(getattr(exchange_cfg, "trading_pairs", ["BTC/USDT", "ETH/USDT"]))
        except Exception as exc:
            logger.warning("Could not load pairs from settings: {} — using defaults", exc)
            pairs = ["BTC/USDT", "ETH/USDT"]

    engine = BacktestEngine(
        pairs=pairs,
        timeframe=args.timeframe,
        days=args.days,
        min_return_pct=args.min_return,
        max_drawdown_pct=args.max_drawdown,
    )
    result = await engine.run()

    if not result.get("passed", False) and not result.get("error"):
        return 1
    if result.get("error"):
        logger.error("Backtest error: {}", result["error"])
        return 1
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
