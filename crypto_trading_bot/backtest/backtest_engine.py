"""Backtesting engine — downloads OHLCV data via CCXT, caches as CSV, and simulates trades."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from loguru import logger

from backtest.data_loader import HistoricalDataLoader
from backtest.performance_metrics import PerformanceMetrics
from backtest.simulator import TradeSimulator
from core.exceptions import BacktestError

_CSV_CACHE_DIR = Path("data/backtest_cache")
_REPORTS_DIR = Path("data/reports")


class BacktestEngine:
    """Event-driven backtesting engine with CCXT data download and CSV caching.

    Feeds accumulated OHLCV data to a strategy's ``analyze()`` method on each
    candle, then simulates fills with configurable slippage and fees via
    :class:`TradeSimulator`.

    Usage::

        engine = BacktestEngine()
        result = await engine.run(
            strategy=MomentumStrategy(symbols=["BTC/USDT"]),
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
        )
    """

    def __init__(
        self,
        exchange_id: str = "gateio",
        fee_rate: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        self._data_loader = HistoricalDataLoader(exchange_id=exchange_id)
        self._simulator = TradeSimulator()
        self._metrics_calc = PerformanceMetrics()
        self._fee_rate = fee_rate
        self._slippage_pct = slippage_pct
        _CSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        strategy: Any,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10_000.0,
    ) -> "BacktestResult":
        """Run a full event-driven backtest.

        Args:
            strategy: Strategy instance with an ``analyze(ohlcv_df, symbol)``
                method returning a signal dict or ``None``.
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            timeframe: OHLCV timeframe (e.g. ``"1h"``).
            start_date: Backtest start date (UTC).
            end_date: Backtest end date (UTC).
            initial_capital: Starting capital in quote currency.

        Returns:
            :class:`BacktestResult` with equity curve, trades, and metrics.

        Raises:
            BacktestError: When data cannot be loaded or the strategy fails critically.
        """
        strategy_name = getattr(strategy, "name", type(strategy).__name__)
        logger.info(
            "BacktestEngine starting: {} | {} | {} | {} → {} | capital={:.2f}",
            strategy_name,
            symbol,
            timeframe,
            start_date.date(),
            end_date.date(),
            initial_capital,
        )

        df = await self._load_ohlcv(symbol, timeframe, start_date, end_date)
        if df.empty:
            raise BacktestError(
                "No historical data available for the requested period.",
                symbol=symbol,
            )

        capital = initial_capital
        equity_curve: List[float] = [capital]
        trades: List[dict] = []
        open_position: Optional[dict] = None
        df_rows = list(df.iterrows())

        for idx, (ts, row) in enumerate(df_rows):
            # Slice the original DataFrame up to and including the current bar
            # to avoid repeated pd.concat() copies inside the loop.
            candle_buffer = df.iloc[: idx + 1]

            ohlcv_dict = row.to_dict()
            ohlcv_dict["timestamp"] = ts

            # ── Check SL/TP on open position ────────────────────────────
            if open_position is not None:
                hit = self._simulator.check_sl_tp_hit(open_position, ohlcv_dict)
                if hit:
                    closed = self._simulator.simulate_exit(
                        open_position, ohlcv_dict, reason=hit, fee_rate=self._fee_rate
                    )
                    capital += closed["pnl"] + closed["capital_used"]
                    trades.append(closed)
                    equity_curve.append(capital)
                    open_position = None

            # ── Generate signal when flat ────────────────────────────────
            if open_position is None:
                signal = self._get_signal(strategy, candle_buffer, symbol)
                if signal and signal.get("side") in ("long", "short"):
                    signal.setdefault("symbol", symbol)
                    open_position = self._simulator.simulate_entry(
                        signal, ohlcv_dict, capital, fee_rate=self._fee_rate
                    )
                    capital -= open_position["capital_used"]
            else:
                # ── Check strategy exit signal ───────────────────────────
                try:
                    should_exit_fn = getattr(strategy, "should_exit", None)
                    if should_exit_fn is not None and should_exit_fn(open_position, ohlcv_dict):
                        closed = self._simulator.simulate_exit(
                            open_position, ohlcv_dict, reason="signal", fee_rate=self._fee_rate
                        )
                        capital += closed["pnl"] + closed["capital_used"]
                        trades.append(closed)
                        equity_curve.append(capital)
                        open_position = None
                except Exception as exc:
                    logger.warning("Strategy exit-check error at {}: {}", ts, exc)

        # ── Close any remaining open position at last bar ────────────────
        if open_position is not None and not df.empty:
            last_row = df.iloc[-1].to_dict()
            last_row["timestamp"] = df.index[-1]
            closed = self._simulator.simulate_exit(
                open_position, last_row, reason="end_of_data", fee_rate=self._fee_rate
            )
            capital += closed["pnl"] + closed["capital_used"]
            trades.append(closed)
            equity_curve.append(capital)

        # ── Build result ─────────────────────────────────────────────────
        return self._build_result(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trades=trades,
            equity_curve=equity_curve,
        )

    # ------------------------------------------------------------------
    # OHLCV data loading with CSV cache
    # ------------------------------------------------------------------

    async def _load_ohlcv(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame, trying CSV cache first."""
        csv_path = self._csv_path(symbol, timeframe, start_date, end_date)
        if csv_path.exists():
            logger.info("Loading OHLCV from CSV cache: {}", csv_path)
            try:
                df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True)
                if not df.empty:
                    logger.info(
                        "Loaded {} bars from CSV cache for {}/{}", len(df), symbol, timeframe
                    )
                    return df
            except Exception as exc:
                logger.warning("CSV cache read failed ({}), re-fetching: {}", csv_path, exc)

        logger.info(
            "Fetching OHLCV from exchange for {}/{} ({} → {})",
            symbol,
            timeframe,
            start_date.date(),
            end_date.date(),
        )
        try:
            df = await self._data_loader.fetch_from_exchange(
                symbol, timeframe, start_date, end_date
            )
        except Exception as exc:
            raise BacktestError(f"Data download failed: {exc}", symbol=symbol) from exc

        if not df.empty:
            self._save_csv(df, csv_path)

        return df

    def _csv_path(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Path:
        """Return the CSV cache file path for the given parameters."""
        safe_symbol = re.sub(r"[^A-Za-z0-9_]", "_", symbol)
        safe_tf = re.sub(r"[^A-Za-z0-9_]", "_", timeframe)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        filename = f"{safe_symbol}_{safe_tf}_{start_str}_{end_str}.csv"
        return _CSV_CACHE_DIR / filename

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        """Persist OHLCV *df* to a CSV file at *path*."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df_out = df.reset_index()
            df_out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
            logger.debug("Saved {} OHLCV rows to {}", len(df), path)
        except Exception as exc:
            logger.error("Failed to save CSV cache to {}: {}", path, exc)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_signal(strategy: Any, candle_buffer: pd.DataFrame, symbol: str) -> Optional[dict]:
        """Call ``strategy.analyze(candle_buffer, symbol)`` and return signal dict."""
        try:
            result = strategy.analyze(candle_buffer, symbol)
            return result  # expected to be a dict or None
        except Exception as exc:
            logger.warning("Strategy analyze() error: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        trades: List[dict],
        equity_curve: List[float],
    ) -> "BacktestResult":
        """Compute metrics and build the :class:`BacktestResult`."""
        metrics = self._metrics_calc.calculate_all(trades, equity_curve)

        durations = [t.get("duration_hours", 0.0) for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
        best_trade = max(pnl_pcts) if pnl_pcts else 0.0
        worst_trade = min(pnl_pcts) if pnl_pcts else 0.0
        final_capital = equity_curve[-1] if equity_curve else initial_capital
        total_return_pct = (final_capital - initial_capital) / initial_capital * 100

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            expectancy=metrics["expectancy"],
            calmar_ratio=metrics["calmar_ratio"],
            avg_trade_duration_hours=avg_duration,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            losing_trades=metrics["losing_trades"],
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=metrics.get("monthly_returns", {}),
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Holds all results from a completed :class:`BacktestEngine` run."""

    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    expectancy: float
    calmar_ratio: float
    avg_trade_duration_hours: float
    best_trade_pct: float
    worst_trade_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    monthly_returns: dict = field(default_factory=dict)
