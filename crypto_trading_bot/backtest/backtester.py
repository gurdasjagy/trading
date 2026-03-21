"""Main backtesting engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from backtest.data_loader import HistoricalDataLoader
from backtest.performance_metrics import PerformanceMetrics
from backtest.simulator import TradeSimulator
from core.exceptions import BacktestError


@dataclass
class BacktestResult:
    """Holds all results from a completed backtest run."""

    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    expectancy: float
    calmar_ratio: float
    avg_trade_duration_hours: float
    best_trade_pct: float
    worst_trade_pct: float
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """Main backtesting engine.

    Runs a full vectorised-style event-driven backtest over historical OHLCV
    data, delegating trade simulation to :class:`TradeSimulator` and metric
    calculation to :class:`PerformanceMetrics`.
    """

    def __init__(
        self,
        data_loader: Optional[HistoricalDataLoader] = None,
        exchange_id: str = "binance",
    ) -> None:
        self._data_loader = data_loader or HistoricalDataLoader(exchange_id=exchange_id)
        self._simulator = TradeSimulator()
        self._metrics_calc = PerformanceMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        strategy: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10_000.0,
    ) -> BacktestResult:
        """Run a full backtest of *strategy* on *symbol*.

        Args:
            strategy: Strategy instance with a ``generate_signal(row, capital)``
                method returning a signal dict or ``None``.
            symbol: Trading pair symbol (e.g. ``"BTC/USDT"``).
            start_date: Backtest start date.
            end_date: Backtest end date.
            initial_capital: Starting capital in quote currency.

        Returns:
            :class:`BacktestResult` with all performance metrics.

        Raises:
            BacktestError: If data loading or strategy execution fails.
        """
        strategy_name = getattr(strategy, "name", type(strategy).__name__)
        logger.info(
            "Backtest starting: {} on {} ({} → {}, capital={:.2f})",
            strategy_name,
            symbol,
            start_date.date(),
            end_date.date(),
            initial_capital,
        )

        # ── Load data ────────────────────────────────────────────────────
        timeframe = getattr(strategy, "timeframe", "1h")
        try:
            df = await self._data_loader.load(symbol, timeframe, start_date, end_date)
        except Exception as exc:
            raise BacktestError(f"Data loading failed: {exc}", symbol=symbol) from exc

        if df.empty:
            raise BacktestError(
                "No historical data available for the requested period.", symbol=symbol
            )

        # ── Event loop ───────────────────────────────────────────────────
        capital = initial_capital
        equity_curve: List[float] = [capital]
        trades: List[dict] = []
        open_position: Optional[dict] = None

        for ts, row in df.iterrows():
            ohlcv = row.to_dict()
            ohlcv["timestamp"] = ts

            # Check SL/TP on open position
            if open_position is not None:
                hit = self._simulator.check_sl_tp_hit(open_position, ohlcv)
                if hit:
                    closed = self._simulator.simulate_exit(open_position, ohlcv, reason=hit)
                    capital += closed["capital_used"] + closed["pnl"]
                    trades.append(closed)
                    equity_curve.append(capital)
                    open_position = None

            # Generate strategy signal (only when flat)
            if open_position is None:
                try:
                    signal = strategy.generate_signal(ohlcv, capital)
                except Exception as exc:
                    logger.warning("Strategy signal error at {}: {}", ts, exc)
                    signal = None

                if signal and signal.get("side") in ("long", "short"):
                    signal.setdefault("symbol", symbol)
                    open_position = self._simulator.simulate_entry(signal, ohlcv, capital)
                    capital -= open_position["capital_used"]
            elif open_position is not None:
                # Check if strategy wants to exit early
                try:
                    exit_signal = getattr(strategy, "should_exit", None)
                    if exit_signal is not None and exit_signal(open_position, ohlcv):
                        closed = self._simulator.simulate_exit(
                            open_position, ohlcv, reason="signal"
                        )
                        capital += closed["pnl"] + closed["capital_used"]
                        trades.append(closed)
                        equity_curve.append(capital)
                        open_position = None
                except Exception as exc:
                    logger.warning("Strategy exit-check error at {}: {}", ts, exc)

        # Close any still-open position at last bar
        if open_position is not None and not df.empty:
            last_row = df.iloc[-1].to_dict()
            last_row["timestamp"] = df.index[-1]
            closed = self._simulator.simulate_exit(open_position, last_row, reason="end_of_data")
            capital += closed["pnl"] + closed["capital_used"]
            trades.append(closed)
            equity_curve.append(capital)

        # ── Metrics ──────────────────────────────────────────────────────
        metrics = self._metrics_calc.calculate_all(trades, equity_curve)

        durations = [t.get("duration_hours", 0.0) for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
        best_trade = max(pnl_pcts) if pnl_pcts else 0.0
        worst_trade = min(pnl_pcts) if pnl_pcts else 0.0
        final_capital = equity_curve[-1] if equity_curve else initial_capital
        total_return_pct = (final_capital - initial_capital) / initial_capital * 100

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            win_rate=metrics["win_rate"],
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            losing_trades=metrics["losing_trades"],
            profit_factor=metrics["profit_factor"],
            expectancy=metrics["expectancy"],
            calmar_ratio=metrics["calmar_ratio"],
            avg_trade_duration_hours=avg_duration,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=metrics.get("monthly_returns", {}),
        )

        self._print_report(result)
        return result

    async def optimize(
        self,
        strategy_class: Type,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        param_ranges: dict,
        method: str = "grid",
    ) -> dict:
        """Optimise *strategy_class* parameters over the given date range.

        Args:
            strategy_class: Strategy class to optimise.
            symbol: Trading pair.
            start_date: Start date.
            end_date: End date.
            param_ranges: Parameter search space.
            method: ``"grid"`` or ``"random"``.

        Returns:
            Best parameter set and associated metrics.
        """
        from backtest.optimizer import StrategyOptimizer  # lazy import

        optimizer = StrategyOptimizer(data_loader=self._data_loader)
        return await optimizer.optimize(
            strategy_class, symbol, start_date, end_date, param_ranges, method
        )

    async def walk_forward(
        self,
        strategy: Any,
        symbol: str,
        window_days: int = 90,
        step_days: int = 30,
        initial_capital: float = 10_000.0,
    ) -> List[BacktestResult]:
        """Walk-forward analysis over rolling windows.

        Args:
            strategy: Strategy instance.
            symbol: Trading pair.
            window_days: Length of each test window in days.
            step_days: Days to advance between windows.
            initial_capital: Starting capital for each window.

        Returns:
            List of :class:`BacktestResult` for each window.
        """
        # Determine the overall date range from the data loader
        now = datetime.now(timezone.utc)
        overall_start = now - timedelta(days=365)
        overall_end = now - timedelta(days=1)

        results: List[BacktestResult] = []
        window_start = overall_start

        while window_start + timedelta(days=window_days) <= overall_end:
            window_end = window_start + timedelta(days=window_days)
            logger.info("Walk-forward window: {} → {}", window_start.date(), window_end.date())
            try:
                result = await self.run(strategy, symbol, window_start, window_end, initial_capital)
                results.append(result)
            except BacktestError as exc:
                logger.warning("Walk-forward window failed: {}", exc)
            window_start += timedelta(days=step_days)

        logger.info("Walk-forward complete: {} windows", len(results))
        return results

    async def monte_carlo(
        self,
        trades: List[dict],
        n_simulations: int = 1000,
        initial_capital: float = 10_000.0,
    ) -> dict:
        """Monte Carlo simulation by resampling trade returns.

        Args:
            trades: Historical trades with ``pnl_pct`` fields.
            n_simulations: Number of resampled equity curves to generate.
            initial_capital: Starting capital for each simulation.

        Returns:
            Dict with ``median_final_capital``, ``p5_final_capital``,
            ``p95_final_capital``, ``max_drawdown_mean``, and
            ``probability_of_ruin`` (capital < 50% of initial).
        """
        if not trades:
            logger.warning("No trades provided for Monte Carlo simulation")
            return {}

        returns = [t.get("pnl_pct", 0.0) / 100.0 for t in trades]
        final_capitals: List[float] = []
        max_drawdowns: List[float] = []
        ruin_count = 0

        for _ in range(n_simulations):
            simulated = random.choices(returns, k=len(returns))
            equity = [initial_capital]
            for r in simulated:
                equity.append(equity[-1] * (1 + r))
            final_capitals.append(equity[-1])
            max_drawdowns.append(self._metrics_calc.max_drawdown(equity))
            if equity[-1] < initial_capital * 0.5:
                ruin_count += 1

        final_capitals.sort()
        p5_idx = int(0.05 * n_simulations)
        p95_idx = int(0.95 * n_simulations)

        return {
            "n_simulations": n_simulations,
            "median_final_capital": final_capitals[n_simulations // 2],
            "p5_final_capital": final_capitals[p5_idx],
            "p95_final_capital": final_capitals[p95_idx],
            "max_drawdown_mean": sum(max_drawdowns) / len(max_drawdowns),
            "probability_of_ruin": ruin_count / n_simulations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_report(self, result: BacktestResult) -> None:
        """Log a human-readable summary of *result*."""
        logger.info(
            "\n"
            "═══════════════════════ BACKTEST REPORT ════════════════════════\n"
            "  Strategy   : {strategy}\n"
            "  Symbol     : {symbol}\n"
            "  Period     : {start} → {end}\n"
            "  Capital    : {initial:.2f} → {final:.2f} ({ret:+.2f}%)\n"
            "  Trades     : {total} (W={wins} / L={losses})\n"
            "  Win Rate   : {wr:.1%}   Profit Factor: {pf:.2f}\n"
            "  Sharpe     : {sharpe:.3f}   Sortino: {sortino:.3f}\n"
            "  Max DD     : {dd:.2f}%   Calmar: {calmar:.3f}\n"
            "  Expectancy : {exp:.4f}   Avg Duration: {dur:.1f}h\n"
            "  Best Trade : {best:+.2f}%   Worst: {worst:+.2f}%\n"
            "═══════════════════════════════════════════════════════════════",
            strategy=result.strategy_name,
            symbol=result.symbol,
            start=result.start_date.date(),
            end=result.end_date.date(),
            initial=result.initial_capital,
            final=result.final_capital,
            ret=result.total_return_pct,
            total=result.total_trades,
            wins=result.winning_trades,
            losses=result.losing_trades,
            wr=result.win_rate,
            pf=result.profit_factor,
            sharpe=result.sharpe_ratio,
            sortino=result.sortino_ratio,
            dd=result.max_drawdown_pct,
            calmar=result.calmar_ratio,
            exp=result.expectancy,
            dur=result.avg_trade_duration_hours,
            best=result.best_trade_pct,
            worst=result.worst_trade_pct,
        )
