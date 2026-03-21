"""Integration tests — cross-module interactions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.data_loader import HistoricalDataLoader
from backtest.performance_metrics import PerformanceMetrics
from backtest.simulator import TradeSimulator
from config.settings import Settings

# ── Settings load ─────────────────────────────────────────────────────────


class TestSettingsLoad:
    def test_settings_load_with_defaults(self):
        """Settings can be instantiated without a .env file using defaults."""
        settings = Settings(
            TRADING_MODE="paper",
            SECRET_KEY="integration-test-secret-key-32!!",
        )
        assert settings.trading_mode == "paper"
        assert settings.is_paper_trading is True
        assert settings.is_live_trading is False

    def test_settings_risk_defaults(self):
        """Default risk settings fall within safe ranges."""
        settings = Settings(TRADING_MODE="paper")
        assert 0 < settings.risk.max_position_size_pct <= 100
        assert settings.risk.max_open_positions >= 1
        assert settings.risk.risk_reward_min >= 1.0

    def test_settings_exchange_defaults(self):
        """Default exchange settings have sensible values."""
        settings = Settings(TRADING_MODE="paper")
        assert len(settings.exchange.trading_pairs) > 0
        assert settings.exchange.default_leverage >= 1
        assert settings.exchange.order_type in ("limit", "market")


# ── PerformanceMetrics — full calculation ─────────────────────────────────


def _make_trades(n_wins: int, n_losses: int) -> list:
    """Build a list of sample trade dicts."""
    trades = []
    for i in range(n_wins):
        entry = datetime(2024, 1, 1) + timedelta(days=i)
        exit_ = entry + timedelta(hours=6)
        trades.append(
            {
                "id": f"win-{i}",
                "symbol": "BTC/USDT",
                "pnl": 100.0,
                "pnl_pct": 2.0,
                "entry_time": entry,
                "exit_time": exit_,
            }
        )
    for i in range(n_losses):
        entry = datetime(2024, 1, 1) + timedelta(days=n_wins + i)
        exit_ = entry + timedelta(hours=3)
        trades.append(
            {
                "id": f"loss-{i}",
                "symbol": "ETH/USDT",
                "pnl": -50.0,
                "pnl_pct": -1.0,
                "entry_time": entry,
                "exit_time": exit_,
            }
        )
    return trades


class TestPerformanceMetricsFull:
    def test_full_metrics_calculation(self):
        """calculate_all returns a complete metrics dict for a set of trades."""
        pm = PerformanceMetrics()
        trades = _make_trades(n_wins=7, n_losses=3)
        equity = [10_000.0 + (i * 70 - (3 - i) * 20) for i in range(10)]
        metrics = pm.calculate_all(trades, equity_curve=equity)

        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown_pct" in metrics
        assert "total_trades" in metrics
        assert metrics["total_trades"] == 10

    def test_win_rate_correct(self):
        """Win rate equals n_wins / total for a balanced sample."""
        pm = PerformanceMetrics()
        trades = _make_trades(n_wins=6, n_losses=4)
        equity = [10_000 + i * 30 for i in range(10)]
        metrics = pm.calculate_all(trades, equity)
        assert metrics["win_rate"] == pytest.approx(0.6, abs=1e-6)

    def test_empty_trades_returns_zeroed_metrics(self):
        """Empty trade list returns zeroed metrics (no exception)."""
        pm = PerformanceMetrics()
        metrics = pm.calculate_all([], equity_curve=[10_000, 10_000])
        assert metrics["total_trades"] == 0
        assert metrics["sharpe_ratio"] == 0.0

    def test_monthly_returns_aggregated(self):
        """monthly_returns groups trades by YYYY-MM correctly."""
        pm = PerformanceMetrics()
        trades = [
            {"pnl": 50.0, "pnl_pct": 1.0, "exit_time": datetime(2024, 3, 10)},
            {"pnl": 30.0, "pnl_pct": 0.5, "exit_time": datetime(2024, 3, 20)},
            {"pnl": -20.0, "pnl_pct": -0.5, "exit_time": datetime(2024, 4, 5)},
        ]
        result = pm.monthly_returns(trades)
        assert "2024-03" in result
        assert result["2024-03"] == pytest.approx(1.5, abs=1e-6)


# ── HistoricalDataLoader ──────────────────────────────────────────────────


class TestBacktesterDataLoaderInit:
    def test_data_loader_initializes(self, tmp_path):
        """HistoricalDataLoader initializes without raising."""
        loader = HistoricalDataLoader(exchange_id="binance", cache_dir=tmp_path)
        assert loader is not None

    def test_data_loader_creates_cache_dir(self, tmp_path):
        """HistoricalDataLoader creates the cache directory if it does not exist."""
        cache = tmp_path / "ohlcv_cache"
        HistoricalDataLoader(exchange_id="binance", cache_dir=cache)
        assert cache.exists()

    def test_data_loader_default_exchange(self, tmp_path):
        """Default exchange_id is 'binance'."""
        loader = HistoricalDataLoader(cache_dir=tmp_path)
        assert loader.exchange_id == "binance"


# ── TradeSimulator — entry / exit ─────────────────────────────────────────


class TestBacktestSimulatorEntryExit:
    def _ohlcv_row(self) -> dict:
        return {
            "timestamp": datetime(2024, 1, 15, 12, 0),
            "open": 45_000.0,
            "high": 46_000.0,
            "low": 44_500.0,
            "close": 45_800.0,
            "volume": 1234.5,
        }

    def test_simulate_entry_returns_position(self):
        """simulate_entry returns a dict with expected keys."""
        sim = TradeSimulator()
        signal = {"symbol": "BTC/USDT", "side": "long", "size": 0.1}
        pos = sim.simulate_entry(signal, self._ohlcv_row(), capital=10_000.0)
        assert "entry_price" in pos
        assert "quantity" in pos
        assert pos["status"] == "open"

    def test_simulate_entry_deducts_fee(self):
        """simulate_entry deducts a fee from the position."""
        sim = TradeSimulator()
        signal = {"symbol": "BTC/USDT", "side": "long", "size": 0.1}
        pos = sim.simulate_entry(signal, self._ohlcv_row(), capital=10_000.0, fee_rate=0.001)
        assert pos["entry_fee"] > 0.0

    def test_simulate_exit_closes_position(self):
        """simulate_exit returns a closed position with pnl calculated."""
        sim = TradeSimulator()
        signal = {"symbol": "BTC/USDT", "side": "long", "size": 0.1}
        position = sim.simulate_entry(signal, self._ohlcv_row(), capital=10_000.0)
        exit_row = {**self._ohlcv_row(), "close": 46_000.0, "open": 46_000.0}
        closed = sim.simulate_exit(position, exit_row, reason="take_profit")
        assert closed["status"] == "closed"
        assert "pnl" in closed

    def test_simulate_short_pnl_negative_on_rise(self):
        """A short position loses money when price rises."""
        sim = TradeSimulator()
        signal = {"symbol": "BTC/USDT", "side": "short", "size": 0.1}
        row = self._ohlcv_row()
        position = sim.simulate_entry(signal, row, capital=10_000.0)
        # Price rises after entry — short loses
        exit_row = {**row, "close": 47_000.0, "open": 47_000.0}
        closed = sim.simulate_exit(position, exit_row, reason="stop_loss")
        assert closed["pnl"] < 0
