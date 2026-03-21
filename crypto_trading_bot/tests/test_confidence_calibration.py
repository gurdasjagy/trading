"""Tests for confidence calibration and signal conflict resolution."""

from __future__ import annotations

import pytest


def test_calibrator_no_data_returns_raw():
    """With no historical data, calibrator should return raw confidence unchanged."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()
    assert cal.calibrate("test_strategy", 0.7) == 0.7


def test_calibrator_below_min_trades_returns_raw():
    """With fewer than 20 trades, calibrator should still return raw confidence."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()
    # Record only 10 trades (below _min_trades_for_calibration = 20)
    for _ in range(8):
        cal.record_outcome("young_strategy", 0.7, won=True)
    for _ in range(2):
        cal.record_outcome("young_strategy", 0.7, won=False)

    result = cal.calibrate("young_strategy", 0.7)
    assert result == 0.7


def test_calibrator_adjusts_overconfident_strategy():
    """Strategy with 30% win rate but avg 0.7 confidence should be scaled down."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()

    # Record 30 trades: 9 wins (30 % win rate), all with confidence 0.7
    for _ in range(9):
        cal.record_outcome("overconfident", 0.7, won=True)
    for _ in range(21):
        cal.record_outcome("overconfident", 0.7, won=False)

    calibrated = cal.calibrate("overconfident", 0.7)
    # Expected: 0.3 / 0.7 * 0.7 = 0.3
    assert calibrated < 0.5
    assert calibrated == pytest.approx(0.3, abs=0.05)


def test_calibrator_boosts_underconfident_strategy():
    """Strategy with 80% win rate but avg 0.5 confidence should be scaled up."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()

    # 24 wins, 6 losses (80 % win rate), all with confidence 0.5
    for _ in range(24):
        cal.record_outcome("underconfident", 0.5, won=True)
    for _ in range(6):
        cal.record_outcome("underconfident", 0.5, won=False)

    calibrated = cal.calibrate("underconfident", 0.5)
    # Expected: 0.8 / 0.5 * 0.5 = 0.8
    assert calibrated > 0.7


def test_calibrator_clamps_to_one():
    """Calibrated confidence should never exceed 1.0."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()

    # Perfect win rate with low entry confidence → would scale above 1.0 without clamping
    for _ in range(25):
        cal.record_outcome("perfect", 0.3, won=True)

    calibrated = cal.calibrate("perfect", 0.9)
    assert calibrated <= 1.0


def test_calibrator_clamps_to_zero():
    """Calibrated confidence should never go below 0.0."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()

    # All losses — calibration factor = 0 / avg_conf = 0
    for _ in range(20):
        cal.record_outcome("terrible", 0.9, won=False)

    calibrated = cal.calibrate("terrible", 0.9)
    assert calibrated >= 0.0


def test_calibrator_exact_match():
    """Strategy with win rate exactly equal to avg confidence returns unchanged value."""
    from strategy.strategy_manager import ConfidenceCalibrator

    cal = ConfidenceCalibrator()

    # 50% win rate, 0.5 avg confidence → calibration factor = 1.0
    for _ in range(10):
        cal.record_outcome("balanced", 0.5, won=True)
    for _ in range(10):
        cal.record_outcome("balanced", 0.5, won=False)

    calibrated = cal.calibrate("balanced", 0.5)
    assert calibrated == pytest.approx(0.5, abs=0.01)


def test_signal_conflict_resolution():
    """Conflicting LONG and SHORT signals for same symbol should keep the strongest."""
    from strategy.strategy_manager import StrategyManager

    mgr = StrategyManager.__new__(StrategyManager)

    signals = [
        {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.6, "strategy": "trend"},
        {"symbol": "BTC/USDT", "direction": "short", "confidence": 0.8, "strategy": "mean_rev"},
        {"symbol": "ETH/USDT", "direction": "long", "confidence": 0.7, "strategy": "trend"},
    ]

    resolved = mgr._resolve_conflicts(signals)

    btc_signals = [s for s in resolved if s["symbol"] == "BTC/USDT"]
    assert len(btc_signals) == 1
    assert btc_signals[0]["direction"] == "short"
    assert btc_signals[0]["confidence"] == 0.8

    eth_signals = [s for s in resolved if s["symbol"] == "ETH/USDT"]
    assert len(eth_signals) == 1


def test_signal_conflict_resolution_no_conflict():
    """Signals with only one direction per symbol should all be kept."""
    from strategy.strategy_manager import StrategyManager

    mgr = StrategyManager.__new__(StrategyManager)

    signals = [
        {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.6, "strategy": "trend"},
        {"symbol": "ETH/USDT", "direction": "short", "confidence": 0.8, "strategy": "mean_rev"},
    ]

    resolved = mgr._resolve_conflicts(signals)
    assert len(resolved) == 2


def test_signal_conflict_resolution_empty():
    """Empty signal list should return empty list."""
    from strategy.strategy_manager import StrategyManager

    mgr = StrategyManager.__new__(StrategyManager)
    resolved = mgr._resolve_conflicts([])
    assert resolved == []
