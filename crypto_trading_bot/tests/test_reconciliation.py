"""Tests for the startup state reconciliation module."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.reconciliation import StateReconciler


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_position(symbol: str, side: str = "long", amount: float = 0.1, entry: float = 50000.0):
    """Return a minimal mock exchange Position object."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.side = MagicMock()
    pos.side.value = side
    pos.amount = amount
    pos.entry_price = entry
    pos.current_price = entry
    return pos


def _make_db_record(
    symbol: str,
    strategy: str = "trend_following",
    stop_loss: float = 48000.0,
    take_profit=None,
):
    """Return a minimal mock ActivePosition DB record."""
    rec = MagicMock()
    rec.id = 1
    rec.symbol = symbol
    rec.strategy_name = strategy
    rec.stop_loss = stop_loss
    rec.take_profit = take_profit if take_profit is not None else [52000.0, 55000.0]
    return rec


def _make_tracker(symbol: str):
    """Return a minimal mock PositionTracker."""
    t = MagicMock()
    t.symbol = symbol
    t.stop_loss = None
    t.take_profit = []
    t.strategy = "unknown"
    return t


def _make_session_ctx(session: MagicMock):
    """Return an async-context-manager mock that yields *session*."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_result(scalar=None, scalars_list=None):
    """Return a MagicMock that mimics a SQLAlchemy async execute result."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    scalars_proxy = MagicMock()
    scalars_proxy.all.return_value = scalars_list or []
    result.scalars.return_value = scalars_proxy
    return result


# ---------------------------------------------------------------------------
# reconcile_state — happy path (all positions matched)
# ---------------------------------------------------------------------------


class TestReconcileStateMatched:
    @pytest.fixture
    def reconciler(self):
        exchange = AsyncMock()
        position_manager = AsyncMock()
        alert_manager = AsyncMock()
        settings = MagicMock()
        settings.risk.auto_close_orphaned_positions = False
        return StateReconciler(
            exchange=exchange,
            position_manager=position_manager,
            alert_manager=alert_manager,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_matched_position_restores_metadata(self, reconciler):
        """A position found in both exchange and DB should have SL/TP restored."""
        symbol = "BTC/USDT"
        live_pos = _make_position(symbol)
        db_rec = _make_db_record(symbol, stop_loss=48000.0, take_profit=[52000.0])
        tracker = _make_tracker(symbol)

        reconciler._exchange.get_positions.return_value = [live_pos]
        reconciler._position_manager.sync_positions = AsyncMock()
        reconciler._position_manager.get_position = AsyncMock(return_value=tracker)

        # First execute call returns the per-symbol record; second returns all records
        session = MagicMock()
        session.execute = AsyncMock(
            side_effect=[
                _make_result(scalar=db_rec),   # SELECT WHERE symbol = ?
                _make_result(scalars_list=[db_rec]),  # SELECT all (stale cleanup)
            ]
        )

        with patch("core.reconciliation.get_async_session", return_value=_make_session_ctx(session)):
            summary = await reconciler.reconcile_state()

        assert summary["live_positions"] == 1
        assert summary["matched"] == 1
        assert summary["orphaned"] == 0
        assert len(summary["errors"]) == 0
        assert tracker.stop_loss == db_rec.stop_loss
        assert tracker.strategy == db_rec.strategy_name

    @pytest.mark.asyncio
    async def test_no_live_positions_cleans_stale_db(self, reconciler):
        """When exchange returns no positions, stale DB records should be removed."""
        db_rec = _make_db_record("ETH/USDT")

        reconciler._exchange.get_positions.return_value = []
        reconciler._position_manager.sync_positions = AsyncMock()

        session = MagicMock()
        session.execute = AsyncMock(return_value=_make_result(scalars_list=[db_rec]))
        session.get = AsyncMock(return_value=db_rec)

        with patch("core.reconciliation.get_async_session", return_value=_make_session_ctx(session)):
            summary = await reconciler.reconcile_state()

        assert summary["live_positions"] == 0
        assert summary["matched"] == 0
        assert summary["stale_db"] == 1
        # Verify that the stale record was retrieved and deleted from the session
        session.get.assert_awaited_once()
        session.delete.assert_called_once_with(db_rec)


# ---------------------------------------------------------------------------
# reconcile_state — orphaned positions
# ---------------------------------------------------------------------------


class TestReconcileStateOrphaned:
    @pytest.fixture
    def reconciler_no_autoclose(self):
        exchange = AsyncMock()
        position_manager = AsyncMock()
        alert_manager = AsyncMock()
        settings = MagicMock()
        settings.risk.auto_close_orphaned_positions = False
        return StateReconciler(
            exchange=exchange,
            position_manager=position_manager,
            alert_manager=alert_manager,
            settings=settings,
        )

    @pytest.fixture
    def reconciler_autoclose(self):
        exchange = AsyncMock()
        position_manager = AsyncMock()
        alert_manager = AsyncMock()
        settings = MagicMock()
        settings.risk.auto_close_orphaned_positions = True
        return StateReconciler(
            exchange=exchange,
            position_manager=position_manager,
            alert_manager=alert_manager,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_orphaned_position_sends_alert(self, reconciler_no_autoclose):
        """An orphaned position (no DB record) should trigger an alert."""
        symbol = "SOL/USDT"
        live_pos = _make_position(symbol)

        reconciler_no_autoclose._exchange.get_positions.return_value = [live_pos]
        reconciler_no_autoclose._position_manager.sync_positions = AsyncMock()

        session = MagicMock()
        session.execute = AsyncMock(
            side_effect=[
                _make_result(scalar=None),         # no DB record → orphaned
                _make_result(scalars_list=[]),      # stale cleanup
            ]
        )

        with patch("core.reconciliation.get_async_session", return_value=_make_session_ctx(session)):
            summary = await reconciler_no_autoclose.reconcile_state()

        assert summary["orphaned"] == 1
        assert summary["matched"] == 0
        reconciler_no_autoclose._alert_manager.send_alert.assert_awaited_once()
        # Should NOT have auto-closed
        reconciler_no_autoclose._exchange.close_position.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_orphaned_position_auto_close(self, reconciler_autoclose):
        """When auto_close_orphaned_positions=True, orphaned positions are closed."""
        symbol = "BNB/USDT"
        live_pos = _make_position(symbol)

        reconciler_autoclose._exchange.get_positions.return_value = [live_pos]
        reconciler_autoclose._position_manager.sync_positions = AsyncMock()

        session = MagicMock()
        session.execute = AsyncMock(
            side_effect=[
                _make_result(scalar=None),
                _make_result(scalars_list=[]),
            ]
        )

        with patch("core.reconciliation.get_async_session", return_value=_make_session_ctx(session)):
            summary = await reconciler_autoclose.reconcile_state()

        assert summary["orphaned"] == 1
        reconciler_autoclose._exchange.close_position.assert_awaited_once_with(symbol)


# ---------------------------------------------------------------------------
# reconcile_state — exchange fetch failure
# ---------------------------------------------------------------------------


class TestReconcileStateFetchFailure:
    @pytest.mark.asyncio
    async def test_exchange_fetch_failure_returns_error_summary(self):
        """When the exchange raises, reconcile_state returns an error summary."""
        exchange = AsyncMock()
        exchange.get_positions.side_effect = RuntimeError("exchange down")
        position_manager = AsyncMock()

        reconciler = StateReconciler(
            exchange=exchange,
            position_manager=position_manager,
        )
        summary = await reconciler.reconcile_state()

        assert summary["live_positions"] == 0
        assert len(summary["errors"]) >= 1
        assert "exchange down" in summary["errors"][0]


# ---------------------------------------------------------------------------
# _restore_position_metadata
# ---------------------------------------------------------------------------


class TestRestorePositionMetadata:
    @pytest.mark.asyncio
    async def test_restore_sets_stop_loss_and_take_profit(self):
        tracker = _make_tracker("BTC/USDT")
        position_manager = AsyncMock()
        position_manager.get_position.return_value = tracker

        db_rec = _make_db_record(
            "BTC/USDT", strategy="scalping", stop_loss=47000.0, take_profit=[51000.0, 53000.0]
        )

        reconciler = StateReconciler(
            exchange=AsyncMock(),
            position_manager=position_manager,
        )
        await reconciler._restore_position_metadata("BTC/USDT", db_rec)

        assert tracker.stop_loss == 47000.0
        assert tracker.take_profit == [51000.0, 53000.0]
        assert tracker.strategy == "scalping"

    @pytest.mark.asyncio
    async def test_restore_handles_scalar_take_profit(self):
        """take_profit stored as a scalar float should be wrapped in a list."""
        tracker = _make_tracker("ETH/USDT")
        position_manager = AsyncMock()
        position_manager.get_position.return_value = tracker

        db_rec = MagicMock()
        db_rec.stop_loss = 3000.0
        db_rec.take_profit = 3500.0  # scalar, not list
        db_rec.strategy_name = "trend"

        reconciler = StateReconciler(
            exchange=AsyncMock(),
            position_manager=position_manager,
        )
        await reconciler._restore_position_metadata("ETH/USDT", db_rec)

        assert tracker.take_profit == [3500.0]

    @pytest.mark.asyncio
    async def test_restore_handles_missing_tracker(self):
        """Should not raise if the position is missing from the in-memory manager."""
        position_manager = AsyncMock()
        position_manager.get_position.return_value = None

        db_rec = _make_db_record("XRP/USDT")

        reconciler = StateReconciler(
            exchange=AsyncMock(),
            position_manager=position_manager,
        )
        # Should complete without raising
        await reconciler._restore_position_metadata("XRP/USDT", db_rec)

