"""Add forex trade history, daily performance, and active positions tables.

Revision ID: 0001_add_forex_tables
Revises:
Create Date: 2026-03-15

Non-destructive migration — adds three new forex_ prefixed tables without
touching any existing crypto tables.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_add_forex_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── forex_trade_history ───────────────────────────────────────────────
    op.create_table(
        "forex_trade_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("order_id", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("lot_size", sa.Float, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("stop_loss_price", sa.Float, nullable=True),
        sa.Column("take_profit_price", sa.Float, nullable=True),
        sa.Column("stop_loss_pips", sa.Float, nullable=True),
        sa.Column("take_profit_pips", sa.Float, nullable=True),
        sa.Column("pip_pnl", sa.Float, nullable=True),
        sa.Column("usd_pnl", sa.Float, nullable=True),
        sa.Column("spread_at_entry", sa.Float, nullable=True),
        sa.Column("leverage", sa.Integer, default=20),
        sa.Column("margin_used", sa.Float, nullable=True),
        sa.Column("swap_cost", sa.Float, default=0.0),
        sa.Column("commission", sa.Float, default=0.0),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("session", sa.String(20), nullable=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("exit_reason", sa.String(50), nullable=True),
        sa.Column("max_favorable_pips", sa.Float, nullable=True),
        sa.Column("max_adverse_pips", sa.Float, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_forex_trade_history_order_id", "forex_trade_history", ["order_id"])
    op.create_index("ix_forex_trade_history_symbol", "forex_trade_history", ["symbol"])
    op.create_index("ix_forex_trade_history_strategy", "forex_trade_history", ["strategy"])
    op.create_index("ix_forex_trade_history_session", "forex_trade_history", ["session"])
    op.create_index("ix_forex_trade_history_entry_time", "forex_trade_history", ["entry_time"])
    op.create_index("ix_forex_trade_history_exit_time", "forex_trade_history", ["exit_time"])

    # ── forex_daily_performance ───────────────────────────────────────────
    op.create_table(
        "forex_daily_performance",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("starting_equity", sa.Float, nullable=False),
        sa.Column("ending_equity", sa.Float, nullable=True),
        sa.Column("total_pnl_usd", sa.Float, nullable=True),
        sa.Column("total_pnl_pips", sa.Float, nullable=True),
        sa.Column("total_trades", sa.Integer, default=0),
        sa.Column("wins", sa.Integer, default=0),
        sa.Column("losses", sa.Integer, default=0),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("profit_factor", sa.Float, nullable=True),
        sa.Column("max_drawdown_pct", sa.Float, nullable=True),
        sa.Column("best_trade_pips", sa.Float, nullable=True),
        sa.Column("worst_trade_pips", sa.Float, nullable=True),
        sa.Column("avg_win_pips", sa.Float, nullable=True),
        sa.Column("avg_loss_pips", sa.Float, nullable=True),
        sa.Column("total_lots_traded", sa.Float, nullable=True),
        sa.Column("total_commission", sa.Float, default=0.0),
        sa.Column("total_swap", sa.Float, default=0.0),
        sa.Column("london_trades", sa.Integer, default=0),
        sa.Column("london_pnl_pips", sa.Float, default=0.0),
        sa.Column("ny_trades", sa.Integer, default=0),
        sa.Column("ny_pnl_pips", sa.Float, default=0.0),
        sa.Column("asian_trades", sa.Integer, default=0),
        sa.Column("asian_pnl_pips", sa.Float, default=0.0),
        sa.Column("sydney_trades", sa.Integer, default=0),
        sa.Column("sydney_pnl_pips", sa.Float, default=0.0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_forex_daily_performance_date", "forex_daily_performance", ["date"])

    # ── forex_active_positions ────────────────────────────────────────────
    op.create_table(
        "forex_active_positions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("order_id", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False, unique=True),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("lot_size", sa.Float, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("stop_loss_price", sa.Float, nullable=True),
        sa.Column("take_profit_prices", sa.JSON, nullable=True),
        sa.Column("leverage", sa.Integer, default=20),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("session", sa.String(20), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trailing_stop_active", sa.Boolean, default=False),
        sa.Column("break_even_active", sa.Boolean, default=False),
        sa.Column("partial_closes", sa.JSON, nullable=True),
    )
    op.create_index("ix_forex_active_positions_symbol", "forex_active_positions", ["symbol"])


def downgrade() -> None:
    op.drop_table("forex_active_positions")
    op.drop_table("forex_daily_performance")
    op.drop_table("forex_trade_history")
