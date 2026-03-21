"""SQLAlchemy ORM models for the trading bot database."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import AsyncGenerator, Optional

from sqlalchemy import JSON, Boolean, Column, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase

try:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    _ASYNC_SQLALCHEMY_AVAILABLE = True
except ImportError:
    _ASYNC_SQLALCHEMY_AVAILABLE = False


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy/sell or long/short
    trade_type = Column(String(10), nullable=False, default="futures")  # futures/spot
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    amount = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    fee = Column(Float, default=0.0)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    strategy = Column(String(50), nullable=True)
    signals_used = Column(JSON, nullable=True)
    ai_reasoning = Column(Text, nullable=True)
    opened_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc), index=True)
    closed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="open")  # open/closed/cancelled
    exchange = Column(String(20), nullable=True)
    order_id = Column(String(100), nullable=True)
    exit_reason = Column(String(50), nullable=True)  # tp/sl/manual/circuit_breaker


class TradeHistory(Base):
    """Canonical trade history model — written by PositionManager.close_position().

    Stores completed trades with full entry/exit data for analysis, P&L reporting,
    and historical back-loading of DailyPnLManager on restart.

    Column mapping vs problem-statement names:
      price          → entry_price
      filled_price   → exit_price (actual fill)
      size           → amount
      funding_fee    → funding_costs
      duration       → duration_seconds
    """
    __tablename__ = "trade_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), nullable=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # long/short
    order_type = Column(String(20), nullable=False, default="market")  # market/limit
    size = Column(Float, nullable=False)  # Amount in base asset (contracts / lots)
    price = Column(Float, nullable=False)  # Entry price
    filled_price = Column(Float, nullable=True)  # Actual exit fill price
    leverage = Column(Integer, default=1)
    pnl = Column(Float, nullable=True)  # Gross P&L in quote currency
    pnl_pct = Column(Float, nullable=True)  # P&L as % of margin
    fees = Column(Float, default=0.0)
    entry_time = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc), index=True)
    close_time = Column(DateTime, nullable=True, index=True)
    duration = Column(Integer, nullable=True)  # Duration in seconds
    strategy = Column(String(50), nullable=True, index=True)
    notes = Column(Text, nullable=True)
    # exit_reason: stop_loss | take_profit | trailing_tp | manual | stale_position | circuit_breaker
    exit_reason = Column(String(50), nullable=True, index=True)
    exchange = Column(String(20), nullable=True)
    # Additional metadata
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    liquidation_price = Column(Float, nullable=True)
    funding_fee = Column(Float, default=0.0)  # Funding costs accumulated over position lifetime
    realized_pnl = Column(Float, nullable=True)  # Net realized P&L (after fees & funding)
    margin_used = Column(Float, nullable=True)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # long/short
    amount = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    leverage = Column(Integer, default=1)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    margin_used = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    status = Column(String(20), default="open")  # open/closed
    strategy = Column(String(50), nullable=True)
    exchange = Column(String(20), nullable=True)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # long/short/neutral
    strength = Column(Float, nullable=False)  # 0-100
    source = Column(String(50), nullable=False)
    strategy = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc), index=True)
    signal_metadata = Column(JSON, nullable=True)
    acted_on = Column(Boolean, default=False)
    confluence_score = Column(Float, nullable=True)


class NewsItem(Base):
    __tablename__ = "news_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)
    source = Column(String(100), nullable=False)
    url = Column(String(1000), nullable=True, unique=True)
    category = Column(String(50), nullable=True)  # REGULATORY/TECHNICAL/etc.
    impact_level = Column(String(20), nullable=True)  # CRITICAL/HIGH/MEDIUM/LOW/NOISE
    direction = Column(String(10), nullable=True)  # BULLISH/BEARISH/NEUTRAL
    affected_assets = Column(JSON, nullable=True)  # list of symbols
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    timestamp = Column(DateTime, nullable=False, index=True)
    processed = Column(Boolean, default=False)
    ai_analysis = Column(JSON, nullable=True)


class SentimentSnapshot(Base):
    __tablename__ = "sentiment_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    score = Column(Float, nullable=False)  # -1 to 1
    label = Column(String(30), nullable=False)  # very_bullish/bullish/neutral/bearish/very_bearish
    sources_breakdown = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc), index=True)


class DailyPerformance(Base):
    __tablename__ = "daily_performance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)  # YYYY-MM-DD
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    trades_count = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    best_trade = Column(Float, nullable=True)
    worst_trade = Column(Float, nullable=True)
    fees_paid = Column(Float, default=0.0)


class SystemState(Base):
    __tablename__ = "system_states"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc), index=True)
    circuit_breaker_active = Column(Boolean, default=False)
    active_strategies = Column(JSON, nullable=True)
    risk_level = Column(String(20), nullable=True)  # LOW/MEDIUM/HIGH/CRITICAL
    market_regime = Column(String(30), nullable=True)
    total_balance = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)


class AIMemory(Base):
    __tablename__ = "ai_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    context_type = Column(String(50), nullable=False)  # trade_analysis/market_context/etc.
    content = Column(Text, nullable=False)
    relevance_score = Column(Float, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    symbol = Column(String(20), nullable=True)
    tags = Column(JSON, nullable=True)


class AlertLog(Base):
    __tablename__ = "alert_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False)
    channel = Column(String(20), nullable=False)  # telegram/discord/email
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    success = Column(Boolean, default=True)
    error_message = Column(String(500), nullable=True)


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    start_date = Column(String(10), nullable=False)
    end_date = Column(String(10), nullable=False)
    total_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    parameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


class ActivePosition(Base):
    """Tracks currently open positions for crash-recovery / reconciliation.

    Every time :class:`~execution.trade_executor.TradeExecutor` opens or
    modifies a trade this record is upserted so that on the next startup the
    bot can reconcile its in-memory state with what actually exists on the
    exchange.
    """

    __tablename__ = "active_positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # exchange_id: the position identifier returned by the exchange (may be
    # None for exchanges that use symbol as the position key).
    exchange_id = Column(String(100), nullable=True)
    symbol = Column(String(20), nullable=False, index=True, unique=True)
    strategy_name = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # long/short
    amount = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(JSON, nullable=True)  # list of TP price levels
    opened_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


class ActiveOrder(Base):
    """Tracks open SL / TP / entry orders placed by the bot.

    Stored so that on restart the reconciler can verify these orders are still
    live on the exchange and re-attach them to the relevant position tracker.
    """

    __tablename__ = "active_orders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # exchange_id: the order ID returned by the exchange.
    exchange_id = Column(String(100), nullable=False, unique=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy_name = Column(String(50), nullable=True)
    # order_type: "entry" | "stop_loss" | "take_profit"
    order_type = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy/sell
    amount = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)   # limit / entry price
    stop_loss = Column(Float, nullable=True)     # trigger price for SL orders
    take_profit = Column(Float, nullable=True)   # trigger price for TP orders
    created_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


class DailyPnLRecord(Base):
    """Persistent daily P&L record, written at daily reset.

    Loaded on startup to initialise :class:`~risk.daily_pnl_manager.DailyPnLManager`
    and :class:`~risk.drawdown_protector.DrawdownProtector` with real historical data
    rather than in-memory-only state that is lost on restarts.
    """

    __tablename__ = "daily_pnl_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)  # UTC calendar date
    starting_equity = Column(Float, nullable=False)
    ending_equity = Column(Float, nullable=True)
    total_pnl = Column(Float, nullable=True)
    total_pnl_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    best_trade_pnl = Column(Float, nullable=True)
    worst_trade_pnl = Column(Float, nullable=True)
    fees_paid = Column(Float, default=0.0)
    funding_costs = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# Forex-specific models
# ---------------------------------------------------------------------------


class ForexTradeHistory(Base):
    """Forex trade history with pip-based metrics."""

    __tablename__ = "forex_trade_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), nullable=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)  # XAU/USD, XAUUSD
    side = Column(String(10), nullable=False)  # long/short
    lot_size = Column(Float, nullable=False)  # 0.01 = micro lot
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    stop_loss_pips = Column(Float, nullable=True)
    take_profit_pips = Column(Float, nullable=True)
    pip_pnl = Column(Float, nullable=True)  # P&L in pips
    usd_pnl = Column(Float, nullable=True)  # P&L in USD
    spread_at_entry = Column(Float, nullable=True)  # Spread in pips at entry
    leverage = Column(Integer, default=20)
    margin_used = Column(Float, nullable=True)
    swap_cost = Column(Float, default=0.0)  # Overnight swap/funding cost
    commission = Column(Float, default=0.0)
    strategy = Column(String(50), nullable=True, index=True)
    session = Column(String(20), nullable=True, index=True)  # london/new_york/asian/sydney
    entry_time = Column(DateTime(timezone=True), nullable=False, index=True)
    exit_time = Column(DateTime(timezone=True), nullable=True, index=True)
    duration_seconds = Column(Integer, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # sl/tp1/tp2/tp3/trailing/manual/margin_call
    max_favorable_pips = Column(Float, nullable=True)  # MFE
    max_adverse_pips = Column(Float, nullable=True)  # MAE
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))


class ForexDailyPerformance(Base):
    """Daily forex performance record with session breakdown."""

    __tablename__ = "forex_daily_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    starting_equity = Column(Float, nullable=False)
    ending_equity = Column(Float, nullable=True)
    total_pnl_usd = Column(Float, nullable=True)
    total_pnl_pips = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    best_trade_pips = Column(Float, nullable=True)
    worst_trade_pips = Column(Float, nullable=True)
    avg_win_pips = Column(Float, nullable=True)
    avg_loss_pips = Column(Float, nullable=True)
    total_lots_traded = Column(Float, nullable=True)
    total_commission = Column(Float, default=0.0)
    total_swap = Column(Float, default=0.0)
    # Session breakdown
    london_trades = Column(Integer, default=0)
    london_pnl_pips = Column(Float, default=0.0)
    ny_trades = Column(Integer, default=0)
    ny_pnl_pips = Column(Float, default=0.0)
    asian_trades = Column(Integer, default=0)
    asian_pnl_pips = Column(Float, default=0.0)
    sydney_trades = Column(Integer, default=0)
    sydney_pnl_pips = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))


class ForexActivePosition(Base):
    """Track currently open forex positions for crash recovery."""

    __tablename__ = "forex_active_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), nullable=True)
    symbol = Column(String(20), nullable=False, index=True, unique=True)
    side = Column(String(10), nullable=False)
    lot_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_prices = Column(JSON, nullable=True)  # [TP1, TP2, TP3]
    leverage = Column(Integer, default=20)
    strategy = Column(String(50), nullable=True)
    session = Column(String(20), nullable=True)
    opened_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))
    trailing_stop_active = Column(Boolean, default=False)
    break_even_active = Column(Boolean, default=False)
    partial_closes = Column(JSON, nullable=True)  # [{pct: 40, price: 3050, pips: 200}]


# ---------------------------------------------------------------------------
# Async database session factory
# ---------------------------------------------------------------------------

_async_engine = None
_async_session_factory: Optional["async_sessionmaker[AsyncSession]"] = None


def _get_database_url() -> str:
    """Return the DATABASE_URL from the environment, converting sync drivers to async."""
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/trading_bot.db")
    # Convert common sync driver prefixes to their async equivalents
    replacements = {
        "postgresql://": "postgresql+asyncpg://",
        "postgres://": "postgresql+asyncpg://",
        "sqlite:///": "sqlite+aiosqlite:///",
    }
    for old, new in replacements.items():
        if url.startswith(old):
            return url.replace(old, new, 1)
    return url


def get_async_engine():
    """Return a lazily-created async SQLAlchemy engine."""
    global _async_engine
    if not _ASYNC_SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "sqlalchemy[asyncio] is required for async database access. "
            "Install with: pip install 'sqlalchemy[asyncio]' aiosqlite asyncpg"
        )
    if _async_engine is None:
        db_url = _get_database_url()
        connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}
        _async_engine = create_async_engine(
            db_url,
            echo=False,
            future=True,
            connect_args=connect_args,
        )
    return _async_engine


def get_async_session_factory() -> "async_sessionmaker[AsyncSession]":
    """Return a lazily-created async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields an :class:`AsyncSession`.

    Usage::

        async with get_async_session() as session:
            session.add(trade)
            await session.commit()
    """
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all ORM tables in the database (idempotent)."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
