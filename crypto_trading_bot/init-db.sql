-- init-db.sql — Trading Bot Database Schema
-- PostgreSQL 16 — Optimized for 4vCPU / 8GB RAM VPS

-- ═══════════════════════════════════════════════
-- Trade Journal (core)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        TEXT UNIQUE NOT NULL,
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    quantity        DOUBLE PRECISION NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    fees_paid       DOUBLE PRECISION DEFAULT 0,
    slippage_cost   DOUBLE PRECISION DEFAULT 0,
    signal_confidence DOUBLE PRECISION DEFAULT 0,
    regime_at_entry TEXT DEFAULT 'unknown',
    leverage        INTEGER DEFAULT 1,
    exchange        TEXT DEFAULT 'gateio',
    is_testnet      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime_at_entry);

-- ═══════════════════════════════════════════════
-- Order Lifecycle (audit trail)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS order_lifecycle (
    id              BIGSERIAL PRIMARY KEY,
    client_order_id TEXT NOT NULL,
    exchange_order_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    original_qty    DOUBLE PRECISION NOT NULL,
    original_price  DOUBLE PRECISION,
    final_state     TEXT NOT NULL,
    filled_qty      DOUBLE PRECISION DEFAULT 0,
    avg_fill_price  DOUBLE PRECISION DEFAULT 0,
    total_fees      DOUBLE PRECISION DEFAULT 0,
    strategy_tag    TEXT,
    signal_confidence DOUBLE PRECISION DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    state_history   JSONB DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON order_lifecycle(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_created ON order_lifecycle(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_state ON order_lifecycle(final_state);

-- ═══════════════════════════════════════════════
-- Daily PnL Summary
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS daily_pnl (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE UNIQUE NOT NULL,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    total_fees      DOUBLE PRECISION DEFAULT 0,
    total_slippage  DOUBLE PRECISION DEFAULT 0,
    trade_count     INTEGER DEFAULT 0,
    win_count       INTEGER DEFAULT 0,
    loss_count      INTEGER DEFAULT 0,
    max_drawdown_pct DOUBLE PRECISION DEFAULT 0,
    strategy_pnl    JSONB DEFAULT '{}'::jsonb,
    symbol_pnl      JSONB DEFAULT '{}'::jsonb,
    regime_pnl      JSONB DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════
-- System Events (degradation, circuit breaker, etc.)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS system_events (
    id              BIGSERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical', 'emergency')),
    message         TEXT NOT NULL,
    details         JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_time ON system_events(created_at);

-- ═══════════════════════════════════════════════
-- Latency Snapshots (periodic from Rust engine)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS latency_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    tick_to_book_p50    INTEGER,
    tick_to_book_p99    INTEGER,
    book_to_signal_p50  INTEGER,
    book_to_signal_p99  INTEGER,
    signal_to_order_p50 INTEGER,
    signal_to_order_p99 INTEGER,
    order_to_ack_p50    INTEGER,
    order_to_ack_p99    INTEGER,
    end_to_end_p50      INTEGER,
    end_to_end_p99      INTEGER,
    sample_count        BIGINT DEFAULT 0,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-cleanup: keep 90 days of trade data, 30 days of events/latency
-- (Run via pg_cron or application-level cron)

-- ═══════════════════════════════════════════════
-- Market Impact Observations
-- (logged per-trade for model calibration)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS market_impact_observations (
    id                  BIGSERIAL PRIMARY KEY,
    symbol              TEXT NOT NULL,
    order_size_usd      DOUBLE PRECISION NOT NULL,
    mid_price_at_order  DOUBLE PRECISION NOT NULL,
    bid_depth_usd       DOUBLE PRECISION NOT NULL,
    ask_depth_usd       DOUBLE PRECISION NOT NULL,
    spread_bps          DOUBLE PRECISION NOT NULL,
    estimated_impact_bps DOUBLE PRECISION NOT NULL,
    actual_slippage_bps  DOUBLE PRECISION,
    liquidity_ratio     DOUBLE PRECISION NOT NULL,
    was_split           BOOLEAN DEFAULT FALSE,
    num_slices          INTEGER DEFAULT 1,
    is_buy              BOOLEAN NOT NULL,
    recorded_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_impact_symbol ON market_impact_observations(symbol);
CREATE INDEX IF NOT EXISTS idx_impact_time ON market_impact_observations(recorded_at);

-- ═══════════════════════════════════════════════
-- PnL Attribution Breakdown
-- (periodic snapshots from cold path engine)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS pnl_attribution (
    id              BIGSERIAL PRIMARY KEY,
    period_date     DATE NOT NULL,
    dimension       TEXT NOT NULL,    -- 'strategy', 'symbol', 'session', 'regime', 'confidence'
    bucket_label    TEXT NOT NULL,    -- e.g. 'imbalance', 'BTC_USDT', 'london', 'trending'
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    total_fees      DOUBLE PRECISION DEFAULT 0,
    total_slippage  DOUBLE PRECISION DEFAULT 0,
    trade_count     INTEGER DEFAULT 0,
    win_count       INTEGER DEFAULT 0,
    loss_count      INTEGER DEFAULT 0,
    win_rate        DOUBLE PRECISION DEFAULT 0,
    profit_factor   DOUBLE PRECISION DEFAULT 0,
    sharpe_approx   DOUBLE PRECISION DEFAULT 0,
    total_volume    DOUBLE PRECISION DEFAULT 0,
    avg_hold_secs   DOUBLE PRECISION DEFAULT 0,
    recorded_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(period_date, dimension, bucket_label)
);

CREATE INDEX IF NOT EXISTS idx_pnl_attr_date ON pnl_attribution(period_date);
CREATE INDEX IF NOT EXISTS idx_pnl_attr_dimension ON pnl_attribution(dimension);

-- ═══════════════════════════════════════════════
-- Degradation Events
-- (state transitions logged by graceful degradation manager)
-- ═══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS degradation_events (
    id              BIGSERIAL PRIMARY KEY,
    from_level      TEXT NOT NULL,
    to_level        TEXT NOT NULL,
    cpu_percent     DOUBLE PRECISION,
    memory_percent  DOUBLE PRECISION,
    disk_percent    DOUBLE PRECISION,
    actions_taken   JSONB DEFAULT '[]'::jsonb,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_degrad_time ON degradation_events(recorded_at);

-- ═══════════════════════════════════════════════
-- Auto-cleanup: keep 90 days of trades, 30 days of events/latency
-- ═══════════════════════════════════════════════
-- These should be run via pg_cron or application-level scheduled task:
--   DELETE FROM latency_snapshots WHERE recorded_at < NOW() - INTERVAL '30 days';
--   DELETE FROM system_events WHERE created_at < NOW() - INTERVAL '30 days';
--   DELETE FROM market_impact_observations WHERE recorded_at < NOW() - INTERVAL '90 days';
--   DELETE FROM degradation_events WHERE recorded_at < NOW() - INTERVAL '30 days';
