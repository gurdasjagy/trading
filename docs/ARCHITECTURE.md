# Trading Bot Architecture

## Overview

This document describes the complete system architecture of the institutional-grade crypto trading bot, including the Rust hot path, Python cold path, IPC layer, risk management flow, and deployment topology.

## System Components

### 1. Rust Hot Path (Execution Engine)

The Rust execution engine handles all latency-critical operations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust Execution Engine                     │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   WebSocket  │───▶│  Orderbook   │───▶│   Strategy   │  │
│  │   Ingestion  │    │  Management  │    │  Evaluator   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Tick Broadcast│    │  Pre-Trade   │    │  Execution   │  │
│  │   (SHM IPC)  │    │     Risk     │    │   Gateway    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `rust_engine/src/main.rs`: Main orchestrator (1800+ lines)
- `rust_engine/src/strategy_engine.rs`: VPIN + orderbook imbalance strategy
- `rust_engine/src/execution_gateway.rs`: Order submission and fill handling
- `rust_engine/src/pre_trade_risk.rs`: Synchronous risk checks (VaR, Kelly, correlation)
- `rust_engine/src/position_lifecycle.rs`: Real-time PnL tracking and reversal detection

**Performance:**
- WebSocket tick processing: < 10µs per tick
- Pre-trade risk check: < 1µs
- Order submission: < 100µs end-to-end

### 2. Python Cold Path (ML & Analytics)

The Python side handles all non-latency-critical operations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Trading Logic                      │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  AI/ML Brain │───▶│    Regime    │───▶│  Portfolio   │  │
│  │  (LSTM/TFT)  │    │   Detection  │    │  Optimizer   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Feature    │    │  Risk Manager│    │  Dashboard   │  │
│  │  Engineering │    │  (VaR, Kelly)│    │  (FastAPI)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `crypto_trading_bot/core/engine.py`: Main trading engine (2000+ lines)
- `crypto_trading_bot/ai/brain.py`: ML orchestrator
- `crypto_trading_bot/ai/prediction/feature_engine.py`: Feature extraction (120+ features)
- `crypto_trading_bot/ai/prediction/lstm_model.py`: LSTM with walk-forward validation
- `crypto_trading_bot/ai/prediction/walk_forward_validator.py`: Time-series cross-validation
- `crypto_trading_bot/monitoring/dashboard.py`: Real-time WebSocket dashboard

**Update Frequency:**
- ML model retraining: Every 6 hours
- Regime detection: Every 5 minutes
- Portfolio rebalancing: Every 1 minute

### 3. IPC Layer (Shared Memory)

The IPC layer uses `/dev/shm` shared memory for zero-copy, lock-free communication:

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Memory IPC                         │
│                                                              │
│  Rust → Python:                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /dev/shm/bridge_ticks (8192 slots, 128 bytes each)  │   │
│  │  - Tick data (bid, ask, mid, VPIN, imbalance)        │   │
│  │  - Sequence number for consistency                    │   │
│  │  - Timestamp for staleness detection                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /dev/shm/bridge_exec (2048 slots, 128 bytes each)   │   │
│  │  - Execution confirmations (fills, cancels, rejects) │   │
│  │  - PnL attribution data                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Python → Rust:                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /dev/shm/bridge_portfolio (64 entries, 32 bytes)    │   │
│  │  - Target portfolio weights (-1.0 to 1.0)            │   │
│  │  - Confidence levels (0.0 to 1.0)                     │   │
│  │  - Max position sizes                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /dev/shm/regime_weights (112 bytes)                 │   │
│  │  - Market regime (trending/ranging/choppy)           │   │
│  │  - Volatility regime (low/moderate/high/extreme)     │   │
│  │  - Position scale multiplier (0.0 to 4.0)            │   │
│  │  - Seqlock for consistency                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `rust_engine/src/bridge_ipc/tick_broadcast.rs`: Tick data writer
- `rust_engine/src/bridge_ipc/portfolio_receiver.rs`: Portfolio weight reader
- `rust_engine/src/bridge_ipc/exec_confirm_broadcast.rs`: Execution confirmation writer
- `rust_engine/src/regime_shm.rs`: Regime weight reader (seqlock pattern)
- `rust_engine/src/bridge_ipc/health_monitor.rs`: IPC health monitoring

**Message Formats:**
- All messages use fixed-size structs (no heap allocation)
- Little-endian byte order
- Sequence numbers for consistency checks
- Timestamps for staleness detection

### 4. Risk Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Risk Management Flow                      │
│                                                              │
│  Signal Generation (Python)                                  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Pre-Trade Risk (Rust)                                │   │
│  │  - VaR check (99% confidence, 10-day horizon)         │   │
│  │  - Kelly Criterion position sizing                    │   │
│  │  - Correlation-based exposure limits (30% max)        │   │
│  │  - Leverage limits (max 125x)                         │   │
│  │  - Position slot limits (max 3 concurrent)            │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Order Execution (Rust)                               │   │
│  │  - Submit to exchange                                 │   │
│  │  - Track fills and partial fills                      │   │
│  │  - Update position lifecycle                          │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Position Lifecycle (Rust)                            │   │
│  │  - Tick-by-tick PnL tracking                          │   │
│  │  - Peak PnL tracking                                  │   │
│  │  - Reversal detection (30% from peak)                 │   │
│  │  - Automatic position reduction                       │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Post-Trade Analytics (Python)                        │   │
│  │  - PnL attribution                                    │   │
│  │  - Performance metrics (Sharpe, Sortino, Calmar)     │   │
│  │  - Trade journal persistence                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Risk Controls:**
1. **Pre-Trade (Rust):**
   - VaR: 99% confidence, 10-day horizon, max 20% of capital
   - Kelly Criterion: Half-Kelly with 55% WR, 1.5 W/L ratio
   - Correlation: Max 30% exposure in correlated assets
   - Leverage: Max 125x (configurable per exchange)
   - Position slots: Max 3 concurrent positions

2. **Intra-Trade (Rust):**
   - Tick-by-tick PnL tracking
   - Peak PnL tracking with reversal detection
   - Automatic close on 30% reversal from peak
   - Hard stop loss at 2% max loss

3. **Post-Trade (Python):**
   - Daily PnL limits (2% max loss)
   - Drawdown limits (10% max drawdown)
   - Consecutive loss limits (5 max)
   - Circuit breaker on extreme conditions

### 5. Deployment Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Deployment                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  rust-engine container                                │   │
│  │  - Rust binary (release build)                        │   │
│  │  - Shared memory mounts (/dev/shm)                    │   │
│  │  - Network: host mode (low latency)                   │   │
│  │  - CPU affinity: cores 0-3                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  python-brain container                               │   │
│  │  - Python 3.11+ with PyTorch                          │   │
│  │  - Shared memory mounts (/dev/shm)                    │   │
│  │  - Network: bridge mode                               │   │
│  │  - CPU affinity: cores 4-7                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  dashboard container                                  │   │
│  │  - FastAPI + Uvicorn                                  │   │
│  │  - WebSocket server (port 8080)                       │   │
│  │  - Network: bridge mode                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  postgres container                                   │   │
│  │  - Trade history persistence                          │   │
│  │  - Performance metrics                                │   │
│  │  - Volume: /var/lib/postgresql/data                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Shared Memory Configuration:**
- `/dev/shm` mounted with `tmpfs` (RAM-backed)
- Size: 512MB (sufficient for all IPC channels)
- Permissions: 0666 (world-readable/writable for cross-container access)

**Network Configuration:**
- Rust engine: Host mode for minimal latency to exchange
- Python brain: Bridge mode (no direct exchange access)
- Dashboard: Bridge mode with port 8080 exposed

**Resource Allocation:**
- Rust engine: 4 cores (0-3), 2GB RAM
- Python brain: 4 cores (4-7), 4GB RAM
- Dashboard: 1 core, 1GB RAM
- PostgreSQL: 1 core, 2GB RAM

## Data Flow

### 1. Market Data Flow

```
Exchange WebSocket
    │
    ▼
Rust: WebSocket Ingestion (< 10µs)
    │
    ▼
Rust: Orderbook Update (< 5µs)
    │
    ├──▶ Rust: Strategy Evaluator (< 50µs)
    │
    └──▶ SHM: Tick Broadcast (< 1µs)
         │
         ▼
    Python: Feature Engineering (~ 100ms)
         │
         ▼
    Python: ML Prediction (~ 50ms)
         │
         ▼
    Python: Regime Detection (~ 200ms)
```

### 2. Signal Flow

```
Python: ML Signal Generation
    │
    ▼
SHM: Portfolio Weights Write
    │
    ▼
Rust: Portfolio Receiver Read (< 1µs)
    │
    ▼
Rust: Pre-Trade Risk Check (< 1µs)
    │
    ▼
Rust: Order Submission (< 100µs)
    │
    ▼
Exchange: Order Acknowledgement
    │
    ▼
Rust: Fill Handling (< 10µs)
    │
    ▼
SHM: Execution Confirmation Broadcast
    │
    ▼
Python: Trade Journal Update
```

## Performance Characteristics

### Latency Targets

| Component | Target | Actual |
|-----------|--------|--------|
| WebSocket tick processing | < 10µs | 5-8µs |
| Pre-trade risk check | < 1µs | 0.5-0.8µs |
| Order submission | < 100µs | 50-80µs |
| SHM write | < 1µs | 0.3-0.5µs |
| SHM read | < 1µs | 0.2-0.4µs |
| ML prediction | < 100ms | 50-80ms |
| Regime detection | < 500ms | 200-300ms |

### Throughput

| Metric | Capacity |
|--------|----------|
| Ticks per second | 100,000+ |
| Orders per second | 30 (rate limited) |
| Positions tracked | 8 concurrent |
| ML predictions per minute | 60 |

## Monitoring & Observability

### Metrics Exposed

1. **IPC Health (`/health` endpoint):**
   - Message rates (ticks/sec, portfolio updates/sec)
   - Latency histograms (p50, p95, p99)
   - Error counts (stale data, corrupted messages)
   - Sequence gaps

2. **Trading Metrics (`/api/metrics`):**
   - Orders submitted/rejected
   - Fill rate
   - Average latency
   - Active positions
   - Circuit breaker state

3. **Performance Metrics (`/api/performance`):**
   - Sharpe ratio
   - Sortino ratio
   - Max drawdown
   - Win rate
   - Profit factor

### Logging

- Rust: `tracing` crate with JSON output
- Python: `loguru` with structured logging
- Log levels: DEBUG, INFO, WARN, ERROR, CRITICAL
- Log rotation: Daily, 7-day retention

## Security

### API Keys

- Stored in environment variables (never in code)
- Separate keys for testnet and mainnet
- Read-only keys for monitoring

### Network Security

- Rust engine: Direct exchange access (TLS 1.3)
- Python brain: No direct exchange access
- Dashboard: HTTP Basic Auth (configurable)
- WebSocket: Token-based authentication

### Data Security

- Shared memory: World-readable (no sensitive data)
- Trade history: PostgreSQL with encrypted volume
- API keys: Environment variables only

## Disaster Recovery

### Failure Modes

1. **Rust engine crash:**
   - Automatic restart via Docker
   - Positions recovered from exchange REST API
   - Shared memory state rebuilt

2. **Python brain crash:**
   - Rust engine continues with last known regime
   - Safe defaults applied (conservative position sizing)
   - Automatic restart via Docker

3. **Exchange connectivity loss:**
   - Circuit breaker triggered
   - All positions closed via REST API fallback
   - Reconnection with exponential backoff

4. **Shared memory corruption:**
   - Automatic detection via magic bytes and sequence numbers
   - Shared memory region recreated
   - State recovered from exchange

### Backup & Recovery

- Trade history: PostgreSQL with daily backups
- Configuration: Git-tracked
- Shared memory: Ephemeral (no backup needed)
- Model weights: Versioned in S3/Git LFS

## Future Enhancements

1. **Multi-Exchange Support:**
   - Parallel Rust engines for each exchange
   - Unified Python brain for cross-exchange arbitrage

2. **FPGA Acceleration:**
   - Orderbook processing on FPGA
   - Sub-microsecond latency

3. **Physical Colocation:**
   - Deploy Rust engine in exchange datacenter
   - < 1ms round-trip latency

4. **Advanced ML:**
   - Transformer models for multi-horizon prediction
   - Reinforcement learning for execution optimization

## References

- [Rust Engine Source](../rust_engine/)
- [Python Trading Logic](../crypto_trading_bot/)
- [IPC Protocol Specification](./IPC_PROTOCOL.md) (TODO)
- [Risk Management Specification](./RISK_MANAGEMENT.md) (TODO)
