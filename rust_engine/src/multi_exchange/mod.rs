//! Multi-Exchange Feature Module
//!
//! This module implements the full multi-exchange / cross-exchange capability
//! for the institutional trading bot. When `USE_MULTI_EXCHANGE=on`, the engine:
//!
//! - Maintains simultaneous WebSocket connections to Gate.io, Binance, and Bybit
//! - Fuses order books into a Consolidated Global Order Book
//! - Enables Smart Order Routing (SOR) across all three exchanges
//! - Detects cross-exchange funding rate arbitrage opportunities
//! - Monitors cross-venue margin health and delta neutrality
//!
//! When `USE_MULTI_EXCHANGE=off` (default), this module is not initialized
//! and the engine runs in single-exchange mode with zero performance impact.

pub mod global_book;
pub mod sor;
pub mod funding_arb;
pub mod margin_monitor;
pub mod ws_ingestion_multi;
pub mod cross_exchange_mm;
pub mod stat_arb;
pub mod funding_arb_engine;
pub mod funding_arb_executor;
pub mod funding_arb_risk;

// Re-export commonly used types
pub use global_book::{ExchangeId, GlobalBookRegistry, GlobalOrderBook, SharedGlobalBook};
pub use sor::{SmartOrderRouter, SorConfig, SorResult, OrderSlice};
pub use funding_arb::{CrossExchangeFundingArb, FundingArbOpportunity, FundingRateData};
pub use margin_monitor::{CrossVenueMarginMonitor, ExchangeMarginHealth, MarginImbalanceAlert};
pub use cross_exchange_mm::{CrossExchangeMarketMaker, CrossExchangeMMConfig, MakerOrder, HedgePosition, MakerOrderStatus};
pub use stat_arb::{StatArbEngine, StatArbConfig, StatArbPosition, StatArbExitReason};
pub use funding_arb_engine::{FundingArbEngine, FundingArbEngineConfig, FundingArbState};
pub use funding_arb_executor::{DualLegExecutor, DualLegResult, LegStatus};
pub use funding_arb_risk::{PreTradeValidator, PreTradeResult, ExitReason};
