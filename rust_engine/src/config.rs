//! Shared configuration types used by both the binary and library targets.
//!
//! Issue 1 changes:
//! - Removed `SharedBooks` type alias (DashMap-based) — replaced by flat book architecture.
//! - Added `SymbolRegistry` for mapping symbol strings to compact u16 IDs.
//! - Added `ThreadTopology` for core-affinity pinning configuration.
//! - Added `FlatBookSymbolConfig` for per-symbol tick sizes.
//! - Kept: `EngineConfig`, `ExchangeConfig`, `RiskConfig` (with new fields).

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::flat_book::FlatBookConfig;
use crate::orderbook::RustOrderBook;
use crate::strategy_engine::StrategyConfig;

// ---------------------------------------------------------------------------
// Legacy SharedBooks type alias (kept for backward compatibility with
// ws_ingestion.rs and lib.rs PyO3 target)
// ---------------------------------------------------------------------------

/// Shared orderbook state across all Tokio tasks (legacy — used by library target).
/// The binary target uses FlatOrderBook + SPSC ring buffers instead.
pub type SharedBooks = Arc<DashMap<String, RustOrderBook>>;

// ---------------------------------------------------------------------------
// Symbol Registry
// ---------------------------------------------------------------------------

/// Maps symbol strings (e.g. "BTC_USDT") to compact u16 IDs.
///
/// Used throughout the hot path to avoid String comparisons.
/// Symbol ID 0 is reserved (invalid/uninitialized).
#[derive(Debug, Clone)]
pub struct SymbolRegistry {
    /// symbol string → u16 ID
    name_to_id: HashMap<String, u16>,
    /// u16 ID → symbol string
    id_to_name: Vec<String>,
}

impl SymbolRegistry {
    /// Create a new registry from a list of symbol names.
    /// IDs are assigned starting from 1 (0 = invalid).
    pub fn new(symbols: &[String]) -> Self {
        let mut name_to_id = HashMap::new();
        let mut id_to_name = vec!["INVALID".to_string()]; // ID 0 = invalid
        for (i, sym) in symbols.iter().enumerate() {
            let id = (i + 1) as u16;
            name_to_id.insert(sym.clone(), id);
            id_to_name.push(sym.clone());
        }
        Self { name_to_id, id_to_name }
    }

    /// Look up a symbol's numeric ID by name. Returns 0 if not found.
    #[inline]
    pub fn get_id(&self, name: &str) -> u16 {
        self.name_to_id.get(name).copied().unwrap_or(0)
    }

    /// Look up a symbol's name by ID. Returns "INVALID" for ID 0 or out-of-range.
    #[inline]
    pub fn get_name(&self, id: u16) -> &str {
        self.id_to_name.get(id as usize).map(|s| s.as_str()).unwrap_or("INVALID")
    }

    /// Total number of registered symbols (excluding the invalid ID 0).
    pub fn len(&self) -> usize {
        self.id_to_name.len() - 1
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all symbol IDs (excluding 0).
    pub fn all_ids(&self) -> Vec<u16> {
        (1..=self.len() as u16).collect()
    }
}

// ---------------------------------------------------------------------------
// Thread Topology
// ---------------------------------------------------------------------------

/// Core-affinity pinning configuration for the 32-core machine.
///
/// ```text
/// Cores 0-1:   OS interrupts, kernel threads, Docker daemon
/// Core 2:      Rust WS Ingestion Thread — Gate.io
/// Core 3:      Rust L2/L3 Orderbook Builder Thread
/// Core 4:      Rust Signal/Strategy Evaluator Thread + Market Impact
/// Core 5:      Rust Execution Router Thread + Order Lifecycle + Latency
/// Core 6:      Rust Telemetry/Journaling Thread
/// Cores 7-10:  Rust Microstructure Analytics
/// Cores 16+:   Python services (non-isolated)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadTopology {
    /// Core ID for Gate.io WS ingestion thread.
    #[serde(default = "default_ws_gateio_core")]
    pub ws_gateio_core: usize,

    /// Core ID for orderbook builder thread.
    #[serde(default = "default_book_builder_core")]
    pub book_builder_core: usize,
    /// Core ID for strategy evaluator thread.
    #[serde(default = "default_strategy_core")]
    pub strategy_core: usize,
    /// Core ID for execution router thread.
    #[serde(default = "default_execution_core")]
    pub execution_core: usize,
    /// Core ID for telemetry/journaling thread.
    #[serde(default = "default_telemetry_core")]
    pub telemetry_core: usize,
    /// Core IDs for microstructure analytics workers.
    #[serde(default = "default_micro_cores")]
    pub microstructure_cores: Vec<usize>,
}

fn default_ws_gateio_core() -> usize { 2 }
fn default_book_builder_core() -> usize { 4 }
fn default_strategy_core() -> usize { 5 }
fn default_execution_core() -> usize { 6 }
fn default_telemetry_core() -> usize { 7 }
fn default_micro_cores() -> Vec<usize> { vec![8, 9, 10, 11] }

impl ThreadTopology {
    /// Auto-detect the optimal thread topology based on available CPU cores.
    ///
    /// Layouts:
    ///   - 1-2 cores:  Minimal — WS + book on core 0, strategy + exec + telemetry on core 1
    ///   - 3-4 cores:  Constrained — core 0 reserved for OS, spread across 1-3
    ///   - 5-8 cores:  Standard — dedicated cores for each component
    ///   - 9+ cores:   High-performance — original 32-core layout with micro cores
    pub fn auto_detect() -> Self {
        let cpu_count = core_affinity::get_core_ids()
            .map(|ids| ids.len())
            .unwrap_or(4); // Fallback to 4 if detection fails

        match cpu_count {
            1 => {
                Self {
                    ws_gateio_core: 0,
                    book_builder_core: 0,
                    strategy_core: 0,
                    execution_core: 0,
                    telemetry_core: 0,
                    microstructure_cores: vec![],
                }
            }
            2 => {
                Self {
                    ws_gateio_core: 1,
                    book_builder_core: 1,
                    strategy_core: 1,
                    execution_core: 1,
                    telemetry_core: 0,
                    microstructure_cores: vec![],
                }
            }
            3 => {
                Self {
                    ws_gateio_core: 1,
                    book_builder_core: 1,
                    strategy_core: 2,
                    execution_core: 2,
                    telemetry_core: 0,
                    microstructure_cores: vec![],
                }
            }
            4 => {
                Self {
                    ws_gateio_core: 1,
                    book_builder_core: 1,
                    strategy_core: 2,
                    execution_core: 2,
                    telemetry_core: 3,
                    microstructure_cores: vec![3],
                }
            }
            5..=8 => {
                let last = cpu_count - 1;
                Self {
                    ws_gateio_core: 1,
                    book_builder_core: 2.min(last),
                    strategy_core: 3.min(last),
                    execution_core: 4.min(last),
                    telemetry_core: 5.min(last),
                    microstructure_cores: if cpu_count >= 7 { vec![6.min(last)] } else { vec![] },
                }
            }
            _ => {
                Self {
                    ws_gateio_core: 2,
                    book_builder_core: 3,
                    strategy_core: 4,
                    execution_core: 5,
                    telemetry_core: 6,
                    microstructure_cores: vec![7, 8, 9, 10],
                }
            }
        }
    }

    /// Check if this topology is valid for the current machine.
    pub fn validate(&self) -> Result<(), String> {
        let cpu_count = core_affinity::get_core_ids()
            .map(|ids| ids.len())
            .unwrap_or(4);

        let max_core = [
            self.ws_gateio_core,
            self.book_builder_core,
            self.strategy_core,
            self.execution_core,
            self.telemetry_core,
        ].into_iter()
            .chain(self.microstructure_cores.iter().copied())
            .max()
            .unwrap_or(0);

        if max_core >= cpu_count {
            Err(format!(
                "ThreadTopology references core {} but only {} cores available. \
                 Use THREAD_TOPOLOGY=auto for automatic detection.",
                max_core, cpu_count
            ))
        } else {
            Ok(())
        }
    }
}

impl Default for ThreadTopology {
    fn default() -> Self {
        // Auto-detect by default instead of hardcoded 32-core layout
        Self::auto_detect()
    }
}

// ---------------------------------------------------------------------------
// Per-Symbol FlatBook Config
// ---------------------------------------------------------------------------

/// Per-symbol configuration for the flat array orderbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatBookSymbolConfig {
    /// Symbol name (e.g. "BTC_USDT").
    pub symbol: String,
    /// Tick size in USDT (e.g. 0.1 for BTC).
    #[serde(default = "default_tick_size")]
    pub tick_size: f64,
    /// Maximum number of levels per side.
    #[serde(default = "default_max_levels")]
    pub max_levels: usize,
}

/// Task 7: Pair-specific trading profile configuration (Phase 2 Feature 8).
///
/// Allows per-symbol customization of:
/// - Imbalance threshold for signal generation
/// - VPIN bucket size for toxicity detection
/// - Trailing stop ATR multiplier
/// - Maximum leverage cap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairProfile {
    /// Symbol name (e.g. "BTC_USDT").
    pub symbol: String,
    /// Imbalance threshold for entry signals (e.g., 0.05 = 5%).
    #[serde(default = "default_imbalance_threshold")]
    pub imbalance_threshold: f64,
    /// VPIN bucket size in USD (e.g., 500000.0 for BTC).
    #[serde(default = "default_vpin_bucket_size")]
    pub vpin_bucket_size: f64,
    /// Trailing stop ATR multiplier (e.g., 3.0 = 3x ATR).
    #[serde(default = "default_trailing_stop_atr_multiplier")]
    pub trailing_stop_atr_multiplier: f64,
    /// Maximum leverage for this pair (e.g., 20).
    #[serde(default = "default_max_leverage_pair")]
    pub max_leverage: i32,
}

fn default_imbalance_threshold() -> f64 { 0.03 }
fn default_vpin_bucket_size() -> f64 { 100_000.0 }
fn default_trailing_stop_atr_multiplier() -> f64 { 2.0 }
fn default_max_leverage_pair() -> i32 { 20 }

fn default_tick_size() -> f64 { 0.1 }
fn default_max_levels() -> usize { 10_000 }

impl FlatBookSymbolConfig {
    /// Convert to the internal `FlatBookConfig` representation.
    pub fn to_flat_config(&self) -> FlatBookConfig {
        use crate::fixed_point::FixedPrice;
        FlatBookConfig {
            tick_size_fp: (self.tick_size * FixedPrice::PRECISION as f64).round() as i64,
            max_levels: self.max_levels,
            reference_price_fp: 0, // Will be set on first snapshot
        }
    }
}

// ---------------------------------------------------------------------------
// Institutional Feature Configs
// ---------------------------------------------------------------------------

/// Feature 1: Hardware Timestamp configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HwTimestampConfig {
    #[serde(default)]
    pub enabled: bool,
}

/// Feature 2: Tick Store configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickStoreConfig {
    #[serde(default = "default_tick_store_enabled")]
    pub enabled: bool,
    #[serde(default = "default_tick_store_path")]
    pub base_path: String,
    #[serde(default = "default_tick_retention_days")]
    pub retention_days: u32,
    #[serde(default = "default_tick_sync_interval")]
    pub sync_interval_ms: u64,
    #[serde(default = "default_tick_prealloc")]
    pub prealloc_ticks: usize,
}

fn default_tick_store_enabled() -> bool { true }
fn default_tick_store_path() -> String { "/data/ticks".to_string() }
fn default_tick_retention_days() -> u32 { 30 }
fn default_tick_sync_interval() -> u64 { 1000 }
fn default_tick_prealloc() -> usize { 10_000_000 }

impl Default for TickStoreConfig {
    fn default() -> Self {
        Self {
            enabled: default_tick_store_enabled(),
            base_path: default_tick_store_path(),
            retention_days: default_tick_retention_days(),
            sync_interval_ms: default_tick_sync_interval(),
            prealloc_ticks: default_tick_prealloc(),
        }
    }
}

/// Feature 3: VaR Engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRConfig {
    #[serde(default = "default_var_enabled")]
    pub enabled: bool,
    #[serde(default = "default_var_limit_pct")]
    pub limit_pct: f64,
    #[serde(default = "default_var_confidence")]
    pub confidence_level: f64,
    #[serde(default = "default_var_update_interval")]
    pub update_interval_ms: u64,
    #[serde(default = "default_var_rolling_window")]
    pub rolling_window_minutes: usize,
    #[serde(default = "default_var_circuit_breaker")]
    pub circuit_breaker_threshold_pct: f64,
}

fn default_var_enabled() -> bool { true }
fn default_var_limit_pct() -> f64 { 0.05 }
fn default_var_confidence() -> f64 { 0.99 }
fn default_var_update_interval() -> u64 { 1000 }
fn default_var_rolling_window() -> usize { 1000 }
fn default_var_circuit_breaker() -> f64 { 90.0 }

impl Default for VaRConfig {
    fn default() -> Self {
        Self {
            enabled: default_var_enabled(),
            limit_pct: default_var_limit_pct(),
            confidence_level: default_var_confidence(),
            update_interval_ms: default_var_update_interval(),
            rolling_window_minutes: default_var_rolling_window(),
            circuit_breaker_threshold_pct: default_var_circuit_breaker(),
        }
    }
}

/// Feature 4: Gamma Hedging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaHedgingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_options_source")]
    pub options_data_source: String,
    #[serde(default = "default_flip_alert_threshold")]
    pub flip_alert_threshold: f64,
    #[serde(default = "default_implied_vol")]
    pub default_implied_vol: f64,
    #[serde(default = "default_risk_free_rate")]
    pub risk_free_rate: f64,
    #[serde(default)]
    pub tracked_symbols: Vec<String>,
}

fn default_options_source() -> String { "deribit".to_string() }
fn default_flip_alert_threshold() -> f64 { 0.2 }
fn default_implied_vol() -> f64 { 0.50 }
fn default_risk_free_rate() -> f64 { 0.05 }

impl Default for GammaHedgingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            options_data_source: default_options_source(),
            flip_alert_threshold: default_flip_alert_threshold(),
            default_implied_vol: default_implied_vol(),
            risk_free_rate: default_risk_free_rate(),
            tracked_symbols: vec!["BTC_USDT".to_string(), "ETH_USDT".to_string()],
        }
    }
}

/// Feature 5: Arbitrage Engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_arb_min_spread")]
    pub min_spread_bps: i64,
    #[serde(default = "default_arb_max_size")]
    pub max_size_usdt: f64,
    #[serde(default = "default_arb_max_latency")]
    pub max_latency_us: u64,
    #[serde(default)]
    pub enabled_exchanges: Vec<String>,
}

fn default_arb_min_spread() -> i64 { 10 }
fn default_arb_max_size() -> f64 { 10000.0 }
fn default_arb_max_latency() -> u64 { 500 }

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_spread_bps: default_arb_min_spread(),
            max_size_usdt: default_arb_max_size(),
            max_latency_us: default_arb_max_latency(),
            enabled_exchanges: vec!["gateio".to_string()],
        }
    }
}

/// Feature 6: Fee Optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeOptimizerConfig {
    #[serde(default = "default_fee_optimizer_enabled")]
    pub enabled: bool,
    #[serde(default = "default_fee_exchange")]
    pub exchange: String,
    #[serde(default = "default_fee_refresh_hours")]
    pub refresh_hours: u32,
    #[serde(default = "default_prefer_maker")]
    pub prefer_maker_orders: bool,
}

fn default_fee_optimizer_enabled() -> bool { true }
fn default_fee_exchange() -> String { "gateio".to_string() }
fn default_fee_refresh_hours() -> u32 { 24 }
fn default_prefer_maker() -> bool { true }

impl Default for FeeOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: default_fee_optimizer_enabled(),
            exchange: default_fee_exchange(),
            refresh_hours: default_fee_refresh_hours(),
            prefer_maker_orders: default_prefer_maker(),
        }
    }
}

/// Feature 7: Adaptive TWAP configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTwapConfig {
    #[serde(default = "default_adaptive_twap_enabled")]
    pub enabled: bool,
    #[serde(default = "default_target_participation")]
    pub target_participation: f64,
    #[serde(default = "default_max_twap_participation")]
    pub max_participation: f64,
    #[serde(default = "default_max_twap_spread")]
    pub max_spread_bps: f64,
    #[serde(default = "default_vpin_high")]
    pub vpin_high_threshold: f64,
    #[serde(default = "default_vpin_medium")]
    pub vpin_medium_threshold: f64,
    #[serde(default = "default_min_slice_pct")]
    pub min_slice_pct: f64,
}

fn default_adaptive_twap_enabled() -> bool { true }
fn default_target_participation() -> f64 { 0.05 }
fn default_max_twap_participation() -> f64 { 0.30 }
fn default_max_twap_spread() -> f64 { 20.0 }
fn default_vpin_high() -> f64 { 0.7 }
fn default_vpin_medium() -> f64 { 0.5 }
fn default_min_slice_pct() -> f64 { 0.01 }

impl Default for AdaptiveTwapConfig {
    fn default() -> Self {
        Self {
            enabled: default_adaptive_twap_enabled(),
            target_participation: default_target_participation(),
            max_participation: default_max_twap_participation(),
            max_spread_bps: default_max_twap_spread(),
            vpin_high_threshold: default_vpin_high(),
            vpin_medium_threshold: default_vpin_medium(),
            min_slice_pct: default_min_slice_pct(),
        }
    }
}

/// Feature 8: Alert Manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    #[serde(default = "default_alert_enabled")]
    pub enabled: bool,
    #[serde(default = "default_rate_limit")]
    pub rate_limit_per_minute: u32,
    #[serde(default = "default_dedup_window")]
    pub dedup_window_secs: u64,
    #[serde(default = "default_alert_min_priority")]
    pub min_priority: String,
    #[serde(default)]
    pub enabled_channels: Vec<String>,
}

fn default_alert_enabled() -> bool { true }
fn default_rate_limit() -> u32 { 10 }
fn default_dedup_window() -> u64 { 300 }
fn default_alert_min_priority() -> String { "info".to_string() }

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            enabled: default_alert_enabled(),
            rate_limit_per_minute: default_rate_limit(),
            dedup_window_secs: default_dedup_window(),
            min_priority: default_alert_min_priority(),
            enabled_channels: vec!["telegram".to_string(), "console".to_string()],
        }
    }
}

// ---------------------------------------------------------------------------
// Shared Memory Config
// ---------------------------------------------------------------------------

/// Configuration for shared memory communication between Rust and Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemConfig {
    /// Path to the regime weights shared memory buffer.
    #[serde(default = "default_regime_shm_path")]
    pub regime_shm_path: String,
    /// Path to the telemetry shared memory buffer.
    #[serde(default = "default_telemetry_shm_path")]
    pub telemetry_shm_path: String,
}

fn default_regime_shm_path() -> String { "/dev/shm/regime_weights".to_string() }
fn default_telemetry_shm_path() -> String { "/dev/shm/engine_telemetry".to_string() }

impl Default for SharedMemConfig {
    fn default() -> Self {
        Self {
            regime_shm_path: default_regime_shm_path(),
            telemetry_shm_path: default_telemetry_shm_path(),
        }
    }
}

// ---------------------------------------------------------------------------
// Existing Config Types (preserved for backward compatibility)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    /// Exchange identifier, e.g. "gateio".
    #[serde(alias = "exchange_id")]
    pub name: String,
    pub symbols: Vec<String>,
    pub ws_url: String,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub passphrase: Option<String>,
    #[serde(default)]
    pub testnet: bool,
    pub rest_url: Option<String>,
    #[serde(default = "default_max_leverage")]
    pub max_leverage: i32,
    /// Whether this exchange is enabled (for multi-exchange arbitrage)
    #[serde(default = "default_exchange_enabled")]
    pub enabled: bool,
}

fn default_exchange_enabled() -> bool { true }

fn default_max_leverage() -> i32 { 20 }

/// Strategy configuration for TOML file loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStrategyConfig {
    pub name: String,
    #[serde(default)]
    pub params: serde_json::Value,
    #[serde(default = "default_leverage")]
    pub default_leverage: i32,
    #[serde(default = "default_max_position")]
    pub max_position_usdt: f64,
    #[serde(default = "default_min_spread")]
    pub min_spread_bps: f64,
    #[serde(default = "default_min_imbalance")]
    pub min_imbalance_threshold: f64,
    #[serde(default = "default_vpin")]
    pub vpin_toxicity_threshold: f64,
}

fn default_leverage() -> i32 { 5 }
fn default_max_position() -> f64 { 1000.0 }
fn default_min_spread() -> f64 { 1.0 }
fn default_min_imbalance() -> f64 { 0.15 }
fn default_vpin() -> f64 { 0.7 }

/// Risk management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    #[serde(default = "default_max_drawdown")]
    pub max_drawdown_pct: f64,
    #[serde(default = "default_max_daily_loss")]
    pub max_daily_loss_usdt: f64,
    #[serde(default = "default_max_open_positions")]
    pub max_open_positions: usize,
    #[serde(default = "default_circuit_breaker")]
    pub circuit_breaker_loss_pct: f64,
    #[serde(default = "default_position_size")]
    pub position_size_pct: f64,
    /// Maximum consecutive losing trades before circuit breaker trips.
    #[serde(default)]
    pub max_consecutive_losses: Option<u64>,
    /// Maximum daily drawdown as fraction (0.05 = 5%) before halt.
    #[serde(default)]
    pub max_daily_drawdown: Option<f64>,
    /// Maximum loss on a single trade in USDT.
    #[serde(default)]
    pub max_single_loss_usdt: Option<f64>,
    /// Maximum total position value in USDT.
    #[serde(default)]
    pub max_position_usdt: Option<f64>,
}

fn default_max_drawdown() -> f64 { 5.0 }
fn default_max_daily_loss() -> f64 { 200.0 }
fn default_max_open_positions() -> usize { 3 }
fn default_circuit_breaker() -> f64 { 3.0 }
fn default_position_size() -> f64 { 2.0 }

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_drawdown_pct: default_max_drawdown(),
            max_daily_loss_usdt: default_max_daily_loss(),
            max_open_positions: default_max_open_positions(),
            circuit_breaker_loss_pct: default_circuit_breaker(),
            position_size_pct: default_position_size(),
            max_consecutive_losses: None,
            max_daily_drawdown: None,
            max_single_loss_usdt: None,
            max_position_usdt: None,
        }
    }
}

/// Top-level engine configuration loaded from TOML file or JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub exchanges: Vec<ExchangeConfig>,
    pub strategy: StrategyConfig,
    #[serde(default)]
    pub risk: RiskConfig,
    /// Top-level symbol list (used when exchanges don't specify their own).
    #[serde(default)]
    pub symbols: Vec<String>,
    /// ZeroMQ PUB address for telemetry (legacy — kept for backward compat).
    #[serde(alias = "telemetry_addr", default = "default_telemetry_bind")]
    pub zmq_telemetry_bind: String,
    /// ZeroMQ PULL address for config updates (legacy — kept for backward compat).
    #[serde(alias = "config_pull_addr", default = "default_config_bind")]
    pub zmq_config_bind: String,
    /// Path to the regime state JSON file.
    #[serde(default = "default_regime_path")]
    pub regime_file_path: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    // Legacy API key fields
    pub gateio_api_key: Option<String>,
    pub gateio_api_secret: Option<String>,
    // === Forex credentials (Mandate 3) ===
    /// MT5/TradFi login ID for forex execution.
    #[serde(default)]
    pub forex_login: Option<String>,
    /// MT5/TradFi password.
    #[serde(default)]
    pub forex_password: Option<String>,
    /// MT5/TradFi server address (e.g., "GateIO-TradFi").
    #[serde(default)]
    pub forex_server: Option<String>,
    // === NEW FIELDS (Issue 1) ===
    /// Thread topology for core-affinity pinning.
    #[serde(default)]
    pub thread_topology: ThreadTopology,
    /// Per-symbol flat book configurations.
    #[serde(default)]
    pub flat_book_configs: Vec<FlatBookSymbolConfig>,
    /// Shared memory configuration for Rust↔Python communication.
    #[serde(default)]
    pub shared_mem: SharedMemConfig,
    /// Task 8: Per-symbol pair profiles (Phase 2 Feature 8).
    #[serde(default)]
    pub pair_profiles: HashMap<String, PairProfile>,
    // === Institutional Features Configuration ===
    /// Feature 1: Hardware Timestamps
    #[serde(default, alias = "hw_timestamps")]
    pub hw_timestamp: HwTimestampConfig,
    /// Feature 2: Tick Store
    #[serde(default)]
    pub tick_store: TickStoreConfig,
    /// Feature 3: VaR Engine
    #[serde(default, alias = "var_engine")]
    pub var_config: VaRConfig,
    /// Feature 4: Gamma Hedging
    #[serde(default)]
    pub gamma_hedging: GammaHedgingConfig,
    /// Feature 5: Arbitrage Engine
    #[serde(default)]
    pub arbitrage: ArbitrageConfig,
    /// Feature 6: Fee Optimizer
    #[serde(default)]
    pub fee_optimizer: FeeOptimizerConfig,
    /// Feature 7: Adaptive TWAP
    #[serde(default)]
    pub adaptive_twap: AdaptiveTwapConfig,
    /// Feature 8: Alert Manager
    #[serde(default)]
    pub alert_manager: AlertManagerConfig,
}

// ---------------------------------------------------------------------------
// Forex Symbol Detection (Mandate 3)
// ---------------------------------------------------------------------------

/// Known forex and precious-metals symbols that must be routed through the
/// Forex Execution Gateway instead of the crypto gateway.
const FOREX_SYMBOLS: &[&str] = &[
    "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
    "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY", "CADCHF", "CADJPY",
    "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURNZD", "GBPAUD",
    "GBPCAD", "GBPCHF", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY",
    "XAU_USD", "XAG_USD",
];

/// Returns `true` if the given symbol should be routed through the Forex
/// gateway rather than the crypto gateway (Gate.io).
///
/// Matching is case-insensitive and also recognises the underscore
/// variant (e.g. "XAU_USD").
#[inline]
pub fn is_forex_symbol(symbol: &str) -> bool {
    let upper = symbol.to_ascii_uppercase();
    // Strip underscores for matching: "XAU_USD" → "XAUUSD"
    let normalized: String = upper.chars().filter(|c| *c != '_').collect();
    FOREX_SYMBOLS.iter().any(|&fx| {
        let fx_norm: String = fx.chars().filter(|c| *c != '_').collect();
        fx_norm == normalized
    })
}

fn default_telemetry_bind() -> String { "tcp://127.0.0.1:5555".to_string() }
fn default_config_bind() -> String { "tcp://127.0.0.1:5556".to_string() }
fn default_regime_path() -> String { "/dev/shm/regime_state.json".to_string() }
fn default_log_level() -> String { "info".to_string() }

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            exchanges: vec![
                ExchangeConfig {
                    name: "gateio".to_string(),
                    symbols: vec!["BTC_USDT".to_string(), "ETH_USDT".to_string()],
                    ws_url: "wss://ws.gate.com/v4/ws/futures/usdt".to_string(),
                    api_key: None,
                    secret_key: None,
                    passphrase: None,
                    testnet: false,
                    rest_url: Some("https://api.gateio.ws/api/v4".to_string()),
                    max_leverage: 20,
                    enabled: true,
                },
            ],
            strategy: StrategyConfig::default(),
            risk: RiskConfig::default(),
            symbols: vec!["BTC_USDT".to_string(), "ETH_USDT".to_string(), "XAUT_USDT".to_string()],
            zmq_telemetry_bind: default_telemetry_bind(),
            zmq_config_bind: default_config_bind(),
            regime_file_path: default_regime_path(),
            log_level: default_log_level(),
            gateio_api_key: None,
            gateio_api_secret: None,
            forex_login: None,
            forex_password: None,
            forex_server: None,
            thread_topology: ThreadTopology::default(),
            flat_book_configs: vec![],
            shared_mem: SharedMemConfig::default(),
            pair_profiles: HashMap::new(),
            // Institutional Features (defaults)
            hw_timestamp: HwTimestampConfig::default(),
            tick_store: TickStoreConfig::default(),
            var_config: VaRConfig::default(),
            gamma_hedging: GammaHedgingConfig::default(),
            arbitrage: ArbitrageConfig::default(),
            fee_optimizer: FeeOptimizerConfig::default(),
            adaptive_twap: AdaptiveTwapConfig::default(),
            alert_manager: AlertManagerConfig::default(),
        }
    }
}

/// Build a `SymbolRegistry` from the engine config.
/// Collects all unique symbols from all exchanges.
pub fn build_symbol_registry(config: &EngineConfig) -> SymbolRegistry {
    let mut all_symbols: Vec<String> = config.symbols.clone();
    for exchange in &config.exchanges {
        for sym in &exchange.symbols {
            if !all_symbols.contains(sym) {
                all_symbols.push(sym.clone());
            }
        }
    }
    SymbolRegistry::new(&all_symbols)
}

/// Build per-symbol FlatBookConfig from the engine config.
/// Falls back to defaults for symbols not explicitly configured.
pub fn build_flat_book_configs(config: &EngineConfig, registry: &SymbolRegistry) -> HashMap<u16, FlatBookConfig> {
    let mut configs = HashMap::new();
    
    // Explicit configs from TOML
    let explicit: HashMap<String, &FlatBookSymbolConfig> = config.flat_book_configs
        .iter()
        .map(|c| (c.symbol.clone(), c))
        .collect();
    
    for id in registry.all_ids() {
        let name = registry.get_name(id);
        let flat_config = if let Some(sym_config) = explicit.get(name) {
            sym_config.to_flat_config()
        } else {
            // Default config based on symbol type
            let tick_size = if name.starts_with("BTC") {
                0.1
            } else if name.starts_with("ETH") {
                0.01
            } else {
                0.01
            };
            FlatBookSymbolConfig {
                symbol: name.to_string(),
                tick_size,
                max_levels: 10_000,
            }.to_flat_config()
        };
        configs.insert(id, flat_config);
    }
    
    configs
}
