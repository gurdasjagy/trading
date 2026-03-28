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
use serde::{de::{self, Deserializer}, Deserialize, Serialize};

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
    /// Core ID for Binance WS ingestion thread (multi-exchange mode only).
    #[serde(default = "default_ws_binance_core")]
    pub ws_binance_core: usize,
    /// Core ID for Bybit WS ingestion thread (multi-exchange mode only).
    #[serde(default = "default_ws_bybit_core")]
    pub ws_bybit_core: usize,

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
fn default_ws_binance_core() -> usize { 12 }
fn default_ws_bybit_core() -> usize { 13 }
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
    ///   - 9-13 cores: Extended — adds micro cores
    ///   - 14+ cores:  Multi-exchange — dedicated cores for Binance/Bybit WS
    pub fn auto_detect() -> Self {
        let cpu_count = core_affinity::get_core_ids()
            .map(|ids| ids.len())
            .unwrap_or(4); // Fallback to 4 if detection fails

        match cpu_count {
            1 => {
                Self {
                    ws_gateio_core: 0,
                    ws_binance_core: 0,
                    ws_bybit_core: 0,
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
                    ws_binance_core: 1,
                    ws_bybit_core: 1,
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
                    ws_binance_core: 1,
                    ws_bybit_core: 1,
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
                    ws_binance_core: 1,
                    ws_bybit_core: 1,
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
                    ws_binance_core: 1,  // Share with Gate.io
                    ws_bybit_core: 1,    // Share with Gate.io
                    book_builder_core: 2.min(last),
                    strategy_core: 3.min(last),
                    execution_core: 4.min(last),
                    telemetry_core: 5.min(last),
                    microstructure_cores: if cpu_count >= 7 { vec![6.min(last)] } else { vec![] },
                }
            }
            9..=13 => {
                Self {
                    ws_gateio_core: 2,
                    ws_binance_core: 2,  // Share with Gate.io
                    ws_bybit_core: 2,    // Share with Gate.io
                    book_builder_core: 3,
                    strategy_core: 4,
                    execution_core: 5,
                    telemetry_core: 6,
                    microstructure_cores: vec![7, 8, 9, 10],
                }
            }
            _ => {
                // 14+ cores: dedicated cores for multi-exchange WS ingestion
                Self {
                    ws_gateio_core: 2,
                    ws_binance_core: 12,
                    ws_bybit_core: 13,
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
            self.ws_binance_core,
            self.ws_bybit_core,
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

/// CONFIG 1: Per-symbol trading profile configuration.
///
/// Allows per-symbol customization of:
/// - Imbalance threshold for signal generation
/// - VPIN bucket size for toxicity detection
/// - Trailing stop ATR multiplier
/// - Maximum leverage cap
/// - Maximum position size (contracts)
/// - Stop-loss percentage
/// - Take-profit percentage
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
    // ── CONFIG 1: Additional per-symbol overrides ──
    /// Maximum position size in contracts for this pair.
    #[serde(default)]
    pub max_position_size: Option<i64>,
    /// Leverage override for this pair (separate from max_leverage).
    /// If set, the engine uses this instead of the global strategy leverage.
    #[serde(default)]
    pub leverage: Option<i32>,
    /// Stop-loss percentage (e.g., 1.5 = 1.5% below entry).
    #[serde(default)]
    pub sl_pct: Option<f64>,
    /// Take-profit percentage (e.g., 3.0 = 3.0% above entry).
    #[serde(default)]
    pub tp_pct: Option<f64>,
    /// Tick size override for this pair (e.g., 0.01 for BTC).
    /// Used by FEAT 10 spread widening for more accurate tick estimation.
    #[serde(default)]
    pub tick_size: Option<f64>,
}

fn default_imbalance_threshold() -> f64 { 0.03 }
fn default_vpin_bucket_size() -> f64 { 100_000.0 }
fn default_trailing_stop_atr_multiplier() -> f64 { 2.0 }
fn default_max_leverage_pair() -> i32 { 20 }

impl PairProfile {
    /// Get the effective leverage for this pair, falling back to the given default.
    pub fn effective_leverage(&self, default: i32) -> i32 {
        self.leverage.unwrap_or(default).min(self.max_leverage)
    }

    /// Get the stop-loss price for a given entry price and side.
    /// Returns None if sl_pct is not configured.
    pub fn stop_loss_price(&self, entry_price: f64, is_long: bool) -> Option<f64> {
        self.sl_pct.map(|pct| {
            if is_long {
                entry_price * (1.0 - pct / 100.0)
            } else {
                entry_price * (1.0 + pct / 100.0)
            }
        })
    }

    /// Get the take-profit price for a given entry price and side.
    /// Returns None if tp_pct is not configured.
    pub fn take_profit_price(&self, entry_price: f64, is_long: bool) -> Option<f64> {
        self.tp_pct.map(|pct| {
            if is_long {
                entry_price * (1.0 + pct / 100.0)
            } else {
                entry_price * (1.0 - pct / 100.0)
            }
        })
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PairProfilesToml {
    Map(HashMap<String, PairProfile>),
    Seq(Vec<PairProfile>),
}

fn deserialize_pair_profiles<'de, D>(deserializer: D) -> Result<HashMap<String, PairProfile>, D::Error>
where
    D: Deserializer<'de>,
{
    let input = Option::<PairProfilesToml>::deserialize(deserializer)?;
    let Some(input) = input else {
        return Ok(HashMap::new());
    };

    match input {
        PairProfilesToml::Map(map) => Ok(map),
        PairProfilesToml::Seq(list) => {
            let mut map = HashMap::new();
            for profile in list {
                let key = profile.symbol.clone();
                if map.contains_key(&key) {
                    return Err(de::Error::custom(format!("duplicate pair_profiles symbol '{}'", key)));
                }
                map.insert(key, profile);
            }
            Ok(map)
        }
    }
}

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
// Auto-Protection Configuration (Feature 2)
// ---------------------------------------------------------------------------

/// Configuration for automatic SL/TP protection of unprotected positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoProtectionConfig {
    /// Default stop-loss percentage (e.g. 2.0 = 2% adverse move from entry).
    #[serde(default = "default_sl_pct")]
    pub default_sl_pct: f64,
    /// Default take-profit percentage (e.g. 3.0 = 3% favorable move from entry).
    #[serde(default = "default_tp_pct")]
    pub default_tp_pct: f64,
    /// Use ATR-based SL/TP instead of fixed percentage.
    #[serde(default)]
    pub use_atr_based: bool,
    /// ATR multiplier for stop-loss (used when use_atr_based=true).
    #[serde(default = "default_atr_sl_multiplier")]
    pub atr_sl_multiplier: f64,
    /// ATR multiplier for take-profit (used when use_atr_based=true).
    #[serde(default = "default_atr_tp_multiplier")]
    pub atr_tp_multiplier: f64,
    /// How often to scan for unprotected positions (in snapshot count).
    #[serde(default = "default_scan_interval")]
    pub scan_interval: u64,
}

fn default_sl_pct() -> f64 { 2.0 }
fn default_tp_pct() -> f64 { 3.0 }
fn default_atr_sl_multiplier() -> f64 { 2.0 }
fn default_atr_tp_multiplier() -> f64 { 3.0 }
fn default_scan_interval() -> u64 { 50 }

impl Default for AutoProtectionConfig {
    fn default() -> Self {
        Self {
            default_sl_pct: default_sl_pct(),
            default_tp_pct: default_tp_pct(),
            use_atr_based: false,
            atr_sl_multiplier: default_atr_sl_multiplier(),
            atr_tp_multiplier: default_atr_tp_multiplier(),
            scan_interval: default_scan_interval(),
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent State Configuration (Feature 4)
// ---------------------------------------------------------------------------

/// Configuration for SQLite-based persistent state storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentStateConfig {
    /// Enable persistent state storage.
    #[serde(default = "default_persistent_enabled")]
    pub enabled: bool,
    /// Path to the SQLite database file.
    #[serde(default = "default_db_path")]
    pub db_path: String,
}

fn default_persistent_enabled() -> bool { false }
fn default_db_path() -> String { "./trading_state.db".to_string() }

impl Default for PersistentStateConfig {
    fn default() -> Self {
        Self {
            enabled: default_persistent_enabled(),
            db_path: default_db_path(),
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-Exchange Configuration
// ---------------------------------------------------------------------------

/// Smart Order Router configuration (TOML format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SorConfigToml {
    /// Minimum order size (USDT) to trigger splitting across exchanges.
    #[serde(default = "default_sor_min_split")]
    pub min_split_size_usdt: f64,
    /// Maximum number of exchanges to split across (1-3).
    #[serde(default = "default_sor_max_venues")]
    pub max_venues: usize,
    /// Maximum slippage tolerance in basis points.
    #[serde(default = "default_sor_max_slippage")]
    pub max_slippage_bps: f64,
    /// Prefer maker orders when spread allows.
    #[serde(default = "default_sor_prefer_maker")]
    pub prefer_maker: bool,
}

fn default_sor_min_split() -> f64 { 5000.0 }
fn default_sor_max_venues() -> usize { 3 }
fn default_sor_max_slippage() -> f64 { 30.0 }
fn default_sor_prefer_maker() -> bool { true }

impl Default for SorConfigToml {
    fn default() -> Self {
        Self {
            min_split_size_usdt: default_sor_min_split(),
            max_venues: default_sor_max_venues(),
            max_slippage_bps: default_sor_max_slippage(),
            prefer_maker: default_sor_prefer_maker(),
        }
    }
}

/// Funding Rate Arbitrage configuration (TOML format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbConfigToml {
    /// Minimum net rate spread to consider actionable (e.g., 0.00005 = 0.005%).
    #[serde(default = "default_funding_min_net_rate")]
    pub min_net_rate: f64,
    /// Minimum annualized APR to consider actionable (e.g., 0.10 = 10%).
    #[serde(default = "default_funding_min_apr")]
    pub min_annualized_apr: f64,
    /// Refresh interval in seconds.
    #[serde(default = "default_funding_refresh_interval")]
    pub refresh_interval_secs: u64,
}

fn default_funding_min_net_rate() -> f64 { 0.00005 }
fn default_funding_min_apr() -> f64 { 0.10 }
fn default_funding_refresh_interval() -> u64 { 60 }

impl Default for FundingArbConfigToml {
    fn default() -> Self {
        Self {
            min_net_rate: default_funding_min_net_rate(),
            min_annualized_apr: default_funding_min_apr(),
            refresh_interval_secs: default_funding_refresh_interval(),
        }
    }
}

/// Cross-Venue Margin Monitor configuration (TOML format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginMonitorConfigToml {
    /// Minimum acceptable margin ratio before alert (e.g., 0.30 = 30%).
    #[serde(default = "default_margin_min_ratio")]
    pub min_margin_ratio: f64,
    /// Critical margin ratio threshold (e.g., 0.15 = 15%).
    #[serde(default = "default_margin_critical_ratio")]
    pub critical_margin_ratio: f64,
    /// Refresh interval in seconds.
    #[serde(default = "default_margin_refresh_interval")]
    pub refresh_interval_secs: u64,
}

fn default_margin_min_ratio() -> f64 { 0.30 }
fn default_margin_critical_ratio() -> f64 { 0.15 }
fn default_margin_refresh_interval() -> u64 { 30 }

impl Default for MarginMonitorConfigToml {
    fn default() -> Self {
        Self {
            min_margin_ratio: default_margin_min_ratio(),
            critical_margin_ratio: default_margin_critical_ratio(),
            refresh_interval_secs: default_margin_refresh_interval(),
        }
    }
}

// ---------------------------------------------------------------------------
// Spot-Futures Arbitrage Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Spot-Futures (Cash and Carry) funding rate arbitrage engine.
///
/// Controls all parameters for the institutional-grade spot-futures arb system:
/// - Position limits, leverage, hold times
/// - Minimum APR thresholds, take-profit targets
/// - Margin rebalancing parameters
/// - Hedge ratio tolerances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotFuturesConfig {
    /// Master toggle for the spot-futures engine.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum number of simultaneous spot-futures positions (default: 1).
    #[serde(default = "default_sf_max_positions")]
    pub max_positions: u32,
    /// Leverage for the futures short leg (1, 2, or 3). Default: 1.
    #[serde(default = "default_sf_short_leverage")]
    pub short_leverage: i32,
    /// Minimum projected APR (%) to enter a position. Default: 10.
    #[serde(default = "default_sf_min_apr")]
    pub min_apr_pct: f64,
    /// Maximum hold time in hours. Default: 72.
    #[serde(default = "default_sf_max_hold_hours")]
    pub max_hold_hours: f64,
    /// Margin ratio threshold to trigger rebalancing. Default: 0.30 (30%).
    #[serde(default = "default_sf_margin_rebalance_threshold")]
    pub margin_rebalance_threshold: f64,
    /// V1: same exchange only (spot + futures on same exchange). Default: true.
    #[serde(default = "default_sf_same_exchange_only")]
    pub same_exchange_only: bool,
    /// Spot order type for entry: "limit" or "market". Default: "limit".
    #[serde(default = "default_sf_spot_order_type")]
    pub spot_order_type: String,
    /// Number of historical funding rate observations to store per symbol. Default: 24.
    #[serde(default = "default_sf_funding_history_depth")]
    pub funding_history_depth: usize,
    /// Enable margin rebalancing (transfers between spot/futures wallets). Default: true.
    #[serde(default = "default_sf_rebalance_enabled")]
    pub rebalance_enabled: bool,
    /// Maximum hedge ratio deviation from 1.0 before corrective action. Default: 0.05.
    #[serde(default = "default_sf_hedge_ratio_tolerance")]
    pub hedge_ratio_tolerance: f64,
    /// Take profit when accumulated funding exceeds this % of deployed capital. Default: 0.5.
    #[serde(default = "default_sf_take_profit_pct")]
    pub take_profit_pct: f64,
    /// Maximum percentage of total capital to allocate per position. Default: 0.90 (90%).
    #[serde(default = "default_sf_max_position_pct")]
    pub max_position_pct: f64,
    /// Number of consecutive negative funding periods before exit. Default: 2.
    #[serde(default = "default_sf_negative_funding_exit_periods")]
    pub negative_funding_exit_periods: u32,
}

fn default_sf_max_positions() -> u32 { 1 }
fn default_sf_short_leverage() -> i32 { 1 }
fn default_sf_min_apr() -> f64 { 10.0 }
fn default_sf_max_hold_hours() -> f64 { 72.0 }
fn default_sf_margin_rebalance_threshold() -> f64 { 0.30 }
fn default_sf_same_exchange_only() -> bool { true }
fn default_sf_spot_order_type() -> String { "limit".to_string() }
fn default_sf_funding_history_depth() -> usize { 24 }
fn default_sf_rebalance_enabled() -> bool { true }
fn default_sf_hedge_ratio_tolerance() -> f64 { 0.05 }
fn default_sf_take_profit_pct() -> f64 { 0.5 }
fn default_sf_max_position_pct() -> f64 { 0.90 }
fn default_sf_negative_funding_exit_periods() -> u32 { 2 }

impl Default for SpotFuturesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_positions: default_sf_max_positions(),
            short_leverage: default_sf_short_leverage(),
            min_apr_pct: default_sf_min_apr(),
            max_hold_hours: default_sf_max_hold_hours(),
            margin_rebalance_threshold: default_sf_margin_rebalance_threshold(),
            same_exchange_only: default_sf_same_exchange_only(),
            spot_order_type: default_sf_spot_order_type(),
            funding_history_depth: default_sf_funding_history_depth(),
            rebalance_enabled: default_sf_rebalance_enabled(),
            hedge_ratio_tolerance: default_sf_hedge_ratio_tolerance(),
            take_profit_pct: default_sf_take_profit_pct(),
            max_position_pct: default_sf_max_position_pct(),
            negative_funding_exit_periods: default_sf_negative_funding_exit_periods(),
        }
    }
}

/// Multi-Exchange Feature configuration.
///
/// When `USE_MULTI_EXCHANGE=on`, this configuration controls:
/// - Simultaneous connections to Gate.io, Binance, and Bybit
/// - Consolidated Global Order Book fusion
/// - Smart Order Routing across all exchanges
/// - Cross-exchange funding rate arbitrage detection
/// - Cross-venue margin health monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiExchangeConfig {
    /// Master toggle — mirrors USE_MULTI_EXCHANGE env var.
    #[serde(default)]
    pub enabled: bool,
    /// Binance Futures API key.
    pub binance_api_key: Option<String>,
    /// Binance Futures API secret.
    pub binance_secret_key: Option<String>,
    /// Use Binance Futures testnet.
    #[serde(default)]
    pub binance_testnet: bool,
    /// Bybit v5 API key.
    pub bybit_api_key: Option<String>,
    /// Bybit v5 API secret.
    pub bybit_secret_key: Option<String>,
    /// Use Bybit testnet.
    #[serde(default)]
    pub bybit_testnet: bool,
    /// Smart Order Router configuration.
    #[serde(default)]
    pub sor: SorConfigToml,
    /// Funding rate arbitrage configuration.
    #[serde(default)]
    pub funding_arb: FundingArbConfigToml,
    /// Margin monitor configuration.
    #[serde(default)]
    pub margin_monitor: MarginMonitorConfigToml,
    /// Maximum open positions when multi-exchange is enabled (default: 5).
    #[serde(default = "default_multi_max_positions")]
    pub max_open_positions: u32,
    /// Spot-Futures (Cash and Carry) arbitrage configuration.
    #[serde(default)]
    pub spot_futures: SpotFuturesConfig,
}

fn default_multi_max_positions() -> u32 { 5 }

impl Default for MultiExchangeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            binance_api_key: None,
            binance_secret_key: None,
            binance_testnet: false,
            bybit_api_key: None,
            bybit_secret_key: None,
            bybit_testnet: false,
            sor: SorConfigToml::default(),
            funding_arb: FundingArbConfigToml::default(),
            margin_monitor: MarginMonitorConfigToml::default(),
            max_open_positions: default_multi_max_positions(),
            spot_futures: SpotFuturesConfig::default(),
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
    #[serde(default, deserialize_with = "deserialize_pair_profiles")]
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
    // === Auto-Protection Configuration (Feature 2) ===
    /// Configurable SL/TP auto-protection for unprotected positions.
    #[serde(default)]
    pub auto_protection: AutoProtectionConfig,
    // === Persistent State Configuration (Feature 4) ===
    /// SQLite-based persistent state storage.
    #[serde(default)]
    pub persistent_state: PersistentStateConfig,
    // === Multi-Exchange Feature (USE_MULTI_EXCHANGE) ===
    /// Master toggle for multi-exchange mode.
    #[serde(default)]
    pub multi_exchange_enabled: bool,
    /// Multi-exchange configuration (Binance, Bybit credentials and settings).
    #[serde(default)]
    pub multi_exchange: MultiExchangeConfig,
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
                    ws_url: "wss://fx-ws.gateio.ws/v4/ws/usdt".to_string(),
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
            auto_protection: AutoProtectionConfig::default(),
            persistent_state: PersistentStateConfig::default(),
            multi_exchange_enabled: false,
            multi_exchange: MultiExchangeConfig::default(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct PairProfileWrapper {
        #[serde(default, deserialize_with = "deserialize_pair_profiles")]
        pair_profiles: HashMap<String, PairProfile>,
    }

    #[test]
    fn deserializes_pair_profiles_from_array_of_tables() {
        let toml_str = r#"
            [[pair_profiles]]
            symbol = "BTC_USDT"
            imbalance_threshold = 0.05
            vpin_bucket_size = 500000.0
            trailing_stop_atr_multiplier = 3.0
            max_leverage = 20

            [[pair_profiles]]
            symbol = "ETH_USDT"
            imbalance_threshold = 0.03
            vpin_bucket_size = 200000.0
            trailing_stop_atr_multiplier = 2.0
            max_leverage = 25
        "#;

        let wrapper: PairProfileWrapper = toml::from_str(toml_str).expect("toml should parse");
        assert_eq!(wrapper.pair_profiles.len(), 2);
        assert!(wrapper.pair_profiles.contains_key("BTC_USDT"));
        assert!(wrapper.pair_profiles.contains_key("ETH_USDT"));
        assert_eq!(
            wrapper.pair_profiles["BTC_USDT"].imbalance_threshold,
            0.05
        );
    }
}
