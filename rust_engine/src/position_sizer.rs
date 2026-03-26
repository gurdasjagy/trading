//! Directive 2: Exchange-Aware Position Sizer with Quanto Multiplier.
//!
//! Gate.io Futures does NOT accept fractional contracts. One contract is NOT
//! one coin — it is `quanto_multiplier` of the base asset. For example:
//!   - BTC_USDT: `quanto_multiplier = 0.0001` → 1 contract = 0.0001 BTC
//!   - ETH_USDT: `quanto_multiplier = 0.01`   → 1 contract = 0.01 ETH
//!
//! The old code blindly rounded to `max(1, floor(amount))`, which caused
//! massive over-leveraging on expensive coins (1 BTC contract at $60k is
//! only ~$6 notional, but the old code thought it was buying 1 BTC).
//!
//! # Sizing Algorithm
//!
//! 1. `target_notional_usdt = account_balance * risk_pct * leverage`
//! 2. `target_base_amount = target_notional_usdt / entry_price`
//! 3. `contracts = floor(target_base_amount / quanto_multiplier)`
//! 4. Clamp to `[order_size_min, order_size_max]`
//! 5. Verify margin requirement: `contracts * quanto_multiplier * price / leverage <= available_margin`
//!
//! # Contract Info Fetching
//!
//! At startup, the engine fetches `/api/v4/futures/usdt/contracts/{symbol}`
//! for each configured symbol and caches the result.

use std::collections::HashMap;
use tracing::{info, warn, error};

use crate::realized_vol::VolatilityRegime;

// ═══════════════════════════════════════════════════════════════════════════
// Contract Specification
// ═══════════════════════════════════════════════════════════════════════════

/// Cached contract specification from Gate.io.
#[derive(Debug, Clone)]
pub struct ContractSpec {
    /// Gate.io contract name (e.g., "BTC_USDT").
    pub contract: String,
    /// How much of the base asset one contract represents.
    /// e.g., 0.0001 BTC per contract for BTC_USDT.
    pub quanto_multiplier: f64,
    /// Minimum order size in contracts.
    pub order_size_min: i64,
    /// Maximum order size in contracts.
    pub order_size_max: i64,
    /// Minimum leverage.
    pub leverage_min: i32,
    /// Maximum leverage.
    pub leverage_max: i32,
    /// Mark price precision (number of decimals).
    pub mark_price_precision: u32,
    /// Order price precision (number of decimals).
    pub order_price_precision: u32,
    /// Maintenance margin rate.
    pub maintenance_rate: f64,
    /// Maker fee rate.
    pub maker_fee_rate: f64,
    /// Taker fee rate.
    pub taker_fee_rate: f64,
}

impl ContractSpec {
    /// Parse from Gate.io REST API response JSON.
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        let contract = v.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if contract.is_empty() {
            return None;
        }

        let quanto_multiplier = parse_f64(v, "quanto_multiplier").unwrap_or(1.0);
        let order_size_min = v.get("order_size_min")
            .and_then(|v| v.as_i64())
            .unwrap_or(1);
        let order_size_max = v.get("order_size_max")
            .and_then(|v| v.as_i64())
            .unwrap_or(1_000_000);
        let leverage_min = v.get("leverage_min")
            .and_then(|v| v.as_i64())
            .unwrap_or(1) as i32;
        let leverage_max = v.get("leverage_max")
            .and_then(|v| v.as_i64())
            .unwrap_or(100) as i32;
        let maintenance_rate = parse_f64(v, "maintenance_rate").unwrap_or(0.005);
        let maker_fee_rate = parse_f64(v, "maker_fee_rate").unwrap_or(-0.000_25);
        let taker_fee_rate = parse_f64(v, "taker_fee_rate").unwrap_or(0.000_75);

        Some(Self {
            contract,
            quanto_multiplier,
            order_size_min,
            order_size_max,
            leverage_min,
            leverage_max,
            mark_price_precision: 8,
            order_price_precision: 8,
            maintenance_rate,
            maker_fee_rate,
            taker_fee_rate,
        })
    }

    /// Calculate the notional value of N contracts at a given price.
    #[inline]
    pub fn notional_usdt(&self, contracts: i64, price: f64) -> f64 {
        contracts.abs() as f64 * self.quanto_multiplier * price
    }

    /// Calculate the margin required for N contracts at a given price and leverage.
    #[inline]
    pub fn margin_required(&self, contracts: i64, price: f64, leverage: i32) -> f64 {
        self.notional_usdt(contracts, price) / leverage.max(1) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Position Sizing Engine
// ═══════════════════════════════════════════════════════════════════════════

/// Sizing result returned by the position sizer.
#[derive(Debug, Clone)]
pub struct SizingResult {
    /// Number of integer contracts to trade.
    pub contracts: i64,
    /// Notional value in USDT.
    pub notional_usdt: f64,
    /// Margin required for this position.
    pub margin_required: f64,
    /// Whether the size was clamped to min/max.
    pub was_clamped: bool,
    /// Reason if the sizing was rejected (None = approved).
    pub rejection: Option<String>,
}

/// Exchange-aware position sizing engine.
///
/// Thread-safe: contract specs are populated once at startup and then
/// read-only from the hot path.
pub struct PositionSizer {
    /// Contract specs keyed by normalized symbol name.
    specs: HashMap<String, ContractSpec>,
    /// Default quanto multiplier for unknown contracts.
    default_quanto_multiplier: f64,
}

impl PositionSizer {
    pub fn new() -> Self {
        Self {
            specs: HashMap::with_capacity(32),
            default_quanto_multiplier: 1.0,
        }
    }

    /// Register a contract specification (called at startup after REST fetch).
    pub fn register_spec(&mut self, spec: ContractSpec) {
        info!(
            "[sizer] Registered {}: quanto={}, min={}, max={}, lev={}-{}x, maker={:.4}%, taker={:.4}%",
            spec.contract, spec.quanto_multiplier,
            spec.order_size_min, spec.order_size_max,
            spec.leverage_min, spec.leverage_max,
            spec.maker_fee_rate * 100.0, spec.taker_fee_rate * 100.0,
        );
        self.specs.insert(spec.contract.clone(), spec);
    }

    /// Get the contract spec for a symbol.
    pub fn get_spec(&self, symbol: &str) -> Option<&ContractSpec> {
        self.specs.get(symbol)
    }

    /// Calculate adaptive leverage based on realized volatility (FEATURE 8).
    ///
    /// # Formula
    /// - ATR/price > 2% → leverage = 2x (high volatility)
    /// - ATR/price < 0.5% → leverage = 10x (low volatility)
    /// - Linear interpolation between 0.5% and 2%
    /// - Clamped to [2x, 20x]
    ///
    /// # Arguments
    /// * `atr` — Average True Range
    /// * `price` — Current price
    ///
    /// # Returns
    /// Adaptive leverage (2-20x)
    pub fn calculate_adaptive_leverage(&self, atr: f64, price: f64) -> i32 {
        if price <= 0.0 || atr <= 0.0 {
            return 5; // Default leverage
        }

        let atr_pct = (atr / price) * 100.0;

        let leverage = if atr_pct > 2.0 {
            2 // High volatility: conservative leverage
        } else if atr_pct < 0.5 {
            10 // Low volatility: aggressive leverage
        } else {
            // Linear interpolation: 2x at 2%, 10x at 0.5%
            // slope = (10 - 2) / (0.5 - 2.0) = -5.333
            // leverage = 10 + (-5.333) * (atr_pct - 0.5)
            let slope = (10.0 - 2.0) / (0.5 - 2.0);
            let lev = 10.0 + slope * (atr_pct - 0.5);
            lev.round() as i32
        };

        leverage.clamp(2, 20)
    }

    /// Calculate the correct integer contract count for a trade.
    ///
    /// # Arguments
    /// * `symbol` — Gate.io contract name (e.g., "BTC_USDT")
    /// * `target_notional_usdt` — Target trade size in USDT
    /// * `entry_price` — Expected entry price
    /// * `leverage` — Leverage to use
    /// * `available_margin` — Available margin in USDT
    pub fn calculate_contracts(
        &self,
        symbol: &str,
        target_notional_usdt: f64,
        entry_price: f64,
        leverage: i32,
        available_margin: f64,
    ) -> SizingResult {
        let spec = match self.specs.get(symbol) {
            Some(s) => s,
            None => {
                warn!(
                    "[sizer] No spec for {} — using default quanto_multiplier={}",
                    symbol, self.default_quanto_multiplier,
                );
                return self.calculate_with_default(
                    symbol,
                    target_notional_usdt,
                    entry_price,
                    leverage,
                    available_margin,
                );
            }
        };

        if entry_price <= 0.0 {
            return SizingResult {
                contracts: 0,
                notional_usdt: 0.0,
                margin_required: 0.0,
                was_clamped: false,
                rejection: Some("Invalid entry price".to_string()),
            };
        }

        // Step 1: target_base_amount = target_notional / entry_price
        let target_base_amount = target_notional_usdt / entry_price;

        // Step 2: contracts = floor(target_base_amount / quanto_multiplier)
        let raw_contracts = (target_base_amount / spec.quanto_multiplier).floor() as i64;

        // Step 3: Clamp to [order_size_min, order_size_max]
        let mut contracts = raw_contracts.max(spec.order_size_min).min(spec.order_size_max);
        let was_clamped = contracts != raw_contracts;

        if was_clamped {
            info!(
                "[sizer] {} clamped: raw={} -> {} (min={}, max={})",
                symbol, raw_contracts, contracts, spec.order_size_min, spec.order_size_max,
            );
        }

        // Step 4: Verify margin requirement
        let _notional = spec.notional_usdt(contracts, entry_price);
        let margin_req = spec.margin_required(contracts, entry_price, leverage);

        if margin_req > available_margin {
            // Scale down to fit available margin
            let max_notional = available_margin * leverage as f64;
            let max_base = max_notional / entry_price;
            contracts = (max_base / spec.quanto_multiplier).floor() as i64;
            contracts = contracts.max(spec.order_size_min);

            let new_margin = spec.margin_required(contracts, entry_price, leverage);
            if new_margin > available_margin {
                return SizingResult {
                    contracts: 0,
                    notional_usdt: 0.0,
                    margin_required: new_margin,
                    was_clamped: true,
                    rejection: Some(format!(
                        "Insufficient margin: need {:.2} USDT, have {:.2} USDT",
                        new_margin, available_margin,
                    )),
                };
            }

            warn!(
                "[sizer] {} scaled for margin: {} contracts, notional={:.2} USDT, margin={:.2} USDT",
                symbol, contracts, spec.notional_usdt(contracts, entry_price), new_margin,
            );
        }

        // Step 5: Final validation
        if contracts < spec.order_size_min {
            return SizingResult {
                contracts: 0,
                notional_usdt: 0.0,
                margin_required: 0.0,
                was_clamped: true,
                rejection: Some(format!(
                    "Below minimum order size: {} < {}",
                    contracts, spec.order_size_min,
                )),
            };
        }

        let final_notional = spec.notional_usdt(contracts, entry_price);
        let final_margin = spec.margin_required(contracts, entry_price, leverage);

        info!(
            "[sizer] {} sized: {} contracts, notional={:.2} USDT, margin={:.2} USDT (quanto={})",
            symbol, contracts, final_notional, final_margin, spec.quanto_multiplier,
        );

        SizingResult {
            contracts,
            notional_usdt: final_notional,
            margin_required: final_margin,
            was_clamped,
            rejection: None,
        }
    }

    /// Fallback sizing when no spec is available (unknown contract).
    fn calculate_with_default(
        &self,
        _symbol: &str,
        target_notional_usdt: f64,
        entry_price: f64,
        leverage: i32,
        available_margin: f64,
    ) -> SizingResult {
        if entry_price <= 0.0 {
            return SizingResult {
                contracts: 0,
                notional_usdt: 0.0,
                margin_required: 0.0,
                was_clamped: false,
                rejection: Some("Invalid entry price".to_string()),
            };
        }

        let target_base = target_notional_usdt / entry_price;
        let contracts = (target_base / self.default_quanto_multiplier).floor() as i64;
        let contracts = contracts.max(1);
        let notional = contracts as f64 * self.default_quanto_multiplier * entry_price;
        let margin_req = notional / leverage.max(1) as f64;

        if margin_req > available_margin {
            return SizingResult {
                contracts: 0,
                notional_usdt: 0.0,
                margin_required: margin_req,
                was_clamped: false,
                rejection: Some(format!(
                    "Insufficient margin (default sizing): need {:.2}, have {:.2}",
                    margin_req, available_margin,
                )),
            };
        }

        SizingResult {
            contracts,
            notional_usdt: notional,
            margin_required: margin_req,
            was_clamped: false,
            rejection: None,
        }
    }
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FEATURE 9: Volatility-Adjusted Position Sizing
// ═══════════════════════════════════════════════════════════════════════════

/// FEAT 9: Volatility regime thresholds and scale factors as specified:
/// - Low vol (< 20% annualized): 1.5x base size
/// - Normal vol (20-50%): 1.0x base size
/// - High vol (50-80%): 0.5x base size
/// - Extreme vol (> 80%): 0.25x base size
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolAdjustedRegime {
    /// Low volatility (< 20% annualized) — increase position size
    Low,
    /// Normal volatility (20-50%) — baseline position size
    Normal,
    /// High volatility (50-80%) — reduce position size
    High,
    /// Extreme volatility (> 80%) — aggressively reduce position size
    Extreme,
}

impl VolAdjustedRegime {
    /// Classify volatility into a FEAT 9 regime.
    pub fn from_annualized_vol(vol_pct: f64) -> Self {
        if vol_pct < 20.0 {
            VolAdjustedRegime::Low
        } else if vol_pct < 50.0 {
            VolAdjustedRegime::Normal
        } else if vol_pct < 80.0 {
            VolAdjustedRegime::High
        } else {
            VolAdjustedRegime::Extreme
        }
    }

    /// Get the position size multiplier for this regime.
    pub fn scale_factor(&self) -> f64 {
        match self {
            VolAdjustedRegime::Low => 1.5,
            VolAdjustedRegime::Normal => 1.0,
            VolAdjustedRegime::High => 0.5,
            VolAdjustedRegime::Extreme => 0.25,
        }
    }

    /// Get the regime name.
    pub fn name(&self) -> &'static str {
        match self {
            VolAdjustedRegime::Low => "low",
            VolAdjustedRegime::Normal => "normal",
            VolAdjustedRegime::High => "high",
            VolAdjustedRegime::Extreme => "extreme",
        }
    }
}

/// FEAT 9: Result of volatility-adjusted sizing.
#[derive(Debug, Clone)]
pub struct VolAdjustedSizingResult {
    /// The final sizing result.
    pub sizing: SizingResult,
    /// Volatility regime used.
    pub regime: VolAdjustedRegime,
    /// Annualized volatility percentage.
    pub volatility_pct: f64,
    /// Scale factor applied.
    pub vol_scale_factor: f64,
    /// Kelly fraction applied (if Kelly is enabled).
    pub kelly_fraction: Option<f64>,
    /// Original target notional before adjustment.
    pub original_notional_usdt: f64,
    /// Adjusted target notional after volatility scaling.
    pub adjusted_notional_usdt: f64,
}

impl PositionSizer {
    /// FEAT 9: Calculate position size adjusted for current realized volatility.
    ///
    /// Uses the FEAT 9 thresholds:
    /// - < 20% annualized vol: 1.5x base size
    /// - 20-50%: 1.0x base size
    /// - 50-80%: 0.5x base size
    /// - > 80%: 0.25x base size
    ///
    /// Optionally applies Kelly Criterion for optimal sizing.
    pub fn calculate_vol_adjusted(
        &self,
        symbol: &str,
        base_notional_usdt: f64,
        entry_price: f64,
        leverage: i32,
        available_margin: f64,
        annualized_vol_pct: f64,
        kelly_params: Option<&KellyParams>,
    ) -> VolAdjustedSizingResult {
        // Step 1: Classify regime and get scale factor
        let regime = VolAdjustedRegime::from_annualized_vol(annualized_vol_pct);
        let vol_scale = regime.scale_factor();

        // Step 2: Apply Kelly Criterion if parameters provided
        let (kelly_fraction, kelly_scale) = match kelly_params {
            Some(params) => {
                let fraction = params.half_kelly_fraction();
                // Kelly fraction scales the position; use it as a multiplier
                // but clamp to [0.1, 2.0] to avoid extremes
                let scale = fraction.clamp(0.1, 2.0);
                (Some(fraction), scale)
            }
            None => (None, 1.0),
        };

        // Step 3: Calculate adjusted notional
        let adjusted_notional = base_notional_usdt * vol_scale * kelly_scale;

        info!(
            "[sizer] FEAT9 vol-adjusted {}: vol={:.1}% regime={} scale={:.2}x kelly={:.3} -> notional ${:.2} (base ${:.2})",
            symbol, annualized_vol_pct, regime.name(), vol_scale,
            kelly_scale, adjusted_notional, base_notional_usdt
        );

        // Step 4: Calculate contracts using adjusted notional
        let sizing = self.calculate_contracts(
            symbol,
            adjusted_notional,
            entry_price,
            leverage,
            available_margin,
        );

        VolAdjustedSizingResult {
            sizing,
            regime,
            volatility_pct: annualized_vol_pct,
            vol_scale_factor: vol_scale,
            kelly_fraction,
            original_notional_usdt: base_notional_usdt,
            adjusted_notional_usdt: adjusted_notional,
        }
    }

    /// FEAT 9: Convenience method using a RealizedVolatilityCalculator directly.
    ///
    /// Reads the current volatility from the calculator and delegates to
    /// `calculate_vol_adjusted`.
    pub fn calculate_with_vol_calculator(
        &self,
        symbol: &str,
        base_notional_usdt: f64,
        entry_price: f64,
        leverage: i32,
        available_margin: f64,
        vol_calculator: &crate::realized_vol::RealizedVolatilityCalculator,
        kelly_params: Option<&KellyParams>,
    ) -> VolAdjustedSizingResult {
        let vol_pct = vol_calculator.get_volatility();
        self.calculate_vol_adjusted(
            symbol,
            base_notional_usdt,
            entry_price,
            leverage,
            available_margin,
            vol_pct,
            kelly_params,
        )
    }
}

/// FEAT 9: Kelly Criterion parameters for optimal position sizing.
#[derive(Debug, Clone)]
pub struct KellyParams {
    /// Historical win rate (0.0 to 1.0).
    pub win_rate: f64,
    /// Average win / average loss ratio (e.g., 2.0 = wins are 2x losses).
    pub win_loss_ratio: f64,
}

impl KellyParams {
    /// Create new Kelly parameters.
    pub fn new(win_rate: f64, win_loss_ratio: f64) -> Self {
        Self {
            win_rate: win_rate.clamp(0.01, 0.99),
            win_loss_ratio: win_loss_ratio.max(0.01),
        }
    }

    /// Calculate full Kelly fraction: f* = (p * b - q) / b
    /// where p = win_rate, b = win/loss ratio, q = 1 - p
    pub fn full_kelly_fraction(&self) -> f64 {
        let p = self.win_rate;
        let b = self.win_loss_ratio;
        let q = 1.0 - p;
        let f = (p * b - q) / b;
        f.max(0.0) // Never go negative (no edge)
    }

    /// Calculate half-Kelly fraction (safer: f*/2).
    pub fn half_kelly_fraction(&self) -> f64 {
        self.full_kelly_fraction() * 0.5
    }

    /// Calculate quarter-Kelly fraction (very conservative: f*/4).
    pub fn quarter_kelly_fraction(&self) -> f64 {
        self.full_kelly_fraction() * 0.25
    }
}

impl Default for KellyParams {
    fn default() -> Self {
        Self {
            win_rate: 0.55,
            win_loss_ratio: 1.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Upgrade 2: Dynamic Position Sizing — Kelly Criterion + Volatility Scaling
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for dynamic position sizing using Kelly Criterion,
/// volatility scaling, drawdown scaling, and confidence weighting.
#[derive(Debug, Clone)]
pub struct DynamicSizingParams {
    /// Win rate (0.0 to 1.0). Historical probability of winning trades.
    pub win_rate: f64,
    /// Average win / average loss ratio. E.g., 2.0 means wins are 2x losses.
    pub avg_win_loss_ratio: f64,
    /// Realized volatility (annualized, e.g., 0.50 = 50%).
    pub realized_volatility: f64,
    /// Target volatility (annualized, e.g., 0.15 = 15%).
    pub target_volatility: f64,
    /// Current drawdown percentage (0.0 to 1.0).
    pub current_drawdown_pct: f64,
    /// Maximum allowed drawdown percentage (e.g., 0.10 = 10%).
    pub max_drawdown_pct: f64,
    /// Signal confidence from the Alpha Oracle (0.0 to 1.0).
    pub signal_confidence: f64,
}

impl Default for DynamicSizingParams {
    fn default() -> Self {
        Self {
            win_rate: 0.55,
            avg_win_loss_ratio: 1.5,
            realized_volatility: 0.50,
            target_volatility: 0.15,
            current_drawdown_pct: 0.0,
            max_drawdown_pct: 0.10,
            signal_confidence: 0.7,
        }
    }
}

/// Result of the dynamic sizing calculation, including breakdown of scalars.
#[derive(Debug, Clone)]
pub struct DynamicSizingResult {
    /// Final sizing result (contracts, notional, etc.)
    pub sizing: SizingResult,
    /// Half-Kelly fraction used.
    pub kelly_fraction: f64,
    /// Volatility scalar applied.
    pub vol_scalar: f64,
    /// Drawdown scalar applied.
    pub dd_scalar: f64,
    /// Confidence scalar applied.
    pub conf_scalar: f64,
    /// Final adjusted risk percentage.
    pub adjusted_risk_pct: f64,
    /// Original base risk percentage.
    pub base_risk_pct: f64,
}

impl PositionSizer {
    /// Calculate dynamically-sized contract count using Kelly Criterion,
    /// volatility scaling, drawdown scaling, and confidence weighting.
    ///
    /// # Kelly Criterion (Half-Kelly)
    /// `f* = ((p * b) - q) / b` where p=win_rate, b=win/loss ratio, q=1-p
    /// We use half-Kelly (`f*/2`) for safety margin.
    ///
    /// # Volatility Scaling
    /// `vol_scalar = target_vol / realized_vol`, clamped to [0.25, 2.0]
    ///
    /// # Drawdown Scaling
    /// `dd_scalar = 1.0 - (current_dd / max_dd)`, clamped to [0.1, 1.0]
    ///
    /// # Confidence Weighting
    /// `conf_scalar = signal_confidence`, clamped to [0.3, 1.0]
    ///
    /// # Final Size
    /// `adjusted_risk = base_risk * kelly * vol_scalar * dd_scalar * conf_scalar`
    pub fn calculate_dynamic_contracts(
        &self,
        symbol: &str,
        base_risk_pct: f64,
        entry_price: f64,
        leverage: i32,
        available_margin: f64,
        params: &DynamicSizingParams,
    ) -> DynamicSizingResult {
        // 1. Half-Kelly fraction
        let p = params.win_rate.clamp(0.01, 0.99);
        let b = params.avg_win_loss_ratio.max(0.01);
        let q = 1.0 - p;
        let kelly_full = ((p * b) - q) / b;
        let kelly_fraction = kelly_full.max(0.0) * 0.5; // Half-Kelly for safety

        // 2. Volatility scaling
        let vol_scalar = (params.target_volatility / params.realized_volatility.max(0.001))
            .clamp(0.25, 2.0);

        // 3. Drawdown scaling
        let dd_scalar = if params.max_drawdown_pct > 0.0 {
            (1.0 - params.current_drawdown_pct / params.max_drawdown_pct)
                .clamp(0.1, 1.0)
        } else {
            1.0
        };

        // 4. Confidence scalar
        let conf_scalar = params.signal_confidence.clamp(0.3, 1.0);

        // 5. Final adjusted risk percentage
        // If Kelly suggests zero (negative edge), use a minimal risk
        let kelly_adj = if kelly_fraction > 0.001 { kelly_fraction } else { 0.1 };
        let adjusted_risk_pct = base_risk_pct * kelly_adj * vol_scalar * dd_scalar * conf_scalar;

        // Clamp to reasonable bounds: min 0.1%, max 5% of balance
        let final_risk_pct = adjusted_risk_pct.clamp(0.001, 0.05);

        // 6. Calculate target notional and size
        let target_notional = available_margin * final_risk_pct * leverage as f64;

        info!(
            "[sizer] Dynamic sizing for {}: kelly={:.3}, vol_sc={:.2}, dd_sc={:.2}, conf={:.2} -> risk={:.4}% (base={:.2}%)",
            symbol, kelly_fraction, vol_scalar, dd_scalar, conf_scalar,
            final_risk_pct * 100.0, base_risk_pct * 100.0,
        );

        let sizing = self.calculate_contracts(
            symbol,
            target_notional,
            entry_price,
            leverage,
            available_margin,
        );

        DynamicSizingResult {
            sizing,
            kelly_fraction,
            vol_scalar,
            dd_scalar,
            conf_scalar,
            adjusted_risk_pct: final_risk_pct,
            base_risk_pct,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Contract Info Fetcher
// ═══════════════════════════════════════════════════════════════════════════

/// Fetch contract specifications from Gate.io REST API.
///
/// Called once at startup. Populates the PositionSizer's spec cache.
pub async fn fetch_contract_specs(
    client: &reqwest::Client,
    base_url: &str,
    _api_key: &str,
    _api_secret: &[u8],
    symbols: &[String],
) -> Vec<ContractSpec> {
    let mut specs = Vec::with_capacity(symbols.len());
    let all_path = "/futures/usdt/contracts";
    let url = format!("{}{}", base_url, all_path);

    info!("[sizer] Fetching contract specs from {}", url);

    // Fetch all contracts in one call (more efficient than per-symbol)
    // NOTE: /futures/usdt/contracts is a PUBLIC endpoint — no authentication needed.
    // Sending auth headers to the testnet proxy causes HTTP 502 Bad Gateway.
    match client
        .get(&url)
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        if let Some(arr) = data.as_array() {
                            let symbol_set: std::collections::HashSet<String> =
                                symbols.iter().cloned().collect();

                            for item in arr {
                                let name = item
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default();
                                if symbol_set.contains(name) || symbol_set.is_empty() {
                                    if let Some(spec) = ContractSpec::from_json(item) {
                                        specs.push(spec);
                                    }
                                }
                            }
                            info!(
                                "[sizer] Fetched {} contract specs ({} total available)",
                                specs.len(),
                                arr.len()
                            );
                        }
                    }
                    Err(e) => error!("[sizer] Failed to parse contract specs: {}", e),
                }
            } else {
                error!(
                    "[sizer] Contract specs fetch failed: HTTP {}",
                    resp.status()
                );
            }
        }
        Err(e) => error!("[sizer] Contract specs request failed: {}", e),
    }

    specs
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn parse_f64(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key)
        .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn btc_spec() -> ContractSpec {
        ContractSpec {
            contract: "BTC_USDT".to_string(),
            quanto_multiplier: 0.0001,
            order_size_min: 1,
            order_size_max: 1_000_000,
            leverage_min: 1,
            leverage_max: 100,
            mark_price_precision: 8,
            order_price_precision: 2,
            maintenance_rate: 0.005,
            maker_fee_rate: -0.000_25,
            taker_fee_rate: 0.000_75,
        }
    }

    fn eth_spec() -> ContractSpec {
        ContractSpec {
            contract: "ETH_USDT".to_string(),
            quanto_multiplier: 0.01,
            order_size_min: 1,
            order_size_max: 1_000_000,
            leverage_min: 1,
            leverage_max: 100,
            mark_price_precision: 8,
            order_price_precision: 2,
            maintenance_rate: 0.005,
            maker_fee_rate: -0.000_25,
            taker_fee_rate: 0.000_75,
        }
    }

    #[test]
    fn test_btc_sizing_100_usdt() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(btc_spec());

        // Want $100 notional at $60,000 BTC
        let result = sizer.calculate_contracts(
            "BTC_USDT", 100.0, 60000.0, 10, 1000.0,
        );
        assert!(result.rejection.is_none());
        // $100 / $60000 = 0.001667 BTC
        // 0.001667 / 0.0001 = 16.67 → floor = 16 contracts
        assert_eq!(result.contracts, 16);
        // Verify notional: 16 * 0.0001 * 60000 = 96.0
        assert!((result.notional_usdt - 96.0).abs() < 0.01);
    }

    #[test]
    fn test_eth_sizing_500_usdt() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(eth_spec());

        // Want $500 notional at $3,000 ETH
        let result = sizer.calculate_contracts(
            "ETH_USDT", 500.0, 3000.0, 5, 1000.0,
        );
        assert!(result.rejection.is_none());
        // $500 / $3000 = 0.1667 ETH
        // 0.1667 / 0.01 = 16.67 → floor = 16 contracts
        assert_eq!(result.contracts, 16);
    }

    #[test]
    fn test_margin_rejection() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(btc_spec());

        // Want $10,000 notional but only $50 margin at 10x leverage
        let result = sizer.calculate_contracts(
            "BTC_USDT", 10000.0, 60000.0, 10, 50.0,
        );
        // Max affordable: $50 * 10 = $500 notional
        // $500 / $60000 / 0.0001 = 83 contracts → margin = 83 * 0.0001 * 60000 / 10 = $49.8
        assert!(result.rejection.is_none());
        assert!(result.contracts <= 83);
    }

    #[test]
    fn test_never_sends_fractional() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(btc_spec());

        // Very small target
        let result = sizer.calculate_contracts(
            "BTC_USDT", 5.0, 60000.0, 10, 1000.0,
        );
        // $5 / $60000 / 0.0001 = 0.833 → floor = 0, clamped to min=1
        assert_eq!(result.contracts, 1);
    }

    // ─── FEAT 9 Tests ───

    #[test]
    fn test_vol_regime_classification() {
        assert_eq!(VolAdjustedRegime::from_annualized_vol(10.0), VolAdjustedRegime::Low);
        assert_eq!(VolAdjustedRegime::from_annualized_vol(30.0), VolAdjustedRegime::Normal);
        assert_eq!(VolAdjustedRegime::from_annualized_vol(60.0), VolAdjustedRegime::High);
        assert_eq!(VolAdjustedRegime::from_annualized_vol(90.0), VolAdjustedRegime::Extreme);
    }

    #[test]
    fn test_vol_scale_factors() {
        assert!((VolAdjustedRegime::Low.scale_factor() - 1.5).abs() < 0.01);
        assert!((VolAdjustedRegime::Normal.scale_factor() - 1.0).abs() < 0.01);
        assert!((VolAdjustedRegime::High.scale_factor() - 0.5).abs() < 0.01);
        assert!((VolAdjustedRegime::Extreme.scale_factor() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_vol_adjusted_sizing_low_vol() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(btc_spec());

        // Low vol (15%) should give 1.5x base size
        let result = sizer.calculate_vol_adjusted(
            "BTC_USDT", 100.0, 60000.0, 10, 10000.0,
            15.0, // < 20% = Low
            None,
        );

        assert_eq!(result.regime, VolAdjustedRegime::Low);
        assert!((result.vol_scale_factor - 1.5).abs() < 0.01);
        assert!((result.adjusted_notional_usdt - 150.0).abs() < 0.01);
        assert!(result.sizing.contracts > 0);
    }

    #[test]
    fn test_vol_adjusted_sizing_extreme_vol() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(btc_spec());

        // Extreme vol (90%) should give 0.25x base size
        let result = sizer.calculate_vol_adjusted(
            "BTC_USDT", 1000.0, 60000.0, 10, 10000.0,
            90.0, // > 80% = Extreme
            None,
        );

        assert_eq!(result.regime, VolAdjustedRegime::Extreme);
        assert!((result.vol_scale_factor - 0.25).abs() < 0.01);
        assert!((result.adjusted_notional_usdt - 250.0).abs() < 0.01);
    }

    #[test]
    fn test_vol_adjusted_with_kelly() {
        let mut sizer = PositionSizer::new();
        sizer.register_spec(btc_spec());

        let kelly = KellyParams::new(0.60, 2.0);
        // Full Kelly = (0.60 * 2.0 - 0.40) / 2.0 = 0.40
        // Half Kelly = 0.20
        assert!((kelly.half_kelly_fraction() - 0.20).abs() < 0.01);

        // Normal vol + Kelly should apply both scalars
        let result = sizer.calculate_vol_adjusted(
            "BTC_USDT", 1000.0, 60000.0, 10, 10000.0,
            30.0, // Normal vol
            Some(&kelly),
        );

        assert_eq!(result.regime, VolAdjustedRegime::Normal);
        assert!(result.kelly_fraction.is_some());
        // adjusted = 1000 * 1.0 (vol) * 0.20 (kelly, clamped to 0.1 min)
        // Kelly half = 0.20, clamped to [0.1, 2.0] = 0.20
        assert!((result.adjusted_notional_usdt - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_kelly_params() {
        // Edge case: no edge (win_rate < break-even)
        let no_edge = KellyParams::new(0.30, 1.0);
        assert_eq!(no_edge.full_kelly_fraction(), 0.0);
        assert_eq!(no_edge.half_kelly_fraction(), 0.0);

        // Good edge
        let good_edge = KellyParams::new(0.55, 1.5);
        let f = good_edge.full_kelly_fraction();
        assert!(f > 0.0);
        assert!(f < 1.0);
    }
}
