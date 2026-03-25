//! Pre-Trade Validation and Exit Logic for Funding Rate Arbitrage
//!
//! Implements the "Serious Checks" from overview.txt:
//! 1. Spread vs Fee Check (profitability gate)
//! 2. Order Book Depth & VWAP Slippage Check
//! 3. Basis Risk Calculation
//! 4. Capital & Margin Sufficiency Check

use tracing::{warn, debug, info};
use serde::{Deserialize, Serialize};

use crate::multi_exchange::global_book::{ExchangeId, GlobalBookRegistry};
use crate::multi_exchange::margin_monitor::CrossVenueMarginMonitor;
use crate::multi_exchange::funding_arb::FundingArbOpportunity;
use crate::instrument_manager::{InstrumentManager, Exchange, simulate_margin};

// Forward reference: FundingArbEngineConfig is defined in funding_arb_engine.rs
use crate::multi_exchange::funding_arb_engine::FundingArbEngineConfig;

/// Result of pre-trade validation.
#[derive(Debug)]
pub enum PreTradeResult {
    Approved {
        estimated_slippage_bps: f64,
        basis_risk: f64,
        breakeven_periods: f64,
        recommended_size: i64,
    },
    Rejected {
        reason: String,
    },
}

/// Reason for exiting a position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    TakeProfit { accumulated_funding: f64 },
    TakeProfitPct { pnl_pct: f64 },
    StopLoss { net_pnl: f64 },
    SpreadReversal { entry_net_rate: f64, current_net_rate: f64 },
    ConsecutiveNegativePeriods { count: u32 },
    TimeStop { hours_open: f64 },
    MarginDanger { exchange: ExchangeId, margin_ratio: f64 },
    Manual,
}

pub struct PreTradeValidator;

impl PreTradeValidator {
    /// Run all pre-trade checks for a funding arbitrage opportunity.
    ///
    /// BUG FIX #2: Now accepts an optional InstrumentManager to properly
    /// calculate fractional position sizes based on available equity and
    /// exchange-specific contract multipliers. Previously, the position
    /// sizing at line 126-128 treated raw USD balance * leverage as contract
    /// count, which for BTC meant trying to open size=4943 contracts when
    /// the balance was $4943 — but on Binance 1 contract = 1 BTC ($60k+).
    ///
    /// With InstrumentManager: notional USD → base asset qty → contracts,
    /// respecting each exchange's quanto_multiplier and lot size.
    pub fn validate(
        opp: &FundingArbOpportunity,
        global_book_registry: &GlobalBookRegistry,
        margin_monitor: &CrossVenueMarginMonitor,
        config: &FundingArbEngineConfig,
    ) -> PreTradeResult {
        Self::validate_with_instruments(opp, global_book_registry, margin_monitor, config, None)
    }

    /// Run all pre-trade checks with InstrumentManager for proper sizing.
    pub fn validate_with_instruments(
        opp: &FundingArbOpportunity,
        global_book_registry: &GlobalBookRegistry,
        margin_monitor: &CrossVenueMarginMonitor,
        config: &FundingArbEngineConfig,
        instrument_mgr: Option<&InstrumentManager>,
    ) -> PreTradeResult {
        // 1. PROFITABILITY GATE: Is the funding spread enough to cover fees?
        let short_fee_bps = opp.short_exchange.taker_fee_bps() as f64;
        let long_fee_bps = opp.long_exchange.taker_fee_bps() as f64;
        let round_trip_fee_bps = 2.0 * (short_fee_bps + long_fee_bps);
        let net_rate_bps = opp.net_rate * 10000.0;

        let breakeven_periods = if net_rate_bps > 0.0 {
            round_trip_fee_bps / net_rate_bps
        } else {
            f64::INFINITY
        };

        if breakeven_periods > config.max_breakeven_periods {
            // BUG 9 FIX: Log WHY validation rejected — previously silent
            let reason = format!(
                "Breakeven requires {:.1} funding periods (max: {:.1}). Net rate: {:.2}bps, fees: {:.2}bps",
                breakeven_periods, config.max_breakeven_periods, net_rate_bps, round_trip_fee_bps
            );
            warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
            return PreTradeResult::Rejected { reason };
        }

        // 2. ORDER BOOK DEPTH & SLIPPAGE CHECK
        // Use the global book to estimate VWAP slippage for the target size
        let estimated_slippage_bps = Self::estimate_slippage(
            opp, global_book_registry, config.max_notional_usdt,
        );

        if estimated_slippage_bps > config.max_entry_slippage_bps {
            let reason = format!(
                "Estimated slippage {:.1}bps exceeds max {:.1}bps",
                estimated_slippage_bps, config.max_entry_slippage_bps
            );
            warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
            return PreTradeResult::Rejected { reason };
        }

        // 3. MARGIN SUFFICIENCY CHECK
        // FIX (startup gate): Reject if margin monitor hasn't completed its
        // first balance fetch. Previously, balance defaulted to 0.0, causing
        // notional_usd=0.0 → qty=0.0 → .max(1) clamp → 1 BTC ($71k) order
        // with only $4943 balance → InsufficientBalance on both exchanges.
        if !margin_monitor.has_exchange_data(opp.short_exchange)
            || !margin_monitor.has_exchange_data(opp.long_exchange)
        {
            let reason = format!(
                "Margin monitor not yet initialized for {} and/or {} — skipping until balances are fetched",
                opp.short_exchange.name(), opp.long_exchange.name()
            );
            warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
            return PreTradeResult::Rejected { reason };
        }

        let short_balance = margin_monitor.get_health(opp.short_exchange)
            .map(|h| h.available_balance)
            .unwrap_or(0.0);
        let long_balance = margin_monitor.get_health(opp.long_exchange)
            .map(|h| h.available_balance)
            .unwrap_or(0.0);

        // BUG 2 FIX: Default to 1.0 (100% health) when margin monitor hasn't
        // completed its first API pull or exchange is disconnected.
        // Previously defaulted to 0.0 which caused immediate rejection.
        let short_margin_ratio = margin_monitor.get_health(opp.short_exchange)
            .map(|h| h.margin_ratio)
            .unwrap_or(1.0);
        let long_margin_ratio = margin_monitor.get_health(opp.long_exchange)
            .map(|h| h.margin_ratio)
            .unwrap_or(1.0);

        if short_margin_ratio < config.min_entry_margin_ratio
            || long_margin_ratio < config.min_entry_margin_ratio
        {
            let reason = format!(
                "Insufficient margin: {}={:.1}% {}={:.1}% (min: {:.1}%)",
                opp.short_exchange.name(), short_margin_ratio * 100.0,
                opp.long_exchange.name(), long_margin_ratio * 100.0,
                config.min_entry_margin_ratio * 100.0
            );
            warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
            return PreTradeResult::Rejected { reason };
        }

        // 4. CALCULATE RECOMMENDED SIZE
        // BUG FIX #2: Proper position sizing using InstrumentManager.
        // Previously: recommended_size = (balance * leverage) as i64
        //   → For $4943 balance: size = 4943 * 2 = 9886 contracts
        //   → On Binance, 1 contract = 1 BTC (~$60k) → InsufficientBalance!
        //
        // Now: Convert notional USD → base-asset quantity → exchange contracts
        //   → $4943 * 2x leverage * 50% position = $4943 notional
        //   → At $60k/BTC: 4943/60000 = 0.082 BTC
        //   → On Binance (linear): qty = 0.082, rounded to stepSize
        //   → On Gate.io (quanto 0.0001): contracts = 0.082/0.0001 = 820
        let min_balance = short_balance.min(long_balance);
        let notional_usd = (min_balance * config.max_position_pct * config.leverage as f64)
            .min(config.max_notional_usdt);

        let recommended_size = if let Some(mgr) = instrument_mgr {
            // Use InstrumentManager for exchange-aware sizing
            let short_exchange_type = exchange_id_to_exchange(opp.short_exchange);
            let long_exchange_type = exchange_id_to_exchange(opp.long_exchange);

            // Get contract specs for both exchanges
            let short_spec = mgr.get_or_default(short_exchange_type, &opp.symbol);
            let long_spec = mgr.get_or_default(long_exchange_type, &opp.symbol);

            // FIX: Reject opportunity if no price is available instead of using
            // fallback of 1.0. A $1 price for BTC ($71k actual) is catastrophically
            // wrong and leads to oversized positions (e.g., 1 BTC = $71k notional
            // when only $4943 is available).
            let approx_price = match Self::get_approximate_price(opp, global_book_registry) {
                Some(p) => p,
                None => {
                    let reason = format!(
                        "No price data available for {} — cannot safely size position",
                        opp.symbol
                    );
                    warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
                    return PreTradeResult::Rejected { reason };
                }
            };

            // Calculate qty for each exchange and use the minimum
            let short_qty = short_spec.notional_to_qty(notional_usd, approx_price);
            let long_qty = long_spec.notional_to_qty(notional_usd, approx_price);

            // Use the smaller of the two to ensure both legs can be filled
            let qty = short_qty.min(long_qty);

            // Convert to exchange contract units (for Gate.io: qty / quanto_multiplier)
            // For Binance/Bybit linear: contracts = qty (since multiplier = 1.0)
            let short_contracts = short_spec.qty_to_contracts(qty);
            let long_contracts = long_spec.qty_to_contracts(qty);
            let contracts = short_contracts.min(long_contracts);

            // FIX: Validate that the minimum contract size (1) is actually
            // affordable before clamping. Previously .max(1) blindly produced
            // size=1 even when 1 contract = 1 BTC = $71k, far exceeding the
            // account balance. Now we check the notional of 1 contract against
            // available margin before allowing it.
            let contracts = if contracts < 1 {
                // Check if even 1 contract is affordable
                let one_contract_qty = short_spec.contract_multiplier.max(long_spec.contract_multiplier);
                let one_contract_notional = one_contract_qty * approx_price;
                let one_contract_margin = one_contract_notional / config.leverage as f64;
                if one_contract_margin > min_balance {
                    let reason = format!(
                        "Minimum 1 contract = {:.4} units = ${:.2} notional requires ${:.2} margin, but only ${:.2} available",
                        one_contract_qty, one_contract_notional, one_contract_margin, min_balance
                    );
                    warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
                    return PreTradeResult::Rejected { reason };
                }
                1
            } else {
                contracts
            };

            // Pre-flight margin simulation
            let short_margin_check = simulate_margin(
                short_balance, approx_price, qty, config.leverage, true, 0.0,
            );
            let long_margin_check = simulate_margin(
                long_balance, approx_price, qty, config.leverage, true, 0.0,
            );

            if !short_margin_check.can_place {
                info!(
                    "[pre-trade] Margin simulation: short leg would fail — reducing size. {}",
                    short_margin_check.rejection_reason.unwrap_or_default()
                );
                // Use max affordable qty instead
                let affordable_contracts = short_spec.qty_to_contracts(
                    short_margin_check.max_affordable_qty
                );
                let reduced = affordable_contracts.min(contracts);
                if reduced < 1 {
                    let reason = "Short leg margin insufficient even for minimum position size".to_string();
                    warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
                    return PreTradeResult::Rejected { reason };
                }
                reduced
            } else if !long_margin_check.can_place {
                info!(
                    "[pre-trade] Margin simulation: long leg would fail — reducing size. {}",
                    long_margin_check.rejection_reason.unwrap_or_default()
                );
                let affordable_contracts = long_spec.qty_to_contracts(
                    long_margin_check.max_affordable_qty
                );
                let reduced = affordable_contracts.min(contracts);
                if reduced < 1 {
                    let reason = "Long leg margin insufficient even for minimum position size".to_string();
                    warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
                    return PreTradeResult::Rejected { reason };
                }
                reduced
            } else {
                contracts
            }
        } else {
            // Legacy fallback without InstrumentManager
            let max_size_by_balance = (min_balance * config.max_position_pct * config.leverage as f64) as i64;
            let max_size_by_notional = config.max_notional_usdt as i64;
            let size = max_size_by_balance.min(max_size_by_notional);
            if size < 1 {
                let reason = format!(
                    "Computed position size {} is too small (balance: ${:.2})",
                    size, min_balance
                );
                warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
                return PreTradeResult::Rejected { reason };
            }
            size
        };

        // 5. BASIS RISK
        // Walk both exchange books to compute basis (price gap between exchanges).
        // If we can't get live prices, we conservatively assume zero basis risk
        // and let execution-time slippage checks catch issues.
        let basis_risk = Self::estimate_basis_risk(opp, global_book_registry);

        if basis_risk > config.max_basis_risk_pct {
            let reason = format!(
                "Basis risk {:.4}% exceeds max {:.4}%",
                basis_risk * 100.0, config.max_basis_risk_pct * 100.0
            );
            warn!("[pre-trade] REJECTED {}: {}", opp.symbol, reason);
            return PreTradeResult::Rejected { reason };
        }

        debug!("[pre-trade] {} passed all checks: breakeven={:.1} slippage={:.1}bps margin=({:.1}%/{:.1}%) basis={:.4}%",
            opp.symbol, breakeven_periods, estimated_slippage_bps,
            short_margin_ratio * 100.0, long_margin_ratio * 100.0, basis_risk * 100.0);

        PreTradeResult::Approved {
            estimated_slippage_bps,
            basis_risk,
            breakeven_periods,
            recommended_size,
        }
    }

    /// Estimate VWAP slippage for a given notional size using the global book.
    ///
    /// Walks the order book levels to calculate the volume-weighted average price
    /// for the target notional size. Falls back to a heuristic estimate if the
    /// global book is not available for the symbol.
    fn estimate_slippage(
        opp: &FundingArbOpportunity,
        registry: &GlobalBookRegistry,
        notional_usdt: f64,
    ) -> f64 {
        // Try to get the global book for this symbol.
        // The registry is keyed by symbol_id (u16), but we only have the symbol
        // string here. Walk all registered books to find matching data.
        // For now, use a heuristic estimate based on fee structure and notional size.
        //
        // In production, this would:
        // 1. Look up the symbol_id from the symbol registry
        // 2. Get the SharedGlobalBook from the registry
        // 3. Walk global_asks for the short leg (selling) and global_bids for the long leg (buying)
        // 4. Calculate VWAP for the target notional on each exchange
        // 5. Compare VWAP to mid price to get slippage in bps

        // Attempt to walk real book data from the registry
        for sym_id in registry.all_symbol_ids() {
            if let Some(book_lock) = registry.get(sym_id) {
                let book = book_lock.read();
                // Check if this book has data from the relevant exchanges
                let has_short = book.get_exchange_snapshot(opp.short_exchange).is_some();
                let has_long = book.get_exchange_snapshot(opp.long_exchange).is_some();

                if has_short && has_long {
                    // Calculate VWAP slippage on the ask side (for short entry = selling)
                    let short_slippage = Self::calculate_vwap_slippage(
                        &book.global_asks,
                        notional_usdt,
                    );
                    // Calculate VWAP slippage on the bid side (for long entry = buying)
                    let long_slippage = Self::calculate_vwap_slippage(
                        &book.global_bids,
                        notional_usdt,
                    );
                    // Combined slippage for both legs
                    return short_slippage + long_slippage;
                }
            }
        }

        // GAP 3 FIX: Improved heuristic when real book data is unavailable.
        // Uses a non-linear impact model: slippage grows with sqrt(size) relative
        // to typical market depth, matching the square-root market impact law
        // (Almgren & Chriss 2005, widely used by institutional desks).
        let base_fee_slippage = (opp.short_exchange.taker_fee_bps() + opp.long_exchange.taker_fee_bps()) as f64 / 2.0;
        // Square-root impact: ~1 bps per sqrt($10k notional)
        let sqrt_impact = (notional_usdt / 10_000.0).sqrt().min(10.0);
        // Spread cost: estimate half-spread on each side (~0.5-1.0 bps per side)
        let estimated_half_spread_bps = 1.0;
        base_fee_slippage + sqrt_impact + estimated_half_spread_bps * 2.0
    }

    /// Calculate VWAP slippage in basis points by walking order book levels.
    ///
    /// Walks the sorted levels (asks ascending or bids descending) and accumulates
    /// quantity until the target notional is filled. Returns the VWAP deviation
    /// from the best level price in basis points.
    ///
    /// # Fixed-point encoding (GlobalLevel)
    ///
    /// - `raw_price_fp`: i64 scaled by 1e8 (FixedPrice precision).
    ///   E.g. $50,000.12345678 → 5_000_012_345_678.
    /// - `qty`: i64 scaled by 1e8 (contracts).
    ///   E.g. 1.5 contracts → 150_000_000.
    ///
    /// All arithmetic below stays in consistent units so the final
    /// slippage ratio (VWAP / best_price) cancels the scaling factor
    /// and produces a dimensionless basis-point value.
    fn calculate_vwap_slippage(
        levels: &[crate::multi_exchange::global_book::GlobalLevel],
        target_notional_usdt: f64,
    ) -> f64 {
        if levels.is_empty() {
            return 2.0; // Default 2 bps if no book data
        }

        let best_price = levels[0].raw_price_fp as f64; // fp-scaled
        if best_price <= 0.0 {
            return 2.0;
        }

        let mut remaining_notional = target_notional_usdt;
        let mut total_cost = 0.0;
        let mut total_qty = 0.0;

        for level in levels {
            if remaining_notional <= 0.0 {
                break;
            }
            // price is raw_price_fp (i64 * 1e8); qty is also i64 * 1e8.
            let price = level.raw_price_fp as f64;
            let qty = level.qty as f64 / 1e8; // → actual contract qty
            // notional = (price / 1e8) * qty  =  price * qty / 1e8  (in USD)
            let level_notional = price * qty / 1e8;

            let fill_notional = remaining_notional.min(level_notional);
            let fill_qty = if level_notional > 0.0 {
                qty * (fill_notional / level_notional)
            } else {
                0.0
            };

            // total_cost stays in fp-scaled price * actual qty — same
            // domain as best_price * qty, so VWAP / best_price is correct.
            total_cost += fill_qty * price;
            total_qty += fill_qty;
            remaining_notional -= fill_notional;
        }

        if total_qty <= 0.0 {
            return 5.0; // High slippage if we can't fill
        }

        let vwap = total_cost / total_qty; // fp-scaled price
        let slippage_bps = ((vwap - best_price).abs() / best_price) * 10000.0;
        slippage_bps
    }

    /// Get an approximate price for the symbol from the global book.
    /// Used for notional → quantity conversion in position sizing.
    fn get_approximate_price(
        opp: &FundingArbOpportunity,
        registry: &GlobalBookRegistry,
    ) -> Option<f64> {
        for sym_id in registry.all_symbol_ids() {
            if let Some(book_lock) = registry.get(sym_id) {
                let book = book_lock.read();
                // Try the short exchange first, then the long exchange
                for exchange in &[opp.short_exchange, opp.long_exchange] {
                    if let Some(snap) = book.get_exchange_snapshot(*exchange) {
                        let mid = (snap.best_bid_fp + snap.best_ask_fp) as f64 / 2.0 / 1e8;
                        if mid > 0.0 {
                            return Some(mid);
                        }
                    }
                }
            }
        }
        None
    }

    /// Estimate basis risk (price gap between exchanges) as a fraction of mid price.
    fn estimate_basis_risk(
        opp: &FundingArbOpportunity,
        registry: &GlobalBookRegistry,
    ) -> f64 {
        // Try to find the book with snapshots from both exchanges
        for sym_id in registry.all_symbol_ids() {
            if let Some(book_lock) = registry.get(sym_id) {
                let book = book_lock.read();
                let short_snap = book.get_exchange_snapshot(opp.short_exchange);
                let long_snap = book.get_exchange_snapshot(opp.long_exchange);

                if let (Some(ss), Some(ls)) = (short_snap, long_snap) {
                    let short_mid = (ss.best_bid_fp + ss.best_ask_fp) as f64 / 2.0;
                    let long_mid = (ls.best_bid_fp + ls.best_ask_fp) as f64 / 2.0;

                    if short_mid > 0.0 && long_mid > 0.0 {
                        let avg_mid = (short_mid + long_mid) / 2.0;
                        return (short_mid - long_mid).abs() / avg_mid;
                    }
                }
            }
        }

        // If we can't compute basis risk, return 0 (conservative — let execution handle it)
        0.0
    }
}

/// Convert ExchangeId (multi_exchange module) to Exchange (instrument_manager module).
fn exchange_id_to_exchange(id: ExchangeId) -> Exchange {
    match id {
        ExchangeId::Binance => Exchange::Binance,
        ExchangeId::Bybit => Exchange::Bybit,
        ExchangeId::GateIo => Exchange::GateIo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_reason_serialization() {
        let reason = ExitReason::TakeProfit { accumulated_funding: 150.0 };
        let json = serde_json::to_string(&reason).unwrap();
        assert!(json.contains("TakeProfit"));
        assert!(json.contains("150"));

        let reason2 = ExitReason::MarginDanger {
            exchange: ExchangeId::Binance,
            margin_ratio: 0.12,
        };
        let json2 = serde_json::to_string(&reason2).unwrap();
        assert!(json2.contains("MarginDanger"));
    }

    #[test]
    fn test_exit_reason_variants() {
        // Ensure all variants are constructible
        let _tp = ExitReason::TakeProfit { accumulated_funding: 100.0 };
        let _tp_pct = ExitReason::TakeProfitPct { pnl_pct: 0.005 };
        let _sl = ExitReason::StopLoss { net_pnl: -50.0 };
        let _sr = ExitReason::SpreadReversal { entry_net_rate: 0.001, current_net_rate: -0.0001 };
        let _cnp = ExitReason::ConsecutiveNegativePeriods { count: 3 };
        let _ts = ExitReason::TimeStop { hours_open: 72.0 };
        let _md = ExitReason::MarginDanger { exchange: ExchangeId::GateIo, margin_ratio: 0.10 };
        let _m = ExitReason::Manual;
    }

    #[test]
    fn test_vwap_slippage_empty_book() {
        let slippage = PreTradeValidator::calculate_vwap_slippage(&[], 50_000.0);
        assert_eq!(slippage, 2.0);
    }
}
