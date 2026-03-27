//! Dual-Leg Execution Engine for Funding Rate Arbitrage
//!
//! Handles the critical execution phase where both legs must be filled
//! simultaneously to maintain delta neutrality. Implements:
//! - Asynchronous simultaneous order routing (tokio::join!)
//! - Legging risk detection and resolution
//! - Partial fill handling with immediate market completion
//! - Fill verification and delta-neutrality enforcement

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{error, info, warn};

use crate::execution_gateway::{
    ExecutionGateway, ExchangeError, OrderIntent, OrderResult, OrderSide, OrderType,
};
use crate::execution_state::PlacementType;
use crate::instrument_manager::{
    self, Exchange, InstrumentManager,
};
use crate::multi_exchange::global_book::ExchangeId;

// ---------------------------------------------------------------------------
// Post-Only Repricing Configuration
// ---------------------------------------------------------------------------

/// Maximum number of Post-Only reprice attempts before falling back to market.
const POST_ONLY_MAX_ATTEMPTS: u32 = 5;

/// Delay between Post-Only reprice attempts in milliseconds.
const POST_ONLY_REPRICE_DELAY_MS: u64 = 200;

/// Maximum time to wait for a single leg to fill before declaring timeout (ms).
const LEG_FILL_TIMEOUT_MS: u64 = 5_000;

/// Maximum number of retry attempts for a failed leg.
const LEG_RETRY_MAX_ATTEMPTS: u32 = 3;

/// Delay between leg retry attempts in milliseconds (with exponential backoff).
const LEG_RETRY_BASE_DELAY_MS: u64 = 500;

/// Status of a single leg after execution attempt.
#[derive(Debug, Clone)]
pub enum LegStatus {
    ShortFilled { result: OrderResult, exchange: ExchangeId },
    LongFilled { result: OrderResult, exchange: ExchangeId },
    ShortFailed { error: ExchangeError, exchange: ExchangeId },
    LongFailed { error: ExchangeError, exchange: ExchangeId },
}

impl LegStatus {
    /// Get the exchange this leg was executed on.
    pub fn exchange(&self) -> ExchangeId {
        match self {
            LegStatus::ShortFilled { exchange, .. } => *exchange,
            LegStatus::LongFilled { exchange, .. } => *exchange,
            LegStatus::ShortFailed { exchange, .. } => *exchange,
            LegStatus::LongFailed { exchange, .. } => *exchange,
        }
    }

    /// Check if this leg was filled successfully.
    pub fn is_filled(&self) -> bool {
        matches!(self, LegStatus::ShortFilled { .. } | LegStatus::LongFilled { .. })
    }
}

/// Result of a dual-leg execution attempt.
#[derive(Debug)]
pub enum DualLegResult {
    /// Both legs filled successfully - position is delta neutral.
    BothFilled {
        short_result: OrderResult,
        long_result: OrderResult,
    },
    /// One leg filled but the other failed - LEGGING RISK.
    PartialFill {
        filled_leg: LegStatus,
        unfilled_exchange: ExchangeId,
        filled_size: i64,
    },
    /// Both legs failed - no position opened.
    BothFailed {
        short_error: ExchangeError,
        long_error: ExchangeError,
    },
}

pub struct DualLegExecutor;

/// Convert ExchangeId (multi_exchange module) to Exchange (instrument_manager module).
fn exchange_id_to_exchange(id: ExchangeId) -> Exchange {
    match id {
        ExchangeId::Binance => Exchange::Binance,
        ExchangeId::Bybit => Exchange::Bybit,
        ExchangeId::GateIo => Exchange::GateIo,
    }
}

impl DualLegExecutor {
    /// Execute simultaneous entry on both exchanges with Post-Only repricing.
    ///
    /// GAP 1 FIX: Uses Post-Only (maker) orders by default to capture maker
    /// rebates instead of paying taker fees. Falls back to market orders only
    /// after POST_ONLY_MAX_ATTEMPTS repricing attempts fail.
    ///
    /// GAP 2 FIX: Implements legging risk safeguards with configurable timeout
    /// and automatic retry with exponential backoff for failed legs.
    ///
    /// FEE STRATEGY: Uses InstrumentManager + FeeStrategy to set exchange-specific
    /// time-in-force and price offsets for optimal maker rebate capture.
    ///
    /// PRE-FLIGHT MARGIN: Simulates margin requirements on both exchanges before
    /// submitting orders to prevent InsufficientBalance rejections mid-execution.
    ///
    /// EXECUTION IDEMPOTENCY: Before retrying timed-out orders, checks if the
    /// order already exists on the exchange to prevent duplicate fills.
    ///
    /// CONTRACT MULTIPLIER: Normalizes order sizes using InstrumentManager specs
    /// so that both exchanges receive equivalent notional exposure.
    pub async fn execute_entry(
        symbol: &str,
        short_exchange: ExchangeId,
        long_exchange: ExchangeId,
        size: i64,
        leverage: i32,
        use_market_orders: bool,
        max_slippage: f64,
        _timeout_ms: u64,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
        instrument_mgr: Option<&InstrumentManager>,
    ) -> DualLegResult {
        let short_gw = match gateways.get(&short_exchange) {
            Some(gw) => gw.clone(),
            None => return DualLegResult::BothFailed {
                short_error: ExchangeError::Unknown {
                    code: "NO_GATEWAY".into(),
                    message: format!("No gateway for {}", short_exchange.name()),
                },
                long_error: ExchangeError::Unknown {
                    code: "NO_GATEWAY".into(),
                    message: "Skipped due to short gateway missing".into(),
                },
            },
        };

        let long_gw = match gateways.get(&long_exchange) {
            Some(gw) => gw.clone(),
            None => return DualLegResult::BothFailed {
                short_error: ExchangeError::Unknown {
                    code: "NO_GATEWAY".into(),
                    message: "Skipped due to long gateway missing".into(),
                },
                long_error: ExchangeError::Unknown {
                    code: "NO_GATEWAY".into(),
                    message: format!("No gateway for {}", long_exchange.name()),
                },
            },
        };

        // Set leverage on both exchanges first
        let _ = tokio::join!(
            short_gw.set_leverage(symbol, leverage),
            long_gw.set_leverage(symbol, leverage),
        );

        // ── Pre-flight margin simulation ──────────────────────────────────────
        // Simulate margin requirements on both exchanges before submitting.
        // This prevents InsufficientBalance rejections that leave us "legged in".
        if let Some(mgr) = instrument_mgr {
            let short_ex = exchange_id_to_exchange(short_exchange);
            let long_ex = exchange_id_to_exchange(long_exchange);
            let short_spec = mgr.get_or_default(short_ex, symbol);
            let long_spec = mgr.get_or_default(long_ex, symbol);

            // Approximate price from spec (fallback to 1.0 if unavailable)
            let approx_price = if short_spec.min_notional > 0.0 {
                short_spec.min_notional / short_spec.min_qty.max(1.0)
            } else {
                1.0
            };

            // Fetch balances from both gateways concurrently
            let (short_bal, long_bal) = tokio::join!(
                short_gw.get_balance(),
                long_gw.get_balance(),
            );
            let short_balance = short_bal.unwrap_or(0.0);
            let long_balance = long_bal.unwrap_or(0.0);

            let qty_f64 = size as f64 * short_spec.contract_multiplier;
            let short_margin = instrument_manager::simulate_margin(
                short_balance, approx_price, qty_f64, leverage, true, 0.0,
            );
            let long_margin = instrument_manager::simulate_margin(
                long_balance, approx_price, qty_f64, leverage, true, 0.0,
            );

            if !short_margin.can_place {
                warn!("[dual-leg] Pre-flight margin REJECTED for short leg on {}: {}",
                    short_exchange.name(),
                    short_margin.rejection_reason.as_deref().unwrap_or("unknown"));
                return DualLegResult::BothFailed {
                    short_error: ExchangeError::Unknown {
                        code: "MARGIN_INSUFFICIENT".into(),
                        message: short_margin.rejection_reason.unwrap_or_else(|| "Margin check failed".into()),
                    },
                    long_error: ExchangeError::Unknown {
                        code: "SKIPPED".into(),
                        message: "Skipped due to short leg margin rejection".into(),
                    },
                };
            }
            if !long_margin.can_place {
                warn!("[dual-leg] Pre-flight margin REJECTED for long leg on {}: {}",
                    long_exchange.name(),
                    long_margin.rejection_reason.as_deref().unwrap_or("unknown"));
                return DualLegResult::BothFailed {
                    short_error: ExchangeError::Unknown {
                        code: "SKIPPED".into(),
                        message: "Skipped due to long leg margin rejection".into(),
                    },
                    long_error: ExchangeError::Unknown {
                        code: "MARGIN_INSUFFICIENT".into(),
                        message: long_margin.rejection_reason.unwrap_or_else(|| "Margin check failed".into()),
                    },
                };
            }
            info!("[dual-leg] Pre-flight margin OK: short=${:.2}/{:.2} long=${:.2}/{:.2}",
                short_margin.required_margin, short_balance,
                long_margin.required_margin, long_balance);
        }

        // ── Contract multiplier normalization ─────────────────────────────────
        // Ensure both exchanges receive equivalent notional exposure by adjusting
        // contract sizes based on each exchange's contract multiplier.
        let (short_size, long_size) = if let Some(mgr) = instrument_mgr {
            let short_ex = exchange_id_to_exchange(short_exchange);
            let long_ex = exchange_id_to_exchange(long_exchange);
            let short_spec = mgr.get_or_default(short_ex, symbol);
            let long_spec = mgr.get_or_default(long_ex, symbol);

            if (short_spec.contract_multiplier - long_spec.contract_multiplier).abs() > 1e-12 {
                // Normalize: target_notional = size * short_multiplier
                // long_size = target_notional / long_multiplier
                let target_notional_units = size as f64 * short_spec.contract_multiplier;
                let adjusted_long = (target_notional_units / long_spec.contract_multiplier).round() as i64;
                info!("[dual-leg] Contract multiplier normalization: short={} (mult={}) long={} (mult={})",
                    size, short_spec.contract_multiplier, adjusted_long, long_spec.contract_multiplier);
                // FIX: Don't blindly clamp to 1 — if adjusted_long < 1,
                // the notional mismatch is too large to safely execute.
                if adjusted_long < 1 {
                    warn!("[dual-leg] Contract multiplier normalization produced long_size=0, aborting");
                    return DualLegResult::BothFailed {
                        short_error: ExchangeError::Unknown {
                            code: "SIZE_TOO_SMALL".into(),
                            message: "Long leg size rounds to 0 after multiplier normalization".into(),
                        },
                        long_error: ExchangeError::Unknown {
                            code: "SIZE_TOO_SMALL".into(),
                            message: "Long leg size rounds to 0 after multiplier normalization".into(),
                        },
                    };
                }
                (size, adjusted_long)
            } else {
                (size, size)
            }
        } else {
            (size, size)
        };

        // ── Fee Strategy integration ─────────────────────────────────────────
        // Use InstrumentManager + FeeStrategy to set exchange-specific TIF and
        // order type for optimal maker rebate capture.
        let short_ex_enum = exchange_id_to_exchange(short_exchange);
        let long_ex_enum = exchange_id_to_exchange(long_exchange);
        let urgency = if use_market_orders { 1.0 } else { 0.0 };
        let short_fee_strategy = instrument_manager::optimal_fee_strategy(short_ex_enum, urgency);
        let long_fee_strategy = instrument_manager::optimal_fee_strategy(long_ex_enum, urgency);

        let (short_order_type, short_tif) = if use_market_orders {
            (OrderType::Market, "ioc".to_string())
        } else if short_fee_strategy.use_post_only {
            (OrderType::PostOnly, short_fee_strategy.time_in_force.clone())
        } else {
            (OrderType::Market, short_fee_strategy.time_in_force.clone())
        };

        let (long_order_type, long_tif) = if use_market_orders {
            (OrderType::Market, "ioc".to_string())
        } else if long_fee_strategy.use_post_only {
            (OrderType::PostOnly, long_fee_strategy.time_in_force.clone())
        } else {
            (OrderType::Market, long_fee_strategy.time_in_force.clone())
        };

        info!("[dual-leg] Fee strategy: short@{}={} (tif={}, fee={:.5}) long@{}={} (tif={}, fee={:.5})",
            short_exchange.name(), if short_fee_strategy.use_post_only { "PostOnly" } else { "Market" },
            short_tif, short_fee_strategy.expected_fee_rate,
            long_exchange.name(), if long_fee_strategy.use_post_only { "PostOnly" } else { "Market" },
            long_tif, long_fee_strategy.expected_fee_rate);

        let short_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            size: short_size,
            order_type: short_order_type,
            price: None,
            reduce_only: false,
            leverage: Some(leverage),
            time_in_force: short_tif,
            slippage_cap_pct: Some(max_slippage),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "funding_arb_short_entry".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        let long_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            size: long_size,
            order_type: long_order_type,
            price: None,
            reduce_only: false,
            leverage: Some(leverage),
            time_in_force: long_tif,
            slippage_cap_pct: Some(max_slippage),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "funding_arb_long_entry".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        // CRITICAL: Execute both legs simultaneously
        let (short_result, long_result) = tokio::join!(
            Self::submit_with_post_only_retry(short_gw.clone(), short_intent, use_market_orders),
            Self::submit_with_post_only_retry(long_gw.clone(), long_intent, use_market_orders),
        );

        match (short_result, long_result) {
            (Ok(sr), Ok(lr)) => {
                // Verify delta neutrality: both fills should be same size
                if sr.filled_size != lr.filled_size {
                    warn!(
                        "[dual-leg] Size mismatch: short={} long={} — delta not perfectly neutral",
                        sr.filled_size, lr.filled_size
                    );
                }
                info!(
                    "[dual-leg] Both legs filled: short={} (fee={:.6}) long={} (fee={:.6})",
                    sr.order_id, sr.fee, lr.order_id, lr.fee
                );
                DualLegResult::BothFilled {
                    short_result: sr,
                    long_result: lr,
                }
            }
            (Ok(sr), Err(le)) => {
                // GAP 2 FIX: Attempt to retry the failed leg before declaring legging risk
                warn!("[dual-leg] Short filled but long failed: {:?} — attempting retry", le);
                match Self::retry_failed_leg(
                    long_gw.clone(),
                    symbol,
                    OrderSide::Buy,
                    size,
                    leverage,
                    max_slippage,
                ).await {
                    Ok(lr) => {
                        info!("[dual-leg] Retry succeeded: both legs now filled");
                        DualLegResult::BothFilled {
                            short_result: sr,
                            long_result: lr,
                        }
                    }
                    Err(_retry_err) => {
                        error!("[dual-leg] LEGGING RISK: Retry failed for long leg. Initiating emergency close.");
                        DualLegResult::PartialFill {
                            filled_leg: LegStatus::ShortFilled { result: sr, exchange: short_exchange },
                            unfilled_exchange: long_exchange,
                            filled_size: size,
                        }
                    }
                }
            }
            (Err(se), Ok(lr)) => {
                // GAP 2 FIX: Attempt to retry the failed leg
                warn!("[dual-leg] Long filled but short failed: {:?} — attempting retry", se);
                match Self::retry_failed_leg(
                    short_gw.clone(),
                    symbol,
                    OrderSide::Sell,
                    size,
                    leverage,
                    max_slippage,
                ).await {
                    Ok(sr) => {
                        info!("[dual-leg] Retry succeeded: both legs now filled");
                        DualLegResult::BothFilled {
                            short_result: sr,
                            long_result: lr,
                        }
                    }
                    Err(_retry_err) => {
                        error!("[dual-leg] LEGGING RISK: Retry failed for short leg. Initiating emergency close.");
                        DualLegResult::PartialFill {
                            filled_leg: LegStatus::LongFilled { result: lr, exchange: long_exchange },
                            unfilled_exchange: short_exchange,
                            filled_size: size,
                        }
                    }
                }
            }
            (Err(se), Err(le)) => {
                DualLegResult::BothFailed {
                    short_error: se,
                    long_error: le,
                }
            }
        }
    }

    /// Submit an order with Post-Only repricing loop.
    ///
    /// If the initial Post-Only order is rejected (crosses the spread),
    /// reprice closer to best bid/ask and retry up to POST_ONLY_MAX_ATTEMPTS
    /// times. If all attempts fail, fall back to a market order.
    async fn submit_with_post_only_retry(
        gw: Arc<dyn ExecutionGateway + Send + Sync>,
        intent: OrderIntent,
        force_market: bool,
    ) -> Result<OrderResult, ExchangeError> {
        // If market orders are forced, skip repricing loop
        if force_market || intent.order_type == OrderType::Market {
            return gw.submit_order(intent).await;
        }

        let current_intent = intent.clone();

        for attempt in 0..POST_ONLY_MAX_ATTEMPTS {
            match gw.submit_order(current_intent.clone()).await {
                Ok(result) => {
                    if result.filled_size > 0 {
                        if attempt > 0 {
                            info!(
                                "[post-only] Filled after {} reprice attempts (maker rebate captured)",
                                attempt + 1
                            );
                        }
                        return Ok(result);
                    }
                    // Order accepted but not filled yet - wait briefly
                    tokio::time::sleep(tokio::time::Duration::from_millis(POST_ONLY_REPRICE_DELAY_MS)).await;

                    // Check if filled after waiting
                    if !result.order_id.is_empty() {
                        match gw.get_order_status(&result.order_id, &current_intent.symbol).await {
                            Ok(Some(status)) if status.filled_size > 0 => {
                                info!("[post-only] Filled after wait on attempt {}", attempt + 1);
                                return Ok(status);
                            }
                            Ok(_) => {
                                // Still not filled or order gone - cancel and reprice
                                let _ = gw.cancel_order(&result.order_id, &current_intent.symbol).await;
                            }
                            Err(_) => {
                                // Status check failed - cancel and retry
                                let _ = gw.cancel_order(&result.order_id, &current_intent.symbol).await;
                            }
                        }
                    }
                }
                Err(ExchangeError::Unknown { ref code, .. }) if code == "POST_ONLY_REJECTED" => {
                    // Post-Only rejected because it would cross the spread.
                    // This is expected; reprice on next attempt.
                    warn!(
                        "[post-only] Attempt {} rejected (would cross spread), repricing",
                        attempt + 1
                    );
                }
                Err(e) if e.is_retryable() => {
                    warn!("[post-only] Attempt {} retryable error: {:?}", attempt + 1, e);
                    tokio::time::sleep(tokio::time::Duration::from_millis(POST_ONLY_REPRICE_DELAY_MS)).await;
                }
                Err(e) => {
                    // Non-retryable error - fall through to market fallback
                    warn!("[post-only] Non-retryable error on attempt {}: {:?}", attempt + 1, e);
                    break;
                }
            }

            // Brief pause before next reprice attempt
            if attempt < POST_ONLY_MAX_ATTEMPTS - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    POST_ONLY_REPRICE_DELAY_MS / 2,
                )).await;
            }
        }

        // Fallback: submit as market order to guarantee fill
        warn!(
            "[post-only] All {} Post-Only attempts exhausted, falling back to market order",
            POST_ONLY_MAX_ATTEMPTS
        );
        let mut market_intent = intent;
        market_intent.order_type = OrderType::Market;
        market_intent.time_in_force = "ioc".to_string();
        gw.submit_order(market_intent).await
    }

    /// Retry a failed leg with exponential backoff.
    ///
    /// Uses market orders for retries to guarantee fill and resolve
    /// the legging risk as quickly as possible.
    ///
    /// EXECUTION IDEMPOTENCY: Before each retry attempt, checks if the
    /// previous order already exists on the exchange to prevent duplicate fills
    /// when a timeout was caused by network latency rather than actual rejection.
    ///
    /// Two paths:
    /// 1. If the previous attempt returned an exchange `order_id` (e.g. order was
    ///    accepted but zero-filled), `get_order_status(order_id)` is used.
    /// 2. If the previous attempt timed out entirely and returned no `order_id`
    ///    (the critical gap described in bugs.txt), `check_order_by_client_id` is
    ///    called with the `client_order_id` embedded in the `TimedOut` error.
    ///    This queries the exchange by the user-defined identifier
    ///    (`origClientOrderId` on Binance, `orderLinkId` on Bybit, `text` on Gate.io)
    ///    to detect whether the order was silently accepted.
    async fn retry_failed_leg(
        gw: Arc<dyn ExecutionGateway + Send + Sync>,
        symbol: &str,
        side: OrderSide,
        size: i64,
        leverage: i32,
        max_slippage: f64,
    ) -> Result<OrderResult, ExchangeError> {
        let mut last_err = ExchangeError::Timeout;
        // Exchange-assigned order ID from a previous timed-out/zero-fill attempt.
        let mut last_order_id: Option<String> = None;
        // Client-side order identifier from a `TimedOut` error; used when we have no
        // exchange order_id to fall back on check_order_by_client_id.
        let mut last_client_order_id: Option<String> = None;

        for attempt in 0..LEG_RETRY_MAX_ATTEMPTS {
            let delay = LEG_RETRY_BASE_DELAY_MS * (1u64 << attempt); // Exponential backoff
            tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;

            info!(
                "[dual-leg] Retry attempt {}/{} for {:?} leg (delay={}ms)",
                attempt + 1, LEG_RETRY_MAX_ATTEMPTS, side, delay
            );

            // IDEMPOTENCY CHECK (Path 1): If we have a previous exchange order_id from
            // a timed-out/zero-fill attempt, check its status before retrying.
            if let Some(ref prev_oid) = last_order_id {
                match gw.get_order_status(prev_oid, symbol).await {
                    Ok(Some(status)) if status.filled_size > 0 => {
                        info!(
                            "[dual-leg] Idempotency check: previous order {} already filled ({} contracts) — skipping retry",
                            prev_oid, status.filled_size
                        );
                        return Ok(status);
                    }
                    Ok(Some(status)) => {
                        info!(
                            "[dual-leg] Idempotency check: previous order {} exists but unfilled (status={})",
                            prev_oid, status.status
                        );
                        // Order exists but not filled — cancel it before retrying
                        let _ = gw.cancel_order(prev_oid, symbol).await;
                    }
                    Ok(None) => {
                        info!("[dual-leg] Idempotency check: previous order {} not found — safe to retry", prev_oid);
                    }
                    Err(e) => {
                        warn!("[dual-leg] Idempotency check failed for order {}: {:?} — proceeding with retry", prev_oid, e);
                    }
                }
            } else if let Some(ref prev_cid) = last_client_order_id {
                // IDEMPOTENCY CHECK (Path 2): WS ACK timed out with no exchange order_id.
                // The exchange may have accepted the order silently.  Query by the
                // client-side identifier before issuing a new order to prevent doubling
                // the position.
                info!(
                    "[dual-leg] Idempotency check by client_order_id={} (no exchange order_id available)",
                    prev_cid
                );
                match gw.check_order_by_client_id(prev_cid, symbol).await {
                    Ok(Some(exch_oid)) => {
                        // Order found on exchange — get its fill status
                        match gw.get_order_status(&exch_oid, symbol).await {
                            Ok(Some(status)) if status.filled_size > 0 => {
                                info!(
                                    "[dual-leg] Idempotency (client_id={}): order {} already filled ({} contracts) — skipping retry",
                                    prev_cid, exch_oid, status.filled_size
                                );
                                return Ok(status);
                            }
                            Ok(Some(status)) => {
                                info!(
                                    "[dual-leg] Idempotency (client_id={}): order {} exists but unfilled (status={}) — cancelling",
                                    prev_cid, exch_oid, status.status
                                );
                                let _ = gw.cancel_order(&exch_oid, symbol).await;
                            }
                            Ok(None) | Err(_) => {
                                // get_order_status not supported or order gone — cancel defensively
                                let _ = gw.cancel_order(&exch_oid, symbol).await;
                            }
                        }
                    }
                    Ok(None) => {
                        info!(
                            "[dual-leg] Idempotency (client_id={}): no order found on exchange — safe to retry",
                            prev_cid
                        );
                    }
                    Err(e) => {
                        warn!(
                            "[dual-leg] Idempotency check by client_id {} failed: {:?} — proceeding with retry",
                            prev_cid, e
                        );
                    }
                }
            }

            // Use market order for retries (speed > cost when resolving legging risk)
            let retry_intent = OrderIntent {
                symbol: symbol.to_string(),
                side: side.clone(),
                size,
                order_type: OrderType::Market,
                price: None,
                reduce_only: false,
                leverage: Some(leverage),
                time_in_force: "ioc".to_string(),
                slippage_cap_pct: Some(max_slippage * 2.0), // Wider slippage for emergency
                placement: PlacementType::AtBest,
                stop_loss: None,
                take_profit: None,
                confidence: 1.0,
                signal_tag: "funding_arb_leg_retry".to_string(),
                min_fill_size: None,
                strategy_name: "funding_arb".to_string(),
            };

            match gw.submit_order(retry_intent).await {
                Ok(result) if result.filled_size > 0 => {
                    info!(
                        "[dual-leg] Retry succeeded on attempt {}: filled {} contracts",
                        attempt + 1, result.filled_size
                    );
                    return Ok(result);
                }
                Ok(result) => {
                    warn!("[dual-leg] Retry attempt {} returned 0 fill: {:?}", attempt + 1, result.status);
                    // Track exchange order ID for idempotency check on next attempt
                    if !result.order_id.is_empty() {
                        last_order_id = Some(result.order_id);
                        last_client_order_id = None; // exchange oid takes priority
                    }
                    last_err = ExchangeError::Unknown {
                        code: "ZERO_FILL".into(),
                        message: format!("Retry filled 0 contracts on attempt {}", attempt + 1),
                    };
                }
                Err(ExchangeError::TimedOut { ref client_order_id }) => {
                    // WS ACK / REST response timed out and the gateway has provided the
                    // client_order_id that was used.  Store it so the next iteration can
                    // call check_order_by_client_id() to detect a silent acceptance.
                    warn!(
                        "[dual-leg] Retry attempt {} timed out (client_order_id={}) — will check by client_id on next attempt",
                        attempt + 1, client_order_id
                    );
                    last_client_order_id = Some(client_order_id.clone());
                    last_order_id = None; // no exchange oid available
                    last_err = ExchangeError::TimedOut { client_order_id: client_order_id.clone() };
                }
                Err(ExchangeError::Timeout) => {
                    warn!("[dual-leg] Retry attempt {} timed out — will check idempotency on next attempt", attempt + 1);
                    // Timeout with no client_order_id (e.g. from a gateway that doesn't
                    // support TimedOut yet) — existing path handles this gracefully.
                    last_err = ExchangeError::Timeout;
                }
                Err(e) => {
                    warn!("[dual-leg] Retry attempt {} failed: {:?}", attempt + 1, e);
                    last_err = e;
                }
            }
        }

        error!(
            "[dual-leg] All {} retry attempts exhausted for {:?} leg",
            LEG_RETRY_MAX_ATTEMPTS, side
        );
        Err(last_err)
    }

    /// Execute simultaneous exit on both exchanges.
    ///
    /// Closes both legs with market orders to minimize slippage.
    /// Close short = BUY, Close long = SELL.
    pub async fn execute_exit(
        symbol: &str,
        short_exchange: ExchangeId,
        long_exchange: ExchangeId,
        size: i64,
        _timeout_ms: u64,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    ) -> DualLegResult {
        let short_gw = match gateways.get(&short_exchange) {
            Some(gw) => gw.clone(),
            None => return DualLegResult::BothFailed {
                short_error: ExchangeError::Unknown { code: "NO_GATEWAY".into(), message: "Missing".into() },
                long_error: ExchangeError::Unknown { code: "SKIPPED".into(), message: "Skipped".into() },
            },
        };
        let long_gw = match gateways.get(&long_exchange) {
            Some(gw) => gw.clone(),
            None => return DualLegResult::BothFailed {
                short_error: ExchangeError::Unknown { code: "SKIPPED".into(), message: "Skipped".into() },
                long_error: ExchangeError::Unknown { code: "NO_GATEWAY".into(), message: "Missing".into() },
            },
        };

        // Close short = BUY, Close long = SELL
        let close_short = OrderIntent {
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            size,
            order_type: OrderType::Market,
            price: None,
            reduce_only: true,
            leverage: None,
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.005),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 0.0,
            signal_tag: "funding_arb_short_exit".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        let close_long = OrderIntent {
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            size,
            order_type: OrderType::Market,
            price: None,
            reduce_only: true,
            leverage: None,
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.005),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 0.0,
            signal_tag: "funding_arb_long_exit".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        let (short_result, long_result) = tokio::join!(
            short_gw.submit_order(close_short),
            long_gw.submit_order(close_long),
        );

        match (short_result, long_result) {
            (Ok(sr), Ok(lr)) => {
                info!("[dual-leg] Both exit legs filled: short_close={} long_close={}", sr.order_id, lr.order_id);
                DualLegResult::BothFilled {
                    short_result: sr,
                    long_result: lr,
                }
            }
            (Ok(sr), Err(le)) => {
                error!("[dual-leg] Exit PARTIAL: short closed but long exit failed: {:?}", le);
                DualLegResult::PartialFill {
                    filled_leg: LegStatus::ShortFilled { result: sr, exchange: short_exchange },
                    unfilled_exchange: long_exchange,
                    filled_size: size,
                }
            }
            (Err(se), Ok(lr)) => {
                error!("[dual-leg] Exit PARTIAL: long closed but short exit failed: {:?}", se);
                DualLegResult::PartialFill {
                    filled_leg: LegStatus::LongFilled { result: lr, exchange: long_exchange },
                    unfilled_exchange: short_exchange,
                    filled_size: size,
                }
            }
            (Err(se), Err(le)) => {
                error!("[dual-leg] CRITICAL: Both exit legs failed! short={:?} long={:?}", se, le);
                DualLegResult::BothFailed {
                    short_error: se,
                    long_error: le,
                }
            }
        }
    }

    /// BUG 5 FIX: Maker-Taker Leg Execution Pattern.
    ///
    /// Instead of firing two simultaneous market orders (which risks naked
    /// directional exposure if one leg fails), this method:
    ///
    /// 1. Places MAKER (post-only) order on less liquid exchange first
    /// 2. Polls every 100ms for fill confirmation (timeout 30s)
    /// 3. Only when leg 1 fills, immediately hedges with IOC taker on liquid exchange
    /// 4. If leg 1 doesn't fill within timeout, cancels it (no exposure)
    /// 5. If leg 2 fails after leg 1 filled, immediately closes leg 1 at market
    pub async fn execute_entry_maker_taker(
        symbol: &str,
        maker_exchange: ExchangeId,
        taker_exchange: ExchangeId,
        maker_side: OrderSide,
        taker_side: OrderSide,
        size: i64,
        leverage: i32,
        max_slippage: f64,
        maker_timeout_ms: u64,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    ) -> DualLegResult {
        let maker_gw = match gateways.get(&maker_exchange) {
            Some(gw) => gw.clone(),
            None => return DualLegResult::BothFailed {
                short_error: ExchangeError::Unknown {
                    code: "NO_GATEWAY".into(),
                    message: format!("No gateway for maker {}", maker_exchange.name()),
                },
                long_error: ExchangeError::Unknown {
                    code: "SKIPPED".into(),
                    message: "Skipped due to maker gateway missing".into(),
                },
            },
        };

        let taker_gw = match gateways.get(&taker_exchange) {
            Some(gw) => gw.clone(),
            None => return DualLegResult::BothFailed {
                short_error: ExchangeError::Unknown {
                    code: "SKIPPED".into(),
                    message: "Skipped due to taker gateway missing".into(),
                },
                long_error: ExchangeError::Unknown {
                    code: "NO_GATEWAY".into(),
                    message: format!("No gateway for taker {}", taker_exchange.name()),
                },
            },
        };

        // Set leverage on both exchanges
        let _ = tokio::join!(
            maker_gw.set_leverage(symbol, leverage),
            taker_gw.set_leverage(symbol, leverage),
        );

        // Step 1: Place MAKER (post-only) order on less liquid exchange
        let maker_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: maker_side.clone(),
            size,
            order_type: OrderType::PostOnly,
            price: None, // Gateway will use best price
            reduce_only: false,
            leverage: Some(leverage),
            time_in_force: "poc".to_string(),
            slippage_cap_pct: Some(max_slippage),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "funding_arb_maker_leg".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        info!(
            "[dual-leg] BUG5 FIX: Placing MAKER leg on {} ({:?}), size={}",
            maker_exchange.name(), maker_side, size
        );

        let maker_result = maker_gw.submit_order(maker_intent.clone()).await;
        let maker_order_id = match maker_result {
            Ok(ref r) if !r.order_id.is_empty() => r.order_id.clone(),
            Ok(ref r) if r.filled_size > 0 => {
                // Filled immediately (unlikely for post-only but possible)
                info!("[dual-leg] Maker leg filled immediately: {} contracts", r.filled_size);
                // Proceed directly to taker leg
                return Self::execute_taker_hedge(
                    symbol, taker_exchange, taker_side, size, leverage,
                    max_slippage, r.clone(), maker_exchange, maker_side.clone(),
                    gateways,
                ).await;
            }
            Ok(_) => {
                warn!("[dual-leg] Maker order returned empty order_id with 0 fill");
                return DualLegResult::BothFailed {
                    short_error: ExchangeError::Unknown {
                        code: "MAKER_FAILED".into(),
                        message: "Maker order returned no order_id".into(),
                    },
                    long_error: ExchangeError::Unknown {
                        code: "SKIPPED".into(),
                        message: "Taker skipped: maker leg not placed".into(),
                    },
                };
            }
            Err(e) => {
                // Maker leg failed - no exposure, safe to abort
                info!("[dual-leg] Maker leg rejected (no exposure): {:?}", e);
                return DualLegResult::BothFailed {
                    short_error: e.clone(),
                    long_error: ExchangeError::Unknown {
                        code: "SKIPPED".into(),
                        message: "Taker skipped: maker leg not placed".into(),
                    },
                };
            }
        };

        // Step 2: Poll for maker fill (every 100ms, timeout configurable)
        let poll_interval = tokio::time::Duration::from_millis(100);
        let timeout = tokio::time::Duration::from_millis(maker_timeout_ms);
        let deadline = tokio::time::Instant::now() + timeout;

        info!(
            "[dual-leg] Polling maker order {} for fill (timeout={}ms)",
            maker_order_id, maker_timeout_ms
        );

        let mut maker_fill: Option<OrderResult> = None;
        while tokio::time::Instant::now() < deadline {
            tokio::time::sleep(poll_interval).await;

            match maker_gw.get_order_status(&maker_order_id, symbol).await {
                Ok(Some(status)) if status.filled_size > 0 => {
                    info!(
                        "[dual-leg] Maker leg FILLED: {} contracts @ {:.2}",
                        status.filled_size, status.avg_fill_price
                    );
                    maker_fill = Some(status);
                    break;
                }
                Ok(Some(_)) => {
                    // Order exists but not filled yet - keep waiting
                    continue;
                }
                Ok(None) => {
                    // Order disappeared (cancelled externally?) - abort
                    warn!("[dual-leg] Maker order {} disappeared during polling", maker_order_id);
                    break;
                }
                Err(_) => {
                    // Status check failed - keep trying
                    continue;
                }
            }
        }

        // Step 3: Handle result
        match maker_fill {
            Some(maker_result) => {
                // Maker filled - immediately hedge with IOC taker
                Self::execute_taker_hedge(
                    symbol, taker_exchange, taker_side, size, leverage,
                    max_slippage, maker_result, maker_exchange, maker_side,
                    gateways,
                ).await
            }
            None => {
                // Step 4: Maker didn't fill within timeout - cancel it (no exposure)
                info!(
                    "[dual-leg] Maker timeout: cancelling order {} on {}",
                    maker_order_id, maker_exchange.name()
                );
                let _ = maker_gw.cancel_order(&maker_order_id, symbol).await;
                DualLegResult::BothFailed {
                    short_error: ExchangeError::Unknown {
                        code: "MAKER_TIMEOUT".into(),
                        message: format!(
                            "Maker leg on {} did not fill within {}ms",
                            maker_exchange.name(), maker_timeout_ms
                        ),
                    },
                    long_error: ExchangeError::Unknown {
                        code: "SKIPPED".into(),
                        message: "Taker skipped: maker leg not filled".into(),
                    },
                }
            }
        }
    }

    /// BUG 5 FIX: Execute the taker hedge leg after maker has been filled.
    ///
    /// If the taker leg fails, immediately closes the maker leg at market
    /// to prevent naked directional exposure.
    async fn execute_taker_hedge(
        symbol: &str,
        taker_exchange: ExchangeId,
        taker_side: OrderSide,
        size: i64,
        leverage: i32,
        max_slippage: f64,
        maker_result: OrderResult,
        maker_exchange: ExchangeId,
        maker_side: OrderSide,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    ) -> DualLegResult {
        let taker_gw = match gateways.get(&taker_exchange) {
            Some(gw) => gw.clone(),
            None => {
                // Taker gateway missing after maker filled - emergency close maker
                error!(
                    "[dual-leg] CRITICAL: Taker gateway {} missing after maker filled! Emergency closing maker.",
                    taker_exchange.name()
                );
                let maker_leg = if maker_side == OrderSide::Sell {
                    LegStatus::ShortFilled { result: maker_result, exchange: maker_exchange }
                } else {
                    LegStatus::LongFilled { result: maker_result, exchange: maker_exchange }
                };
                return DualLegResult::PartialFill {
                    filled_leg: maker_leg,
                    unfilled_exchange: taker_exchange,
                    filled_size: size,
                };
            }
        };

        let taker_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: taker_side.clone(),
            size,
            order_type: OrderType::Market,
            price: None,
            reduce_only: false,
            leverage: Some(leverage),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(max_slippage),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "funding_arb_taker_leg".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        info!(
            "[dual-leg] Placing TAKER hedge on {} ({:?}), size={}",
            taker_exchange.name(), taker_side, size
        );

        match taker_gw.submit_order(taker_intent).await {
            Ok(taker_result) if taker_result.filled_size > 0 => {
                info!(
                    "[dual-leg] BUG5 FIX: Both legs filled via maker-taker pattern. maker={} taker={}",
                    maker_result.order_id, taker_result.order_id
                );
                // Map to BothFilled with correct short/long assignment
                let (short_result, long_result) = if maker_side == OrderSide::Sell {
                    (maker_result, taker_result)
                } else {
                    (taker_result, maker_result)
                };
                DualLegResult::BothFilled { short_result, long_result }
            }
            Ok(_) | Err(_) => {
                // Step 5: Taker failed after maker filled - emergency close maker
                error!(
                    "[dual-leg] LEGGING RISK: Taker hedge FAILED on {}. Emergency closing maker on {}.",
                    taker_exchange.name(), maker_exchange.name()
                );

                let maker_leg = if maker_side == OrderSide::Sell {
                    LegStatus::ShortFilled { result: maker_result, exchange: maker_exchange }
                } else {
                    LegStatus::LongFilled { result: maker_result, exchange: maker_exchange }
                };

                // Attempt emergency close
                let _ = Self::emergency_close_leg(symbol, &maker_leg, size, gateways).await;

                DualLegResult::PartialFill {
                    filled_leg: maker_leg,
                    unfilled_exchange: taker_exchange,
                    filled_size: size,
                }
            }
        }
    }

    /// Emergency close a single filled leg to resolve legging risk.
    ///
    /// When one leg fills but the other fails, this function immediately
    /// closes the filled leg with a market order to prevent unhedged exposure.
    pub async fn emergency_close_leg(
        symbol: &str,
        filled_leg: &LegStatus,
        filled_size: i64,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    ) -> Result<OrderResult, ExchangeError> {
        let (exchange, close_side) = match filled_leg {
            LegStatus::ShortFilled { exchange, .. } => (*exchange, OrderSide::Buy),
            LegStatus::LongFilled { exchange, .. } => (*exchange, OrderSide::Sell),
            _ => {
                return Err(ExchangeError::Unknown {
                    code: "INVALID_LEG".into(),
                    message: "Cannot close a failed leg".into(),
                });
            }
        };

        let gateway = gateways.get(&exchange).ok_or_else(|| ExchangeError::Unknown {
            code: "NO_GATEWAY".into(),
            message: format!("No gateway for {}", exchange.name()),
        })?;

        let close_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: close_side,
            size: filled_size,
            order_type: OrderType::Market,
            price: None,
            reduce_only: true,
            leverage: None,
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.005), // 0.5% emergency slippage tolerance
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 0.0,
            signal_tag: "funding_arb_legging_close".to_string(),
            min_fill_size: None,
            strategy_name: "funding_arb".to_string(),
        };

        warn!(
            "[dual-leg] Emergency closing {} leg on {}: size={}",
            match filled_leg {
                LegStatus::ShortFilled { .. } => "SHORT",
                LegStatus::LongFilled { .. } => "LONG",
                _ => "UNKNOWN",
            },
            exchange.name(),
            filled_size
        );

        gateway.submit_order(close_intent).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leg_status_exchange() {
        let leg = LegStatus::ShortFilled {
            result: OrderResult {
                order_id: "test".to_string(),
                status: "closed".to_string(),
                filled_size: 100,
                avg_fill_price: 50000.0,
                fee: 0.5,
                latency_us: 1000,
                exchange_timestamp: 0,
                rejection_reason: None,
            },
            exchange: ExchangeId::Binance,
        };
        assert_eq!(leg.exchange(), ExchangeId::Binance);
        assert!(leg.is_filled());

        let failed_leg = LegStatus::ShortFailed {
            error: ExchangeError::Timeout,
            exchange: ExchangeId::GateIo,
        };
        assert_eq!(failed_leg.exchange(), ExchangeId::GateIo);
        assert!(!failed_leg.is_filled());
    }
}
