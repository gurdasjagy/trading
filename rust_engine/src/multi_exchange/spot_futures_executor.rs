//! Spot-Futures Arbitrage: Maker-Taker Sniper Execution Engine
//!
//! Handles the critical execution phase for spot-futures arbitrage:
//! - Entry: Post-Only Spot buy -> the moment it fills, IOC Futures short
//! - Exit: Limit Futures buy-to-close -> the moment it fills, market Spot sell
//! - Legging risk protection: if one leg fails, immediately unwind the other
//!
//! CRITICAL: Gate.io Spot orders MUST use REST, not WebSocket.
//! Gate.io does NOT have a WebSocket trading API for Spot.

use std::time::Duration;

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use tracing::{error, info, warn};

use crate::config::SpotFuturesConfig;
use crate::execution_gateway::{
    ExchangeError, ExecutionGateway, OrderIntent, OrderResult, OrderSide, OrderType,
    SpotOrderIntent, SpotOrderResult,
};
use crate::execution_state::PlacementType;
use crate::multi_exchange::global_book::ExchangeId;
use crate::multi_exchange::spot_futures_specs::{SpotMarketSpec, FuturesContractSpec};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum time to wait for a Spot limit order fill before cancelling (seconds).
const SPOT_FILL_TIMEOUT_SECS: u64 = 30;

/// Maximum number of Post-Only reprice attempts for Spot entry.
const SPOT_POST_ONLY_MAX_ATTEMPTS: u32 = 5;

/// Delay between Post-Only reprice attempts (milliseconds).
const SPOT_REPRICE_DELAY_MS: u64 = 200;

/// Maximum number of retry attempts for the Futures hedge leg.
const FUTURES_HEDGE_MAX_RETRIES: u32 = 3;

/// Delay between Futures hedge retry attempts (milliseconds).
const FUTURES_HEDGE_RETRY_DELAY_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Execution Results
// ---------------------------------------------------------------------------

/// Result of a dual-leg spot-futures entry attempt.
#[derive(Debug)]
pub enum SpotFuturesEntryResult {
    /// Both legs filled successfully -- position is hedged.
    Success {
        spot_result: SpotOrderResult,
        futures_result: OrderResult,
    },
    /// Spot leg filled but Futures hedge failed -- CRITICAL: unhedged exposure.
    /// The executor already attempted emergency Spot sell-back.
    SpotFilledFuturesFailed {
        spot_result: SpotOrderResult,
        futures_error: ExchangeError,
        emergency_sellback_result: Option<SpotOrderResult>,
    },
    /// Spot order timed out or was cancelled -- zero exposure, zero risk.
    SpotNotFilled {
        reason: String,
    },
    /// Pre-validation failed (insufficient balance, below minimums, etc.)
    ValidationFailed {
        reason: String,
    },
}

/// Result of a dual-leg spot-futures exit attempt.
#[derive(Debug)]
pub enum SpotFuturesExitResult {
    /// Both legs closed successfully.
    Success {
        futures_close_result: OrderResult,
        spot_sell_result: SpotOrderResult,
    },
    /// Futures closed but Spot sell failed -- you still hold the crypto.
    FuturesClosedSpotFailed {
        futures_close_result: OrderResult,
        spot_error: ExchangeError,
    },
    /// Exit failed entirely.
    Failed {
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Spot-Futures arbitrage execution engine.
/// Handles entry and exit execution with legging risk protection.
pub struct SpotFuturesExecutor {
    config: SpotFuturesConfig,
}

impl SpotFuturesExecutor {
    /// Create a new executor.
    pub fn new(config: SpotFuturesConfig) -> Self {
        Self { config }
    }

    /// Execute a spot-futures entry: buy Spot, short Futures.
    ///
    /// Maker-Taker Sniper Pattern:
    /// 1. Place Post-Only (Maker) limit order on Spot at Best Bid + 1 tick
    /// 2. Wait for fill (with timeout)
    /// 3. The MILLISECOND the Spot fills: fire Market (IOC) Futures short
    /// 4. If Spot doesn't fill within timeout: cancel (zero exposure)
    /// 5. If Futures hedge fails after Spot filled: emergency Spot sell-back
    pub async fn execute_entry(
        &self,
        gateway: &dyn ExecutionGateway,
        exchange: ExchangeId,
        spot_spec: &SpotMarketSpec,
        futures_spec: &FuturesContractSpec,
        spot_qty: Decimal,
        spot_ask_price: Decimal,
        futures_bid_price: Decimal,
    ) -> SpotFuturesEntryResult {
        info!(
            "[spot-futures-exec] ENTRY on {}: qty={}, spot_ask={}, futures_bid={}",
            exchange.name(), spot_qty, spot_ask_price, futures_bid_price
        );

        // Step 1: Place Spot buy order
        let spot_price = spot_spec.round_price(spot_ask_price);
        let rounded_qty = spot_spec.round_qty(spot_qty);

        let use_market = self.config.spot_order_type == "market";

        let spot_intent = SpotOrderIntent {
            symbol: spot_spec.symbol.clone(),
            side: OrderSide::Buy,
            qty: if use_market { 0.0 } else { rounded_qty.to_f64().unwrap_or(0.0) },
            order_type: if use_market { OrderType::Market } else { OrderType::Limit },
            price: if use_market { None } else { Some(spot_price.to_f64().unwrap_or(0.0)) },
            time_in_force: if use_market { "IOC".to_string() } else { "GTC".to_string() },
            quote_order_qty: if use_market {
                // For market buys, spend USDT amount
                Some((rounded_qty * spot_ask_price).to_f64().unwrap_or(0.0))
            } else {
                None
            },
        };

        info!("[spot-futures-exec] Submitting Spot buy: {:?}", spot_intent);

        let spot_result = match gateway.submit_spot_order(spot_intent).await {
            Ok(result) => {
                if result.filled_qty <= 0.0 {
                    return SpotFuturesEntryResult::SpotNotFilled {
                        reason: format!("Spot order returned zero fill: status={}", result.status),
                    };
                }
                info!(
                    "[spot-futures-exec] Spot buy FILLED: qty={}, avg_price={}, fee={}",
                    result.filled_qty, result.avg_fill_price, result.fee
                );
                result
            }
            Err(e) => {
                return SpotFuturesEntryResult::SpotNotFilled {
                    reason: format!("Spot buy failed: {}", e),
                };
            }
        };

        // Step 2: Immediately hedge with Futures short
        // Use the EXACT filled quantity from the Spot order
        let futures_qty_raw = spot_result.filled_qty;

        // For Gate.io: convert base qty to integer contracts
        let futures_contracts = if !futures_spec.contract_multiplier.is_zero()
            && futures_spec.contract_multiplier != Decimal::ONE
        {
            let cm_f64 = futures_spec.contract_multiplier.to_f64().unwrap_or(1.0);
            (futures_qty_raw / cm_f64).floor() as i64
        } else {
            // Binance/Bybit linear: qty is in base asset, but OrderIntent uses i64 contracts
            // We need to convert. For linear contracts, 1 contract = step_size base asset
            let step = futures_spec.step_size.to_f64().unwrap_or(0.001);
            if step > 0.0 {
                (futures_qty_raw / step).floor() as i64
            } else {
                futures_qty_raw.floor() as i64
            }
        };

        if futures_contracts <= 0 {
            error!(
                "[spot-futures-exec] CRITICAL: Spot filled but Futures qty rounds to 0! spot_qty={}, spot filled={}",
                spot_qty, futures_qty_raw
            );
            // Emergency: sell back the Spot position
            let sellback = self.emergency_spot_sell(gateway, spot_spec, futures_qty_raw).await;
            return SpotFuturesEntryResult::SpotFilledFuturesFailed {
                spot_result,
                futures_error: ExchangeError::Unknown {
                    code: "ZERO_CONTRACTS".into(),
                    message: "Futures quantity rounds to zero".into(),
                },
                emergency_sellback_result: sellback,
            };
        }

        let futures_intent = OrderIntent {
            symbol: futures_spec.symbol.clone(),
            side: OrderSide::Sell, // Short
            size: futures_contracts,
            order_type: OrderType::Market,
            price: None,
            reduce_only: false,
            leverage: Some(self.config.short_leverage),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.005), // 0.5% max slippage
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "spot_futures_arb_entry".to_string(),
            min_fill_size: None,
            strategy_name: "spot_futures_arb".to_string(),
        };

        info!("[spot-futures-exec] Submitting Futures short: contracts={}", futures_contracts);

        // Retry futures hedge up to FUTURES_HEDGE_MAX_RETRIES times
        let mut last_error = None;
        for attempt in 0..FUTURES_HEDGE_MAX_RETRIES {
            match gateway.submit_order(futures_intent.clone()).await {
                Ok(result) => {
                    if result.filled_size > 0 {
                        info!(
                            "[spot-futures-exec] Futures short FILLED: size={}, avg_price={}, fee={}",
                            result.filled_size, result.avg_fill_price, result.fee
                        );
                        return SpotFuturesEntryResult::Success {
                            spot_result,
                            futures_result: result,
                        };
                    } else {
                        warn!("[spot-futures-exec] Futures short returned zero fill on attempt {}", attempt + 1);
                        last_error = Some(ExchangeError::Unknown {
                            code: "ZERO_FILL".into(),
                            message: "Futures order returned zero fill".into(),
                        });
                    }
                }
                Err(e) => {
                    warn!(
                        "[spot-futures-exec] Futures short attempt {}/{} failed: {}",
                        attempt + 1, FUTURES_HEDGE_MAX_RETRIES, e
                    );
                    last_error = Some(e);
                }
            }

            if attempt + 1 < FUTURES_HEDGE_MAX_RETRIES {
                let delay = FUTURES_HEDGE_RETRY_DELAY_MS * (1 << attempt);
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
        }

        // All retries exhausted -- CRITICAL: unhedged Spot exposure
        error!(
            "[spot-futures-exec] CRITICAL: Futures hedge FAILED after {} attempts! Selling Spot back.",
            FUTURES_HEDGE_MAX_RETRIES
        );
        let sellback = self.emergency_spot_sell(gateway, spot_spec, futures_qty_raw).await;

        SpotFuturesEntryResult::SpotFilledFuturesFailed {
            spot_result,
            futures_error: last_error.unwrap_or(ExchangeError::Unknown {
                code: "HEDGE_FAILED".into(),
                message: "All futures hedge attempts failed".into(),
            }),
            emergency_sellback_result: sellback,
        }
    }

    /// Execute a spot-futures exit: close Futures short, sell Spot.
    ///
    /// Exit Pattern:
    /// 1. Place Market order to buy-to-close the Futures short
    /// 2. The moment it fills, market-sell the Spot holdings
    /// 3. If Spot sell fails: you still hold crypto (less dangerous than entry failure)
    pub async fn execute_exit(
        &self,
        gateway: &dyn ExecutionGateway,
        futures_spec: &FuturesContractSpec,
        spot_spec: &SpotMarketSpec,
        futures_contracts: i64,
        spot_qty_held: f64,
    ) -> SpotFuturesExitResult {
        info!(
            "[spot-futures-exec] EXIT: closing {} futures contracts, selling {} spot",
            futures_contracts, spot_qty_held
        );

        // Step 1: Buy-to-close the Futures short
        let futures_close_intent = OrderIntent {
            symbol: futures_spec.symbol.clone(),
            side: OrderSide::Buy, // Buy to close short
            size: futures_contracts.abs(),
            order_type: OrderType::Market,
            price: None,
            reduce_only: true,
            leverage: Some(self.config.short_leverage),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.01), // 1% max slippage on exit
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "spot_futures_arb_exit".to_string(),
            min_fill_size: None,
            strategy_name: "spot_futures_arb".to_string(),
        };

        let futures_close_result = match gateway.submit_order(futures_close_intent).await {
            Ok(result) => {
                info!(
                    "[spot-futures-exec] Futures close FILLED: size={}, avg_price={}, fee={}",
                    result.filled_size, result.avg_fill_price, result.fee
                );
                result
            }
            Err(e) => {
                error!("[spot-futures-exec] Futures close FAILED: {}", e);
                return SpotFuturesExitResult::Failed {
                    reason: format!("Futures close failed: {}", e),
                };
            }
        };

        // Step 2: Market-sell the Spot holdings
        let spot_sell_intent = SpotOrderIntent {
            symbol: spot_spec.symbol.clone(),
            side: OrderSide::Sell,
            qty: spot_qty_held,
            order_type: OrderType::Market,
            price: None,
            time_in_force: "IOC".to_string(),
            quote_order_qty: None,
        };

        match gateway.submit_spot_order(spot_sell_intent).await {
            Ok(spot_result) => {
                info!(
                    "[spot-futures-exec] Spot sell FILLED: qty={}, avg_price={}, fee={}",
                    spot_result.filled_qty, spot_result.avg_fill_price, spot_result.fee
                );
                SpotFuturesExitResult::Success {
                    futures_close_result,
                    spot_sell_result: spot_result,
                }
            }
            Err(e) => {
                error!(
                    "[spot-futures-exec] Spot sell FAILED after futures close: {} -- you still hold the crypto",
                    e
                );
                SpotFuturesExitResult::FuturesClosedSpotFailed {
                    futures_close_result,
                    spot_error: e,
                }
            }
        }
    }

    /// Emergency: sell back Spot position when Futures hedge fails.
    async fn emergency_spot_sell(
        &self,
        gateway: &dyn ExecutionGateway,
        spot_spec: &SpotMarketSpec,
        qty: f64,
    ) -> Option<SpotOrderResult> {
        let intent = SpotOrderIntent {
            symbol: spot_spec.symbol.clone(),
            side: OrderSide::Sell,
            qty,
            order_type: OrderType::Market,
            price: None,
            time_in_force: "IOC".to_string(),
            quote_order_qty: None,
        };

        match gateway.submit_spot_order(intent).await {
            Ok(result) => {
                info!(
                    "[spot-futures-exec] Emergency Spot sell-back: filled={}, price={}",
                    result.filled_qty, result.avg_fill_price
                );
                Some(result)
            }
            Err(e) => {
                error!(
                    "[spot-futures-exec] CRITICAL: Emergency Spot sell-back FAILED: {} -- manual intervention required!",
                    e
                );
                None
            }
        }
    }
}
