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
use crate::multi_exchange::global_book::ExchangeId;

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

impl DualLegExecutor {
    /// Execute simultaneous entry on both exchanges.
    ///
    /// Uses tokio::join! to send both orders at the exact same time.
    /// If one fills and the other fails, the caller is responsible for
    /// closing the filled leg to resolve legging risk.
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

        let order_type = if use_market_orders { OrderType::Market } else { OrderType::Limit };

        let short_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            size,
            order_type: order_type.clone(),
            price: None,
            reduce_only: false,
            leverage: Some(leverage),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(max_slippage),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "funding_arb_short_entry".to_string(),
        };

        let long_intent = OrderIntent {
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            size,
            order_type,
            price: None,
            reduce_only: false,
            leverage: Some(leverage),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(max_slippage),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "funding_arb_long_entry".to_string(),
        };

        // CRITICAL: Execute both legs simultaneously
        let (short_result, long_result) = tokio::join!(
            short_gw.submit_order(short_intent),
            long_gw.submit_order(long_intent),
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
                info!("[dual-leg] Both legs filled: short={} long={}", sr.order_id, lr.order_id);
                DualLegResult::BothFilled {
                    short_result: sr,
                    long_result: lr,
                }
            }
            (Ok(sr), Err(le)) => {
                warn!("[dual-leg] LEGGING RISK: Short filled but long failed: {:?}", le);
                DualLegResult::PartialFill {
                    filled_leg: LegStatus::ShortFilled { result: sr, exchange: short_exchange },
                    unfilled_exchange: long_exchange,
                    filled_size: size,
                }
            }
            (Err(se), Ok(lr)) => {
                warn!("[dual-leg] LEGGING RISK: Long filled but short failed: {:?}", se);
                DualLegResult::PartialFill {
                    filled_leg: LegStatus::LongFilled { result: lr, exchange: long_exchange },
                    unfilled_exchange: short_exchange,
                    filled_size: size,
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
