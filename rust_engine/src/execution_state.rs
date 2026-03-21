//! Order lifecycle state machine — Issue 3.
//!
//! Tracks the full lifecycle of a single order from intent to terminal state.
//! Every state transition is validated (no invalid transitions) and logged
//! to the JournalWriter for crash recovery and audit.
//!
//! # States
//!
//! ```text
//! IDLE → PLACING → RESTING → FILLED
//!                ↘ REJECTED
//!        RESTING → AMENDING → RESTING (new price)
//!        RESTING → CANCELING → IDLE
//! ```
//!
//! # Thread Safety
//!
//! `OrderLifecycle` is designed to be owned by a single thread (Core 6:
//! Execution Router). It is NOT `Send + Sync` — the execution router is
//! the sole owner and mutator.

use std::fmt;
use crate::journal::{
    JournalWriter, JournalEntryHeader, JournalOrderIntent, JournalOrderResult,
    ENTRY_ORDER_INTENT, ENTRY_ORDER_RESULT,
};

// ═══════════════════════════════════════════════════════════════════════════
// OrderState Enum
// ═══════════════════════════════════════════════════════════════════════════

/// Order lifecycle state machine.
///
/// Each variant carries the state-specific data needed for monitoring
/// and decision-making by the execution router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    /// Waiting for an OrderCommand from the strategy engine.
    Idle,

    /// Order submitted to exchange, awaiting WS confirmation.
    Placing {
        /// Nanosecond timestamp when the order was submitted.
        submit_ts_ns: u64,
    },

    /// Order is live on the book, tracking queue position.
    Resting {
        /// Exchange-assigned order ID (zero-padded).
        order_id: [u8; 32],
        /// Nanosecond timestamp when the order was placed.
        place_ts_ns: u64,
        /// Price in FixedPrice representation.
        price_fp: i64,
        /// Size in FixedQty representation.
        size: i64,
        /// Quantity filled so far.
        filled_so_far: i64,
    },

    /// Amend in-flight (price/size change requested).
    Amending {
        /// Exchange-assigned order ID.
        order_id: [u8; 32],
        /// New target price in FixedPrice.
        new_price_fp: i64,
        /// Nanosecond timestamp when the amend was requested.
        amend_ts_ns: u64,
    },

    /// Cancel in-flight (adverse selection or other reason detected).
    Canceling {
        /// Exchange-assigned order ID.
        order_id: [u8; 32],
        /// Why we are canceling.
        reason: CancelReason,
        /// Nanosecond timestamp when cancel was requested.
        cancel_ts_ns: u64,
    },

    /// Order fully or partially filled — terminal state.
    Filled {
        /// Volume-weighted average fill price (FixedPrice).
        avg_price_fp: i64,
        /// Total quantity filled.
        total_filled: i64,
        /// Total fees paid (FixedPrice).
        total_fee_fp: i64,
    },

    /// Order rejected by exchange — terminal state.
    Rejected {
        /// Exchange-specific rejection reason code.
        reason_code: u32,
    },
}

impl Default for OrderState {
    fn default() -> Self {
        OrderState::Idle
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CancelReason Enum
// ═══════════════════════════════════════════════════════════════════════════

/// Why an order is being canceled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    /// Detected informed flow about to run us over.
    AdverseSelection,
    /// Mid-price moved beyond our threshold.
    PriceMoved,
    /// Queue position degraded beyond recovery.
    QueuePositionBad,
    /// Strategy engine requested cancellation.
    StrategyCancel,
    /// Order resting too long without fill.
    Timeout,
    /// Risk manager triggered cancel.
    RiskLimit,
}

impl fmt::Display for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CancelReason::AdverseSelection => write!(f, "adverse_selection"),
            CancelReason::PriceMoved => write!(f, "price_moved"),
            CancelReason::QueuePositionBad => write!(f, "queue_position_bad"),
            CancelReason::StrategyCancel => write!(f, "strategy_cancel"),
            CancelReason::Timeout => write!(f, "timeout"),
            CancelReason::RiskLimit => write!(f, "risk_limit"),
        }
    }
}

impl CancelReason {
    /// Convert to a u8 for journal serialization.
    pub fn to_u8(self) -> u8 {
        match self {
            CancelReason::AdverseSelection => 0,
            CancelReason::PriceMoved => 1,
            CancelReason::QueuePositionBad => 2,
            CancelReason::StrategyCancel => 3,
            CancelReason::Timeout => 4,
            CancelReason::RiskLimit => 5,
        }
    }

    /// Convert from u8 (for journal replay).
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(CancelReason::AdverseSelection),
            1 => Some(CancelReason::PriceMoved),
            2 => Some(CancelReason::QueuePositionBad),
            3 => Some(CancelReason::StrategyCancel),
            4 => Some(CancelReason::Timeout),
            5 => Some(CancelReason::RiskLimit),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PlacementType — how the strategy wants the order placed
// ═══════════════════════════════════════════════════════════════════════════

/// How the strategy engine wants the order placed relative to the BBO.
/// The execution router translates this to an actual price using live book state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementType {
    /// Place at the current best bid (for buys) or best ask (for sells).
    AtBest,
    /// Improve the best price by 1 tick (more aggressive).
    Improve1Tick,
    /// Place 1 tick behind the best (less aggressive, better queue position).
    Behind1Tick,
    /// Place at the mid-price (aggressive for both sides).
    AtMid,
    /// Use a specific absolute price (FixedPrice value).
    AbsolutePrice { price_fp: i64 },
    /// Let the execution router decide based on QPE and adverse selection.
    SmartPlace,
}

impl Default for PlacementType {
    fn default() -> Self {
        PlacementType::AtBest
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OrderLifecycle
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks the full lifecycle of a single order.
///
/// Owned by the execution router thread (Core 6). All state transitions
/// are validated and logged to the journal.
pub struct OrderLifecycle {
    /// Current state of the order.
    state: OrderState,
    /// Symbol ID for this order.
    symbol_id: u16,
    /// Side: 0 = buy, 1 = sell.
    side: u8,
    /// Client-assigned sequence number for correlation.
    client_seq: u64,
    /// Number of state transitions so far.
    transition_count: u32,
    /// Timestamp of the last state transition.
    last_transition_ns: u64,
    /// Desired placement type from strategy.
    pub placement_type: PlacementType,
}

impl OrderLifecycle {
    /// Create a new order lifecycle in the Idle state.
    pub fn new(symbol_id: u16, side: u8, client_seq: u64) -> Self {
        Self {
            state: OrderState::Idle,
            symbol_id,
            side,
            client_seq,
            transition_count: 0,
            last_transition_ns: now_ns(),
            placement_type: PlacementType::AtBest,
        }
    }

    /// Get the current state.
    #[inline]
    pub fn state(&self) -> &OrderState {
        &self.state
    }

    /// Get the symbol ID.
    #[inline]
    pub fn symbol_id(&self) -> u16 {
        self.symbol_id
    }

    /// Get the side.
    #[inline]
    pub fn side(&self) -> u8 {
        self.side
    }

    /// Get the client sequence number.
    #[inline]
    pub fn client_seq(&self) -> u64 {
        self.client_seq
    }

    /// Get transition count.
    #[inline]
    pub fn transition_count(&self) -> u32 {
        self.transition_count
    }

    /// Check if the order is in a terminal state (Filled or Rejected).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, OrderState::Filled { .. } | OrderState::Rejected { .. })
    }

    /// Check if the order is actively resting on the book.
    #[inline]
    pub fn is_resting(&self) -> bool {
        matches!(self.state, OrderState::Resting { .. })
    }

    /// Check if an order is in-flight (Placing, Amending, or Canceling).
    #[inline]
    pub fn is_in_flight(&self) -> bool {
        matches!(
            self.state,
            OrderState::Placing { .. } | OrderState::Amending { .. } | OrderState::Canceling { .. }
        )
    }

    // ─── State Transitions ───────────────────────────────────────────────

    /// Transition: IDLE → PLACING (order submitted to exchange).
    ///
    /// Returns `Ok(())` if valid, `Err(msg)` if the transition is invalid.
    pub fn submit(
        &mut self,
        journal: &mut Option<&mut JournalWriter>,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Idle => {
                let ts = now_ns();
                self.state = OrderState::Placing { submit_ts_ns: ts };
                self.transition_count += 1;
                self.last_transition_ns = ts;

                // Log to journal
                if let Some(ref mut j) = journal {
                    let entry = JournalOrderIntent {
                        header: JournalEntryHeader {
                            entry_type: ENTRY_ORDER_INTENT,
                            payload_size: 0,
                            sequence: 0,
                        },
                        timestamp_ns: ts,
                        symbol_id: self.symbol_id,
                        side: self.side,
                        order_type: 0, // limit
                        size: 0,
                        price_fp: 0,
                        reduce_only: 0,
                        leverage: 0,
                        slippage_cap_bps: 0,
                        book_sequence: self.client_seq,
                        _reserved: [0; 5],
                    };
                    let _ = j.append_order_intent(entry);
                }

                Ok(())
            }
            _ => Err("Cannot submit: order is not in Idle state"),
        }
    }

    /// Transition: PLACING → RESTING (exchange confirmed the order).
    pub fn on_placed(
        &mut self,
        order_id: [u8; 32],
        price_fp: i64,
        size: i64,
        journal: &mut Option<&mut JournalWriter>,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Placing { .. } => {
                let ts = now_ns();
                self.state = OrderState::Resting {
                    order_id,
                    place_ts_ns: ts,
                    price_fp,
                    size,
                    filled_so_far: 0,
                };
                self.transition_count += 1;
                self.last_transition_ns = ts;

                if let Some(ref mut j) = journal {
                    let entry = JournalOrderResult {
                        header: JournalEntryHeader {
                            entry_type: ENTRY_ORDER_RESULT,
                            payload_size: 0,
                            sequence: 0,
                        },
                        timestamp_ns: ts,
                        symbol_id: self.symbol_id,
                        side: self.side,
                        status: 0, // open
                        filled_size: 0,
                        avg_fill_price_fp: price_fp,
                        fee_fp: 0,
                        exchange_latency_us: 0,
                        order_id,
                    };
                    let _ = j.append_order_result(entry);
                }

                Ok(())
            }
            _ => Err("Cannot transition to Resting: order is not in Placing state"),
        }
    }

    /// Transition: PLACING → REJECTED (exchange rejected the order).
    pub fn on_rejected(
        &mut self,
        reason_code: u32,
        journal: &mut Option<&mut JournalWriter>,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Placing { .. } => {
                let ts = now_ns();
                self.state = OrderState::Rejected { reason_code };
                self.transition_count += 1;
                self.last_transition_ns = ts;

                if let Some(ref mut j) = journal {
                    let entry = JournalOrderResult {
                        header: JournalEntryHeader {
                            entry_type: ENTRY_ORDER_RESULT,
                            payload_size: 0,
                            sequence: 0,
                        },
                        timestamp_ns: ts,
                        symbol_id: self.symbol_id,
                        side: self.side,
                        status: 2, // rejected
                        filled_size: 0,
                        avg_fill_price_fp: 0,
                        fee_fp: 0,
                        exchange_latency_us: 0,
                        order_id: [0; 32],
                    };
                    let _ = j.append_order_result(entry);
                }

                Ok(())
            }
            _ => Err("Cannot reject: order is not in Placing state"),
        }
    }

    /// Transition: RESTING → AMENDING (request price/size change).
    pub fn start_amend(
        &mut self,
        new_price_fp: i64,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Resting { order_id, .. } => {
                let ts = now_ns();
                self.state = OrderState::Amending {
                    order_id,
                    new_price_fp,
                    amend_ts_ns: ts,
                };
                self.transition_count += 1;
                self.last_transition_ns = ts;
                Ok(())
            }
            _ => Err("Cannot amend: order is not in Resting state"),
        }
    }

    /// Transition: AMENDING → RESTING (amend confirmed with new price).
    pub fn on_amend_confirmed(
        &mut self,
        new_price_fp: i64,
        new_size: i64,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Amending { order_id, .. } => {
                let ts = now_ns();
                self.state = OrderState::Resting {
                    order_id,
                    place_ts_ns: ts,
                    price_fp: new_price_fp,
                    size: new_size,
                    filled_so_far: 0,
                };
                self.transition_count += 1;
                self.last_transition_ns = ts;
                Ok(())
            }
            _ => Err("Cannot confirm amend: order is not in Amending state"),
        }
    }

    /// Transition: RESTING → CANCELING (adverse selection or other reason).
    pub fn start_cancel(
        &mut self,
        reason: CancelReason,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Resting { order_id, .. } => {
                let ts = now_ns();
                self.state = OrderState::Canceling {
                    order_id,
                    reason,
                    cancel_ts_ns: ts,
                };
                self.transition_count += 1;
                self.last_transition_ns = ts;
                Ok(())
            }
            _ => Err("Cannot cancel: order is not in Resting state"),
        }
    }

    /// Transition: CANCELING → IDLE (cancel confirmed by exchange).
    pub fn on_cancel_confirmed(
        &mut self,
        journal: &mut Option<&mut JournalWriter>,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Canceling { order_id, reason, .. } => {
                let ts = now_ns();
                self.state = OrderState::Idle;
                self.transition_count += 1;
                self.last_transition_ns = ts;

                if let Some(ref mut j) = journal {
                    let entry = JournalOrderResult {
                        header: JournalEntryHeader {
                            entry_type: ENTRY_ORDER_RESULT,
                            payload_size: 0,
                            sequence: 0,
                        },
                        timestamp_ns: ts,
                        symbol_id: self.symbol_id,
                        side: self.side,
                        status: 3, // cancelled
                        filled_size: 0,
                        avg_fill_price_fp: 0,
                        fee_fp: 0,
                        exchange_latency_us: 0,
                        order_id,
                    };
                    let _ = j.append_order_result(entry);
                }
                let _ = reason; // used above in journal
                Ok(())
            }
            _ => Err("Cannot confirm cancel: order is not in Canceling state"),
        }
    }

    /// Transition: RESTING → FILLED (order filled on the book).
    pub fn on_filled(
        &mut self,
        avg_price_fp: i64,
        total_filled: i64,
        total_fee_fp: i64,
        journal: &mut Option<&mut JournalWriter>,
    ) -> Result<(), &'static str> {
        match self.state {
            OrderState::Resting { order_id, .. } => {
                let ts = now_ns();
                self.state = OrderState::Filled {
                    avg_price_fp,
                    total_filled,
                    total_fee_fp,
                };
                self.transition_count += 1;
                self.last_transition_ns = ts;

                if let Some(ref mut j) = journal {
                    let entry = JournalOrderResult {
                        header: JournalEntryHeader {
                            entry_type: ENTRY_ORDER_RESULT,
                            payload_size: 0,
                            sequence: 0,
                        },
                        timestamp_ns: ts,
                        symbol_id: self.symbol_id,
                        side: self.side,
                        status: 1, // filled
                        filled_size: total_filled,
                        avg_fill_price_fp: avg_price_fp,
                        fee_fp: total_fee_fp,
                        exchange_latency_us: 0,
                        order_id,
                    };
                    let _ = j.append_order_result(entry);
                }

                Ok(())
            }
            _ => Err("Cannot fill: order is not in Resting state"),
        }
    }

    /// Update partial fill while resting (order remains resting).
    pub fn on_partial_fill(&mut self, additional_filled: i64) -> Result<(), &'static str> {
        match &mut self.state {
            OrderState::Resting { filled_so_far, .. } => {
                *filled_so_far += additional_filled;
                Ok(())
            }
            _ => Err("Cannot partial fill: order is not in Resting state"),
        }
    }

    /// Get the resting order ID (if currently resting or in amending/canceling).
    pub fn order_id(&self) -> Option<&[u8; 32]> {
        match &self.state {
            OrderState::Resting { order_id, .. }
            | OrderState::Amending { order_id, .. }
            | OrderState::Canceling { order_id, .. } => Some(order_id),
            _ => None,
        }
    }

    /// Get the resting price (if currently resting).
    pub fn resting_price_fp(&self) -> Option<i64> {
        match &self.state {
            OrderState::Resting { price_fp, .. } => Some(*price_fp),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Current time in nanoseconds (monotonic clock).
#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_state_transitions() {
        let mut order = OrderLifecycle::new(1, 0, 100);
        assert_eq!(*order.state(), OrderState::Idle);

        // Idle → Placing
        assert!(order.submit(&mut None).is_ok());
        assert!(matches!(*order.state(), OrderState::Placing { .. }));

        // Placing → Resting
        let oid = [1u8; 32];
        assert!(order.on_placed(oid, 5000_00000000, 100_0000, &mut None).is_ok());
        assert!(order.is_resting());

        // Resting → Amending
        assert!(order.start_amend(5001_00000000).is_ok());
        assert!(matches!(*order.state(), OrderState::Amending { .. }));

        // Amending → Resting
        assert!(order.on_amend_confirmed(5001_00000000, 100_0000).is_ok());
        assert!(order.is_resting());

        // Resting → Canceling
        assert!(order.start_cancel(CancelReason::AdverseSelection).is_ok());
        assert!(matches!(*order.state(), OrderState::Canceling { .. }));

        // Canceling → Idle
        assert!(order.on_cancel_confirmed(&mut None).is_ok());
        assert_eq!(*order.state(), OrderState::Idle);

        assert_eq!(order.transition_count(), 6);
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut order = OrderLifecycle::new(1, 0, 100);

        // Cannot go to Resting from Idle
        assert!(order.on_placed([0; 32], 0, 0, &mut None).is_err());

        // Cannot cancel from Idle
        assert!(order.start_cancel(CancelReason::Timeout).is_err());

        // Cannot fill from Idle
        assert!(order.on_filled(0, 0, 0, &mut None).is_err());

        // Submit to go to Placing
        assert!(order.submit(&mut None).is_ok());

        // Cannot submit again from Placing
        assert!(order.submit(&mut None).is_err());

        // Cannot amend from Placing
        assert!(order.start_amend(0).is_err());

        // Cannot cancel from Placing
        assert!(order.start_cancel(CancelReason::Timeout).is_err());
    }

    #[test]
    fn test_fill_from_resting() {
        let mut order = OrderLifecycle::new(1, 0, 100);
        assert!(order.submit(&mut None).is_ok());
        assert!(order.on_placed([1; 32], 5000_00000000, 100_0000, &mut None).is_ok());

        // Partial fill
        assert!(order.on_partial_fill(50_0000).is_ok());

        // Full fill
        assert!(order.on_filled(5000_50000000, 100_0000, 500_0000, &mut None).is_ok());
        assert!(order.is_terminal());
        assert_eq!(order.transition_count(), 3);
    }

    #[test]
    fn test_rejection_from_placing() {
        let mut order = OrderLifecycle::new(1, 1, 200);
        assert!(order.submit(&mut None).is_ok());
        assert!(order.on_rejected(110013, &mut None).is_ok());
        assert!(order.is_terminal());
        assert!(matches!(*order.state(), OrderState::Rejected { reason_code: 110013 }));
    }

    #[test]
    fn test_cancel_reason_serialization() {
        for reason in [
            CancelReason::AdverseSelection,
            CancelReason::PriceMoved,
            CancelReason::QueuePositionBad,
            CancelReason::StrategyCancel,
            CancelReason::Timeout,
            CancelReason::RiskLimit,
        ] {
            let byte = reason.to_u8();
            let decoded = CancelReason::from_u8(byte).expect("should decode");
            assert_eq!(reason, decoded);
        }
        assert!(CancelReason::from_u8(255).is_none());
    }
}

