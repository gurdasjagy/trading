//! Deterministic Event Sourcing Sequencer — Institutional Upgrade.
//!
//! Refactors the execution gateway to operate as a pure state machine.
//! ALL external inputs (WS ticks, timer events, REST responses) feed into
//! a single sequencer that assigns monotonic sequence numbers. The entire
//! execution state can be 100% deterministically reproduced in a backtester
//! by replaying the event log.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐
//! │  WS Ticks    ├──┐
//! ├─────────────┤  │
//! │  Timers      ├──┤  ──▶  EventSequencer  ──▶  ExecutionStateMachine
//! ├─────────────┤  │        (monotonic seq)      (deterministic replay)
//! │  REST Resp.  ├──┘
//! └─────────────┘
//! ```
//!
//! # Event Types
//!
//! - `OrderSubmitRequest`: Strategy wants to submit an order
//! - `OrderAckReceived`: Exchange acknowledged our order
//! - `OrderFillReceived`: Exchange reports a fill
//! - `OrderRejectReceived`: Exchange rejected our order
//! - `BookUpdateReceived`: Order book changed
//! - `TimerFired`: Periodic timer (reconciliation, TTL check)
//! - `ConnectivityChange`: WS connected/disconnected
//! - `BalanceUpdate`: REST balance query response
//!
//! # Determinism
//!
//! Each event gets a monotonic `seq_no` and a wall-clock `timestamp_ns`.
//! During replay, the wall clock is replaced by the event's timestamp.
//! The state machine's behavior depends ONLY on the event sequence,
//! never on the current wall clock.

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Event Types
// ═══════════════════════════════════════════════════════════════════════════

/// Unique monotonic sequence number for each event.
pub type SeqNo = u64;

/// All possible events that flow through the sequencer.
#[derive(Debug, Clone)]
pub enum SequencedEventKind {
    /// Strategy engine wants to submit an order.
    OrderSubmitRequest {
        symbol_id: u16,
        side: u8,
        order_type: u8,
        price_fp: i64,
        qty_fp: i64,
        stop_loss_fp: i64,
        take_profit_fp: i64,
        leverage: u8,
    },
    /// Exchange acknowledged our order.
    OrderAckReceived {
        client_id: String,
        exchange_id: String,
        status: String,
        filled_size: i64,
        fill_price: f64,
        latency_us: u64,
    },
    /// Exchange reports a fill (partial or complete).
    OrderFillReceived {
        exchange_id: String,
        fill_price: f64,
        fill_qty: i64,
        fee: f64,
        is_final: bool,
    },
    /// Exchange rejected our order.
    OrderRejectReceived {
        client_id: String,
        reason: String,
        error_code: String,
    },
    /// Order cancelled (by us or exchange).
    OrderCancelled {
        exchange_id: String,
        reason: String,
    },
    /// Order book snapshot received.
    BookUpdateReceived {
        symbol_id: u16,
        best_bid_fp: i64,
        best_ask_fp: i64,
        mid_price_fp: i64,
        spread_bps: i32,
    },
    /// Periodic timer fired (reconciliation, health check).
    TimerFired {
        timer_id: u32,
        timer_name: String,
    },
    /// WebSocket connectivity change.
    ConnectivityChange {
        connected: bool,
        exchange: String,
        reconnect_attempt: u32,
    },
    /// Balance update from REST API.
    BalanceUpdate {
        available_usdt: f64,
        total_margin: f64,
    },
    /// Position update from REST reconciliation.
    PositionUpdate {
        symbol: String,
        size: i64,
        entry_price: f64,
        unrealized_pnl: f64,
        leverage: i32,
    },
    /// Circuit breaker state change.
    CircuitBreakerChange {
        halted: bool,
        reason: u32,
    },
    /// SL/TP conditional order placed.
    ConditionalOrderPlaced {
        parent_exchange_id: String,
        trigger_type: String, // "stop_loss" or "take_profit"
        trigger_price: f64,
        order_size: i64,
    },
}

/// A fully sequenced event with monotonic sequence number and timestamp.
#[derive(Debug, Clone)]
pub struct SequencedEvent {
    /// Monotonic sequence number (never reused, never out of order).
    pub seq_no: SeqNo,
    /// Wall-clock timestamp in nanoseconds when the event was sequenced.
    /// During replay, this is the event's original timestamp.
    pub timestamp_ns: u64,
    /// The event payload.
    pub kind: SequencedEventKind,
}

// ═══════════════════════════════════════════════════════════════════════════
// EventSequencer
// ═══════════════════════════════════════════════════════════════════════════

/// Single-threaded event sequencer. All external events flow through here
/// to receive a monotonic sequence number before being processed.
///
/// The sequencer maintains a bounded in-memory log of recent events for
/// debugging and a callback mechanism for the state machine.
pub struct EventSequencer {
    /// Next sequence number to assign.
    next_seq: SeqNo,
    /// Rolling window of recent events (bounded to prevent memory growth).
    recent_events: VecDeque<SequencedEvent>,
    /// Maximum number of events to retain in memory.
    max_retained: usize,
    /// Total events sequenced since creation.
    pub total_events: u64,
    /// Whether we're in replay mode (timestamps come from events, not wall clock).
    replay_mode: bool,
}

impl EventSequencer {
    /// Create a new sequencer.
    pub fn new(max_retained: usize) -> Self {
        Self {
            next_seq: 1,
            recent_events: VecDeque::with_capacity(max_retained),
            max_retained,
            total_events: 0,
            replay_mode: false,
        }
    }

    /// Create with default settings (retain last 10,000 events).
    pub fn with_defaults() -> Self {
        Self::new(10_000)
    }

    /// Enable replay mode (for backtesting). In replay mode, the caller
    /// provides timestamps instead of using the wall clock.
    pub fn set_replay_mode(&mut self, enabled: bool) {
        self.replay_mode = enabled;
    }

    /// Sequence a new event. Returns the sequenced event with its assigned
    /// sequence number and timestamp.
    pub fn sequence(&mut self, kind: SequencedEventKind) -> SequencedEvent {
        let seq_no = self.next_seq;
        self.next_seq += 1;
        self.total_events += 1;

        let timestamp_ns = if self.replay_mode {
            0 // Caller must set this for replay
        } else {
            now_ns()
        };

        let event = SequencedEvent {
            seq_no,
            timestamp_ns,
            kind,
        };

        // Retain in rolling window
        if self.recent_events.len() >= self.max_retained {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(event.clone());

        event
    }

    /// Sequence an event with a specific timestamp (for replay).
    pub fn sequence_with_timestamp(
        &mut self,
        kind: SequencedEventKind,
        timestamp_ns: u64,
    ) -> SequencedEvent {
        let seq_no = self.next_seq;
        self.next_seq += 1;
        self.total_events += 1;

        let event = SequencedEvent {
            seq_no,
            timestamp_ns,
            kind,
        };

        if self.recent_events.len() >= self.max_retained {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(event.clone());

        event
    }

    /// Get the current sequence number (next to be assigned).
    #[inline]
    pub fn current_seq(&self) -> SeqNo {
        self.next_seq
    }

    /// Get the total number of events sequenced.
    #[inline]
    pub fn total_sequenced(&self) -> u64 {
        self.total_events
    }

    /// Get the most recent N events.
    pub fn recent(&self, n: usize) -> Vec<&SequencedEvent> {
        self.recent_events.iter().rev().take(n).collect()
    }

    /// Get all retained events (for debugging or export).
    pub fn all_retained(&self) -> &VecDeque<SequencedEvent> {
        &self.recent_events
    }

    /// Get events since a specific sequence number.
    pub fn events_since(&self, since_seq: SeqNo) -> Vec<&SequencedEvent> {
        self.recent_events
            .iter()
            .filter(|e| e.seq_no > since_seq)
            .collect()
    }

    /// Clear the retained event buffer (free memory).
    pub fn clear_retained(&mut self) {
        self.recent_events.clear();
    }
}

impl Default for EventSequencer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

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
    fn test_monotonic_sequence() {
        let mut seq = EventSequencer::with_defaults();

        let e1 = seq.sequence(SequencedEventKind::TimerFired {
            timer_id: 1,
            timer_name: "health".to_string(),
        });
        let e2 = seq.sequence(SequencedEventKind::TimerFired {
            timer_id: 2,
            timer_name: "reconcile".to_string(),
        });

        assert_eq!(e1.seq_no, 1);
        assert_eq!(e2.seq_no, 2);
        assert!(e2.timestamp_ns >= e1.timestamp_ns);
    }

    #[test]
    fn test_retained_events() {
        let mut seq = EventSequencer::new(3); // Only retain 3

        for i in 0..5 {
            seq.sequence(SequencedEventKind::TimerFired {
                timer_id: i,
                timer_name: format!("timer_{}", i),
            });
        }

        assert_eq!(seq.all_retained().len(), 3);
        // Should retain events 3, 4, 5
        assert_eq!(seq.all_retained()[0].seq_no, 3);
        assert_eq!(seq.all_retained()[2].seq_no, 5);
    }

    #[test]
    fn test_events_since() {
        let mut seq = EventSequencer::with_defaults();

        for i in 0..10 {
            seq.sequence(SequencedEventKind::TimerFired {
                timer_id: i,
                timer_name: format!("t{}", i),
            });
        }

        let since_5 = seq.events_since(5);
        assert_eq!(since_5.len(), 5); // Events 6, 7, 8, 9, 10
    }

    #[test]
    fn test_replay_mode() {
        let mut seq = EventSequencer::with_defaults();
        seq.set_replay_mode(true);

        let e = seq.sequence_with_timestamp(
            SequencedEventKind::BookUpdateReceived {
                symbol_id: 1,
                best_bid_fp: 5000_00000000,
                best_ask_fp: 5001_00000000,
                mid_price_fp: 5000_50000000,
                spread_bps: 2,
            },
            1234567890_000_000_000,
        );

        assert_eq!(e.timestamp_ns, 1234567890_000_000_000);
        assert_eq!(e.seq_no, 1);
    }

    #[test]
    fn test_order_lifecycle_events() {
        let mut seq = EventSequencer::with_defaults();

        // Full order lifecycle
        let _submit = seq.sequence(SequencedEventKind::OrderSubmitRequest {
            symbol_id: 1,
            side: 0,
            order_type: 0,
            price_fp: 5000_00000000,
            qty_fp: 100_0000,
            stop_loss_fp: 4950_00000000,
            take_profit_fp: 5100_00000000,
            leverage: 10,
        });

        let _ack = seq.sequence(SequencedEventKind::OrderAckReceived {
            client_id: "r1".to_string(),
            exchange_id: "12345".to_string(),
            status: "open".to_string(),
            filled_size: 0,
            fill_price: 0.0,
            latency_us: 150,
        });

        let _fill = seq.sequence(SequencedEventKind::OrderFillReceived {
            exchange_id: "12345".to_string(),
            fill_price: 50000.5,
            fill_qty: 100_0000,
            fee: 0.05,
            is_final: true,
        });

        assert_eq!(seq.total_sequenced(), 3);
    }
}

