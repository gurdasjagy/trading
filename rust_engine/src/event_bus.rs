//! Event Bus — flume-based MPMC internal event distribution.
//!
//! Provides a high-performance, bounded, multi-producer multi-consumer event bus
//! for distributing internal events between the Rust execution container threads.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────┐     ┌──────────────┐     ┌──────────────┐
//! │ WS Ingest  │────▶│              │────▶│  Risk Engine  │
//! └────────────┘     │              │     └──────────────┘
//!                    │  Event Bus   │
//! ┌────────────┐     │  (flume)     │     ┌──────────────┐
//! │ Execution  │────▶│              │────▶│  Telemetry    │
//! └────────────┘     │              │     └──────────────┘
//!                    │              │
//! ┌────────────┐     │              │     ┌──────────────┐
//! │ Strategy   │────▶│              │────▶│  Dashboard    │
//! └────────────┘     └──────────────┘     └──────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let bus = EventBus::new(4096);
//! let tx = bus.publisher();
//! let rx = bus.subscriber();
//!
//! // Producer thread:
//! tx.try_publish(EngineEvent::TickUpdate { ... });
//!
//! // Consumer thread:
//! while let Ok(event) = rx.recv() {
//!     match event { ... }
//! }
//! ```
//!
//! # Why flume over crossbeam?
//!
//! - flume supports both bounded and unbounded channels
//! - Async-compatible: `.recv_async()` for tokio integration
//! - Slightly faster for MPMC broadcast patterns
//! - API is a drop-in replacement for std::sync::mpsc

use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

// ═══════════════════════════════════════════════════════════════════════════
// Engine Events
// ═══════════════════════════════════════════════════════════════════════════

/// Events distributed through the internal event bus.
///
/// Each variant carries the minimum data needed for consumers to act.
/// Heavy payloads (orderbook snapshots, etc.) are referenced by ID,
/// not copied through the bus.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// A new tick arrived (symbol_id, mid_price_fp, timestamp_ns).
    TickUpdate {
        symbol_id: u16,
        mid_price_fp: i64,
        spread_bps: i32,
        timestamp_ns: u64,
    },

    /// An order was filled on the exchange.
    OrderFill {
        client_order_id: String,
        exchange_order_id: String,
        symbol_id: u16,
        filled_qty: i64,
        fill_price_fp: i64,
        fee_fp: i64,
        is_maker: bool,
        timestamp_ns: u64,
    },

    /// A position was opened.
    PositionOpened {
        symbol_id: u16,
        is_long: bool,
        entry_price_fp: i64,
        size_contracts: i64,
        leverage: i32,
    },

    /// A position was closed.
    PositionClosed {
        symbol_id: u16,
        exit_price_fp: i64,
        pnl_fp: i64,
        reason: String,
    },

    /// Risk alert from the pre-trade risk engine or circuit breaker.
    RiskAlert {
        level: RiskLevel,
        message: String,
        timestamp_ns: u64,
    },

    /// Market regime change detected by Python brain.
    RegimeChange {
        regime: String,
        momentum_weight: f64,
        volatility_weight: f64,
        timestamp_ns: u64,
    },

    /// Circuit breaker tripped or reset.
    CircuitBreakerEvent {
        halted: bool,
        reason: String,
        timestamp_ns: u64,
    },

    /// Portfolio weight target update from Python brain.
    PortfolioTargetUpdate {
        symbol_id: u16,
        target_weight: f64,
        confidence: f64,
        timestamp_ns: u64,
    },

    /// Heartbeat event (for liveness monitoring).
    Heartbeat {
        thread_name: String,
        timestamp_ns: u64,
    },
}

/// Risk alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Informational — no action required.
    Info,
    /// Warning — increased monitoring.
    Warning,
    /// Critical — immediate action required.
    Critical,
    /// Emergency — halt all trading.
    Emergency,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Info => write!(f, "INFO"),
            RiskLevel::Warning => write!(f, "WARN"),
            RiskLevel::Critical => write!(f, "CRIT"),
            RiskLevel::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Publisher (clone-able sender)
// ═══════════════════════════════════════════════════════════════════════════

/// A clone-able event publisher. Multiple threads can hold a publisher
/// and send events concurrently (MPMC pattern).
#[derive(Clone)]
pub struct EventPublisher {
    tx: flume::Sender<EngineEvent>,
    drops: &'static AtomicU64,
}

impl EventPublisher {
    /// Try to publish an event without blocking.
    /// Returns true if the event was successfully enqueued.
    /// Returns false if the channel is full (event is dropped).
    #[inline]
    pub fn try_publish(&self, event: EngineEvent) -> bool {
        match self.tx.try_send(event) {
            Ok(()) => true,
            Err(flume::TrySendError::Full(_)) => {
                self.drops.fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(flume::TrySendError::Disconnected(_)) => {
                self.drops.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Publish an event, blocking if the channel is full.
    /// Only use from non-hot-path threads (e.g., telemetry, dashboard).
    pub fn publish_blocking(&self, event: EngineEvent) -> bool {
        self.tx.send(event).is_ok()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Subscriber (receiver)
// ═══════════════════════════════════════════════════════════════════════════

/// An event subscriber. Each subscriber receives ALL events from the bus.
/// For fan-out, create multiple subscribers from the same bus.
pub struct EventSubscriber {
    rx: flume::Receiver<EngineEvent>,
}

impl EventSubscriber {
    /// Try to receive an event without blocking.
    #[inline]
    pub fn try_recv(&self) -> Option<EngineEvent> {
        self.rx.try_recv().ok()
    }

    /// Block until an event is available.
    pub fn recv(&self) -> Option<EngineEvent> {
        self.rx.recv().ok()
    }

    /// Drain all currently available events into a Vec.
    /// Useful for batch processing.
    pub fn drain(&self) -> Vec<EngineEvent> {
        self.rx.drain().collect()
    }

    /// Check if the channel is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }

    /// Get the number of pending events.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.rx.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Bus (the central hub)
// ═══════════════════════════════════════════════════════════════════════════

/// Central event bus for the execution container.
///
/// Creates bounded flume channels for internal event distribution.
/// Supports multiple publishers (MPMC) and a single subscriber per
/// `subscriber()` call. For fan-out, call `subscriber()` multiple times
/// — each gets its own channel that shares the same sender.
///
/// Note: flume channels are MPMC but each message goes to ONE receiver.
/// For broadcast (one message to ALL receivers), use multiple buses
/// or re-publish from a fan-out thread.
pub struct EventBus {
    tx: flume::Sender<EngineEvent>,
    rx: flume::Receiver<EngineEvent>,
    capacity: usize,
    total_drops: &'static AtomicU64,
    total_published: AtomicU64,
}

impl EventBus {
    /// Create a new bounded event bus with the given capacity.
    ///
    /// Typical capacities:
    /// - 4096: For high-frequency tick events
    /// - 1024: For order/position events
    /// - 256: For risk alerts and regime changes
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = flume::bounded(capacity);
        let drops = Box::leak(Box::new(AtomicU64::new(0)));
        Self {
            tx,
            rx,
            capacity,
            total_drops: drops,
            total_published: AtomicU64::new(0),
        }
    }

    /// Create a publisher (clone-able sender).
    /// Multiple threads can hold publishers concurrently.
    pub fn publisher(&self) -> EventPublisher {
        EventPublisher {
            tx: self.tx.clone(),
            drops: self.total_drops,
        }
    }

    /// Create a subscriber (receiver).
    /// Note: In flume, a cloned receiver shares the same channel —
    /// each message is delivered to exactly ONE receiver (work-stealing).
    pub fn subscriber(&self) -> EventSubscriber {
        EventSubscriber {
            rx: self.rx.clone(),
        }
    }

    /// Get the total number of dropped events (bus full).
    #[inline]
    pub fn total_drops(&self) -> u64 {
        self.total_drops.load(Ordering::Relaxed)
    }

    /// Get the configured capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current queue depth.
    #[inline]
    pub fn pending(&self) -> usize {
        self.rx.len()
    }

    /// Log bus health metrics.
    pub fn log_health(&self) {
        let drops = self.total_drops();
        let pending = self.pending();
        if drops > 0 {
            warn!(
                "[event-bus] Health: capacity={}, pending={}, total_drops={}",
                self.capacity, pending, drops
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: Create a standard set of buses for the engine
// ═══════════════════════════════════════════════════════════════════════════

/// Standard set of event buses for the trading engine.
///
/// Separates high-frequency tick events from lower-frequency execution events
/// to prevent tick floods from blocking order processing.
pub struct EngineEventBuses {
    /// High-frequency bus: tick updates, depth changes (4096 capacity).
    pub market_data: EventBus,
    /// Medium-frequency bus: order fills, position changes (1024 capacity).
    pub execution: EventBus,
    /// Low-frequency bus: risk alerts, regime changes, heartbeats (256 capacity).
    pub control: EventBus,
}

impl EngineEventBuses {
    /// Create the standard engine event bus topology.
    pub fn new() -> Self {
        Self {
            market_data: EventBus::new(4096),
            execution: EventBus::new(1024),
            control: EventBus::new(256),
        }
    }

    /// Log health metrics for all buses.
    pub fn log_health(&self) {
        self.market_data.log_health();
        self.execution.log_health();
        self.control.log_health();
    }
}

impl Default for EngineEventBuses {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_and_receive() {
        let bus = EventBus::new(16);
        let tx = bus.publisher();
        let rx = bus.subscriber();

        let event = EngineEvent::Heartbeat {
            thread_name: "test".to_string(),
            timestamp_ns: 123456789,
        };
        assert!(tx.try_publish(event));
        let received = rx.try_recv();
        assert!(received.is_some());
        match received.unwrap() {
            EngineEvent::Heartbeat { thread_name, .. } => {
                assert_eq!(thread_name, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_bounded_drops() {
        let bus = EventBus::new(2);
        let tx = bus.publisher();

        // Fill the bus
        assert!(tx.try_publish(EngineEvent::Heartbeat {
            thread_name: "1".into(),
            timestamp_ns: 0,
        }));
        assert!(tx.try_publish(EngineEvent::Heartbeat {
            thread_name: "2".into(),
            timestamp_ns: 0,
        }));
        // This should fail (bus full)
        assert!(!tx.try_publish(EngineEvent::Heartbeat {
            thread_name: "3".into(),
            timestamp_ns: 0,
        }));
        assert_eq!(bus.total_drops(), 1);
    }

    #[test]
    fn test_multiple_publishers() {
        let bus = EventBus::new(16);
        let tx1 = bus.publisher();
        let tx2 = bus.publisher();
        let rx = bus.subscriber();

        tx1.try_publish(EngineEvent::Heartbeat {
            thread_name: "pub1".into(),
            timestamp_ns: 0,
        });
        tx2.try_publish(EngineEvent::Heartbeat {
            thread_name: "pub2".into(),
            timestamp_ns: 0,
        });

        assert_eq!(rx.pending_count(), 2);
    }

    #[test]
    fn test_drain() {
        let bus = EventBus::new(16);
        let tx = bus.publisher();
        let rx = bus.subscriber();

        for i in 0..5 {
            tx.try_publish(EngineEvent::Heartbeat {
                thread_name: format!("hb-{}", i),
                timestamp_ns: i as u64,
            });
        }

        let events = rx.drain();
        assert_eq!(events.len(), 5);
        assert!(rx.is_empty());
    }

    #[test]
    fn test_engine_event_buses() {
        let buses = EngineEventBuses::new();
        assert_eq!(buses.market_data.capacity(), 4096);
        assert_eq!(buses.execution.capacity(), 1024);
        assert_eq!(buses.control.capacity(), 256);
    }
}
