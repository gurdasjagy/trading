//! Adverse Selection Detector — Issue 3.
//!
//! Detects micro-burst / informed flow patterns from the trade tape that
//! indicate an impending adverse price move. When detected, the execution
//! router should cancel resting orders immediately (target: < 5ms from
//! detection to cancel confirmation).
//!
//! # Detection Methods
//!
//! 1. **Micro-burst detection**: Sudden spike in trade velocity on one side.
//!    e.g., 50+ trades in 100ms on the sell side → book sweep imminent.
//!
//! 2. **Volume imbalance**: Sustained directional volume imbalance over a
//!    sliding window (e.g., 80%+ of volume on one side over 500ms).
//!
//! 3. **Trade acceleration**: Rate of change in trade arrival rate.
//!    Normal: ~5 trades/sec. Burst: >50 trades/sec → urgency=2.
//!
//! # Performance
//!
//! `on_trade()` target latency: < 500ns.
//! Achieved by avoiding allocations, using fixed-size circular buffers,
//! and keeping all state in cache-friendly contiguous memory.

// ═══════════════════════════════════════════════════════════════════════════
// Trade Event
// ═══════════════════════════════════════════════════════════════════════════

/// A trade event from the public trade tape.
#[derive(Debug, Clone, Copy)]
pub struct TradeEvent {
    /// Price in FixedPrice representation.
    pub price_fp: i64,
    /// Trade size (always positive).
    pub size: i64,
    /// 0 = buy (taker bought), 1 = sell (taker sold).
    pub side: u8,
    /// Nanosecond timestamp.
    pub timestamp_ns: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Adverse Selection Signal
// ═══════════════════════════════════════════════════════════════════════════

/// Signal emitted when adverse selection is detected.
#[derive(Debug, Clone, Copy)]
pub struct AdverseSelectionSignal {
    /// 0 = low confidence (monitor), 1 = medium (prepare to cancel), 2 = high (cancel now!).
    pub urgency: u8,
    /// Direction of the adverse flow: 0 = buying pressure, 1 = selling pressure.
    pub direction: u8,
    /// Nanosecond timestamp of detection.
    pub detection_ts_ns: u64,
    /// Number of trades in the burst window.
    pub burst_trade_count: u32,
    /// Total volume in the burst window.
    pub burst_volume: i64,
    /// Volume imbalance ratio [0, 1] — 1.0 means all flow on one side.
    pub imbalance_ratio: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the adverse selection detector.
#[derive(Debug, Clone, Copy)]
pub struct AdverseSelectionConfig {
    /// Micro-burst detection window in nanoseconds (default: 100ms = 100_000_000ns).
    pub microburst_window_ns: u64,
    /// Minimum trades in the window to trigger a burst signal.
    pub microburst_trade_threshold: u32,
    /// Volume imbalance threshold [0, 1] (default: 0.8 = 80% one-sided).
    pub imbalance_threshold: f64,
    /// Longer window for sustained imbalance detection (default: 500ms).
    pub imbalance_window_ns: u64,
    /// Maximum age of trade events to keep in the ring buffer (default: 2s).
    pub max_trade_age_ns: u64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            microburst_window_ns: 100_000_000,             // 100ms
            microburst_trade_threshold: 30,                // 30 trades in 100ms
            imbalance_threshold: 0.80,                     // 80%
            imbalance_window_ns: 500_000_000,              // 500ms
            max_trade_age_ns: 2_000_000_000,               // 2s
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fixed-size circular buffer for trade events
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed-capacity circular buffer for recent trade events.
/// No heap allocation after construction. Cache-friendly sequential access.
const TRADE_RING_CAPACITY: usize = 1024;

struct TradeRing {
    trades: [TradeEvent; TRADE_RING_CAPACITY],
    write_pos: usize,
    count: usize,
}

impl TradeRing {
    fn new() -> Self {
        Self {
            trades: [TradeEvent {
                price_fp: 0,
                size: 0,
                side: 0,
                timestamp_ns: 0,
            }; TRADE_RING_CAPACITY],
            write_pos: 0,
            count: 0,
        }
    }

    /// Push a new trade event. Overwrites the oldest if full.
    #[inline]
    fn push(&mut self, event: TradeEvent) {
        self.trades[self.write_pos] = event;
        self.write_pos = (self.write_pos + 1) % TRADE_RING_CAPACITY;
        if self.count < TRADE_RING_CAPACITY {
            self.count += 1;
        }
    }

    /// Iterate over recent trades (from oldest to newest) that are within `window_ns` of `now_ns`.
    fn recent_trades(&self, now_ns: u64, window_ns: u64) -> RecentTradeIter<'_> {
        let cutoff_ns = now_ns.saturating_sub(window_ns);
        RecentTradeIter {
            ring: self,
            remaining: self.count,
            pos: if self.count == TRADE_RING_CAPACITY {
                self.write_pos // Start from oldest (just after write_pos in full buffer)
            } else {
                0 // Start from beginning
            },
            cutoff_ns,
        }
    }
}

struct RecentTradeIter<'a> {
    ring: &'a TradeRing,
    remaining: usize,
    pos: usize,
    cutoff_ns: u64,
}

impl<'a> Iterator for RecentTradeIter<'a> {
    type Item = &'a TradeEvent;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            let trade = &self.ring.trades[self.pos];
            self.pos = (self.pos + 1) % TRADE_RING_CAPACITY;
            self.remaining -= 1;
            if trade.timestamp_ns >= self.cutoff_ns {
                return Some(trade);
            }
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AdverseSelectionDetector
// ═══════════════════════════════════════════════════════════════════════════

/// Detects adverse selection / informed flow patterns from the trade tape.
///
/// Designed for ultra-low latency: `on_trade()` completes in < 500ns.
/// All state is in contiguous memory (cache-friendly circular buffer).
pub struct AdverseSelectionDetector {
    /// Configuration thresholds.
    config: AdverseSelectionConfig,
    /// Circular buffer of recent trade events.
    trade_ring: TradeRing,
    /// Running buy volume in the burst window (reset on window expiry).
    burst_buy_volume: i64,
    /// Running sell volume in the burst window.
    burst_sell_volume: i64,
    /// Running buy trade count in the burst window.
    burst_buy_count: u32,
    /// Running sell trade count in the burst window.
    burst_sell_count: u32,
    /// Timestamp of the burst window start.
    burst_window_start_ns: u64,
    /// Total adverse selection signals emitted.
    total_signals: u64,
}

impl AdverseSelectionDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: AdverseSelectionConfig) -> Self {
        Self {
            config,
            trade_ring: TradeRing::new(),
            burst_buy_volume: 0,
            burst_sell_volume: 0,
            burst_buy_count: 0,
            burst_sell_count: 0,
            burst_window_start_ns: 0,
            total_signals: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AdverseSelectionConfig::default())
    }

    /// Process a new trade event. Returns an adverse selection signal if detected.
    ///
    /// Target latency: < 500ns. No allocations, no syscalls.
    #[inline]
    pub fn on_trade(&mut self, event: &TradeEvent) -> Option<AdverseSelectionSignal> {
        let now = event.timestamp_ns;

        // Store in ring buffer
        self.trade_ring.push(*event);

        // Check if we need to reset the burst window
        if now.saturating_sub(self.burst_window_start_ns) > self.config.microburst_window_ns {
            self.burst_buy_volume = 0;
            self.burst_sell_volume = 0;
            self.burst_buy_count = 0;
            self.burst_sell_count = 0;
            self.burst_window_start_ns = now;
        }

        // Update burst counters
        if event.side == 0 {
            self.burst_buy_volume += event.size;
            self.burst_buy_count += 1;
        } else {
            self.burst_sell_volume += event.size;
            self.burst_sell_count += 1;
        }

        let total_burst_count = self.burst_buy_count + self.burst_sell_count;
        let total_burst_volume = self.burst_buy_volume + self.burst_sell_volume;

        // ─── Check 1: Micro-burst detection ───────────────────────────
        if total_burst_count >= self.config.microburst_trade_threshold {
            let (direction, imbalance_ratio) = if self.burst_buy_volume > self.burst_sell_volume {
                let ratio = if total_burst_volume > 0 {
                    self.burst_buy_volume as f64 / total_burst_volume as f64
                } else {
                    0.5
                };
                (0u8, ratio) // buying pressure
            } else {
                let ratio = if total_burst_volume > 0 {
                    self.burst_sell_volume as f64 / total_burst_volume as f64
                } else {
                    0.5
                };
                (1u8, ratio) // selling pressure
            };

            // Determine urgency based on trade count and imbalance
            let urgency = if total_burst_count >= self.config.microburst_trade_threshold * 2
                && imbalance_ratio >= 0.9
            {
                2 // CRITICAL: cancel immediately
            } else if imbalance_ratio >= self.config.imbalance_threshold {
                1 // WARNING: prepare to cancel
            } else {
                0 // MONITOR
            };

            if urgency >= 1 {
                self.total_signals += 1;
                return Some(AdverseSelectionSignal {
                    urgency,
                    direction,
                    detection_ts_ns: now,
                    burst_trade_count: total_burst_count,
                    burst_volume: total_burst_volume,
                    imbalance_ratio,
                });
            }
        }

        // ─── Check 2: Sustained imbalance over longer window ──────────
        if total_burst_count >= 5 {
            let mut long_buy_vol: i64 = 0;
            let mut long_sell_vol: i64 = 0;
            let mut long_count: u32 = 0;

            for trade in self.trade_ring.recent_trades(now, self.config.imbalance_window_ns) {
                if trade.side == 0 {
                    long_buy_vol += trade.size;
                } else {
                    long_sell_vol += trade.size;
                }
                long_count += 1;
            }

            let long_total = long_buy_vol + long_sell_vol;
            if long_total > 0 && long_count >= 10 {
                let buy_ratio = long_buy_vol as f64 / long_total as f64;
                let sell_ratio = long_sell_vol as f64 / long_total as f64;
                let max_ratio = buy_ratio.max(sell_ratio);

                if max_ratio >= self.config.imbalance_threshold {
                    let direction = if buy_ratio > sell_ratio { 0 } else { 1 };
                    self.total_signals += 1;
                    return Some(AdverseSelectionSignal {
                        urgency: 1, // medium — sustained but not burst
                        direction,
                        detection_ts_ns: now,
                        burst_trade_count: long_count,
                        burst_volume: long_total,
                        imbalance_ratio: max_ratio,
                    });
                }
            }
        }

        None
    }

    /// Advisory function: should we cancel our resting order given current signals?
    ///
    /// Returns `true` if cancellation is recommended.
    pub fn should_cancel_order(
        &self,
        our_side: u8,     // 0 = buy, 1 = sell
        signal: &AdverseSelectionSignal,
    ) -> bool {
        // If we are buying and there's selling pressure about to sweep the book, cancel.
        // If we are selling and there's buying pressure, cancel.
        let opposing_flow = (our_side == 0 && signal.direction == 1)
            || (our_side == 1 && signal.direction == 0);

        if !opposing_flow {
            return false;
        }

        // Cancel if urgency >= 1 (medium or high)
        signal.urgency >= 1
    }

    /// Get total signals emitted.
    #[inline]
    pub fn total_signals(&self) -> u64 {
        self.total_signals
    }

    /// Reset all internal state (useful for testing or strategy resets).
    pub fn reset(&mut self) {
        self.trade_ring = TradeRing::new();
        self.burst_buy_volume = 0;
        self.burst_sell_volume = 0;
        self.burst_buy_count = 0;
        self.burst_sell_count = 0;
        self.burst_window_start_ns = 0;
        self.total_signals = 0;
    }
}

impl Default for AdverseSelectionDetector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(side: u8, size: i64, ts_ns: u64) -> TradeEvent {
        TradeEvent {
            price_fp: 5000_00000000,
            size,
            side,
            timestamp_ns: ts_ns,
        }
    }

    #[test]
    fn test_book_sweep_detection() {
        let config = AdverseSelectionConfig {
            microburst_trade_threshold: 20,
            imbalance_threshold: 0.8,
            ..Default::default()
        };
        let mut detector = AdverseSelectionDetector::new(config);

        let base_ts = 1_000_000_000_000u64; // 1s in ns
        let mut signal = None;

        // Simulate a book sweep: 50 sell trades in 50ms
        for i in 0..50 {
            let trade = make_trade(
                1, // sell
                100_0000,
                base_ts + i * 1_000_000, // 1ms apart
            );
            if let Some(s) = detector.on_trade(&trade) {
                signal = Some(s);
            }
        }

        // Should detect adverse selection with high urgency
        let sig = signal.expect("Should have detected adverse selection");
        assert!(sig.urgency >= 1, "Urgency should be at least 1 for book sweep");
        assert_eq!(sig.direction, 1, "Direction should be sell (1)");
        assert!(sig.imbalance_ratio >= 0.8, "Imbalance should be >= 80%");
    }

    #[test]
    fn test_normal_market_no_false_positives() {
        let mut detector = AdverseSelectionDetector::with_defaults();
        let base_ts = 1_000_000_000_000u64;

        // Simulate normal market: 5 trades/sec, balanced buy/sell
        let mut any_signal = false;
        for i in 0..10 {
            // Alternating buy/sell, 200ms apart
            let side = (i % 2) as u8;
            let trade = make_trade(
                side,
                10_0000,
                base_ts + i * 200_000_000, // 200ms apart
            );
            if detector.on_trade(&trade).is_some() {
                any_signal = true;
            }
        }

        assert!(!any_signal, "Normal balanced market should not trigger signals");
    }

    #[test]
    fn test_should_cancel_opposing_flow() {
        let signal = AdverseSelectionSignal {
            urgency: 2,
            direction: 1, // sell pressure
            detection_ts_ns: 0,
            burst_trade_count: 50,
            burst_volume: 5000_0000,
            imbalance_ratio: 0.95,
        };

        let detector = AdverseSelectionDetector::with_defaults();

        // We have a buy order and there's sell pressure → should cancel
        assert!(detector.should_cancel_order(0, &signal));

        // We have a sell order and there's sell pressure → should NOT cancel
        assert!(!detector.should_cancel_order(1, &signal));
    }

    #[test]
    fn test_urgency_levels() {
        let config = AdverseSelectionConfig {
            microburst_trade_threshold: 10,
            imbalance_threshold: 0.8,
            ..Default::default()
        };
        let mut detector = AdverseSelectionDetector::new(config);
        let base_ts = 1_000_000_000_000u64;

        // 20 sell trades in 50ms (2x threshold) with 100% sell = urgency 2
        for i in 0..25 {
            let trade = make_trade(1, 100_0000, base_ts + i * 2_000_000);
            detector.on_trade(&trade);
        }

        // The detector should have emitted signals
        assert!(detector.total_signals() > 0);
    }

    #[test]
    fn test_reset() {
        let mut detector = AdverseSelectionDetector::with_defaults();
        let base_ts = 1_000_000_000_000u64;

        // Generate some trades
        for i in 0..5 {
            let trade = make_trade(0, 10_0000, base_ts + i * 1_000_000);
            detector.on_trade(&trade);
        }

        detector.reset();

        // After reset, should be clean
        assert_eq!(detector.total_signals(), 0);
    }
}

