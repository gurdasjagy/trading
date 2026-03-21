//! Latency Tracker — Production-Grade Latency Measurement & Reporting.
//!
//! Tracks latency at each stage of the trading pipeline:
//!   - tick_to_book: WebSocket message → orderbook update
//!   - book_to_signal: Orderbook snapshot → strategy signal
//!   - signal_to_order: Signal → order submitted to exchange
//!   - order_to_ack: Order submitted → exchange acknowledgement
//!   - end_to_end: WebSocket message → order acknowledged
//!
//! Uses a lock-free histogram implementation based on HDR Histogram
//! principles for O(1) recording with accurate percentile computation.
//!
//! # Memory Usage
//! Each `LatencyHistogram` uses ~4KB (1024 buckets × 4 bytes).
//! Total tracker: ~24KB for 6 histograms.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::info;

/// Number of buckets in each histogram (covers 0-1023 microseconds, then overflow).
const HISTOGRAM_BUCKETS: usize = 1024;
/// Microsecond range per bucket.
const BUCKET_WIDTH_US: u64 = 10;
/// Maximum latency tracked (10.24ms = 1024 × 10µs). Anything above goes to overflow.
const MAX_TRACKED_US: u64 = (HISTOGRAM_BUCKETS as u64) * BUCKET_WIDTH_US;

/// Lock-free histogram for latency tracking.
/// Uses atomic counters per bucket for thread-safe recording.
pub struct LatencyHistogram {
    /// Atomic counters for each bucket.
    buckets: Vec<AtomicU64>,
    /// Total samples recorded.
    total_count: AtomicU64,
    /// Sum of all latency values (for mean calculation).
    total_sum_us: AtomicU64,
    /// Overflow bucket (latencies > MAX_TRACKED_US).
    overflow: AtomicU64,
    /// Maximum observed latency.
    max_us: AtomicU64,
    /// Minimum observed latency (initialized to u64::MAX).
    min_us: AtomicU64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        let mut buckets = Vec::with_capacity(HISTOGRAM_BUCKETS);
        for _ in 0..HISTOGRAM_BUCKETS {
            buckets.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            total_count: AtomicU64::new(0),
            total_sum_us: AtomicU64::new(0),
            overflow: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
            min_us: AtomicU64::new(u64::MAX),
        }
    }

    /// Record a latency value in microseconds. O(1), lock-free.
    #[inline]
    pub fn record(&self, latency_us: u64) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum_us.fetch_add(latency_us, Ordering::Relaxed);

        // Update min/max (racy but close enough for monitoring)
        let current_max = self.max_us.load(Ordering::Relaxed);
        if latency_us > current_max {
            self.max_us.store(latency_us, Ordering::Relaxed);
        }
        let current_min = self.min_us.load(Ordering::Relaxed);
        if latency_us < current_min {
            self.min_us.store(latency_us, Ordering::Relaxed);
        }

        let bucket_idx = (latency_us / BUCKET_WIDTH_US) as usize;
        if bucket_idx < HISTOGRAM_BUCKETS {
            self.buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
        } else {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record using an Instant start time.
    #[inline]
    pub fn record_since(&self, start: Instant) {
        let elapsed = start.elapsed();
        self.record(elapsed.as_micros() as u64);
    }

    /// Get the mean latency in microseconds.
    pub fn mean_us(&self) -> f64 {
        let count = self.total_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.total_sum_us.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Get a percentile value (e.g., 0.50, 0.95, 0.99).
    pub fn percentile_us(&self, p: f64) -> u64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        let target = ((total as f64) * p).ceil() as u64;
        let mut cumulative: u64 = 0;

        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                return (i as u64) * BUCKET_WIDTH_US + BUCKET_WIDTH_US / 2;
            }
        }

        // All remaining are in overflow
        MAX_TRACKED_US
    }

    /// Get a snapshot of all key statistics.
    pub fn snapshot(&self) -> LatencySnapshot {
        let count = self.total_count.load(Ordering::Relaxed);
        let min = self.min_us.load(Ordering::Relaxed);
        LatencySnapshot {
            count,
            mean_us: self.mean_us(),
            min_us: if min == u64::MAX { 0 } else { min },
            max_us: self.max_us.load(Ordering::Relaxed),
            p50_us: self.percentile_us(0.50),
            p95_us: self.percentile_us(0.95),
            p99_us: self.percentile_us(0.99),
            p999_us: self.percentile_us(0.999),
            overflow_count: self.overflow.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters (typically called at start of each day).
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.total_count.store(0, Ordering::Relaxed);
        self.total_sum_us.store(0, Ordering::Relaxed);
        self.overflow.store(0, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
        self.min_us.store(u64::MAX, Ordering::Relaxed);
    }
}

/// Snapshot of latency statistics.
#[derive(Debug, Clone)]
pub struct LatencySnapshot {
    pub count: u64,
    pub mean_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub p999_us: u64,
    pub overflow_count: u64,
}

impl std::fmt::Display for LatencySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} mean={:.0}µs min={}µs p50={}µs p95={}µs p99={}µs p999={}µs max={}µs",
            self.count, self.mean_us, self.min_us,
            self.p50_us, self.p95_us, self.p99_us, self.p999_us, self.max_us
        )
    }
}

/// Complete latency tracker for all pipeline stages.
pub struct PipelineLatencyTracker {
    pub tick_to_book: LatencyHistogram,
    pub book_to_signal: LatencyHistogram,
    pub signal_to_order: LatencyHistogram,
    pub order_to_ack: LatencyHistogram,
    pub end_to_end: LatencyHistogram,
    pub ws_parse_time: LatencyHistogram,
}

impl PipelineLatencyTracker {
    pub fn new() -> Self {
        Self {
            tick_to_book: LatencyHistogram::new(),
            book_to_signal: LatencyHistogram::new(),
            signal_to_order: LatencyHistogram::new(),
            order_to_ack: LatencyHistogram::new(),
            end_to_end: LatencyHistogram::new(),
            ws_parse_time: LatencyHistogram::new(),
        }
    }

    /// Log a full pipeline latency report.
    pub fn log_report(&self) {
        info!("[latency] tick_to_book:   {}", self.tick_to_book.snapshot());
        info!("[latency] book_to_signal: {}", self.book_to_signal.snapshot());
        info!("[latency] signal_to_order:{}", self.signal_to_order.snapshot());
        info!("[latency] order_to_ack:   {}", self.order_to_ack.snapshot());
        info!("[latency] end_to_end:     {}", self.end_to_end.snapshot());
        info!("[latency] ws_parse:       {}", self.ws_parse_time.snapshot());
    }

    /// Get all snapshots as a serializable struct.
    pub fn get_all_snapshots(&self) -> PipelineLatencyReport {
        PipelineLatencyReport {
            tick_to_book: self.tick_to_book.snapshot(),
            book_to_signal: self.book_to_signal.snapshot(),
            signal_to_order: self.signal_to_order.snapshot(),
            order_to_ack: self.order_to_ack.snapshot(),
            end_to_end: self.end_to_end.snapshot(),
            ws_parse_time: self.ws_parse_time.snapshot(),
        }
    }

    /// Reset all histograms (e.g., daily reset).
    pub fn reset_all(&self) {
        self.tick_to_book.reset();
        self.book_to_signal.reset();
        self.signal_to_order.reset();
        self.order_to_ack.reset();
        self.end_to_end.reset();
        self.ws_parse_time.reset();
    }
}

/// Complete latency report for all pipeline stages.
#[derive(Debug)]
pub struct PipelineLatencyReport {
    pub tick_to_book: LatencySnapshot,
    pub book_to_signal: LatencySnapshot,
    pub signal_to_order: LatencySnapshot,
    pub order_to_ack: LatencySnapshot,
    pub end_to_end: LatencySnapshot,
    pub ws_parse_time: LatencySnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let hist = LatencyHistogram::new();
        for i in 0..1000 {
            hist.record(i * 5);  // 0, 5, 10, ..., 4995µs
        }
        assert_eq!(hist.snapshot().count, 1000);
        assert!(hist.snapshot().p50_us > 0);
        assert!(hist.snapshot().p99_us > hist.snapshot().p50_us);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = LatencyHistogram::new();
        let snap = hist.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.mean_us, 0.0);
    }

    #[test]
    fn test_pipeline_tracker() {
        let tracker = PipelineLatencyTracker::new();
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        tracker.tick_to_book.record_since(start);
        assert!(tracker.tick_to_book.snapshot().count == 1);
    }
}
