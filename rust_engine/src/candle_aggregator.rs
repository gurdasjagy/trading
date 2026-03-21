//! Multi-Timeframe Candle Aggregator with Technical Indicators.
//!
//! Builds 1m/5m/15m/1h candles from WebSocket trade stream and computes
//! EMA(20), EMA(50), and RSI(14) using ring buffers for O(1) updates.
//!
//! # Architecture
//!
//! ```text
//! Trade Stream → CandleAggregator → CandleData{1m, 5m, 15m, 1h}
//!                                    ↓
//!                                  EMA(20), EMA(50), RSI(14)
//! ```
//!
//! Follows the pattern from microstructure.rs:EnhancedVpin (lines 200-250)
//! for bucket-based aggregation with ring buffers.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════════════════
// Candle Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A single OHLCV candle with computed technical indicators.
#[derive(Debug, Clone, Copy)]
pub struct CandleData {
    /// Unix timestamp in nanoseconds (candle close time).
    pub timestamp_ns: u64,
    /// Open price.
    pub open: f64,
    /// High price.
    pub high: f64,
    /// Low price.
    pub low: f64,
    /// Close price.
    pub close: f64,
    /// Volume (in base asset units).
    pub volume: f64,
    /// Exponential Moving Average (20-period).
    pub ema20: f64,
    /// Exponential Moving Average (50-period).
    pub ema50: f64,
    /// Relative Strength Index (14-period).
    pub rsi14: f64,
}

impl Default for CandleData {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            open: 0.0,
            high: 0.0,
            low: 0.0,
            close: 0.0,
            volume: 0.0,
            ema20: 0.0,
            ema50: 0.0,
            rsi14: 50.0, // Neutral RSI
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Timeframe Enum
// ═══════════════════════════════════════════════════════════════════════════

/// Supported candle timeframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    /// 1-minute candles.
    M1,
    /// 5-minute candles.
    M5,
    /// 15-minute candles.
    M15,
    /// 1-hour candles.
    H1,
}

impl Timeframe {
    /// Duration of this timeframe in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        match self {
            Timeframe::M1 => 60_000_000_000,           // 1 minute
            Timeframe::M5 => 300_000_000_000,          // 5 minutes
            Timeframe::M15 => 900_000_000_000,         // 15 minutes
            Timeframe::H1 => 3_600_000_000_000,        // 1 hour
        }
    }

    /// Get the candle start timestamp for a given time.
    pub fn candle_start(&self, timestamp_ns: u64) -> u64 {
        let duration = self.duration_ns();
        (timestamp_ns / duration) * duration
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EMA Calculator (Ring Buffer)
// ═══════════════════════════════════════════════════════════════════════════

/// Exponential Moving Average calculator with O(1) updates.
#[derive(Debug, Clone)]
struct EmaCalculator {
    /// EMA period.
    period: usize,
    /// Current EMA value.
    value: f64,
    /// Smoothing factor (alpha = 2 / (period + 1)).
    alpha: f64,
    /// Number of samples received.
    count: usize,
}

impl EmaCalculator {
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            value: 0.0,
            alpha,
            count: 0,
        }
    }

    /// Update EMA with a new price.
    fn update(&mut self, price: f64) {
        if self.count == 0 {
            self.value = price;
        } else {
            self.value = self.alpha * price + (1.0 - self.alpha) * self.value;
        }
        self.count += 1;
    }

    /// Get current EMA value.
    fn get(&self) -> f64 {
        self.value
    }

    /// Check if EMA is warmed up (has enough samples).
    fn is_ready(&self) -> bool {
        self.count >= self.period
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RSI Calculator (Ring Buffer)
// ═══════════════════════════════════════════════════════════════════════════

/// Relative Strength Index calculator with O(1) updates.
#[derive(Debug, Clone)]
struct RsiCalculator {
    /// RSI period.
    period: usize,
    /// Ring buffer of price changes.
    changes: VecDeque<f64>,
    /// Previous close price.
    prev_close: f64,
    /// Average gain (smoothed).
    avg_gain: f64,
    /// Average loss (smoothed).
    avg_loss: f64,
    /// Number of samples received.
    count: usize,
}

impl RsiCalculator {
    fn new(period: usize) -> Self {
        Self {
            period,
            changes: VecDeque::with_capacity(period),
            prev_close: 0.0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            count: 0,
        }
    }

    /// Update RSI with a new close price.
    fn update(&mut self, close: f64) {
        if self.prev_close > 0.0 {
            let change = close - self.prev_close;
            self.changes.push_back(change);

            if self.changes.len() > self.period {
                self.changes.pop_front();
            }

            // Calculate average gain and loss
            if self.changes.len() == self.period {
                let mut gain_sum = 0.0;
                let mut loss_sum = 0.0;

                for &ch in &self.changes {
                    if ch > 0.0 {
                        gain_sum += ch;
                    } else {
                        loss_sum += ch.abs();
                    }
                }

                // Wilder's smoothing
                if self.count < self.period {
                    self.avg_gain = gain_sum / self.period as f64;
                    self.avg_loss = loss_sum / self.period as f64;
                } else {
                    let gain = if change > 0.0 { change } else { 0.0 };
                    let loss = if change < 0.0 { change.abs() } else { 0.0 };
                    self.avg_gain = (self.avg_gain * (self.period as f64 - 1.0) + gain) / self.period as f64;
                    self.avg_loss = (self.avg_loss * (self.period as f64 - 1.0) + loss) / self.period as f64;
                }
            }

            self.count += 1;
        }
        self.prev_close = close;
    }

    /// Get current RSI value (0-100).
    fn get(&self) -> f64 {
        if self.avg_loss == 0.0 {
            return 100.0;
        }
        if self.avg_gain == 0.0 {
            return 0.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Check if RSI is warmed up.
    fn is_ready(&self) -> bool {
        self.count >= self.period
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Candle Builder (Per-Timeframe)
// ═══════════════════════════════════════════════════════════════════════════

/// Builds candles for a single timeframe from trade events.
#[derive(Debug, Clone)]
struct CandleBuilder {
    /// Timeframe for this builder.
    timeframe: Timeframe,
    /// Current candle being built.
    current: CandleData,
    /// Timestamp of the current candle start.
    current_start_ns: u64,
    /// Completed candles (ring buffer).
    completed: VecDeque<CandleData>,
    /// Maximum number of completed candles to retain.
    max_history: usize,
    /// EMA(20) calculator.
    ema20: EmaCalculator,
    /// EMA(50) calculator.
    ema50: EmaCalculator,
    /// RSI(14) calculator.
    rsi14: RsiCalculator,
}

impl CandleBuilder {
    fn new(timeframe: Timeframe, max_history: usize) -> Self {
        Self {
            timeframe,
            current: CandleData::default(),
            current_start_ns: 0,
            completed: VecDeque::with_capacity(max_history),
            max_history,
            ema20: EmaCalculator::new(20),
            ema50: EmaCalculator::new(50),
            rsi14: RsiCalculator::new(14),
        }
    }

    /// Process a trade event.
    ///
    /// # Arguments
    /// * `timestamp_ns` — Trade timestamp in nanoseconds
    /// * `price` — Trade price
    /// * `volume` — Trade volume (in base asset units)
    fn on_trade(&mut self, timestamp_ns: u64, price: f64, volume: f64) {
        let candle_start = self.timeframe.candle_start(timestamp_ns);

        // Check if we need to close the current candle
        if self.current_start_ns > 0 && candle_start > self.current_start_ns {
            self.close_candle();
        }

        // Initialize new candle if needed
        if self.current_start_ns == 0 || candle_start > self.current_start_ns {
            self.current_start_ns = candle_start;
            self.current.timestamp_ns = candle_start + self.timeframe.duration_ns();
            self.current.open = price;
            self.current.high = price;
            self.current.low = price;
            self.current.close = price;
            self.current.volume = 0.0;
        }

        // Update current candle
        self.current.high = self.current.high.max(price);
        self.current.low = self.current.low.min(price);
        self.current.close = price;
        self.current.volume += volume;
    }

    /// Close the current candle and compute indicators.
    fn close_candle(&mut self) {
        if self.current_start_ns == 0 {
            return;
        }

        // Update indicators
        self.ema20.update(self.current.close);
        self.ema50.update(self.current.close);
        self.rsi14.update(self.current.close);

        // Store indicator values in the candle
        self.current.ema20 = self.ema20.get();
        self.current.ema50 = self.ema50.get();
        self.current.rsi14 = self.rsi14.get();

        // Add to completed candles
        self.completed.push_back(self.current);
        if self.completed.len() > self.max_history {
            self.completed.pop_front();
        }

        // Reset current candle
        self.current = CandleData::default();
        self.current_start_ns = 0;
    }

    /// Get the most recent completed candle.
    fn last_candle(&self) -> Option<&CandleData> {
        self.completed.back()
    }

    /// Get the current (in-progress) candle.
    fn current_candle(&self) -> &CandleData {
        &self.current
    }

    /// Check if indicators are warmed up.
    fn is_ready(&self) -> bool {
        self.ema20.is_ready() && self.ema50.is_ready() && self.rsi14.is_ready()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-Timeframe Candle Aggregator
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregates candles across multiple timeframes from a trade stream.
pub struct CandleAggregator {
    /// 1-minute candle builder.
    m1: CandleBuilder,
    /// 5-minute candle builder.
    m5: CandleBuilder,
    /// 15-minute candle builder.
    m15: CandleBuilder,
    /// 1-hour candle builder.
    h1: CandleBuilder,
    /// Total trades processed.
    pub trade_count: u64,
}

impl CandleAggregator {
    /// Create a new candle aggregator.
    ///
    /// # Arguments
    /// * `max_history` — Maximum number of completed candles to retain per timeframe
    pub fn new(max_history: usize) -> Self {
        Self {
            m1: CandleBuilder::new(Timeframe::M1, max_history),
            m5: CandleBuilder::new(Timeframe::M5, max_history),
            m15: CandleBuilder::new(Timeframe::M15, max_history),
            h1: CandleBuilder::new(Timeframe::H1, max_history),
            trade_count: 0,
        }
    }

    /// Process a trade event from the WebSocket stream.
    ///
    /// # Arguments
    /// * `timestamp_ns` — Trade timestamp in nanoseconds
    /// * `price` — Trade price
    /// * `volume` — Trade volume (in base asset units)
    pub fn on_trade(&mut self, timestamp_ns: u64, price: f64, volume: f64) {
        self.m1.on_trade(timestamp_ns, price, volume);
        self.m5.on_trade(timestamp_ns, price, volume);
        self.m15.on_trade(timestamp_ns, price, volume);
        self.h1.on_trade(timestamp_ns, price, volume);
        self.trade_count += 1;
    }

    /// Get the most recent completed candle for a timeframe.
    pub fn get_candle(&self, timeframe: Timeframe) -> Option<&CandleData> {
        match timeframe {
            Timeframe::M1 => self.m1.last_candle(),
            Timeframe::M5 => self.m5.last_candle(),
            Timeframe::M15 => self.m15.last_candle(),
            Timeframe::H1 => self.h1.last_candle(),
        }
    }

    /// Get the current (in-progress) candle for a timeframe.
    pub fn get_current_candle(&self, timeframe: Timeframe) -> &CandleData {
        match timeframe {
            Timeframe::M1 => self.m1.current_candle(),
            Timeframe::M5 => self.m5.current_candle(),
            Timeframe::M15 => self.m15.current_candle(),
            Timeframe::H1 => self.h1.current_candle(),
        }
    }

    /// Check if a timeframe's indicators are warmed up.
    pub fn is_ready(&self, timeframe: Timeframe) -> bool {
        match timeframe {
            Timeframe::M1 => self.m1.is_ready(),
            Timeframe::M5 => self.m5.is_ready(),
            Timeframe::M15 => self.m15.is_ready(),
            Timeframe::H1 => self.h1.is_ready(),
        }
    }

    /// Force close all current candles (useful for testing or end-of-session).
    pub fn close_all_candles(&mut self) {
        self.m1.close_candle();
        self.m5.close_candle();
        self.m15.close_candle();
        self.h1.close_candle();
    }
}

impl Default for CandleAggregator {
    fn default() -> Self {
        Self::new(200) // Default: retain 200 candles per timeframe
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Get current time in nanoseconds.
#[inline]
pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_aggregator_basic() {
        let mut agg = CandleAggregator::new(100);
        let base_time = 1_700_000_000_000_000_000u64; // Some timestamp

        // Feed trades for 2 minutes
        for i in 0..120 {
            let ts = base_time + (i * 1_000_000_000); // 1 second apart
            let price = 50000.0 + (i as f64 * 10.0);
            agg.on_trade(ts, price, 0.1);
        }

        assert_eq!(agg.trade_count, 120);

        // Should have at least one completed 1m candle
        agg.close_all_candles();
        assert!(agg.get_candle(Timeframe::M1).is_some());
    }

    #[test]
    fn test_ema_calculator() {
        let mut ema = EmaCalculator::new(5);
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];

        for &price in &prices {
            ema.update(price);
        }

        assert!(ema.is_ready());
        let value = ema.get();
        assert!(value > 100.0 && value < 106.0);
    }

    #[test]
    fn test_rsi_calculator() {
        let mut rsi = RsiCalculator::new(14);
        
        // Simulate uptrend
        for i in 0..20 {
            rsi.update(100.0 + (i as f64));
        }

        assert!(rsi.is_ready());
        let value = rsi.get();
        assert!(value > 50.0); // Should be bullish
    }

    #[test]
    fn test_timeframe_candle_start() {
        let tf = Timeframe::M1;
        let ts = 1_700_000_000_123_456_789u64;
        let start = tf.candle_start(ts);
        
        // Should be aligned to 1-minute boundary
        assert_eq!(start % tf.duration_ns(), 0);
    }
}
