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
    /// Average Directional Index (14-period).
    pub adx14: f64,
    /// Task 18: Bollinger Bands upper (20, 2.0).
    pub bb_upper: f64,
    /// Task 18: Bollinger Bands middle (SMA20).
    pub bb_middle: f64,
    /// Task 18: Bollinger Bands lower (20, 2.0).
    pub bb_lower: f64,
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
            adx14: 25.0, // Neutral ADX
            bb_upper: 0.0,
            bb_middle: 0.0,
            bb_lower: 0.0,
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

// ═══════════════════════════════════════════════════════════════════════════
// ADX Calculator (Average Directional Index)
// ═══════════════════════════════════════════════════════════════════════════

/// Average Directional Index calculator with O(1) updates.
/// Measures trend strength (0-100). Higher values = stronger trend.
#[derive(Debug, Clone)]
struct AdxCalculator {
    /// ADX period.
    period: usize,
    /// Ring buffer of +DI values.
    plus_di: VecDeque<f64>,
    /// Ring buffer of -DI values.
    minus_di: VecDeque<f64>,
    /// Ring buffer of ADX values.
    adx_values: VecDeque<f64>,
    /// Previous high price.
    prev_high: f64,
    /// Previous low price.
    prev_low: f64,
    /// Previous close price.
    prev_close: f64,
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

impl AdxCalculator {
    fn new(period: usize) -> Self {
        Self {
            period,
            plus_di: VecDeque::with_capacity(period),
            minus_di: VecDeque::with_capacity(period),
            adx_values: VecDeque::with_capacity(period),
            prev_high: 0.0,
            prev_low: 0.0,
            prev_close: 0.0,
        }
    }

    /// Update ADX with new OHLC data.
    fn update(&mut self, high: f64, low: f64, close: f64) {
        if self.prev_high == 0.0 {
            // First candle - initialize
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return;
        }

        // Calculate True Range
        let tr1 = high - low;
        let tr2 = (high - self.prev_close).abs();
        let tr3 = (low - self.prev_close).abs();
        let tr = tr1.max(tr2).max(tr3);

        // Calculate directional movement
        let plus_dm = if high - self.prev_high > self.prev_low - low {
            (high - self.prev_high).max(0.0)
        } else {
            0.0
        };
        let minus_dm = if self.prev_low - low > high - self.prev_high {
            (self.prev_low - low).max(0.0)
        } else {
            0.0
        };

        // Calculate +DI and -DI using Wilder's smoothing
        let alpha = 1.0 / self.period as f64;
        let plus_di_val = if tr > 0.0 { plus_dm / tr } else { 0.0 };
        let minus_di_val = if tr > 0.0 { minus_dm / tr } else { 0.0 };

        self.plus_di.push_back(plus_di_val);
        self.minus_di.push_back(minus_di_val);

        if self.plus_di.len() > self.period {
            self.plus_di.pop_front();
            self.minus_di.pop_front();
        }

        // Calculate DX (Directional Index)
        if self.plus_di.len() >= self.period {
            let avg_plus_di: f64 = self.plus_di.iter().sum::<f64>() / self.period as f64;
            let avg_minus_di: f64 = self.minus_di.iter().sum::<f64>() / self.period as f64;
            let di_sum = avg_plus_di + avg_minus_di;
            let dx = if di_sum > 0.0 {
                100.0 * (avg_plus_di - avg_minus_di).abs() / di_sum
            } else {
                0.0
            };

            // Smooth DX to get ADX
            if self.adx_values.is_empty() {
                self.adx_values.push_back(dx);
            } else {
                let prev_adx = *self.adx_values.back().unwrap();
                let new_adx = alpha * dx + (1.0 - alpha) * prev_adx;
                self.adx_values.push_back(new_adx);
            }

            if self.adx_values.len() > self.period {
                self.adx_values.pop_front();
            }
        }

        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;
    }

    /// Get current ADX value.
    fn get(&self) -> f64 {
        self.adx_values.back().copied().unwrap_or(25.0)
    }

    /// Check if ADX is warmed up.
    fn is_ready(&self) -> bool {
        self.adx_values.len() >= self.period
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bollinger Bands Calculator (Task 17)
// ═══════════════════════════════════════════════════════════════════════════

/// Bollinger Bands calculator with O(1) updates.
/// Bands are calculated as: upper = SMA + (multiplier * std_dev), lower = SMA - (multiplier * std_dev)
#[derive(Debug, Clone)]
struct BollingerBands {
    /// Period for SMA and standard deviation.
    period: usize,
    /// Standard deviation multiplier (typically 2.0).
    multiplier: f64,
    /// Ring buffer of prices.
    prices: VecDeque<f64>,
    /// Running sum for SMA.
    sum: f64,
}

impl BollingerBands {
    fn new(period: usize, multiplier: f64) -> Self {
        Self {
            period,
            multiplier,
            prices: VecDeque::with_capacity(period),
            sum: 0.0,
        }
    }

    /// Update with a new price.
    fn update(&mut self, price: f64) {
        if self.prices.len() >= self.period {
            if let Some(old) = self.prices.pop_front() {
                self.sum -= old;
            }
        }
        
        self.prices.push_back(price);
        self.sum += price;
    }

    /// Get current Bollinger Bands (upper, middle, lower).
    fn get(&self) -> (f64, f64, f64) {
        if self.prices.len() < self.period {
            return (0.0, 0.0, 0.0);
        }

        let sma = self.sum / self.period as f64;
        
        // Calculate standard deviation
        let variance: f64 = self.prices.iter()
            .map(|&p| {
                let diff = p - sma;
                diff * diff
            })
            .sum::<f64>() / self.period as f64;
        
        let std_dev = variance.sqrt();
        
        let upper = sma + (self.multiplier * std_dev);
        let lower = sma - (self.multiplier * std_dev);
        
        (upper, sma, lower)
    }

    /// Check if Bollinger Bands are warmed up.
    fn is_ready(&self) -> bool {
        self.prices.len() >= self.period
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
    /// ADX(14) calculator.
    adx14: AdxCalculator,
    /// Task 17: Bollinger Bands (20, 2.0) calculator.
    bb20: BollingerBands,
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
            adx14: AdxCalculator::new(14),
            bb20: BollingerBands::new(20, 2.0),
        }
    }

    /// Process a trade event.
    ///
    /// CATEGORY 5 FIX: Handles exchange downtime gaps properly.
    /// When trades arrive after a gap (e.g., exchange was down for 30 minutes),
    /// the aggregator detects skipped candle periods and fills them with
    /// flat candles (OHLC = last known close, volume = 0) instead of
    /// creating artificial candles that span the gap.
    ///
    /// # Arguments
    /// * `timestamp_ns` — Trade timestamp in nanoseconds
    /// * `price` — Trade price
    /// * `volume` — Trade volume (in base asset units)
    fn on_trade(&mut self, timestamp_ns: u64, price: f64, volume: f64) {
        let candle_start = self.timeframe.candle_start(timestamp_ns);

        // Check if we need to close the current candle
        if self.current_start_ns > 0 && candle_start > self.current_start_ns {
            // CATEGORY 5 FIX: Detect and handle gaps from exchange downtime.
            // If more than one candle period was skipped, insert flat candles
            // to prevent artificial candle creation that spans the gap.
            let gap_periods = (candle_start - self.current_start_ns) / self.timeframe.duration_ns();
            if gap_periods > 1 {
                // Close the current candle first
                let last_close = self.current.close;
                self.close_candle();

                // Insert flat candles for skipped periods (up to 10 to avoid flooding)
                let fill_count = ((gap_periods - 1) as usize).min(10);
                for i in 0..fill_count {
                    let gap_start = self.current_start_ns
                        + (i as u64 + 1) * self.timeframe.duration_ns();
                    // Flat candle: OHLC = last known close, volume = 0
                    self.current_start_ns = gap_start;
                    self.current.timestamp_ns = gap_start + self.timeframe.duration_ns();
                    self.current.open = last_close;
                    self.current.high = last_close;
                    self.current.low = last_close;
                    self.current.close = last_close;
                    self.current.volume = 0.0;
                    self.close_candle();
                }

                if gap_periods > 11 {
                    tracing::warn!(
                        "[candle-agg] Large gap detected: {} periods skipped for {:?}",
                        gap_periods, self.timeframe
                    );
                }
            } else {
                self.close_candle();
            }
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
        self.adx14.update(self.current.high, self.current.low, self.current.close);
        self.bb20.update(self.current.close);

        // Store indicator values in the candle
        self.current.ema20 = self.ema20.get();
        self.current.ema50 = self.ema50.get();
        self.current.rsi14 = self.rsi14.get();
        self.current.adx14 = self.adx14.get();
        
        // Task 17: Store Bollinger Bands values
        let (bb_upper, bb_middle, bb_lower) = self.bb20.get();
        self.current.bb_upper = bb_upper;
        self.current.bb_middle = bb_middle;
        self.current.bb_lower = bb_lower;

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

    /// Get the most recent completed candle for a timeframe (Task 2).
    /// Returns None if no completed candles exist yet.
    pub fn get_latest_completed(&self, timeframe: Timeframe) -> Option<&CandleData> {
        match timeframe {
            Timeframe::M1 => self.m1.completed.back(),
            Timeframe::M5 => self.m5.completed.back(),
            Timeframe::M15 => self.m15.completed.back(),
            Timeframe::H1 => self.h1.completed.back(),
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
