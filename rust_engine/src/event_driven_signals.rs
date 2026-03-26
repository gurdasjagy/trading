//! CATEGORY 6 FIX: Event-Driven Signal Generation.
//!
//! Generates immediate counter-trade signals when significant market events
//! are detected, such as:
//!   - Large liquidation cascades (counter-trade the forced selling)
//!   - Sudden order book imbalance shifts (>50% in <1 second)
//!   - Funding rate spikes (> 0.1% per period)
//!   - Exchange outage recovery (liquidity vacuum fill)
//!
//! These are distinct from the continuous microstructure signals because
//! they are triggered by discrete events, not ongoing price/depth analysis.

use std::collections::VecDeque;
use tracing::{info, warn, debug};

/// Types of events that can trigger signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// Large liquidation detected (forced selling/buying creates opportunity).
    LiquidationCascade,
    /// Sudden order book imbalance shift (>50% change in <1s).
    ImbalanceShift,
    /// Funding rate spike (>0.1% per period).
    FundingRateSpike,
    /// Large whale order detected on-chain or in order book.
    WhaleOrder,
    /// Exchange recovery after outage (liquidity vacuum).
    ExchangeRecovery,
}

impl EventType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LiquidationCascade => "liquidation_cascade",
            Self::ImbalanceShift => "imbalance_shift",
            Self::FundingRateSpike => "funding_rate_spike",
            Self::WhaleOrder => "whale_order",
            Self::ExchangeRecovery => "exchange_recovery",
        }
    }
}

/// An event-driven trading signal.
#[derive(Debug, Clone)]
pub struct EventSignal {
    /// Type of event that triggered the signal.
    pub event_type: EventType,
    /// Symbol the signal targets.
    pub symbol: String,
    /// Direction: true = long, false = short.
    pub is_long: bool,
    /// Signal urgency (0.0 = low, 1.0 = immediate).
    pub urgency: f64,
    /// Suggested position size multiplier (relative to base size).
    pub size_multiplier: f64,
    /// Maximum holding time in seconds (events are time-sensitive).
    pub max_hold_secs: u64,
    /// Timestamp when the event was detected (nanoseconds).
    pub detected_ns: u64,
    /// Additional context for logging.
    pub context: String,
}

/// Configuration for event-driven signal generation.
#[derive(Debug, Clone)]
pub struct EventSignalConfig {
    /// Minimum liquidation size in USDT to trigger a cascade signal.
    pub min_liquidation_usdt: f64,
    /// Minimum imbalance change (0-1) to trigger an imbalance shift signal.
    pub min_imbalance_change: f64,
    /// Minimum funding rate (absolute) to trigger a funding spike signal.
    pub min_funding_rate: f64,
    /// Cooldown between signals of the same type per symbol (nanoseconds).
    pub cooldown_ns: u64,
    /// Whether event signals are enabled.
    pub enabled: bool,
}

impl Default for EventSignalConfig {
    fn default() -> Self {
        Self {
            min_liquidation_usdt: 100_000.0, // $100k liquidation
            min_imbalance_change: 0.50,      // 50% imbalance shift
            min_funding_rate: 0.001,         // 0.1% funding rate
            cooldown_ns: 30_000_000_000,     // 30 second cooldown
            enabled: true,
        }
    }
}

/// Event-driven signal generator.
pub struct EventDrivenSignals {
    config: EventSignalConfig,
    /// Recent events for deduplication/cooldown.
    recent_events: VecDeque<(EventType, String, u64)>, // (type, symbol, timestamp)
    /// Previous imbalance values per symbol for shift detection.
    prev_imbalances: Vec<(String, f64, u64)>, // (symbol, imbalance, timestamp_ns)
    /// Signal counter for telemetry.
    total_signals_generated: u64,
}

impl EventDrivenSignals {
    /// Create a new event-driven signal generator.
    pub fn new(config: EventSignalConfig) -> Self {
        Self {
            config,
            recent_events: VecDeque::with_capacity(100),
            prev_imbalances: Vec::with_capacity(64),
            total_signals_generated: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EventSignalConfig::default())
    }

    /// Check cooldown: has a signal of this type been generated recently?
    fn is_on_cooldown(&self, event_type: EventType, symbol: &str, now_ns: u64) -> bool {
        self.recent_events.iter().any(|(t, s, ts)| {
            *t == event_type && s == symbol && now_ns - ts < self.config.cooldown_ns
        })
    }

    /// Record a signal generation for cooldown tracking.
    fn record_signal(&mut self, event_type: EventType, symbol: &str, now_ns: u64) {
        self.recent_events.push_back((event_type, symbol.to_string(), now_ns));
        if self.recent_events.len() > 100 {
            self.recent_events.pop_front();
        }
        self.total_signals_generated += 1;
    }

    /// Check for a liquidation cascade event and generate counter-trade signal.
    ///
    /// When a large liquidation occurs, the forced market order creates temporary
    /// price dislocation. Counter-trading (buying into forced selling) captures
    /// the mean-reversion after the cascade.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `liquidation_usdt` - Size of the liquidation in USDT
    /// * `is_long_liquidation` - True if longs were liquidated (price dropped)
    /// * `now_ns` - Current timestamp in nanoseconds
    pub fn on_liquidation(
        &mut self,
        symbol: &str,
        liquidation_usdt: f64,
        is_long_liquidation: bool,
        now_ns: u64,
    ) -> Option<EventSignal> {
        if !self.config.enabled {
            return None;
        }
        if liquidation_usdt < self.config.min_liquidation_usdt {
            return None;
        }
        if self.is_on_cooldown(EventType::LiquidationCascade, symbol, now_ns) {
            return None;
        }

        // Counter-trade: if longs were liquidated (price crashed),
        // buy the dip. If shorts were liquidated (price spiked), short the spike.
        let is_long = is_long_liquidation; // Buy when longs are liquidated (price is low)

        // Size scales with liquidation magnitude
        let size_mult = (liquidation_usdt / self.config.min_liquidation_usdt)
            .sqrt()
            .clamp(0.5, 3.0);

        self.record_signal(EventType::LiquidationCascade, symbol, now_ns);

        let signal = EventSignal {
            event_type: EventType::LiquidationCascade,
            symbol: symbol.to_string(),
            is_long,
            urgency: 0.9, // High urgency - liquidation cascades are time-sensitive
            size_multiplier: size_mult,
            max_hold_secs: 300, // 5 minute max hold for liquidation bounce
            detected_ns: now_ns,
            context: format!(
                "Liquidation: ${:.0} {} — counter-trading with {:.1}x size",
                liquidation_usdt,
                if is_long_liquidation { "longs" } else { "shorts" },
                size_mult
            ),
        };

        info!(
            "[event-signal] {} {} on {} — ${:.0} liquidation counter-trade",
            if is_long { "LONG" } else { "SHORT" },
            symbol,
            liquidation_usdt,
        );

        Some(signal)
    }

    /// Check for sudden imbalance shift and generate signal.
    ///
    /// A rapid shift in order book imbalance (>50% change in <1s) indicates
    /// a sudden change in order flow that precedes a price move.
    pub fn on_imbalance_update(
        &mut self,
        symbol: &str,
        imbalance: f64,
        now_ns: u64,
    ) -> Option<EventSignal> {
        if !self.config.enabled {
            return None;
        }

        // Find previous imbalance for this symbol
        let prev = self.prev_imbalances.iter()
            .find(|(s, _, _)| s == symbol)
            .map(|(_, imb, ts)| (*imb, *ts));

        // Update stored imbalance
        if let Some(idx) = self.prev_imbalances.iter().position(|(s, _, _)| s == symbol) {
            self.prev_imbalances[idx] = (symbol.to_string(), imbalance, now_ns);
        } else {
            self.prev_imbalances.push((symbol.to_string(), imbalance, now_ns));
        }

        if let Some((prev_imb, prev_ts)) = prev {
            let time_diff_ns = now_ns.saturating_sub(prev_ts);
            let imb_change = (imbalance - prev_imb).abs();

            // Only trigger if change happened within 1 second
            if time_diff_ns < 1_000_000_000 && imb_change > self.config.min_imbalance_change {
                if self.is_on_cooldown(EventType::ImbalanceShift, symbol, now_ns) {
                    return None;
                }

                let is_long = imbalance > prev_imb; // Shift toward bids = bullish
                self.record_signal(EventType::ImbalanceShift, symbol, now_ns);

                return Some(EventSignal {
                    event_type: EventType::ImbalanceShift,
                    symbol: symbol.to_string(),
                    is_long,
                    urgency: 0.8,
                    size_multiplier: 1.0,
                    max_hold_secs: 120, // 2 minute max
                    detected_ns: now_ns,
                    context: format!(
                        "Imbalance shift: {:.3} -> {:.3} ({:.1}% change in {:.0}ms)",
                        prev_imb, imbalance, imb_change * 100.0,
                        time_diff_ns as f64 / 1_000_000.0
                    ),
                });
            }
        }

        None
    }

    /// Check for funding rate spike.
    pub fn on_funding_rate(
        &mut self,
        symbol: &str,
        funding_rate: f64,
        now_ns: u64,
    ) -> Option<EventSignal> {
        if !self.config.enabled {
            return None;
        }
        if funding_rate.abs() < self.config.min_funding_rate {
            return None;
        }
        if self.is_on_cooldown(EventType::FundingRateSpike, symbol, now_ns) {
            return None;
        }

        // Short when funding is very positive (shorts collect), long when very negative
        let is_long = funding_rate < 0.0;
        let size_mult = (funding_rate.abs() / self.config.min_funding_rate)
            .sqrt()
            .clamp(0.5, 2.0);

        self.record_signal(EventType::FundingRateSpike, symbol, now_ns);

        Some(EventSignal {
            event_type: EventType::FundingRateSpike,
            symbol: symbol.to_string(),
            is_long,
            urgency: 0.6, // Lower urgency - funding is scheduled
            size_multiplier: size_mult,
            max_hold_secs: 28800, // 8 hours (through funding period)
            detected_ns: now_ns,
            context: format!(
                "Funding spike: {:.4}% — {} to collect funding",
                funding_rate * 100.0,
                if is_long { "LONG" } else { "SHORT" }
            ),
        })
    }

    /// Get total signals generated.
    pub fn total_signals(&self) -> u64 {
        self.total_signals_generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidation_signal() {
        let mut eds = EventDrivenSignals::with_defaults();
        let signal = eds.on_liquidation("BTC_USDT", 500_000.0, true, 1_000_000_000);
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert!(s.is_long); // Counter-trade: buy when longs are liquidated
        assert_eq!(s.event_type, EventType::LiquidationCascade);
    }

    #[test]
    fn test_liquidation_below_threshold() {
        let mut eds = EventDrivenSignals::with_defaults();
        let signal = eds.on_liquidation("BTC_USDT", 1_000.0, true, 1_000_000_000);
        assert!(signal.is_none()); // Below $100k threshold
    }

    #[test]
    fn test_cooldown() {
        let mut eds = EventDrivenSignals::with_defaults();
        let s1 = eds.on_liquidation("BTC_USDT", 500_000.0, true, 1_000_000_000);
        assert!(s1.is_some());
        // Same signal within cooldown should be rejected
        let s2 = eds.on_liquidation("BTC_USDT", 500_000.0, true, 2_000_000_000);
        assert!(s2.is_none());
    }

    #[test]
    fn test_funding_rate_signal() {
        let mut eds = EventDrivenSignals::with_defaults();
        let signal = eds.on_funding_rate("BTC_USDT", 0.005, 1_000_000_000);
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert!(!s.is_long); // Short when funding is positive
    }

    #[test]
    fn test_imbalance_shift() {
        let mut eds = EventDrivenSignals::with_defaults();
        // First update establishes baseline
        let s1 = eds.on_imbalance_update("BTC_USDT", -0.3, 1_000_000_000);
        assert!(s1.is_none()); // First update, no shift
        // Rapid shift within 1 second
        let s2 = eds.on_imbalance_update("BTC_USDT", 0.4, 1_500_000_000);
        assert!(s2.is_some()); // 0.7 shift in 500ms
    }
}
