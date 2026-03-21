use std::collections::{HashMap, BTreeMap, VecDeque};
// use ordered_float::OrderedFloat; // We'll just use bits for now to avoid dep

#[derive(Debug, Clone)]
pub struct SimulatedOrder {
    pub order_id: u64,
    pub price: f64,
    pub qty: f64,
    pub is_bid: bool,
}

pub struct SimulatedOrderbook {
    pub bids: BTreeMap<u64, VecDeque<SimulatedOrder>>, // Key is f64::to_bits() negated for reverse order
    pub asks: BTreeMap<u64, VecDeque<SimulatedOrder>>, // Key is f64::to_bits()
    pub last_trade_price: f64,
    pub sequence: u64,
}

pub struct LatencyModel {
    pub base_latency_us: u64,        // Network RTT
    pub jitter_std_us: u64,          // Latency jitter
    pub queue_delay_us: u64,         // Exchange matching queue
    pub burst_penalty_us: u64,       // Extra latency during high volume
}

pub struct FillModel {
    pub maker_fill_probability: f64,  // Probability of maker order getting filled
    pub partial_fill_ratio: f64,      // Average partial fill ratio
    pub adverse_fill_probability: f64, // Probability of fill only on adverse move
}

pub struct SimulatedEvent {
    pub timestamp_ns: u64,
    pub event_type: String,
}

pub struct MatchingEngine {
    pub orderbooks: HashMap<String, SimulatedOrderbook>,
    pub latency_model: LatencyModel,
    pub fill_model: FillModel,
    pub event_log: Vec<SimulatedEvent>,
}
