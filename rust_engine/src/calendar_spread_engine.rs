use std::collections::{HashMap, VecDeque};
use crate::execution_gateway::OrderSide;

pub struct SpreadPosition {
    pub coin: String,
    pub perp_side: OrderSide,      // Usually LONG perp
    pub quarterly_side: OrderSide,  // Usually SHORT quarterly
    pub perp_size: i64,
    pub quarterly_size: i64,
    pub entry_basis_bps: f64,
    pub entry_timestamp_ns: u64,
    pub current_basis_bps: f64,
    pub unrealized_pnl: f64,
}

pub struct CalendarSpreadEngine {
    pub active_spreads: HashMap<String, SpreadPosition>,
    pub basis_history: HashMap<String, VecDeque<(u64, f64)>>, // (timestamp_ns, basis)
    pub entry_threshold_bps: f64,  // Enter when basis > X bps annualized
    pub exit_threshold_bps: f64,   // Exit when basis < Y bps
    pub max_spread_positions: usize,
}

impl CalendarSpreadEngine {
    pub fn new(entry_threshold_bps: f64, exit_threshold_bps: f64, max_spread_positions: usize) -> Self {
        Self {
            active_spreads: HashMap::new(),
            basis_history: HashMap::new(),
            entry_threshold_bps,
            exit_threshold_bps,
            max_spread_positions,
        }
    }

    pub fn check_opportunities(&mut self) {
        // Evaluate basis history to find entry/exit
    }
}
