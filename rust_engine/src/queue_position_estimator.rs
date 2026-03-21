use crate::fixed_point::FixedPrice;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct QueueTracker {
    pub symbol_id: u16,
    pub our_price: FixedPrice,
    pub our_side: u8, // 0 = BID, 1 = ASK
    pub our_size: i64,
    pub initial_depth_ahead: f64,
    pub cumulative_fills_at_level: f64,
    pub cumulative_cancels_at_level: f64,
    pub submit_timestamp_ns: u64,
    pub last_update_ns: u64,
}

#[derive(Debug, Clone, Default)]
pub struct QueueEstimate {
    pub estimated_position: f64,    // contracts ahead of us
    pub fill_probability: f64,      // 0.0 to 1.0
    pub time_to_fill_estimate_ms: u64,
    pub adverse_selection_score: f64, // probability we only get filled on adverse moves
}

pub struct QueuePositionEstimator {
    active_trackers: HashMap<u64, QueueTracker>, // Key: order_id
}

impl QueuePositionEstimator {
    pub fn new() -> Self {
        Self {
            active_trackers: HashMap::new(),
        }
    }

    pub fn track_order(&mut self, order_id: u64, tracker: QueueTracker) {
        self.active_trackers.insert(order_id, tracker);
    }

    pub fn untrack_order(&mut self, order_id: u64) {
        self.active_trackers.remove(&order_id);
    }

    pub fn estimate(&self, order_id: u64, current_level_depth: f64) -> Option<QueueEstimate> {
        let tracker = self.active_trackers.get(&order_id)?;
        
        let estimated_position = (tracker.initial_depth_ahead 
            - tracker.cumulative_fills_at_level 
            - tracker.cumulative_cancels_at_level).max(0.0);
            
        let fill_probability = if current_level_depth > 0.0 {
            1.0 - (estimated_position / current_level_depth).min(1.0)
        } else {
            1.0
        };
        
        Some(QueueEstimate {
            estimated_position,
            fill_probability,
            time_to_fill_estimate_ms: 1000,
            adverse_selection_score: 1.0 - fill_probability,
        })
    }
}
