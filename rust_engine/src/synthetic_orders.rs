use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub enum SyntheticOrderType {
    Iceberg {
        total_size: i64,
        visible_size: i64,
        refill_delay_ms: u64,
        randomize_visible: bool, // +/- 20% of visible_size
    },
    Sniper {
        target_price: f64,
        size: i64,
        max_wait_ms: u64,
        aggression: f64, // 0.0 = passive, 1.0 = cross immediately
    },
    PeggedToMid {
        size: i64,
        offset_bps: f64,  // Offset from mid price
        max_drift_bps: f64, // Cancel and replace if drifted too far
        update_interval_ms: u64,
    },
    Twap {
        total_size: i64,
        duration_ms: u64,
        num_slices: u32,
        randomize_timing: bool, // +/- 30% of slice interval
        randomize_size: bool,   // +/- 20% of slice size
        max_participation_rate: f64, // Max % of volume
    },
    ImplementationShortfall {
        total_size: i64,
        urgency: f64,     // 0.0 = patient, 1.0 = aggressive
        risk_aversion: f64,
        alpha_decay_halflife_ms: u64,
    },
}

pub struct SyntheticOrderState {
    pub order_type: SyntheticOrderType,
    pub filled_size: i64,
    pub created_ns: u64,
    pub last_update_ns: u64,
}

pub struct SyntheticOrderManager {
    pub active_orders: HashMap<u64, SyntheticOrderState>,
    pub next_id: AtomicU64,
}

impl SyntheticOrderManager {
    pub fn new() -> Self {
        Self {
            active_orders: HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn submit_synthetic(&mut self, order_type: SyntheticOrderType) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let state = SyntheticOrderState {
            order_type,
            filled_size: 0,
            created_ns: 0, // Should be actual time
            last_update_ns: 0,
        };
        self.active_orders.insert(id, state);
        id
    }
}
