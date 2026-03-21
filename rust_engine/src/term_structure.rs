use std::collections::HashMap;

pub struct TermStructureTracker {
    pub perpetual_price: f64,
    pub quarterly_price: f64,
    pub next_quarterly_price: f64,
    pub days_to_expiry: f64,
}

impl TermStructureTracker {
    pub fn new(days_to_expiry: f64) -> Self {
        Self {
            perpetual_price: 0.0,
            quarterly_price: 0.0,
            next_quarterly_price: 0.0,
            days_to_expiry,
        }
    }

    pub fn basis(&self) -> f64 {
        if self.perpetual_price > 0.0 {
            (self.quarterly_price - self.perpetual_price) / self.perpetual_price
        } else {
            0.0
        }
    }

    pub fn annualized_basis(&self) -> f64 {
        if self.days_to_expiry > 0.0 {
            self.basis() * (365.0 / self.days_to_expiry)
        } else {
            0.0
        }
    }

    pub fn is_contango(&self) -> bool {
        self.quarterly_price > self.perpetual_price
    }
}
