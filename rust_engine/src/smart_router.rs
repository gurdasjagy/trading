//! Smart Order Router — Issue 3.
//!
//! Routes orders to the best venue based on a cost model:
//!
//!   total_cost = spread/2 + fee + impact_estimate
//!
//! The router maintains per-venue state (spread, depth, fees, rate limit status)
//! and selects the venue with the lowest expected execution cost.
//!
//! # Venue State
//!
//! Updated from live orderbook snapshots and exchange API responses:
//! - `spread_bps`: current bid-ask spread in basis points
//! - `bid_depth` / `ask_depth`: top-of-book liquidity
//! - `taker_fee_bps`: exchange taker fee
//! - `maker_fee_bps`: exchange maker fee (negative for rebate)
//! - `at_rate_limit`: whether the venue is currently rate-limited
//! - `last_latency_us`: last observed order-to-ack latency
//!
//! # Usage
//!
//! The execution router calls `route()` before submitting each order.
//! If the best venue is at rate limit, the router can either wait or
//! skip to the next venue.

// ═══════════════════════════════════════════════════════════════════════════
// Venue State
// ═══════════════════════════════════════════════════════════════════════════

/// State of a single trading venue (exchange).
#[derive(Debug, Clone)]
pub struct VenueState {
    /// Exchange identifier (e.g., 0 = Gate.io).
    pub exchange_id: u8,
    /// Exchange name for logging.
    pub name: String,
    /// Current bid-ask spread in basis points.
    pub spread_bps: i64,
    /// Bid-side depth in USDT at top N levels.
    pub bid_depth_usdt: f64,
    /// Ask-side depth in USDT at top N levels.
    pub ask_depth_usdt: f64,
    /// Taker fee in basis points (e.g., 5 = 0.05%).
    pub taker_fee_bps: i64,
    /// Maker fee in basis points (negative for rebate, e.g., -1 = -0.01%).
    pub maker_fee_bps: i64,
    /// Whether this venue is currently at rate limit.
    pub at_rate_limit: bool,
    /// Last observed order-to-ack latency in microseconds.
    pub last_latency_us: u64,
    /// Timestamp of last state update (nanoseconds).
    pub last_update_ns: u64,
    /// Whether this venue is enabled for trading.
    pub enabled: bool,
}

impl VenueState {
    /// Create a new venue state with default values.
    pub fn new(exchange_id: u8, name: &str) -> Self {
        Self {
            exchange_id,
            name: name.to_string(),
            spread_bps: 100, // 1% default (conservative)
            bid_depth_usdt: 0.0,
            ask_depth_usdt: 0.0,
            taker_fee_bps: 5,   // 0.05% default
            maker_fee_bps: -1,  // -0.01% rebate default
            at_rate_limit: false,
            last_latency_us: 0,
            last_update_ns: 0,
            enabled: true,
        }
    }

    /// Update spread and depth from an orderbook snapshot.
    pub fn update_from_book(&mut self, spread_bps: i64, bid_depth: f64, ask_depth: f64) {
        self.spread_bps = spread_bps;
        self.bid_depth_usdt = bid_depth;
        self.ask_depth_usdt = ask_depth;
        self.last_update_ns = now_ns();
    }

    /// Mark this venue as rate-limited.
    pub fn set_rate_limited(&mut self, limited: bool) {
        self.at_rate_limit = limited;
        self.last_update_ns = now_ns();
    }

    /// Update latency from a successful order submission.
    pub fn update_latency(&mut self, latency_us: u64) {
        self.last_latency_us = latency_us;
        self.last_update_ns = now_ns();
    }

    /// Check if venue state is stale (no update in >5 seconds).
    pub fn is_stale(&self) -> bool {
        let now = now_ns();
        now.saturating_sub(self.last_update_ns) > 5_000_000_000 // 5 seconds
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Routing Decision
// ═══════════════════════════════════════════════════════════════════════════

/// Why a particular venue was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingReason {
    /// Selected because it has the tightest spread.
    BestSpread,
    /// Selected because it has the lowest fees.
    LowestFees,
    /// Selected because it has the deepest liquidity.
    BestLiquidity,
    /// Only venue available (others at rate limit or disabled).
    OnlyAvailable,
    /// No venue available (all at rate limit or disabled).
    NoVenueAvailable,
}

/// Result of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Which exchange to route to.
    pub exchange_id: u8,
    /// Why this venue was selected.
    pub reason: RoutingReason,
    /// Expected cost in basis points (lower is better).
    pub expected_cost_bps: i64,
}

// ═══════════════════════════════════════════════════════════════════════════
// SmartOrderRouter
// ═══════════════════════════════════════════════════════════════════════════

/// Smart order router that selects the best venue for each order.
///
/// Cost model: `total_cost = spread/2 + fee + impact_estimate`
///
/// The impact estimate is a simple linear model based on order size
/// relative to available depth at the best price level.
pub struct SmartOrderRouter {
    /// Per-venue state.
    venues: Vec<VenueState>,
    /// Impact coefficient: how much each unit of size moves the price.
    /// Default: 1 BPS per 1% of book depth consumed.
    impact_coefficient_bps: f64,
}

impl SmartOrderRouter {
    /// Create a new router with the given venues.
    pub fn new(venues: Vec<VenueState>) -> Self {
        Self {
            venues,
            impact_coefficient_bps: 1.0,
        }
    }

    /// Create a default router with Gate.io venue.
    pub fn default_venues() -> Self {
        let gateio = VenueState {
            exchange_id: 0,
            name: "gateio".to_string(),
            spread_bps: 2,
            bid_depth_usdt: 100_000.0,
            ask_depth_usdt: 100_000.0,
            taker_fee_bps: 5,    // 0.05%
            maker_fee_bps: -1,   // -0.01% rebate
            at_rate_limit: false,
            last_latency_us: 0,
            last_update_ns: now_ns(),
            enabled: true,
        };
        Self::new(vec![gateio])
    }

    /// Set the impact coefficient.
    pub fn set_impact_coefficient(&mut self, coeff: f64) {
        self.impact_coefficient_bps = coeff;
    }

    /// Update a venue's state by exchange_id.
    pub fn update_venue(&mut self, exchange_id: u8, spread_bps: i64, bid_depth: f64, ask_depth: f64) {
        if let Some(venue) = self.venues.iter_mut().find(|v| v.exchange_id == exchange_id) {
            venue.update_from_book(spread_bps, bid_depth, ask_depth);
        }
    }

    /// Mark a venue as rate-limited.
    pub fn set_venue_rate_limited(&mut self, exchange_id: u8, limited: bool) {
        if let Some(venue) = self.venues.iter_mut().find(|v| v.exchange_id == exchange_id) {
            venue.set_rate_limited(limited);
        }
    }

    /// Route an order to the best venue.
    ///
    /// # Arguments
    /// * `order_size_usdt` — notional order size in USDT.
    /// * `is_maker` — whether this is a maker (limit) or taker (market) order.
    ///
    /// # Cost Model
    /// For each venue:
    ///   half_spread = spread_bps / 2
    ///   fee = maker_fee_bps (if is_maker) or taker_fee_bps
    ///   impact = (order_size / available_depth) * impact_coefficient
    ///   total_cost = half_spread + fee + impact
    pub fn route(&self, order_size_usdt: f64, is_maker: bool) -> RoutingDecision {
        let mut best_score = i64::MAX;
        let mut best_venue: u8 = 0;
        let mut best_reason = RoutingReason::NoVenueAvailable;
        let mut found_any = false;

        for venue in &self.venues {
            if !venue.enabled || venue.at_rate_limit {
                continue;
            }

            found_any = true;

            let half_spread = venue.spread_bps / 2;
            let fee = if is_maker {
                venue.maker_fee_bps
            } else {
                venue.taker_fee_bps
            };

            // Impact estimate: linear model based on order size vs depth
            let available_depth = if is_maker {
                // For maker orders, we're joining the book — less impact
                venue.bid_depth_usdt.max(venue.ask_depth_usdt).max(1.0)
            } else {
                // For taker orders, we're consuming from the opposite side
                venue.bid_depth_usdt.min(venue.ask_depth_usdt).max(1.0)
            };

            let impact_bps = if available_depth > 0.0 {
                ((order_size_usdt / available_depth) * self.impact_coefficient_bps * 100.0) as i64
            } else {
                100 // High penalty if no liquidity
            };

            let total_cost = half_spread + fee + impact_bps;

            if total_cost < best_score {
                best_score = total_cost;
                best_venue = venue.exchange_id;
                best_reason = if half_spread * 2 <= venue.spread_bps / 2 {
                    RoutingReason::BestSpread
                } else if fee <= venue.taker_fee_bps / 2 {
                    RoutingReason::LowestFees
                } else {
                    RoutingReason::BestLiquidity
                };
            }
        }

        if !found_any {
            return RoutingDecision {
                exchange_id: 0,
                reason: RoutingReason::NoVenueAvailable,
                expected_cost_bps: i64::MAX,
            };
        }

        // Check if only one venue was available
        let available_count = self.venues.iter().filter(|v| v.enabled && !v.at_rate_limit).count();
        if available_count == 1 {
            best_reason = RoutingReason::OnlyAvailable;
        }

        RoutingDecision {
            exchange_id: best_venue,
            reason: best_reason,
            expected_cost_bps: best_score,
        }
    }

    /// Get a reference to a venue by exchange_id.
    pub fn get_venue(&self, exchange_id: u8) -> Option<&VenueState> {
        self.venues.iter().find(|v| v.exchange_id == exchange_id)
    }

    /// Get a mutable reference to a venue by exchange_id.
    pub fn get_venue_mut(&mut self, exchange_id: u8) -> Option<&mut VenueState> {
        self.venues.iter_mut().find(|v| v.exchange_id == exchange_id)
    }

    /// Get the number of venues.
    pub fn venue_count(&self) -> usize {
        self.venues.len()
    }

    /// Get the number of available (non-rate-limited, enabled) venues.
    pub fn available_venue_count(&self) -> usize {
        self.venues.iter().filter(|v| v.enabled && !v.at_rate_limit).count()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_to_best_spread() {
        let mut gateio = VenueState::new(0, "gateio");
        gateio.spread_bps = 2;
        gateio.taker_fee_bps = 5;
        gateio.bid_depth_usdt = 100_000.0;
        gateio.ask_depth_usdt = 100_000.0;

        let router = SmartOrderRouter::new(vec![gateio]);
        let decision = router.route(1000.0, false);

        assert_eq!(decision.exchange_id, 0, "Should route to Gate.io (tighter spread)");
        assert!(decision.expected_cost_bps < i64::MAX);
    }

    #[test]
    fn test_avoid_rate_limited_venue() {
        let mut gateio = VenueState::new(0, "gateio");
        gateio.spread_bps = 2;
        gateio.at_rate_limit = true; // Rate limited!

        let router = SmartOrderRouter::new(vec![gateio]);
        let decision = router.route(1000.0, false);

        assert_eq!(decision.reason, RoutingReason::NoVenueAvailable);
    }

    #[test]
    fn test_no_venue_available() {
        let mut gateio = VenueState::new(0, "gateio");
        gateio.at_rate_limit = true;

        let router = SmartOrderRouter::new(vec![gateio]);
        let decision = router.route(1000.0, false);

        assert_eq!(decision.reason, RoutingReason::NoVenueAvailable);
    }

    #[test]
    fn test_maker_vs_taker_fee() {
        let mut gateio = VenueState::new(0, "gateio");
        gateio.spread_bps = 4;
        gateio.taker_fee_bps = 10;
        gateio.maker_fee_bps = -2; // rebate
        gateio.bid_depth_usdt = 100_000.0;
        gateio.ask_depth_usdt = 100_000.0;

        let router = SmartOrderRouter::new(vec![gateio]);

        let taker_decision = router.route(1000.0, false);
        let maker_decision = router.route(1000.0, true);

        // Maker should have lower cost due to rebate
        assert!(
            maker_decision.expected_cost_bps < taker_decision.expected_cost_bps,
            "Maker cost ({}) should be less than taker cost ({})",
            maker_decision.expected_cost_bps,
            taker_decision.expected_cost_bps
        );
    }

    #[test]
    fn test_update_venue_state() {
        let mut router = SmartOrderRouter::default_venues();

        router.update_venue(0, 1, 200_000.0, 200_000.0);

        let venue = router.get_venue(0).unwrap();
        assert_eq!(venue.spread_bps, 1);
        assert_eq!(venue.bid_depth_usdt, 200_000.0);
    }

    #[test]
    fn test_venue_count() {
        let router = SmartOrderRouter::default_venues();
        assert_eq!(router.venue_count(), 1);
        assert_eq!(router.available_venue_count(), 1);
    }
}
