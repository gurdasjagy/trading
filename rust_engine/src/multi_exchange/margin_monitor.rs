//! Cross-Venue Inventory and Margin Management
//!
//! Monitors margin health across all connected exchanges and alerts when:
//! - Any exchange falls below the minimum margin ratio (30% default)
//! - Any exchange reaches critical margin ratio (15% default)
//! - Cross-exchange delta neutrality is violated
//!
//! Provides recommendations for margin rebalancing between venues.

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::multi_exchange::global_book::ExchangeId;
use crate::execution_gateway::ExecutionGateway;

// ---------------------------------------------------------------------------
// Exchange Margin Health
// ---------------------------------------------------------------------------

/// Margin health status for a single exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeMarginHealth {
    pub exchange: ExchangeId,
    pub available_balance: f64,
    pub total_equity: f64,
    pub unrealized_pnl: f64,
    pub margin_ratio: f64,   // available / total_equity (0.0 - 1.0)
    pub is_healthy: bool,    // margin_ratio > 0.3 (30% threshold)
    pub is_critical: bool,   // margin_ratio < 0.15 (15% threshold)
    pub updated_ns: u64,
}

impl ExchangeMarginHealth {
    /// Create a new margin health entry.
    pub fn new(
        exchange: ExchangeId,
        available_balance: f64,
        total_equity: f64,
        unrealized_pnl: f64,
    ) -> Self {
        let margin_ratio = if total_equity > 0.0 {
            available_balance / total_equity
        } else {
            1.0  // No equity = 100% margin
        };

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            exchange,
            available_balance,
            total_equity,
            unrealized_pnl,
            margin_ratio,
            is_healthy: margin_ratio > 0.3,
            is_critical: margin_ratio < 0.15,
            updated_ns: now_ns,
        }
    }

    /// Get status string.
    pub fn status(&self) -> &'static str {
        if self.is_critical {
            "critical"
        } else if self.is_healthy {
            "healthy"
        } else {
            "warning"
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "exchange": self.exchange.name(),
            "exchange_id": self.exchange.id_str(),
            "available_balance": self.available_balance,
            "total_equity": self.total_equity,
            "unrealized_pnl": self.unrealized_pnl,
            "margin_ratio": self.margin_ratio,
            "margin_ratio_pct": format!("{:.1}%", self.margin_ratio * 100.0),
            "is_healthy": self.is_healthy,
            "is_critical": self.is_critical,
            "status": self.status(),
            "updated_ns": self.updated_ns,
        })
    }
}

// ---------------------------------------------------------------------------
// Margin Imbalance Alert
// ---------------------------------------------------------------------------

/// Cross-venue margin imbalance alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginImbalanceAlert {
    pub critical_exchange: ExchangeId,
    pub margin_ratio: f64,
    pub recommended_transfer_usdt: f64,
    pub source_exchange: ExchangeId,  // exchange with excess margin
}

impl MarginImbalanceAlert {
    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "critical_exchange": self.critical_exchange.name(),
            "margin_ratio": self.margin_ratio,
            "margin_ratio_pct": format!("{:.1}%", self.margin_ratio * 100.0),
            "recommended_transfer_usdt": self.recommended_transfer_usdt,
            "source_exchange": self.source_exchange.name(),
            "message": format!(
                "Transfer ${:.0} from {} to {} to rebalance margin",
                self.recommended_transfer_usdt,
                self.source_exchange.name(),
                self.critical_exchange.name()
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Cross-Venue Margin Monitor
// ---------------------------------------------------------------------------

/// Cross-venue margin monitor. Runs on the cold path (every 30 seconds).
pub struct CrossVenueMarginMonitor {
    /// Latest margin health per exchange.
    health: HashMap<ExchangeId, ExchangeMarginHealth>,
    /// Minimum acceptable margin ratio before alert (default: 0.30 = 30%).
    min_margin_ratio: f64,
    /// Critical margin ratio threshold (default: 0.15 = 15%).
    critical_margin_ratio: f64,
    /// BUG 4 FIX: Cached last-known balances per exchange.
    /// Used as fallback when API calls fail (with staleness timeout).
    cached_balances: HashMap<ExchangeId, CachedBalance>,
    /// BUG 4 FIX: Set of exchanges confirmed as ready (have valid gateways).
    ready_exchanges: HashMap<ExchangeId, bool>,
}

/// BUG 4 FIX: Cached balance with staleness tracking.
#[derive(Debug, Clone)]
pub struct CachedBalance {
    pub balance: f64,
    pub cached_at_ns: u64,
}

impl CachedBalance {
    /// Maximum age before a cached balance is considered stale (5 minutes).
    const MAX_STALENESS_NS: u64 = 5 * 60 * 1_000_000_000;

    pub fn new(balance: f64) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self { balance, cached_at_ns: now_ns }
    }

    pub fn is_stale(&self) -> bool {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now_ns.saturating_sub(self.cached_at_ns) > Self::MAX_STALENESS_NS
    }
}

impl CrossVenueMarginMonitor {
    /// Create a new margin monitor.
    pub fn new(min_margin_ratio: f64, critical_margin_ratio: f64) -> Self {
        Self {
            health: HashMap::new(),
            min_margin_ratio,
            critical_margin_ratio,
            cached_balances: HashMap::new(),
            ready_exchanges: HashMap::new(),
        }
    }

    /// Create with default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(0.30, 0.15)
    }

    /// Update margin health for a specific exchange.
    pub fn update_health(&mut self, health: ExchangeMarginHealth) {
        if health.is_critical {
            warn!(
                "[margin] CRITICAL: {} margin ratio at {:.1}% (balance: ${:.2})",
                health.exchange.name(),
                health.margin_ratio * 100.0,
                health.available_balance
            );
        } else if !health.is_healthy {
            warn!(
                "[margin] WARNING: {} margin ratio at {:.1}%",
                health.exchange.name(),
                health.margin_ratio * 100.0
            );
        }
        self.health.insert(health.exchange, health);
    }

    /// Get margin health for a specific exchange.
    pub fn get_health(&self, exchange: ExchangeId) -> Option<&ExchangeMarginHealth> {
        self.health.get(&exchange)
    }

    /// Get all exchange health statuses.
    pub fn all_health(&self) -> Vec<&ExchangeMarginHealth> {
        self.health.values().collect()
    }

    /// Check if the margin monitor has fetched balance data for a specific exchange.
    /// Returns false if the exchange has never been polled or if its balance is zero
    /// (which indicates the first fetch hasn't completed yet for funded accounts).
    pub fn has_exchange_data(&self, exchange: ExchangeId) -> bool {
        self.health.get(&exchange)
            .map(|h| h.available_balance > 0.0 || h.total_equity > 0.0)
            .unwrap_or(false)
    }

    /// Check for margin imbalances across all exchanges.
    /// Returns alerts for exchanges below the minimum margin ratio.
    pub fn check_imbalances(&self) -> Vec<MarginImbalanceAlert> {
        let mut alerts = Vec::new();

        // Find exchanges below threshold and exchanges with excess margin
        let mut critical: Vec<&ExchangeMarginHealth> = Vec::new();
        let mut excess: Vec<&ExchangeMarginHealth> = Vec::new();

        for health in self.health.values() {
            if health.margin_ratio < self.min_margin_ratio {
                critical.push(health);
            } else if health.margin_ratio > 0.5 {
                excess.push(health);
            }
        }

        // Sort excess by available balance descending
        excess.sort_by(|a, b| {
            b.available_balance.partial_cmp(&a.available_balance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Generate alerts for critical exchanges
        for crit in &critical {
            // Find best source for rebalancing
            let source = excess.first();
            
            if let Some(src) = source {
                // Calculate recommended transfer
                // Target: bring critical exchange to min_margin_ratio + 10%
                let target_ratio = self.min_margin_ratio + 0.10;
                let needed_balance = crit.total_equity * target_ratio - crit.available_balance;
                let recommended = needed_balance.min(src.available_balance * 0.3);  // Max 30% of source

                if recommended > 10.0 {  // Only recommend if > $10
                    alerts.push(MarginImbalanceAlert {
                        critical_exchange: crit.exchange,
                        margin_ratio: crit.margin_ratio,
                        recommended_transfer_usdt: recommended,
                        source_exchange: src.exchange,
                    });
                }
            }
        }

        alerts
    }

    /// Calculate global delta neutrality.
    /// Returns (total_long_usdt, total_short_usdt, net_delta_usdt).
    pub fn calculate_global_delta(
        &self,
        positions: &HashMap<ExchangeId, Vec<(String, f64, f64)>>, // (symbol, size, price)
    ) -> (f64, f64, f64) {
        let mut total_long = 0.0;
        let mut total_short = 0.0;

        for (_exchange, pos_list) in positions {
            for (_, size, price) in pos_list {
                let notional = size.abs() * price;
                if *size > 0.0 {
                    total_long += notional;
                } else {
                    total_short += notional;
                }
            }
        }

        let net_delta = total_long - total_short;
        (total_long, total_short, net_delta)
    }

    /// Check if the portfolio is delta neutral within tolerance.
    pub fn is_delta_neutral(
        &self,
        positions: &HashMap<ExchangeId, Vec<(String, f64, f64)>>,
        tolerance_pct: f64,
    ) -> bool {
        let (total_long, total_short, net_delta) = self.calculate_global_delta(positions);
        let total_exposure = total_long + total_short;
        
        if total_exposure == 0.0 {
            return true;  // No positions = neutral
        }

        let delta_pct = net_delta.abs() / total_exposure;
        delta_pct <= tolerance_pct
    }

    /// BUG 4 FIX: Mark an exchange gateway as ready/not ready.
    pub fn set_gateway_ready(&mut self, exchange: ExchangeId, ready: bool) {
        self.ready_exchanges.insert(exchange, ready);
        if ready {
            info!("[margin] {} gateway marked as ready", exchange.name());
        } else {
            warn!("[margin] {} gateway marked as NOT ready", exchange.name());
        }
    }

    /// BUG 4 FIX: Check if an exchange gateway is ready.
    pub fn is_gateway_ready(&self, exchange: ExchangeId) -> bool {
        self.ready_exchanges.get(&exchange).copied().unwrap_or(false)
    }

    /// Fetch and update margin health from all active gateways.
    /// Called every 30 seconds from the execution router loop.
    ///
    /// BUG 4 FIX: Skip exchanges with no valid gateway instead of reporting 0%.
    /// Cache last known balance and use if API call fails (with staleness timeout).
    /// Log clearly if any exchange has $0 balance at startup.
    pub async fn refresh_all(
        &mut self,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    ) {
        for (exchange, gateway) in gateways {
            // BUG 4 FIX: Skip exchanges that are not ready
            if !self.is_gateway_ready(*exchange) {
                info!(
                    "[margin] Skipping {} (gateway not ready, keys may be missing)",
                    exchange.name()
                );
                continue;
            }

            match gateway.get_balance().await {
                Ok(balance) => {
                    // BUG 4 FIX: Cache the balance for fallback
                    self.cached_balances.insert(*exchange, CachedBalance::new(balance));

                    // BUG 4 FIX: Log clearly if balance is $0 (startup check)
                    if balance <= 0.0 {
                        warn!(
                            "[margin] {} reports $0 balance — account may not be funded",
                            exchange.name()
                        );
                    }

                    let health = ExchangeMarginHealth::new(
                        *exchange,
                        balance,
                        balance,  // Total equity (simplified)
                        0.0,      // Unrealized PnL (would need position query)
                    );
                    let margin_ratio = health.margin_ratio;
                    self.update_health(health);
                    info!(
                        "[margin] {} balance: ${:.2} (margin: {:.1}%)",
                        exchange.name(),
                        balance,
                        margin_ratio * 100.0
                    );
                }
                Err(e) => {
                    // BUG 4 FIX: Use cached balance if available and not stale
                    if let Some(cached) = self.cached_balances.get(exchange) {
                        if !cached.is_stale() {
                            warn!(
                                "[margin] {} API failed ({:?}), using cached balance ${:.2}",
                                exchange.name(), e, cached.balance
                            );
                            let health = ExchangeMarginHealth::new(
                                *exchange,
                                cached.balance,
                                cached.balance,
                                0.0,
                            );
                            self.update_health(health);
                        } else {
                            error!(
                                "[margin] {} API failed ({:?}) and cached balance is stale — skipping",
                                exchange.name(), e
                            );
                        }
                    } else {
                        error!(
                            "[margin] {} API failed ({:?}) and no cached balance available — skipping",
                            exchange.name(), e
                        );
                    }
                }
            }
        }
    }

    /// Get total balance across all exchanges.
    pub fn total_balance(&self) -> f64 {
        self.health.values().map(|h| h.available_balance).sum()
    }

    /// Get total equity across all exchanges.
    pub fn total_equity(&self) -> f64 {
        self.health.values().map(|h| h.total_equity).sum()
    }

    /// Get total unrealized PnL across all exchanges.
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.health.values().map(|h| h.unrealized_pnl).sum()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let alerts = self.check_imbalances();
        
        serde_json::json!({
            "exchanges": self.health.values()
                .map(|h| h.to_json())
                .collect::<Vec<_>>(),
            "total_balance": self.total_balance(),
            "total_equity": self.total_equity(),
            "total_unrealized_pnl": self.total_unrealized_pnl(),
            "alerts": alerts.iter().map(|a| a.to_json()).collect::<Vec<_>>(),
            "has_critical": self.health.values().any(|h| h.is_critical),
            "has_warning": self.health.values().any(|h| !h.is_healthy),
        })
    }
}

impl Default for CrossVenueMarginMonitor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_margin_health_creation() {
        let health = ExchangeMarginHealth::new(
            ExchangeId::Binance,
            5000.0,   // available
            10000.0,  // total equity
            500.0,    // unrealized PnL
        );

        assert_eq!(health.margin_ratio, 0.5);
        assert!(health.is_healthy);
        assert!(!health.is_critical);
    }

    #[test]
    fn test_critical_margin() {
        let health = ExchangeMarginHealth::new(
            ExchangeId::GateIo,
            1000.0,   // available
            10000.0,  // total equity
            -500.0,   // unrealized PnL
        );

        assert_eq!(health.margin_ratio, 0.1);
        assert!(!health.is_healthy);
        assert!(health.is_critical);
    }

    #[test]
    fn test_imbalance_detection() {
        let mut monitor = CrossVenueMarginMonitor::with_defaults();

        // Gate.io is critical
        monitor.update_health(ExchangeMarginHealth::new(
            ExchangeId::GateIo,
            1000.0,
            10000.0,
            0.0,
        ));

        // Binance has excess
        monitor.update_health(ExchangeMarginHealth::new(
            ExchangeId::Binance,
            8000.0,
            10000.0,
            0.0,
        ));

        let alerts = monitor.check_imbalances();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].critical_exchange, ExchangeId::GateIo);
        assert_eq!(alerts[0].source_exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_delta_neutrality() {
        let monitor = CrossVenueMarginMonitor::with_defaults();

        let mut positions = HashMap::new();
        positions.insert(
            ExchangeId::GateIo,
            vec![("BTC_USDT".to_string(), 1.0, 50000.0)],  // Long 1 BTC
        );
        positions.insert(
            ExchangeId::Binance,
            vec![("BTC_USDT".to_string(), -1.0, 50000.0)],  // Short 1 BTC
        );

        let (long, short, delta) = monitor.calculate_global_delta(&positions);
        assert_eq!(long, 50000.0);
        assert_eq!(short, 50000.0);
        assert_eq!(delta, 0.0);

        assert!(monitor.is_delta_neutral(&positions, 0.01));
    }
}
