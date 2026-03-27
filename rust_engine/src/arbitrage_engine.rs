//! Multi-Exchange Arbitrage Engine.
//!
//! Monitors price disparities across exchanges and executes simultaneous
//! buy/sell orders when spread exceeds threshold.
//!
//! # Architecture
//!
//! The arbitrage engine maintains a real-time view of prices across multiple
//! exchanges and scans for profitable opportunities considering:
//! - Best bid/ask across all venues
//! - Available liquidity at each level
//! - Transaction costs (fees, slippage)
//! - Execution risk (latency, fill rates)
//!
//! # Supported Exchanges
//!
//! - Gate.io (primary)
//! - Binance Futures (via binance_gateway)
//! - OKX (via okx_gateway)
//! - Bybit (via bybit_gateway)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, info};

/// Minimum spread in basis points to consider an arbitrage opportunity.
const ARB_THRESHOLD_BPS: i64 = 10; // 0.1% minimum spread

/// Maximum arbitrage position size in USDT.
const MAX_ARB_SIZE_USDT: f64 = 10000.0;

/// Maximum latency tolerance for arbitrage execution (microseconds).
const MAX_LATENCY_US: u64 = 500;

/// Exchange identifiers.
pub mod exchange_id {
    pub const GATEIO: u8 = 0;
    pub const BINANCE: u8 = 1;
    pub const OKX: u8 = 2;
    pub const BYBIT: u8 = 3;
}

/// Price data from an exchange.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExchangePrice {
    /// Exchange identifier.
    pub exchange_id: u8,
    /// Best bid price in fixed-point (price * 1e8).
    pub bid: i64,
    /// Best ask price in fixed-point (price * 1e8).
    pub ask: i64,
    /// Bid size in fixed-point (qty * 1e8).
    pub bid_size: i64,
    /// Ask size in fixed-point (qty * 1e8).
    pub ask_size: i64,
    /// Timestamp in nanoseconds since epoch.
    pub timestamp_ns: u64,
}

impl ExchangePrice {
    /// Check if the price data is fresh (within MAX_LATENCY_US).
    pub fn is_fresh(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let age_us = (now - self.timestamp_ns) / 1000;
        age_us < MAX_LATENCY_US
    }
    
    /// Get mid price.
    pub fn mid(&self) -> i64 {
        (self.bid + self.ask) / 2
    }
    
    /// Get spread in basis points.
    pub fn spread_bps(&self) -> i64 {
        if self.mid() > 0 {
            (self.ask - self.bid) * 10000 / self.mid()
        } else {
            0
        }
    }
}

/// Arbitrage opportunity detected by the engine.
#[derive(Debug, Clone)]
pub struct ArbOpportunity {
    /// Symbol identifier.
    pub symbol_id: u16,
    /// Symbol name.
    pub symbol: String,
    /// Exchange to buy on (lower ask).
    pub buy_exchange: u8,
    /// Exchange to sell on (higher bid).
    pub sell_exchange: u8,
    /// Buy price (ask on buy exchange).
    pub buy_price: i64,
    /// Sell price (bid on sell exchange).
    pub sell_price: i64,
    /// Spread in basis points.
    pub spread_bps: i64,
    /// Maximum executable size (min of available liquidity).
    pub max_size: i64,
    /// Estimated profit in USDT (before fees).
    pub est_profit_usdt: f64,
    /// Timestamp when opportunity was detected.
    pub detected_at: Instant,
}

impl ArbOpportunity {
    /// Check if the opportunity is still valid (within timeout).
    pub fn is_valid(&self) -> bool {
        self.detected_at.elapsed().as_micros() < MAX_LATENCY_US as u128
    }
    
    /// Calculate net profit after fees.
    pub fn net_profit(&self, maker_fee_bps: f64, taker_fee_bps: f64) -> f64 {
        let gross_profit_bps = self.spread_bps as f64;
        let total_fees_bps = maker_fee_bps + taker_fee_bps;
        let net_bps = gross_profit_bps - total_fees_bps;
        
        let size_usdt = self.max_size as f64 / 1e8;
        size_usdt.min(MAX_ARB_SIZE_USDT) * net_bps / 10000.0
    }
}

/// Arbitrage execution result.
#[derive(Debug, Clone)]
pub struct ArbExecution {
    pub opportunity: ArbOpportunity,
    pub buy_fill_price: i64,
    pub sell_fill_price: i64,
    pub filled_size: i64,
    pub realized_pnl_usdt: f64,
    pub execution_time_us: u64,
    pub success: bool,
}

/// Multi-exchange arbitrage engine.
pub struct ArbitrageEngine {
    /// Latest prices per symbol per exchange.
    /// Key: (symbol_id, exchange_id) -> ExchangePrice
    prices: HashMap<(u16, u8), ExchangePrice>,
    /// Symbol ID to name mapping.
    symbol_names: HashMap<u16, String>,
    /// Best bid across all exchanges per symbol.
    /// Value: (price, exchange_id)
    best_bid: HashMap<u16, (i64, u8)>,
    /// Best ask across all exchanges per symbol.
    best_ask: HashMap<u16, (i64, u8)>,
    /// Opportunities detected counter.
    pub opportunities_detected: AtomicU64,
    /// Arbitrage trades executed counter.
    pub trades_executed: AtomicU64,
    /// Total realized PnL in USDT.
    pub total_pnl_usdt: f64,
    /// Enabled state.
    pub enabled: bool,
    /// Minimum spread threshold (can be adjusted).
    pub min_spread_bps: i64,
    /// Fee estimates per exchange (in bps).
    exchange_fees: HashMap<u8, (f64, f64)>, // (maker_fee, taker_fee)
    /// Last scan timestamp.
    last_scan_ns: u64,
}

impl ArbitrageEngine {
    /// Create a new arbitrage engine.
    pub fn new() -> Self {
        let mut exchange_fees = HashMap::new();
        // Default fee estimates (maker, taker) in basis points
        exchange_fees.insert(exchange_id::GATEIO, (1.5, 3.5));
        exchange_fees.insert(exchange_id::BINANCE, (1.0, 4.0));
        exchange_fees.insert(exchange_id::OKX, (2.0, 5.0));
        exchange_fees.insert(exchange_id::BYBIT, (1.0, 6.0));
        
        Self {
            prices: HashMap::new(),
            symbol_names: HashMap::new(),
            best_bid: HashMap::new(),
            best_ask: HashMap::new(),
            opportunities_detected: AtomicU64::new(0),
            trades_executed: AtomicU64::new(0),
            total_pnl_usdt: 0.0,
            enabled: false, // Disabled by default
            min_spread_bps: ARB_THRESHOLD_BPS,
            exchange_fees,
            last_scan_ns: 0,
        }
    }
    
    /// Enable or disable the arbitrage engine.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            info!("Arbitrage engine enabled with min spread {} bps", self.min_spread_bps);
        } else {
            info!("Arbitrage engine disabled");
        }
    }
    
    /// Register a symbol for arbitrage monitoring.
    pub fn register_symbol(&mut self, symbol_id: u16, name: &str) {
        self.symbol_names.insert(symbol_id, name.to_string());
    }
    
    /// Update price from an exchange.
    pub fn update_price(&mut self, symbol_id: u16, price: ExchangePrice) {
        self.prices.insert((symbol_id, price.exchange_id), price);
        
        // Update best bid if this is the highest
        let current_best_bid = self.best_bid.get(&symbol_id).map(|&(p, _)| p).unwrap_or(0);
        if price.bid > current_best_bid {
            self.best_bid.insert(symbol_id, (price.bid, price.exchange_id));
        } else if price.exchange_id == self.best_bid.get(&symbol_id).map(|&(_, e)| e).unwrap_or(255) {
            // This exchange had the best bid, but it might have changed
            self.recalculate_best_bid(symbol_id);
        }
        
        // Update best ask if this is the lowest
        let current_best_ask = self.best_ask.get(&symbol_id).map(|&(p, _)| p).unwrap_or(i64::MAX);
        if price.ask < current_best_ask && price.ask > 0 {
            self.best_ask.insert(symbol_id, (price.ask, price.exchange_id));
        } else if price.exchange_id == self.best_ask.get(&symbol_id).map(|&(_, e)| e).unwrap_or(255) {
            // This exchange had the best ask, but it might have changed
            self.recalculate_best_ask(symbol_id);
        }
    }
    
    fn recalculate_best_bid(&mut self, symbol_id: u16) {
        let mut best: Option<(i64, u8)> = None;
        for ex_id in [exchange_id::GATEIO, exchange_id::BINANCE, exchange_id::OKX, exchange_id::BYBIT] {
            if let Some(price) = self.prices.get(&(symbol_id, ex_id)) {
                if price.is_fresh() {
                    match best {
                        None => best = Some((price.bid, ex_id)),
                        Some((p, _)) if price.bid > p => best = Some((price.bid, ex_id)),
                        _ => {}
                    }
                }
            }
        }
        if let Some(b) = best {
            self.best_bid.insert(symbol_id, b);
        }
    }
    
    fn recalculate_best_ask(&mut self, symbol_id: u16) {
        let mut best: Option<(i64, u8)> = None;
        for ex_id in [exchange_id::GATEIO, exchange_id::BINANCE, exchange_id::OKX, exchange_id::BYBIT] {
            if let Some(price) = self.prices.get(&(symbol_id, ex_id)) {
                if price.is_fresh() && price.ask > 0 {
                    match best {
                        None => best = Some((price.ask, ex_id)),
                        Some((p, _)) if price.ask < p => best = Some((price.ask, ex_id)),
                        _ => {}
                    }
                }
            }
        }
        if let Some(b) = best {
            self.best_ask.insert(symbol_id, b);
        }
    }
    
    /// Scan for arbitrage opportunities for a specific symbol.
    pub fn scan(&self, symbol_id: u16) -> Option<ArbOpportunity> {
        if !self.enabled {
            return None;
        }
        
        let (best_bid, bid_ex) = self.best_bid.get(&symbol_id)?;
        let (best_ask, ask_ex) = self.best_ask.get(&symbol_id)?;
        
        // Can only arb if best bid > best ask (crossed market)
        // AND the prices are on different exchanges
        if bid_ex == ask_ex {
            return None; // Same exchange, no cross-exchange arb
        }
        
        if *best_bid <= *best_ask {
            return None; // No arb opportunity
        }
        
        // Calculate spread in basis points
        let mid = (*best_bid + *best_ask) / 2;
        let spread_bps = if mid > 0 {
            (*best_bid - *best_ask) * 10000 / mid
        } else {
            0
        };
        
        if spread_bps < self.min_spread_bps {
            return None;
        }
        
        // Get available sizes
        let bid_price_data = self.prices.get(&(symbol_id, *bid_ex))?;
        let ask_price_data = self.prices.get(&(symbol_id, *ask_ex))?;
        
        // Check freshness
        if !bid_price_data.is_fresh() || !ask_price_data.is_fresh() {
            debug!("Stale prices for arb opportunity, skipping");
            return None;
        }
        
        let max_size = bid_price_data.bid_size.min(ask_price_data.ask_size);
        
        // Calculate estimated profit
        let size_usdt = (max_size as f64 / 1e8).min(MAX_ARB_SIZE_USDT);
        let est_profit_usdt = size_usdt * spread_bps as f64 / 10000.0;
        
        self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
        
        let symbol = self.symbol_names
            .get(&symbol_id)
            .cloned()
            .unwrap_or_else(|| format!("SYM_{}", symbol_id));
        
        Some(ArbOpportunity {
            symbol_id,
            symbol,
            buy_exchange: *ask_ex,
            sell_exchange: *bid_ex,
            buy_price: *best_ask,
            sell_price: *best_bid,
            spread_bps,
            max_size,
            est_profit_usdt,
            detected_at: Instant::now(),
        })
    }
    
    /// Scan all registered symbols for opportunities.
    pub fn scan_all(&self) -> Vec<ArbOpportunity> {
        let symbols: Vec<u16> = self.symbol_names.keys().cloned().collect();
        
        symbols.iter()
            .filter_map(|&sym| self.scan(sym))
            .collect()
    }
    
    /// Record an execution result.
    pub fn record_execution(&mut self, execution: ArbExecution) {
        if execution.success {
            self.trades_executed.fetch_add(1, Ordering::Relaxed);
            self.total_pnl_usdt += execution.realized_pnl_usdt;
            
            info!(
                "Arb executed: {} buy@{} sell@{} size={} pnl=${:.4}",
                execution.opportunity.symbol,
                execution.buy_fill_price,
                execution.sell_fill_price,
                execution.filled_size,
                execution.realized_pnl_usdt
            );
        }
    }
    
    /// Get fee estimate for an exchange (maker, taker) in bps.
    pub fn get_fees(&self, exchange_id: u8) -> (f64, f64) {
        self.exchange_fees.get(&exchange_id).copied().unwrap_or((3.0, 6.0))
    }
    
    /// Update fee estimate for an exchange.
    pub fn set_fees(&mut self, exchange_id: u8, maker_bps: f64, taker_bps: f64) {
        self.exchange_fees.insert(exchange_id, (maker_bps, taker_bps));
    }
    
    /// Get statistics.
    pub fn stats(&self) -> ArbStats {
        ArbStats {
            opportunities_detected: self.opportunities_detected.load(Ordering::Relaxed),
            trades_executed: self.trades_executed.load(Ordering::Relaxed),
            total_pnl_usdt: self.total_pnl_usdt,
            enabled: self.enabled,
            min_spread_bps: self.min_spread_bps,
            symbols_monitored: self.symbol_names.len(),
        }
    }
    
    /// Get exchange name from ID.
    pub fn exchange_name(id: u8) -> &'static str {
        match id {
            exchange_id::GATEIO => "Gate.io",
            exchange_id::BINANCE => "Binance",
            exchange_id::OKX => "OKX",
            exchange_id::BYBIT => "Bybit",
            _ => "Unknown",
        }
    }
}

impl Default for ArbitrageEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Arbitrage engine statistics.
#[derive(Debug, Clone)]
pub struct ArbStats {
    pub opportunities_detected: u64,
    pub trades_executed: u64,
    pub total_pnl_usdt: f64,
    pub enabled: bool,
    pub min_spread_bps: i64,
    pub symbols_monitored: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn now_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    #[test]
    fn test_arb_detection() {
        let mut engine = ArbitrageEngine::new();
        engine.set_enabled(true);
        engine.register_symbol(0, "BTC_USDT");
        
        let now = now_ns();
        
        // Gate.io: bid=50000, ask=50010
        engine.update_price(0, ExchangePrice {
            exchange_id: exchange_id::GATEIO,
            bid: 5000000000000,  // 50000.00
            ask: 5001000000000,  // 50010.00
            bid_size: 100000000, // 1 BTC
            ask_size: 100000000,
            timestamp_ns: now,
        });
        
        // Binance: bid=50020, ask=50030 (higher prices)
        engine.update_price(0, ExchangePrice {
            exchange_id: exchange_id::BINANCE,
            bid: 5002000000000,  // 50020.00
            ask: 5003000000000,  // 50030.00
            bid_size: 100000000,
            ask_size: 100000000,
            timestamp_ns: now,
        });
        
        // Should detect arb: buy on Gate.io at 50010, sell on Binance at 50020
        let opp = engine.scan(0);
        assert!(opp.is_some(), "Should detect arbitrage opportunity");
        
        let opp = opp.unwrap();
        assert_eq!(opp.buy_exchange, exchange_id::GATEIO);
        assert_eq!(opp.sell_exchange, exchange_id::BINANCE);
        assert!(opp.spread_bps > 0);
    }
    
    #[test]
    fn test_no_arb_same_exchange() {
        let mut engine = ArbitrageEngine::new();
        engine.set_enabled(true);
        engine.register_symbol(0, "BTC_USDT");
        
        let now = now_ns();
        
        // Only Gate.io prices
        engine.update_price(0, ExchangePrice {
            exchange_id: exchange_id::GATEIO,
            bid: 5000000000000,
            ask: 5001000000000,
            bid_size: 100000000,
            ask_size: 100000000,
            timestamp_ns: now,
        });
        
        // No arb possible with single exchange
        assert!(engine.scan(0).is_none());
    }
}
