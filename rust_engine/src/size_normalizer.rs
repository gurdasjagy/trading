//! BUG 3 FIX: Universal USDT-to-Contracts Size Normalization
//!
//! Converts USDT notional values to exchange-specific contract quantities.
//! Gate.io uses integer contracts with a quanto_multiplier, while Binance
//! and Bybit use fractional base-asset quantities.
//!
//! Without this layer, `size=1` (e.g. 1 Gate.io BTC_USDT contract = 0.0001 BTC)
//! would be interpreted as 1 whole BTC on Binance ($70,000+), causing
//! "insufficient balance". The quanto_multiplier varies per contract
//! (e.g. BTC_USDT=0.0001, ETH_USDT=0.01) and must be fetched dynamically.

use tracing::{info, warn};

use crate::multi_exchange::global_book::ExchangeId;

/// Universal size normalizer for cross-exchange order sizing.
pub struct SizeNormalizer;

impl SizeNormalizer {
    /// Convert a USDT notional value to exchange-specific contract quantity.
    ///
    /// - **Gate.io**: `contracts = usdt_notional / (quanto_multiplier * price)`
    /// - **Binance**: `quantity = usdt_notional / price` (in base currency units)
    /// - **Bybit**:   `quantity = usdt_notional / price` (in base currency units)
    ///
    /// The result is snapped to the exchange's `step_size` and clamped to `min_qty`.
    pub fn usdt_to_exchange_qty(
        exchange: ExchangeId,
        usdt_notional: f64,
        price: f64,
        quanto_multiplier: f64,
        step_size: f64,
        min_qty: f64,
    ) -> Result<f64, String> {
        if price <= 0.0 {
            return Err("Invalid price: must be > 0".into());
        }
        if usdt_notional <= 0.0 {
            return Err("Invalid notional: must be > 0".into());
        }

        let raw_qty = match exchange {
            ExchangeId::GateIo => {
                if quanto_multiplier > 0.0 {
                    usdt_notional / (quanto_multiplier * price)
                } else {
                    warn!("[size-normalizer] Gate.io quanto_multiplier is 0, falling back to direct division");
                    usdt_notional / price
                }
            }
            ExchangeId::Binance | ExchangeId::Bybit => {
                usdt_notional / price
            }
        };

        // Snap to step size (floor to avoid exceeding balance)
        let snapped = if step_size > 0.0 {
            (raw_qty / step_size).floor() * step_size
        } else {
            raw_qty
        };

        if snapped < min_qty {
            return Err(format!(
                "Qty {:.8} below minimum {} for {} (notional=${:.2}, price={:.2})",
                snapped, min_qty, exchange.name(), usdt_notional, price
            ));
        }

        info!(
            "[size-normalizer] {} ${:.2} @ {:.2} -> qty={:.8} (raw={:.8}, step={}, min={})",
            exchange.name(), usdt_notional, price, snapped, raw_qty, step_size, min_qty
        );

        Ok(snapped)
    }

    /// Convert Gate.io integer contracts to equivalent USDT notional.
    ///
    /// `notional = contracts * quanto_multiplier * price`
    pub fn gateio_contracts_to_usdt(
        contracts: i64,
        quanto_multiplier: f64,
        price: f64,
    ) -> f64 {
        contracts.abs() as f64 * quanto_multiplier * price
    }

    /// Convert Gate.io contracts to Binance/Bybit fractional quantity.
    ///
    /// Useful when the strategy engine emits Gate.io contract counts and
    /// we need to convert to base-asset qty for other exchanges.
    pub fn gateio_contracts_to_base_qty(
        contracts: i64,
        quanto_multiplier: f64,
        step_size: f64,
    ) -> f64 {
        let raw_qty = contracts.abs() as f64 * quanto_multiplier;
        if step_size > 0.0 {
            (raw_qty / step_size).floor() * step_size
        } else {
            raw_qty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateio_btc_sizing() {
        // Gate.io BTC_USDT example: quanto_multiplier = 0.0001
        // $100 notional at $70,000 = 100 / (0.0001 * 70000) = 14.28 contracts
        // NOTE: quanto_multiplier varies per contract (e.g. ETH_USDT=0.01)
        let qty = SizeNormalizer::usdt_to_exchange_qty(
            ExchangeId::GateIo, 100.0, 70000.0, 0.0001, 1.0, 1.0,
        ).unwrap();
        assert_eq!(qty, 14.0); // Floored to integer contracts
    }

    #[test]
    fn test_binance_btc_sizing() {
        // Binance BTCUSDT: direct division, step_size = 0.001
        // $100 notional at $70,000 = 0.001428... -> snapped to 0.001
        let qty = SizeNormalizer::usdt_to_exchange_qty(
            ExchangeId::Binance, 100.0, 70000.0, 1.0, 0.001, 0.001,
        ).unwrap();
        assert_eq!(qty, 0.001);
    }

    #[test]
    fn test_bybit_eth_sizing() {
        // Bybit ETHUSDT: direct division, step_size = 0.01
        // $500 notional at $3500 = 0.1428... -> snapped to 0.14
        let qty = SizeNormalizer::usdt_to_exchange_qty(
            ExchangeId::Bybit, 500.0, 3500.0, 1.0, 0.01, 0.01,
        ).unwrap();
        assert!((qty - 0.14).abs() < 1e-10);
    }

    #[test]
    fn test_below_minimum() {
        let result = SizeNormalizer::usdt_to_exchange_qty(
            ExchangeId::Binance, 1.0, 70000.0, 1.0, 0.001, 0.001,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_price() {
        let result = SizeNormalizer::usdt_to_exchange_qty(
            ExchangeId::Binance, 100.0, 0.0, 1.0, 0.001, 0.001,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gateio_contracts_to_usdt() {
        let notional = SizeNormalizer::gateio_contracts_to_usdt(100, 0.0001, 70000.0);
        assert!((notional - 700.0).abs() < 1e-6);
    }

    #[test]
    fn test_gateio_contracts_to_base_qty() {
        // Example: 100 Gate.io BTC_USDT contracts * 0.0001 multiplier = 0.01 BTC
        let qty = SizeNormalizer::gateio_contracts_to_base_qty(100, 0.0001, 0.001);
        assert!((qty - 0.01).abs() < 1e-10);
    }
}
