//! Persistent state storage using SQLite (Feature 4).
//!
//! Provides crash-resilient position and order tracking across engine restarts.
//! On startup, persisted positions are loaded and reconciled with exchange state.
//! On position open/close/update, changes are written to SQLite.

use rusqlite::{Connection, params};
use std::sync::Mutex;
use tracing::{info, warn, error};

/// A persisted position record.
#[derive(Debug, Clone)]
pub struct PersistedPosition {
    pub symbol: String,
    pub side: String,       // "long" or "short"
    pub entry_price: f64,
    pub size: i64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub leverage: i32,
    pub opened_at: i64,     // unix timestamp seconds
    pub closed_at: Option<i64>,
}

/// A persisted order record.
#[derive(Debug, Clone)]
pub struct PersistedOrder {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub status: String,     // "submitted", "filled", "cancelled", "rejected"
    pub price: f64,
    pub size: i64,
    pub created_at: i64,
    pub signal_tag: String,
}

/// SQLite-based persistent state store.
/// Wrapped in a Mutex because rusqlite::Connection is not Sync.
pub struct StateStore {
    conn: Mutex<Connection>,
}

impl StateStore {
    /// Open or create the SQLite database at the given path.
    pub fn open(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open SQLite database at {}: {}", db_path, e))?;

        // Enable WAL mode for better concurrent read performance
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(|e| format!("Failed to set SQLite pragmas: {}", e))?;

        let store = Self { conn: Mutex::new(conn) };
        store.create_tables()?;

        info!("[state-store] SQLite database opened at {}", db_path);
        Ok(store)
    }

    /// Create the required tables if they don't exist.
    fn create_tables(&self) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size INTEGER NOT NULL,
                stop_loss REAL NOT NULL DEFAULT 0.0,
                take_profit REAL NOT NULL DEFAULT 0.0,
                leverage INTEGER NOT NULL DEFAULT 1,
                opened_at INTEGER NOT NULL,
                closed_at INTEGER,
                UNIQUE(symbol, opened_at)
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'submitted',
                price REAL NOT NULL,
                size INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                signal_tag TEXT NOT NULL DEFAULT '',
                UNIQUE(order_id)
            );

            CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
            CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(closed_at) WHERE closed_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            "
        ).map_err(|e| format!("Failed to create tables: {}", e))?;

        info!("[state-store] Database tables initialized");
        Ok(())
    }

    /// Insert a new open position into the database.
    pub fn insert_position(&self, pos: &PersistedPosition) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT OR REPLACE INTO positions (symbol, side, entry_price, size, stop_loss, take_profit, leverage, opened_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                pos.symbol,
                pos.side,
                pos.entry_price,
                pos.size,
                pos.stop_loss,
                pos.take_profit,
                pos.leverage,
                pos.opened_at,
            ],
        ).map_err(|e| format!("Failed to insert position: {}", e))?;
        Ok(())
    }

    /// Update SL/TP for an open position.
    pub fn update_position_sl_tp(&self, symbol: &str, stop_loss: f64, take_profit: f64) -> Result<(), String> {
        let mut updates = Vec::new();
        let mut update_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if stop_loss > 0.0 {
            updates.push("stop_loss = ?");
            update_params.push(Box::new(stop_loss));
        }
        if take_profit > 0.0 {
            updates.push("take_profit = ?");
            update_params.push(Box::new(take_profit));
        }

        if updates.is_empty() {
            return Ok(());
        }

        update_params.push(Box::new(symbol.to_string()));

        let sql = format!(
            "UPDATE positions SET {} WHERE symbol = ? AND closed_at IS NULL",
            updates.join(", ")
        );

        let params_refs: Vec<&dyn rusqlite::types::ToSql> = update_params.iter().map(|p| p.as_ref()).collect();

        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        conn.execute(&sql, params_refs.as_slice())
            .map_err(|e| format!("Failed to update position SL/TP: {}", e))?;
        Ok(())
    }

    /// Mark a position as closed.
    pub fn close_position(&self, symbol: &str) -> Result<(), String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        conn.execute(
            "UPDATE positions SET closed_at = ?1 WHERE symbol = ?2 AND closed_at IS NULL",
            params![now, symbol],
        ).map_err(|e| format!("Failed to close position: {}", e))?;
        Ok(())
    }

    /// Load all open (unclosed) positions from the database.
    pub fn load_open_positions(&self) -> Result<Vec<PersistedPosition>, String> {
        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        let mut stmt = conn.prepare(
            "SELECT symbol, side, entry_price, size, stop_loss, take_profit, leverage, opened_at
             FROM positions WHERE closed_at IS NULL ORDER BY opened_at ASC"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let positions = stmt.query_map([], |row| {
            Ok(PersistedPosition {
                symbol: row.get(0)?,
                side: row.get(1)?,
                entry_price: row.get(2)?,
                size: row.get(3)?,
                stop_loss: row.get(4)?,
                take_profit: row.get(5)?,
                leverage: row.get(6)?,
                opened_at: row.get(7)?,
                closed_at: None,
            })
        }).map_err(|e| format!("Failed to query positions: {}", e))?;

        let mut result = Vec::new();
        for pos in positions {
            match pos {
                Ok(p) => result.push(p),
                Err(e) => warn!("[state-store] Skipping malformed position row: {}", e),
            }
        }
        Ok(result)
    }

    /// Insert or update an order record.
    pub fn upsert_order(&self, order: &PersistedOrder) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT OR REPLACE INTO orders (order_id, symbol, side, status, price, size, created_at, signal_tag)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                order.order_id,
                order.symbol,
                order.side,
                order.status,
                order.price,
                order.size,
                order.created_at,
                order.signal_tag,
            ],
        ).map_err(|e| format!("Failed to upsert order: {}", e))?;
        Ok(())
    }

    /// Update the status of an order.
    pub fn update_order_status(&self, order_id: &str, status: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        conn.execute(
            "UPDATE orders SET status = ?1 WHERE order_id = ?2",
            params![status, order_id],
        ).map_err(|e| format!("Failed to update order status: {}", e))?;
        Ok(())
    }

    /// Close all open positions for symbols NOT present on the exchange.
    /// This handles positions that were closed while the engine was down.
    pub fn reconcile_with_exchange(&self, exchange_symbols: &[String]) -> Result<usize, String> {
        let open_positions = self.load_open_positions()?;
        let mut closed_count = 0;

        for pos in &open_positions {
            if !exchange_symbols.contains(&pos.symbol) {
                info!("[state-store] Position {} no longer on exchange — marking closed", pos.symbol);
                self.close_position(&pos.symbol)?;
                closed_count += 1;
            }
        }

        Ok(closed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_store_roundtrip() {
        let store = StateStore::open(":memory:").expect("open in-memory db");

        let pos = PersistedPosition {
            symbol: "BTC_USDT".to_string(),
            side: "long".to_string(),
            entry_price: 50000.0,
            size: 10,
            stop_loss: 49000.0,
            take_profit: 52000.0,
            leverage: 5,
            opened_at: 1700000000,
            closed_at: None,
        };

        store.insert_position(&pos).expect("insert");
        let loaded = store.load_open_positions().expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].symbol, "BTC_USDT");
        assert_eq!(loaded[0].entry_price, 50000.0);

        // Update SL/TP
        store.update_position_sl_tp("BTC_USDT", 49500.0, 53000.0).expect("update");

        // Close
        store.close_position("BTC_USDT").expect("close");
        let loaded = store.load_open_positions().expect("load after close");
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_order_upsert() {
        let store = StateStore::open(":memory:").expect("open in-memory db");

        let order = PersistedOrder {
            order_id: "ord-123".to_string(),
            symbol: "BTC_USDT".to_string(),
            side: "buy".to_string(),
            status: "submitted".to_string(),
            price: 50000.0,
            size: 10,
            created_at: 1700000000,
            signal_tag: "strategy_signal".to_string(),
        };

        store.upsert_order(&order).expect("upsert");
        store.update_order_status("ord-123", "filled").expect("update status");
    }

    #[test]
    fn test_reconcile_with_exchange() {
        let store = StateStore::open(":memory:").expect("open in-memory db");

        store.insert_position(&PersistedPosition {
            symbol: "BTC_USDT".to_string(),
            side: "long".to_string(),
            entry_price: 50000.0,
            size: 10,
            stop_loss: 0.0,
            take_profit: 0.0,
            leverage: 1,
            opened_at: 1700000000,
            closed_at: None,
        }).expect("insert");

        store.insert_position(&PersistedPosition {
            symbol: "ETH_USDT".to_string(),
            side: "short".to_string(),
            entry_price: 3000.0,
            size: 5,
            stop_loss: 0.0,
            take_profit: 0.0,
            leverage: 1,
            opened_at: 1700000001,
            closed_at: None,
        }).expect("insert");

        // Only BTC_USDT is still on exchange
        let exchange_symbols = vec!["BTC_USDT".to_string()];
        let closed = store.reconcile_with_exchange(&exchange_symbols).expect("reconcile");
        assert_eq!(closed, 1);

        let open = store.load_open_positions().expect("load");
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].symbol, "BTC_USDT");
    }
}
