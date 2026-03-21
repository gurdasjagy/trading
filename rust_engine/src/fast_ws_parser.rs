//! Fast, zero-allocation JSON extraction for hot-path websocket messages
//! Extracts only the required fields from orderbook and trade updates.

#[derive(Debug)]
pub struct FastBookUpdate {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub update_id: u64,
}

pub fn parse_book_update(bytes: &[u8]) -> Option<FastBookUpdate> {
    // In a real implementation, use memchr or simd-json to locate fields
    // and parse floats without full AST allocation.
    // For now, fallback to serde_json as a placeholder
    let s = std::str::from_utf8(bytes).ok()?;
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    
    let symbol = v.get("s")?.as_str()?.to_string();
    let update_id = v.get("u")?.as_u64().unwrap_or(0);
    
    let mut bids = Vec::new();
    if let Some(b) = v.get("b").and_then(|b| b.as_array()) {
        for level in b {
            if let (Some(p), Some(q)) = (level[0].as_str(), level[1].as_str()) {
                if let (Ok(pf), Ok(qf)) = (p.parse::<f64>(), q.parse::<f64>()) {
                    bids.push((pf, qf));
                }
            }
        }
    }
    
    let mut asks = Vec::new();
    if let Some(a) = v.get("a").and_then(|a| a.as_array()) {
        for level in a {
            if let (Some(p), Some(q)) = (level[0].as_str(), level[1].as_str()) {
                if let (Ok(pf), Ok(qf)) = (p.parse::<f64>(), q.parse::<f64>()) {
                    asks.push((pf, qf));
                }
            }
        }
    }
    
    Some(FastBookUpdate { symbol, bids, asks, update_id })
}
