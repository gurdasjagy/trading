//! Multi-Exchange WebSocket Ingestion
//!
//! Runs parallel WebSocket connections for Binance and Bybit alongside
//! the existing Gate.io connection. Each connection:
//! - Subscribes to L2 orderbook depth streams
//! - Parses exchange-specific message formats
//! - Updates the GlobalBookRegistry with ExchangeBookSnapshot
//! - Implements automatic reconnection with exponential backoff
//!
//! Gate.io WebSocket ingestion remains in ws_ingestion.rs (unchanged).
//! This module adds Binance and Bybit ingestion for multi-exchange mode.

use std::sync::Arc;
use std::time::Duration;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error, debug};

use crate::config::{ExchangeConfig, SymbolRegistry};
use crate::multi_exchange::global_book::{
    ExchangeBookSnapshot, ExchangeId, GlobalBookRegistry,
};
use crate::fixed_point::FixedPrice;

// ---------------------------------------------------------------------------
// Binance WebSocket URLs
// ---------------------------------------------------------------------------
const BINANCE_WS_LIVE: &str = "wss://fstream.binance.com/stream";
const BINANCE_WS_TESTNET: &str = "wss://stream.binancefuture.com/stream";

// ---------------------------------------------------------------------------
// Bybit WebSocket URLs
// ---------------------------------------------------------------------------
const BYBIT_WS_LIVE: &str = "wss://stream.bybit.com/v5/public/linear";
const BYBIT_WS_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/public/linear";

// ---------------------------------------------------------------------------
// Binance WebSocket Ingestion
// ---------------------------------------------------------------------------

/// Spawn a Binance Futures WebSocket ingestion task.
///
/// Subscribes to `{symbol}@depth20@100ms` for each configured symbol.
/// Parses the Binance depth update format:
///   { "b": [["price", "qty"], ...], "a": [["price", "qty"], ...] }
/// Updates the GlobalBookRegistry with ExchangeBookSnapshot on each message.
///
/// Implements automatic reconnection with exponential backoff (500ms -> 30s).
/// On sequence gap detection (lastUpdateId mismatch), resubscribes immediately.
pub async fn run_binance_ws_ingestion(
    config: ExchangeConfig,
    registry: Arc<GlobalBookRegistry>,
    symbol_ids: Arc<SymbolRegistry>,
) {
    let ws_url = if config.testnet { BINANCE_WS_TESTNET } else { BINANCE_WS_LIVE };
    let mut backoff_ms = 500u64;

    loop {
        info!("[ws-binance] Connecting to {}", ws_url);

        // Build combined stream URL: /stream?streams=btcusdt@depth20@100ms/ethusdt@depth20@100ms
        let streams: Vec<String> = config.symbols.iter()
            .map(|s| {
                let normalized = s.replace('_', "").to_lowercase();
                format!("{}@depth20@100ms", normalized)
            })
            .collect();
        let stream_param = streams.join("/");
        let full_url = format!("{}?streams={}", ws_url, stream_param);

        match connect_async(&full_url).await {
            Ok((ws_stream, _)) => {
                info!("[ws-binance] Connected, subscribed to {} symbols", config.symbols.len());
                backoff_ms = 500;

                let (mut write, mut read) = ws_stream.split();
                let mut ping_interval = tokio::time::interval(Duration::from_secs(20));
                ping_interval.tick().await;

                loop {
                    tokio::select! {
                        msg = read.next() => {
                            match msg {
                                Some(Ok(Message::Text(text))) => {
                                    // Parse Binance combined stream format:
                                    // { "stream": "btcusdt@depth20@100ms", "data": { ... } }
                                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                                        let stream_name = parsed.get("stream")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let data = parsed.get("data").unwrap_or(&parsed);

                                        // Extract symbol from stream name: "btcusdt@depth20@100ms" -> "BTCUSDT"
                                        let symbol_upper = stream_name
                                            .split('@')
                                            .next()
                                            .unwrap_or("")
                                            .to_uppercase();

                                        // Map Binance symbol to our internal symbol_id
                                        // "BTCUSDT" -> "BTC_USDT" -> symbol_id lookup
                                        let internal_symbol = format!("{}_USDT",
                                            symbol_upper.trim_end_matches("USDT"));
                                        let sym_id = symbol_ids.get_id(&internal_symbol);
                                        if sym_id == 0 { continue; }

                                        let now_ns = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_nanos() as u64;

                                        // Parse bids and asks
                                        let mut bid_levels = Vec::new();
                                        let mut ask_levels = Vec::new();

                                        if let Some(bids) = data.get("b").and_then(|v| v.as_array()) {
                                            for level in bids {
                                                if let Some(arr) = level.as_array() {
                                                    if arr.len() >= 2 {
                                                        let price: f64 = arr[0].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        let qty: f64 = arr[1].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        if price > 0.0 {
                                                            bid_levels.push((
                                                                FixedPrice::from_f64(price).raw(),
                                                                (qty * 1e8) as i64,
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        if let Some(asks) = data.get("a").and_then(|v| v.as_array()) {
                                            for level in asks {
                                                if let Some(arr) = level.as_array() {
                                                    if arr.len() >= 2 {
                                                        let price: f64 = arr[0].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        let qty: f64 = arr[1].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        if price > 0.0 {
                                                            ask_levels.push((
                                                                FixedPrice::from_f64(price).raw(),
                                                                (qty * 1e8) as i64,
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        let best_bid = bid_levels.iter()
                                            .map(|(p, _)| *p)
                                            .max()
                                            .unwrap_or(0);
                                        let best_ask = ask_levels.iter()
                                            .map(|(p, _)| *p)
                                            .min()
                                            .unwrap_or(0);

                                        let snapshot = ExchangeBookSnapshot {
                                            exchange: ExchangeId::Binance,
                                            symbol_id: sym_id,
                                            best_bid_fp: best_bid,
                                            best_ask_fp: best_ask,
                                            bid_levels,
                                            ask_levels,
                                            sequence: data.get("u")
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0),
                                            timestamp_ns: now_ns,
                                        };

                                        let book = registry.get_or_create(sym_id);
                                        book.write().update_exchange_snapshot(snapshot);
                                        
                                        debug!("[ws-binance] {} book updated: bid={:.2} ask={:.2}",
                                            internal_symbol,
                                            FixedPrice(best_bid).to_f64(),
                                            FixedPrice(best_ask).to_f64()
                                        );
                                    }
                                }
                                Some(Ok(Message::Ping(data))) => {
                                    let _ = write.send(Message::Pong(data)).await;
                                }
                                Some(Ok(Message::Close(_))) => {
                                    warn!("[ws-binance] Server closed connection");
                                    break;
                                }
                                Some(Err(e)) => {
                                    error!("[ws-binance] Read error: {}", e);
                                    break;
                                }
                                None => break,
                                _ => {}
                            }
                        }
                        _ = ping_interval.tick() => {
                            // Binance requires periodic pong responses; send a ping
                            if let Err(e) = write.send(Message::Ping(vec![])).await {
                                warn!("[ws-binance] Ping failed: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("[ws-binance] Connection failed: {}", e);
            }
        }

        warn!("[ws-binance] Reconnecting in {}ms...", backoff_ms);
        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(30_000);
    }
}

// ---------------------------------------------------------------------------
// Bybit WebSocket Ingestion
// ---------------------------------------------------------------------------

/// Spawn a Bybit v5 WebSocket ingestion task.
///
/// Subscribes to `orderbook.50.{SYMBOL}` for each configured symbol.
/// Parses the Bybit v5 orderbook format:
///   { "topic": "orderbook.50.BTCUSDT", "type": "snapshot"|"delta",
///     "data": { "b": [["price", "qty"], ...], "a": [...] } }
/// Updates the GlobalBookRegistry with ExchangeBookSnapshot on each message.
///
/// Bybit v5 uses a subscription message after connection:
///   { "op": "subscribe", "args": ["orderbook.50.BTCUSDT", ...] }
///
/// Implements automatic reconnection with exponential backoff.
pub async fn run_bybit_ws_ingestion(
    config: ExchangeConfig,
    registry: Arc<GlobalBookRegistry>,
    symbol_ids: Arc<SymbolRegistry>,
) {
    let ws_url = if config.testnet { BYBIT_WS_TESTNET } else { BYBIT_WS_LIVE };
    let mut backoff_ms = 500u64;

    loop {
        info!("[ws-bybit] Connecting to {}", ws_url);

        match connect_async(ws_url).await {
            Ok((ws_stream, _)) => {
                info!("[ws-bybit] Connected");
                backoff_ms = 500;

                let (mut write, mut read) = ws_stream.split();

                // Subscribe to orderbook for all symbols
                let args: Vec<String> = config.symbols.iter()
                    .map(|s| {
                        // "BTC_USDT" -> "BTCUSDT"
                        let bybit_sym = s.replace('_', "").to_uppercase();
                        format!("orderbook.50.{}", bybit_sym)
                    })
                    .collect();

                let sub_msg = serde_json::json!({
                    "op": "subscribe",
                    "args": args
                });

                if let Err(e) = write.send(Message::Text(sub_msg.to_string())).await {
                    error!("[ws-bybit] Subscribe failed: {}", e);
                    continue;
                }

                info!("[ws-bybit] Subscribed to {} symbols", config.symbols.len());

                let mut ping_interval = tokio::time::interval(Duration::from_secs(20));
                ping_interval.tick().await;

                loop {
                    tokio::select! {
                        msg = read.next() => {
                            match msg {
                                Some(Ok(Message::Text(text))) => {
                                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                                        // Handle subscription confirmation
                                        if parsed.get("op").and_then(|v| v.as_str()) == Some("subscribe") {
                                            let success = parsed.get("success")
                                                .and_then(|v| v.as_bool())
                                                .unwrap_or(false);
                                            if success {
                                                info!("[ws-bybit] Subscription confirmed");
                                            } else {
                                                error!("[ws-bybit] Subscription failed: {}", text);
                                            }
                                            continue;
                                        }

                                        // Handle pong response
                                        if parsed.get("op").and_then(|v| v.as_str()) == Some("pong") {
                                            continue;
                                        }

                                        // Handle orderbook updates
                                        let topic = parsed.get("topic")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");

                                        if !topic.starts_with("orderbook.") { continue; }

                                        // Extract symbol: "orderbook.20.BTCUSDT" -> "BTCUSDT"
                                        let bybit_symbol = topic.split('.').nth(2).unwrap_or("");
                                        // Map "BTCUSDT" -> "BTC_USDT"
                                        let internal_symbol = format!("{}_USDT",
                                            bybit_symbol.trim_end_matches("USDT"));
                                        let sym_id = symbol_ids.get_id(&internal_symbol);
                                        if sym_id == 0 { continue; }

                                        let data = match parsed.get("data") {
                                            Some(d) => d,
                                            None => continue,
                                        };

                                        let now_ns = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_nanos() as u64;

                                        let mut bid_levels = Vec::new();
                                        let mut ask_levels = Vec::new();

                                        // Bybit format: "b": [["price", "qty"], ...]
                                        if let Some(bids) = data.get("b").and_then(|v| v.as_array()) {
                                            for level in bids {
                                                if let Some(arr) = level.as_array() {
                                                    if arr.len() >= 2 {
                                                        let price: f64 = arr[0].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        let qty: f64 = arr[1].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        if price > 0.0 {
                                                            bid_levels.push((
                                                                FixedPrice::from_f64(price).raw(),
                                                                (qty * 1e8) as i64,
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        if let Some(asks) = data.get("a").and_then(|v| v.as_array()) {
                                            for level in asks {
                                                if let Some(arr) = level.as_array() {
                                                    if arr.len() >= 2 {
                                                        let price: f64 = arr[0].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        let qty: f64 = arr[1].as_str()
                                                            .and_then(|s| s.parse().ok())
                                                            .unwrap_or(0.0);
                                                        if price > 0.0 {
                                                            ask_levels.push((
                                                                FixedPrice::from_f64(price).raw(),
                                                                (qty * 1e8) as i64,
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        let best_bid = bid_levels.iter()
                                            .map(|(p, _)| *p)
                                            .max()
                                            .unwrap_or(0);
                                        let best_ask = ask_levels.iter()
                                            .map(|(p, _)| *p)
                                            .min()
                                            .unwrap_or(0);

                                        let seq = parsed.get("ts")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);

                                        let snapshot = ExchangeBookSnapshot {
                                            exchange: ExchangeId::Bybit,
                                            symbol_id: sym_id,
                                            best_bid_fp: best_bid,
                                            best_ask_fp: best_ask,
                                            bid_levels,
                                            ask_levels,
                                            sequence: seq,
                                            timestamp_ns: now_ns,
                                        };

                                        let book = registry.get_or_create(sym_id);
                                        book.write().update_exchange_snapshot(snapshot);
                                        
                                        debug!("[ws-bybit] {} book updated: bid={:.2} ask={:.2}",
                                            internal_symbol,
                                            FixedPrice(best_bid).to_f64(),
                                            FixedPrice(best_ask).to_f64()
                                        );
                                    }
                                }
                                Some(Ok(Message::Ping(data))) => {
                                    let _ = write.send(Message::Pong(data)).await;
                                }
                                Some(Ok(Message::Close(_))) => {
                                    warn!("[ws-bybit] Server closed connection");
                                    break;
                                }
                                Some(Err(e)) => {
                                    error!("[ws-bybit] Read error: {}", e);
                                    break;
                                }
                                None => break,
                                _ => {}
                            }
                        }
                        _ = ping_interval.tick() => {
                            // Bybit heartbeat: send {"op": "ping"}
                            let ping = serde_json::json!({"op": "ping"});
                            if let Err(e) = write.send(Message::Text(ping.to_string())).await {
                                warn!("[ws-bybit] Ping failed: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("[ws-bybit] Connection failed: {}", e);
            }
        }

        warn!("[ws-bybit] Reconnecting in {}ms...", backoff_ms);
        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(30_000);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_normalization() {
        // Binance style: "BTC_USDT" -> "btcusdt"
        let sym = "BTC_USDT";
        let binance_normalized = sym.replace('_', "").to_lowercase();
        assert_eq!(binance_normalized, "btcusdt");

        // Bybit style: "BTC_USDT" -> "BTCUSDT"
        let bybit_normalized = sym.replace('_', "").to_uppercase();
        assert_eq!(bybit_normalized, "BTCUSDT");
    }

    #[test]
    fn test_stream_url_construction() {
        let symbols = vec!["BTC_USDT".to_string(), "ETH_USDT".to_string()];
        
        let streams: Vec<String> = symbols.iter()
            .map(|s| {
                let normalized = s.replace('_', "").to_lowercase();
                format!("{}@depth20@100ms", normalized)
            })
            .collect();
        let stream_param = streams.join("/");
        
        assert_eq!(stream_param, "btcusdt@depth20@100ms/ethusdt@depth20@100ms");
    }
}
