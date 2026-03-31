# DEFINITIVE TRADING BOT FIX — COMPLETE CODING AGENT PROMPT
## Based on full log analysis + exhaustive code audit of the updated codebase

---

## EXECUTIVE SUMMARY: WHY ZERO TRADES STILL

From the logs, every 30-second latency report shows:
```
tick_to_book:   n=0
book_to_signal: n=0
signal_to_order:n=0
```
The entire pipeline is dead. Auth works ($934.22 balance confirmed). WS connects. But **not one byte of market data reaches the book builder**. The Python Alpha Oracle IS generating signals (ETH 7/9 = 78% confluence, SOL 7/8 = 88% — both above 75% threshold) but Rust never reads them.

There are **two independent total failures** running in parallel:

1. **Rust WS parser is broken** — Zero ticks reach the book. Zero Rust-native signals ever fire.
2. **Python to Rust signal bridge is completely unplumbed** — Signals are written to SHM but Rust never reads them.

Fix BOTH. Either one alone is enough to get trades. Together they create a resilient dual-path system.

---

# PART 1 — CRITICAL BUG FIXES (DO THESE FIRST, IN ORDER)

---

## BUG #1 — ROOT CAUSE #1: Gate.io WS Message Format Mismatch

**File:** `rust_engine/src/main.rs` — function `ws_connect_and_ingest_gateio`
**Evidence:** `tick_to_book: n=0` for entire run

**The problem:** Gate.io futures WebSocket sends bid/ask levels as JSON objects:
```json
{"bids": [{"p": "84000.1", "s": 100}], "asks": [{"p": "84001.2", "s": 50}]}
```
The code tries to parse them as JSON arrays:
```rust
if let Some(arr) = level.as_array() {   // ALWAYS None for Gate.io objects
    let price = json_to_f64(&arr[0])... // Never reached
```
`level.as_array()` returns `None` for every single bid/ask entry. Every update silently drops. No panic, no warning — just silence. This is why `tick_to_book: n=0` for the entire run.

Gate.io also sends `event: "all"` for full snapshots and `event: "update"` for deltas. The existing code never handles the "all" event to clear stale book state. Additionally, the `futures.tickers` channel provides fast BBO updates that can bootstrap the book faster.

**REPLACE the entire `ws_connect_and_ingest_gateio` function** (from `async fn ws_connect_and_ingest_gateio(` to its closing `}`) with:

```rust
async fn ws_connect_and_ingest_gateio(
    ring: &'static SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>,
    trades_ring: &'static SpscRingBuffer<spsc::TradeEvent, WS_TO_STRATEGY_TRADES_CAPACITY>,
    config: &ExchangeConfig,
    registry: &SymbolRegistry,
    drop_count: &mut u64,
    msg_count: &mut u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    let (ws_stream, _) = connect_async(config.ws_url.as_str()).await?;
    let (write, mut read) = ws_stream.split();
    let write = Arc::new(tokio::sync::Mutex::new(write));

    info!("[ws-gateio] Connected, subscribing to {} symbols", config.symbols.len());

    // Subscribe to all channels for each symbol
    {
        let mut w = write.lock().await;
        for symbol in &config.symbols {
            // Full orderbook (20 levels, 100ms updates)
            // Gate.io sends event:"all" for snapshots, event:"update" for deltas
            let sub_book = serde_json::json!({
                "time": now_secs(),
                "channel": "futures.order_book",
                "event": "subscribe",
                "payload": [symbol, "20", "100ms"]
            });
            w.send(Message::Text(sub_book.to_string())).await?;

            // Trades feed (for VPIN and CVD)
            let sub_trades = serde_json::json!({
                "time": now_secs(),
                "channel": "futures.trades",
                "event": "subscribe",
                "payload": [symbol]
            });
            w.send(Message::Text(sub_trades.to_string())).await?;

            // Tickers — fast BBO updates, bootstraps book before full snapshot arrives
            let sub_ticker = serde_json::json!({
                "time": now_secs(),
                "channel": "futures.tickers",
                "event": "subscribe",
                "payload": [symbol]
            });
            w.send(Message::Text(sub_ticker.to_string())).await?;
        }
    }

    let mut ping_interval = tokio::time::interval(Duration::from_secs(15));
    ping_interval.tick().await; // consume immediate first tick

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(msg)) => {
                        match msg {
                            Message::Text(text) => {
                                let recv_ns = now_ns();
                                *msg_count += 1;

                                let parsed: serde_json::Value = match serde_json::from_str(&text) {
                                    Ok(v) => v,
                                    Err(_) => continue,
                                };

                                let channel = parsed.get("channel").and_then(|v| v.as_str()).unwrap_or("");
                                let event = parsed.get("event").and_then(|v| v.as_str()).unwrap_or("");

                                // Skip confirmations, pongs, and errors
                                if event == "subscribe" || event == "unsubscribe" || event == "pong" {
                                    debug!("[ws-gateio] Confirmed: channel={}", channel);
                                    continue;
                                }
                                if event == "error" {
                                    warn!("[ws-gateio] WS error msg: {}", &text[..text.len().min(300)]);
                                    continue;
                                }

                                let result = match parsed.get("result") {
                                    Some(r) if !r.is_null() => r,
                                    _ => continue,
                                };

                                match channel {
                                    "futures.order_book" => {
                                        // BUG #1 FIX: Gate.io sends {"p":"price_str","s":qty_int} objects
                                        // event:"all" = full snapshot (must clear book first)
                                        // event:"update" = incremental delta
                                        let is_snapshot = event == "all";

                                        let contract = result.get("contract")
                                            .or_else(|| result.get("s"))
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let sym_id = registry.get_id(contract);
                                        if sym_id == 0 { continue; }

                                        // For "all" snapshots: signal book builder to clear stale data
                                        if is_snapshot {
                                            let sentinel = RawBookUpdate {
                                                symbol_id: sym_id,
                                                side: spsc::side::BID,
                                                update_type: spsc::update_type::SNAPSHOT_START,
                                                _pad: [0; 4],
                                                price: 0,
                                                qty: 0,
                                                sequence: *msg_count,
                                                recv_ns,
                                                snapshot_count: 0,
                                                _pad2: [0; 4],
                                            };
                                            let _ = ring.try_push(sentinel);
                                        }

                                        // Parse bids: Gate.io format is {"p": "price", "s": size}
                                        if let Some(bids) = result.get("bids").and_then(|v| v.as_array()) {
                                            for level in bids {
                                                let price = level.get("p")
                                                    .and_then(|v| v.as_str()
                                                        .and_then(|s| s.parse::<f64>().ok())
                                                        .or_else(|| v.as_f64()))
                                                    .unwrap_or(0.0);
                                                let qty = level.get("s")
                                                    .and_then(|v| v.as_f64()
                                                        .or_else(|| v.as_str()
                                                            .and_then(|s| s.parse::<f64>().ok())))
                                                    .unwrap_or(0.0);
                                                if price <= 0.0 { continue; }
                                                let update = RawBookUpdate {
                                                    symbol_id: sym_id,
                                                    side: spsc::side::BID,
                                                    update_type: spsc::update_type::DELTA,
                                                    _pad: [0; 4],
                                                    price: FixedPrice::from_f64(price).raw(),
                                                    qty: fixed_point::FixedQty::from_f64(qty).raw(),
                                                    sequence: *msg_count,
                                                    recv_ns,
                                                    snapshot_count: 0,
                                                    _pad2: [0; 4],
                                                };
                                                if !ring.try_push(update) { *drop_count += 1; }
                                            }
                                        }

                                        // Parse asks: same format
                                        if let Some(asks) = result.get("asks").and_then(|v| v.as_array()) {
                                            for level in asks {
                                                let price = level.get("p")
                                                    .and_then(|v| v.as_str()
                                                        .and_then(|s| s.parse::<f64>().ok())
                                                        .or_else(|| v.as_f64()))
                                                    .unwrap_or(0.0);
                                                let qty = level.get("s")
                                                    .and_then(|v| v.as_f64()
                                                        .or_else(|| v.as_str()
                                                            .and_then(|s| s.parse::<f64>().ok())))
                                                    .unwrap_or(0.0);
                                                if price <= 0.0 { continue; }
                                                let update = RawBookUpdate {
                                                    symbol_id: sym_id,
                                                    side: spsc::side::ASK,
                                                    update_type: spsc::update_type::DELTA,
                                                    _pad: [0; 4],
                                                    price: FixedPrice::from_f64(price).raw(),
                                                    qty: fixed_point::FixedQty::from_f64(qty).raw(),
                                                    sequence: *msg_count,
                                                    recv_ns,
                                                    snapshot_count: 0,
                                                    _pad2: [0; 4],
                                                };
                                                if !ring.try_push(update) { *drop_count += 1; }
                                            }
                                        }

                                        // End snapshot sentinel
                                        if is_snapshot {
                                            let sentinel = RawBookUpdate {
                                                symbol_id: sym_id,
                                                side: spsc::side::ASK,
                                                update_type: spsc::update_type::SNAPSHOT_END,
                                                _pad: [0; 4],
                                                price: 0, qty: 0,
                                                sequence: *msg_count, recv_ns,
                                                snapshot_count: 0, _pad2: [0; 4],
                                            };
                                            let _ = ring.try_push(sentinel);
                                            info!("[ws-gateio] Full snapshot received for {}", contract);
                                        }
                                    }

                                    "futures.tickers" => {
                                        // Fast BBO updates — parse bid1/ask1/last
                                        let contract = result.get("contract")
                                            .and_then(|v| v.as_str()).unwrap_or("");
                                        let sym_id = registry.get_id(contract);
                                        if sym_id == 0 { continue; }

                                        let parse_price = |field: &str| -> f64 {
                                            result.get(field)
                                                .and_then(|v| v.as_str()
                                                    .and_then(|s| s.parse::<f64>().ok())
                                                    .or_else(|| v.as_f64()))
                                                .unwrap_or(0.0)
                                        };

                                        let bid1 = parse_price("bid1").max(parse_price("highest_bid"));
                                        let ask1 = parse_price("ask1").max(parse_price("lowest_ask"));
                                        let last = parse_price("last");

                                        if bid1 > 0.0 {
                                            let _ = ring.try_push(RawBookUpdate {
                                                symbol_id: sym_id,
                                                side: spsc::side::BID,
                                                update_type: spsc::update_type::DELTA,
                                                _pad: [0; 4],
                                                price: FixedPrice::from_f64(bid1).raw(),
                                                qty: fixed_point::FixedQty::from_f64(1.0).raw(),
                                                sequence: *msg_count, recv_ns,
                                                snapshot_count: 0, _pad2: [0; 4],
                                            });
                                        }
                                        if ask1 > 0.0 {
                                            let _ = ring.try_push(RawBookUpdate {
                                                symbol_id: sym_id,
                                                side: spsc::side::ASK,
                                                update_type: spsc::update_type::DELTA,
                                                _pad: [0; 4],
                                                price: FixedPrice::from_f64(ask1).raw(),
                                                qty: fixed_point::FixedQty::from_f64(1.0).raw(),
                                                sequence: *msg_count, recv_ns,
                                                snapshot_count: 0, _pad2: [0; 4],
                                            });
                                        }
                                        // If no bid/ask yet, use last price to bootstrap book
                                        if bid1 == 0.0 && ask1 == 0.0 && last > 0.0 {
                                            let _ = ring.try_push(RawBookUpdate {
                                                symbol_id: sym_id, side: spsc::side::BID,
                                                update_type: spsc::update_type::DELTA, _pad: [0; 4],
                                                price: FixedPrice::from_f64(last * 0.9999).raw(),
                                                qty: fixed_point::FixedQty::from_f64(1.0).raw(),
                                                sequence: *msg_count, recv_ns,
                                                snapshot_count: 0, _pad2: [0; 4],
                                            });
                                            let _ = ring.try_push(RawBookUpdate {
                                                symbol_id: sym_id, side: spsc::side::ASK,
                                                update_type: spsc::update_type::DELTA, _pad: [0; 4],
                                                price: FixedPrice::from_f64(last * 1.0001).raw(),
                                                qty: fixed_point::FixedQty::from_f64(1.0).raw(),
                                                sequence: *msg_count, recv_ns,
                                                snapshot_count: 0, _pad2: [0; 4],
                                            });
                                        }
                                    }

                                    "futures.trades" => {
                                        // Trade events for VPIN/CVD
                                        // Gate.io format: array of {id, create_time, contract, size, price}
                                        let trades_arr = if result.is_array() {
                                            result.as_array()
                                        } else {
                                            result.get("data").and_then(|v| v.as_array())
                                        };
                                        if let Some(trades) = trades_arr {
                                            for trade in trades {
                                                let price = trade.get("price")
                                                    .and_then(|v| v.as_str()
                                                        .and_then(|s| s.parse::<f64>().ok())
                                                        .or_else(|| v.as_f64()))
                                                    .unwrap_or(0.0);
                                                let size = trade.get("size")
                                                    .and_then(|v| v.as_i64()).unwrap_or(0);
                                                let contract = trade.get("contract")
                                                    .and_then(|v| v.as_str()).unwrap_or("");
                                                let sym_id = registry.get_id(contract);
                                                if sym_id == 0 || price <= 0.0 { continue; }
                                                // Gate.io: positive size = buy, negative = sell
                                                let side = if size >= 0 { 0u8 } else { 1u8 };
                                                let event = spsc::TradeEvent {
                                                    symbol_id: sym_id,
                                                    side,
                                                    _pad: [0; 5],
                                                    price: FixedPrice::from_f64(price).raw(),
                                                    qty: fixed_point::FixedQty::from_f64(size.unsigned_abs() as f64).raw(),
                                                    recv_ns,
                                                    sequence: *msg_count,
                                                };
                                                if !trades_ring.try_push(event) { *drop_count += 1; }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Message::Ping(data) => {
                                let mut w = write.lock().await;
                                let _ = w.send(Message::Pong(data)).await;
                            }
                            Message::Close(_) => {
                                info!("[ws-gateio] Received Close frame");
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                    Some(Err(e)) => return Err(Box::new(e)),
                    None => return Ok(()),
                }
            }
            _ = ping_interval.tick() => {
                let ping_msg = serde_json::json!({"time": now_secs(), "channel": "futures.ping"});
                let mut w = write.lock().await;
                if let Err(e) = w.send(Message::Text(ping_msg.to_string())).await {
                    warn!("[ws-gateio] Ping failed: {}", e);
                    return Err(Box::new(e));
                }
            }
        }
    }
}
```

---

## BUG #2 — Orderbook Builder Must Handle SNAPSHOT_START/END Sentinels

**File:** `rust_engine/src/main.rs` — `orderbook_builder_loop`

The book builder always calls `apply_delta_tracked()` regardless of update type. When Gate.io sends an "all" snapshot, stale levels from a previous connection persist. Also, the FlatOrderBook needs a `clear()` method.

**Step A — Add `clear()` to `flat_book.rs`** (add this method to `impl FlatOrderBook`):

```rust
/// Clear all levels. Called when a full snapshot arrives to remove stale data.
pub fn clear(&mut self) {
    for qty in self.bid_levels.iter_mut() { *qty = FixedQty::ZERO; }
    for qty in self.ask_levels.iter_mut() { *qty = FixedQty::ZERO; }
    self.best_bid_idx = self.config.max_levels; // sentinel = empty
    self.best_ask_idx = self.config.max_levels;
    self.sequence_num = 0;
}
```

**Step B — Replace the `if let Some(update) = ws_ring_gateio.try_pop()` block** in `orderbook_builder_loop`:

```rust
if let Some(update) = ws_ring_gateio.try_pop() {
    let sym_idx = update.symbol_id as usize;
    if sym_idx > 0 && sym_idx <= books.len() {
        let book = &mut books[sym_idx - 1];

        match update.update_type {
            spsc::update_type::SNAPSHOT_START => {
                // Full snapshot incoming: clear stale levels from previous connection
                book.clear();
                continue; // Don't push snapshot yet, wait for SNAPSHOT_END
            }
            spsc::update_type::SNAPSHOT_END => {
                // Snapshot complete — fall through to push snapshot below
                book.set_timestamp_ns(update.recv_ns);
            }
            _ => {
                // Normal delta update
                let price = FixedPrice(update.price);
                let qty = fixed_point::FixedQty(update.qty);
                let is_bid = update.side == spsc::side::BID;
                book.apply_delta_tracked(price, qty, is_bid);
                book.set_timestamp_ns(update.recv_ns);
            }
        }

        // Update shared price for Python signal consumer
        let mid = book.mid_price().to_f64();
        if sym_idx - 1 < shared_prices.len() && mid > 0.0 {
            shared_prices[sym_idx - 1].store(mid.to_bits(), Ordering::Relaxed);
        }

        // Push snapshot to strategy — requires BOTH sides
        if let (Some((bid, _)), Some((ask, _))) = (book.best_bid(), book.best_ask()) {
            let snapshot = BookSnapshot {
                symbol_id: update.symbol_id,
                bid_levels: 10,
                ask_levels: 10,
                _pad: [0; 4],
                best_bid: bid.raw(),
                best_ask: ask.raw(),
                mid_price: book.mid_price().raw(),
                spread_bps: book.spread_bps() as i32,
                imbalance_bps: (book.imbalance(10) * 10000.0) as i32,
                bid_depth_usdt: (book.bid_depth_usdt(10) * FixedPrice::PRECISION as f64) as i64,
                ask_depth_usdt: (book.ask_depth_usdt(10) * FixedPrice::PRECISION as f64) as i64,
                sequence: book.sequence(),
                timestamp_ns: update.recv_ns,
            };
            let _ = strategy_ring.try_push(snapshot);

            // Update GlobalBookRegistry if multi-exchange enabled
            if let Some(ref gbr) = global_book_registry {
                let top_bids = book.get_bids(20);
                let top_asks = book.get_asks(20);
                let gateio_snap = multi_exchange::global_book::ExchangeBookSnapshot {
                    exchange: multi_exchange::ExchangeId::GateIo,
                    symbol_id: update.symbol_id,
                    best_bid_fp: bid.raw(),
                    best_ask_fp: ask.raw(),
                    bid_levels: top_bids.iter().map(|(p, q)| (p.raw(), q.raw())).collect(),
                    ask_levels: top_asks.iter().map(|(p, q)| (p.raw(), q.raw())).collect(),
                    sequence: book.sequence(),
                    timestamp_ns: update.recv_ns,
                };
                let sym_name = registry.get_name(update.symbol_id);
                let gbook = gbr.get_or_create_named(update.symbol_id, sym_name);
                gbook.write().update_exchange_snapshot(gateio_snap);
            }
        }
    }
}
```

---

## BUG #3 — ROOT CAUSE #2: Python SHM Signal Queue Never Consumed by Rust

**Evidence:** Python emits signals (ETH 78%, SOL 88% confluence) every 60s. `signal_queue::SignalQueueConsumer` exists and is ready. But `_signal_adapter` uses `_` prefix (unused). `SignalQueueConsumer::open()` is never called anywhere.

**Three-step fix:**

### Step A — Declare the consumer inside `execution_router_loop`

Find the execution router function (contains `let trading_mode_for_ws = ...`). Add BEFORE `rt.block_on`:

```rust
// BUG #3 FIX: Open the SHM signal queue to consume Python Alpha Oracle signals.
// Python writes TradeIntents every 60 seconds. This consumer is polled every loop tick.
let mut shm_signal_consumer: Option<signal_queue::SignalQueueConsumer> =
    match signal_queue::SignalQueueConsumer::open() {
        Ok(c) => {
            info!("[execution] ✅ SHM signal queue opened — Python Alpha Oracle signals ENABLED");
            Some(c)
        }
        Err(e) => {
            warn!("[execution] SHM signal queue unavailable: {} — Python signals disabled", e);
            None
        }
    };
```

### Step B — Poll the queue every loop iteration

Inside the `rt.block_on(async { ... loop { ... } })`, at the very TOP of each loop iteration — BEFORE the circuit breaker check and BEFORE `exec_ring.try_pop()` — add:

```rust
// ── Poll Python Alpha Oracle SHM Signal Queue ──────────────────────────
// Drain all pending Python signals and convert to OrderCommands.
if let Some(ref mut sq) = shm_signal_consumer {
    while let Some(intent) = sq.try_pop() {
        // Basic validation
        if intent.symbol.is_empty() || intent.size_contracts <= 0 {
            continue;
        }

        let sym_id = registry.get_id(&intent.symbol);
        if sym_id == 0 {
            warn!("[execution] Python signal: unknown symbol '{}' — skipping", intent.symbol);
            continue;
        }

        // Circuit breaker
        if circuit_breaker.is_trading_halted() {
            info!("[execution] Python signal for {} ignored — circuit breaker halted", intent.symbol);
            continue;
        }

        // Position slot
        if !position_slots.try_acquire() {
            warn!("[execution] Python signal for {} dropped — slots full", intent.symbol);
            continue;
        }

        // Get current market price from the shared_prices AtomicU64 array
        // (updated by the book builder every tick)
        let current_price = {
            let idx = (sym_id as usize).saturating_sub(1);
            if idx < shared_prices.len() {
                let bits = shared_prices[idx].load(Ordering::Relaxed);
                if bits != 0 { f64::from_bits(bits) } else { intent.entry_price }
            } else {
                intent.entry_price
            }
        };

        if current_price <= 0.0 {
            warn!("[execution] Python signal for {}: no price available — skipping", intent.symbol);
            position_slots.release();
            continue;
        }

        // Derive SL/TP — prefer Python's values, fall back to 2%/4% defaults
        let sl = intent.stop_loss.filter(|&v| v > 0.0).unwrap_or_else(|| {
            if intent.side == 0 { current_price * 0.98 } else { current_price * 1.02 }
        });
        let tp = intent.take_profit.filter(|&v| v > 0.0).unwrap_or_else(|| {
            if intent.side == 0 { current_price * 1.04 } else { current_price * 0.96 }
        });

        let leverage_clamped = intent.leverage.max(1).min(20) as u8; // cap at 20x for safety

        let cmd = OrderCommand {
            symbol_id: sym_id,
            side: intent.side, // 0=buy/long, 1=sell/short
            order_type: spsc::order_cmd_type::LIMIT,
            leverage: leverage_clamped,
            _pad: [0; 3],
            price: FixedPrice::from_f64(current_price).raw(),
            qty: fixed_point::FixedQty::from_f64(intent.size_contracts as f64).raw(),
            order_id: 0,
            signal_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            max_slippage_bps: 30,
            ttl_ms: 15_000, // 15s TTL — Python signals are slower cadence
            stop_loss_fp: FixedPrice::from_f64(sl).raw(),
            take_profit_fp: FixedPrice::from_f64(tp).raw(),
            placement_type: 0,
            post_only: 0,
            is_close: 0,
            _pad2: [0; 5],
        };

        // Pre-trade risk check
        if let Err(reason) = pre_trade_risk_engine.check(&cmd) {
            warn!("[execution] Python signal for {} failed risk check: {}", intent.symbol, reason);
            position_slots.release();
            continue;
        }

        if exec_ring.try_push(cmd) {
            info!(
                "[execution] 🐍 Python signal queued: {} {} {}x {}contracts @ {:.4} \
                 (conf={:.0}%, {}/{} strategies)",
                if intent.side == 0 { "LONG" } else { "SHORT" },
                intent.symbol, intent.leverage, intent.size_contracts, current_price,
                intent.confidence * 100.0,
                intent.confluence_count, intent.total_strategies,
            );
        } else {
            warn!("[execution] Python signal for {} dropped — exec_ring full", intent.symbol);
            position_slots.release();
        }
    }
}
// ── End Python Signal Queue polling ────────────────────────────────────
```

### Step C — Pass `shared_prices` to execution router

The execution router function needs access to `shared_prices` (the `Arc<Vec<AtomicU64>>` array) to look up current prices for Python signals. This Arc is already created in `main()` for the book builder. Simply clone it and pass it into the execution closure or make it available in scope. If it's already in scope (same thread/closure), no change needed. If not, add it as a parameter to `execution_router_loop` or capture it in the closure.

---

## BUG #4 — PostgreSQL Role "postgres" Does Not Exist

**Evidence from logs:** `FATAL: Role "postgres" does not exist` — repeated every 10-15 seconds, causes Python journal/analytics to fail

**Root cause:** `docker-compose.yml` creates `POSTGRES_USER: trading` (correct) but Alembic migrations and some SQLAlchemy internals try to connect as `postgres` superuser.

**Fix A — Create `crypto_trading_bot/init-db.sql`** (referenced in docker-compose):

```sql
-- init-db.sql: Initialize database with compatibility role
-- Fixes "Role postgres does not exist" errors

-- Create postgres superuser alias for library compatibility
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'postgres') THEN
        CREATE ROLE postgres WITH SUPERUSER CREATEDB CREATEROLE LOGIN PASSWORD 'postgres';
    END IF;
END
$$;

-- Ensure trading user has full access
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading;
ALTER DATABASE trading_bot OWNER TO trading;

-- Core tables
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size BIGINT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION DEFAULT 0.0,
    fee DOUBLE PRECISION DEFAULT 0.0,
    strategy VARCHAR(100),
    signal_source VARCHAR(50) DEFAULT 'rust',
    leverage INTEGER DEFAULT 5,
    order_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size BIGINT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    leverage INTEGER DEFAULT 5,
    strategy VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS regime_snapshots (
    id BIGSERIAL PRIMARY KEY,
    overall_regime VARCHAR(50),
    volatility_regime VARCHAR(50),
    position_scale DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
```

**Fix B — Update `docker-compose.yml` postgres healthcheck:**

```yaml
postgres:
    image: postgres:16-alpine
    network_mode: "host"
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-trading123}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
    healthcheck:
      # Use correct user 'trading' not 'postgres'
      test: ["CMD-SHELL", "pg_isready -U trading -d trading_bot"]
      interval: 10s
      timeout: 5s
      retries: 5
```

---

## BUG #5 — Strategy Thresholds Too Strict for Testnet + PostOnly Still Causing Zero Fills

**File:** `crypto_trading_bot/config/engine_config.toml`

The `[strategy]` section has `max_spread_bps = 10.0` — Gate.io testnet spreads are often 15–50bps. Any snapshot with spread > 10bps gets rejected before the strategy even evaluates. Also `post_only = false` must be confirmed.

**Complete replacement for `[strategy]` and all `[[pair_profiles]]` sections:**

```toml
[strategy]
imbalance_threshold = 0.015   # Lowered — testnet books are thin
max_spread_bps = 150.0         # RAISED from 10 — testnet spreads are 15-80bps
min_bid_depth_usdt = 50.0      # LOWERED from 5000 — testnet is thin
min_ask_depth_usdt = 50.0
min_vpin = 0.0
order_size_contracts = 1
post_only = false              # MUST be false — PostOnly on momentum = zero fills
enabled_symbols = []
leverage = 5
enabled = true

[[pair_profiles]]
symbol = "BTC_USDT"
imbalance_threshold = 0.015
vpin_bucket_size = 5000.0     # Small bucket = warms up fast on testnet
trailing_stop_atr_multiplier = 3.0
max_leverage = 10
max_position_size = 5
leverage = 5
sl_pct = 1.5
tp_pct = 3.0
tick_size = 0.1

[[pair_profiles]]
symbol = "ETH_USDT"
imbalance_threshold = 0.015
vpin_bucket_size = 2000.0
trailing_stop_atr_multiplier = 2.0
max_leverage = 10
max_position_size = 10
leverage = 5
sl_pct = 2.0
tp_pct = 4.0
tick_size = 0.01

[[pair_profiles]]
symbol = "SOL_USDT"
imbalance_threshold = 0.015
vpin_bucket_size = 1000.0
trailing_stop_atr_multiplier = 3.0
max_leverage = 10
max_position_size = 30
leverage = 5
sl_pct = 3.0
tp_pct = 6.0
tick_size = 0.001
```

---

## BUG #6 — VPIN Bucket Too Large for Testnet Volumes

**File:** `rust_engine/src/main.rs` (~line 1274 in strategy_evaluator_loop)

```rust
// FIND:
let mut vpin_calculator = microstructure::EnhancedVpin::new(100_000.0, 50);

// REPLACE WITH:
// BUG #6 FIX: 100k USDT buckets never fill on testnet — use 5k for faster warmup
let mut vpin_calculator = microstructure::EnhancedVpin::new(5_000.0, 50);
```

---

## BUG #7 — `strategy_engine.rs` Spread Gate Blocks All Testnet Signals

**File:** `rust_engine/src/strategy_engine.rs`

```rust
// FIND:
const MAX_SPREAD_BPS: f64 = 200.0;
const MIN_DEPTH_USD: f64 = 100.0;

// REPLACE WITH:
const MAX_SPREAD_BPS: f64 = 500.0; // Testnet can have wide spreads — don't block
const MIN_DEPTH_USD: f64 = 10.0;   // Testnet books are very thin — accept any depth
```

Also lower the confidence threshold for order types:

```rust
// FIND in evaluate():
let (order_type, time_in_force, price) = if confidence > 0.65 {
    ...
} else if confidence > 0.45 {
    ...

// REPLACE WITH:
// BUG #7 FIX: Testnet signals have lower confidence (thin books, no ML weights)
// Lower thresholds so we don't get stuck in PostOnly forever
let (order_type, time_in_force, price) = if confidence > 0.55 {
    // High confidence: IOC (cross spread)
    let aggressive = if side == OrderSide::Buy {
        metrics.mid_price + metrics.mid_price * 0.0005
    } else {
        metrics.mid_price - metrics.mid_price * 0.0005
    };
    (OrderType::Limit, "ioc".to_string(), Some(aggressive))
} else if confidence > 0.30 {
    // Moderate: GTC Limit at mid (fills when price comes to us)
    (OrderType::Limit, "gtc".to_string(), Some(metrics.mid_price))
} else {
    // Low: PostOnly maker
    let half_spread = metrics.mid_price * (metrics.spread_bps / 2.0) / 10_000.0;
    let maker_price = if side == OrderSide::Buy {
        metrics.mid_price - half_spread
    } else {
        metrics.mid_price + half_spread
    };
    (OrderType::PostOnly, "poc".to_string(), Some(maker_price))
};
```

---

## BUG #8 — `TradeIntent` Missing `total_strategies` Field Deserialization

**File:** `rust_engine/src/signal_queue.rs`

Verify the `TradeIntent` struct has `total_strategies: u32` field and that `from_slot` reads bytes 100..104. If missing, add:

In the struct:
```rust
pub total_strategies: u32,
```

In `from_slot`:
```rust
let total_strategies = u32::from_le_bytes(slot[100..104].try_into().unwrap_or([0;4]));
```

And set it in the returned `TradeIntent`.

---

# PART 2 — PROFIT MAXIMIZATION ENHANCEMENTS

Implement these AFTER the bot opens its first trade.

---

## ENHANCEMENT #1 — Per-Asset ATR Stop Calibration

**File:** `rust_engine/src/main.rs` — replace SL/TP calculation in strategy_evaluator_loop:

```rust
// Calibrated per-asset volatility profiles
let sym_for_sl = registry.get_name(snapshot.symbol_id);
let (min_sl, max_sl, rr) = if sym_for_sl.contains("BTC") {
    (0.004, 0.020, 2.5)
} else if sym_for_sl.contains("ETH") {
    (0.005, 0.025, 2.2)
} else if sym_for_sl.contains("SOL") {
    (0.007, 0.040, 2.0)
} else {
    (0.005, 0.030, 2.0)
};
let vol_scale = match metrics.realized_vol_regime.as_str() {
    "Low" => 0.7, "High" => 1.3, "Extreme" => 1.7, _ => 1.0,
};
let atr = exit_evaluator.get_position_atr(snapshot.symbol_id);
let sl_pct = if atr > 0.0 && entry_price > 0.0 {
    ((atr / entry_price) * vol_scale).max(min_sl).min(max_sl)
} else { (min_sl + max_sl) / 2.0 };
let tp_pct = (sl_pct * rr).min(max_sl * rr);
let stop_loss_price = if is_buy { entry_price * (1.0 - sl_pct) } else { entry_price * (1.0 + sl_pct) };
let take_profit_price = if is_buy { entry_price * (1.0 + tp_pct) } else { entry_price * (1.0 - tp_pct) };
```

---

## ENHANCEMENT #2 — Breakeven Stop Migration

**File:** `rust_engine/src/exit_evaluator.rs` — add to `TrailingStopState`:

```rust
pub at_breakeven: bool,   // true after SL moved to entry
pub at_2r_lock: bool,     // true after SL moved to +1R
```
Initialize both to `false`.

**In `strategy_evaluator_loop`**, after trailing stop update logic:

```rust
// Move to breakeven when position is 1R in profit
if let Some(state) = trailing_stops.get_mut(&snapshot.symbol_id) {
    let mid = FixedPrice(snapshot.mid_price).to_f64();
    let risk_dist = (state.entry_price - state.stop_loss).abs();
    if risk_dist > 0.0 {
        let profit = if state.is_long { mid - state.entry_price } else { state.entry_price - mid };
        if profit >= risk_dist && !state.at_breakeven {
            let new_sl = if state.is_long { state.entry_price + risk_dist * 0.05 }
                         else { state.entry_price - risk_dist * 0.05 };
            state.stop_loss = new_sl;
            state.at_breakeven = true;
            info!("[strategy] 🔒 Breakeven: {} SL → {:.4}", symbol_name, new_sl);
            let _ = sl_tp_update_tx.try_send(SlTpUpdateRequest {
                symbol: symbol_name.to_string(),
                symbol_id: snapshot.symbol_id,
                side: if state.is_long { execution_gateway::OrderSide::Buy } else { execution_gateway::OrderSide::Sell },
                size: exit_evaluator.get_position_size(snapshot.symbol_id).unwrap_or(1),
                sl_price: new_sl,
                tp_price: state.take_profit,
                is_update: true,
            });
        }
        if profit >= risk_dist * 2.0 && !state.at_2r_lock {
            let new_sl = if state.is_long { state.entry_price + risk_dist }
                         else { state.entry_price - risk_dist };
            state.stop_loss = new_sl;
            state.at_2r_lock = true;
            info!("[strategy] 🔒 2R Lock: {} SL → {:.4} (locking 1R profit)", symbol_name, new_sl);
            let _ = sl_tp_update_tx.try_send(SlTpUpdateRequest {
                symbol: symbol_name.to_string(),
                symbol_id: snapshot.symbol_id,
                side: if state.is_long { execution_gateway::OrderSide::Buy } else { execution_gateway::OrderSide::Sell },
                size: exit_evaluator.get_position_size(snapshot.symbol_id).unwrap_or(1),
                sl_price: new_sl,
                tp_price: state.take_profit,
                is_update: true,
            });
        }
    }
}
```

---

## ENHANCEMENT #3 — 4-Tier Drawdown Scaling

**File:** `rust_engine/src/main.rs` — replace drawdown_scalar:

```rust
let (drawdown_scalar, should_halt) = if let Some(ref cb) = circuit_breaker {
    let st = cb.get_state();
    let cur = st.current_equity as f64 / 1e8;
    let peak = st.peak_equity as f64 / 1e8;
    if peak > 0.0 && cur > 0.0 {
        let dd = (peak - cur) / peak;
        match () {
            _ if dd >= 0.10 => { error!("[strategy] 🛑 DD {:.1}% HALT", dd*100.0); (0.0, true) }
            _ if dd >= 0.06 => { warn!("[strategy] DD {:.1}% → 25% size", dd*100.0); (0.25, false) }
            _ if dd >= 0.04 => { warn!("[strategy] DD {:.1}% → 50% size", dd*100.0); (0.50, false) }
            _ if dd >= 0.02 => { info!("[strategy] DD {:.1}% → 75% size", dd*100.0); (0.75, false) }
            _ => (1.0, false)
        }
    } else { (1.0, false) }
} else { (1.0, false) };
```

---

## ENHANCEMENT #4 — Funding Rate Hard Filter

**File:** `rust_engine/src/strategy_engine.rs` — replace funding_score in `evaluate()`:

```rust
let is_long_signal = imbalance > 0.0;
// Hard block crowded trades (above 0.03% per 8h = extreme crowding)
if metrics.funding_rate > 0.0003 && is_long_signal && confidence < 0.80 {
    debug!("[strategy] Funding hard-block: {:.4}% rate blocks long", metrics.funding_rate*100.0);
    return None;
}
if metrics.funding_rate < -0.0003 && !is_long_signal && confidence < 0.80 {
    debug!("[strategy] Funding hard-block: {:.4}% rate blocks short", metrics.funding_rate*100.0);
    return None;
}
// Contrarian funding boost — going against crowded side = free carry
let funding_score = if metrics.funding_rate > 0.0002 && !is_long_signal { 1.3 }
    else if metrics.funding_rate < -0.0002 && is_long_signal { 1.3 }
    else if metrics.funding_rate.abs() > 0.0001 { 1.1 }
    else { 1.0 };
```

---

## ENHANCEMENT #5 — Daily Trade Count Limiter

**File:** `rust_engine/src/main.rs` — add to strategy_evaluator_loop state:

```rust
let mut daily_trade_count: u32 = 0;
let mut last_trade_day: u64 = 0;
```

Before `exec_ring.try_push(cmd)`:

```rust
let today = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() / 86400;
if today != last_trade_day { daily_trade_count = 0; last_trade_day = today; }
if daily_trade_count >= 20 {
    debug!("[strategy] Daily limit 20 reached");
    position_slots.release();
    continue;
}
daily_trade_count += 1;
```

---

# PART 3 — ADDITIONAL PROTECTIVE CHECKS

---

## CHECK #1 — Duplicate Position Block

**File:** `rust_engine/src/main.rs` — before `position_slots.try_acquire()` in strategy loop:

```rust
if exit_evaluator.has_position(snapshot.symbol_id) {
    debug!("[strategy] Duplicate block: {} already open", symbol_name);
    continue;
}
```

Add to `exit_evaluator.rs`:
```rust
pub fn has_position(&self, symbol_id: u16) -> bool {
    self.positions.contains_key(&symbol_id)
}
```

---

## CHECK #2 — Spread Anomaly Gate

**File:** `rust_engine/src/main.rs` — at the top of the `if let Some(snapshot) = book_ring.try_pop()` block:

```rust
// Rolling average spread (EMA) — skip on anomalies
static AVG_SPREAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(50_u64.to_bits() as u64);
// Note: using u64 bits storage for f64 average
let avg_bps = f64::from_bits(AVG_SPREAD.load(Ordering::Relaxed));
let cur_spread = snapshot.spread_bps as f64;
let new_avg = avg_bps * 0.9990 + cur_spread * 0.0010;
AVG_SPREAD.store(new_avg.to_bits(), Ordering::Relaxed);
// Skip if spread is 5x wider than rolling average (liquidity crisis / manipulation)
if cur_spread > new_avg * 5.0 && cur_spread > 100.0 {
    debug!("[strategy] Spread anomaly: {}bps > 5x avg {:.1}bps — skip", snapshot.spread_bps, new_avg);
    continue;
}
```

---

## CHECK #3 — Minimum Entry-Side Depth

**File:** `rust_engine/src/strategy_engine.rs` — in `evaluate()` after the existing depth gate:

```rust
// Require minimum depth on the entry side specifically
let entry_depth = if imbalance > 0.0 { metrics.ask_depth_usdt } else { metrics.bid_depth_usdt };
if entry_depth < 1_000.0 {
    debug!("[strategy] Entry-side depth ${:.0} < $1,000 — skip", entry_depth);
    return None;
}
```

---

# PART 4 — VERIFICATION CHECKLIST

After ALL fixes, the logs MUST show in order:

**Within 10 seconds:**
```
[ws-gateio] Full snapshot received for BTC_USDT   ← BUG #1 fixed
[ws-gateio] Full snapshot received for ETH_USDT
[ws-gateio] Full snapshot received for SOL_USDT
```

**At the next 30s latency report:**
```
tick_to_book:   n=XXX mean=Xµs    ← Was n=0, now non-zero
book_to_signal: n=XXX mean=Xµs
```

**Within 60-120 seconds:**
```
[strategy] Signal: Buy size=1.0 confidence=0.XXX imbalance=0.0XX   ← Rust signal
[execution] Routing order #1: sym=BTC_USDT side=BUY
[execution] ✅ Order filled: BTC_USDT 1 contracts @ XXXXX
```

**Within 2 minutes (Python path):**
```
[execution] 🐍 Python signal queued: SHORT ETH_USDT 10x 5contracts
[execution] Routing order #2: sym=ETH_USDT side=SELL
```

**If you STILL see `tick_to_book: n=0` after fix — add this debug log:**

In `ws_connect_and_ingest_gateio`, right after `let parsed: ... = match serde_json::from_str(&text)`:
```rust
if *msg_count <= 5 {
    info!("[ws-gateio] RAW MSG #{}: channel={} event={} result_keys={:?}",
        msg_count, channel, event,
        result.as_object().map(|m| m.keys().collect::<Vec<_>>()));
}
```
This dumps the first 5 messages so you can see EXACTLY what Gate.io is sending and adjust the parser.

---

# PART 5 — SUMMARY TABLE

| # | Sev | File | Root Cause | Fix |
|---|-----|------|------------|-----|
| BUG-1 | 🔴 CRITICAL | main.rs | Gate.io sends `{"p","s"}` objects, code expects arrays → ALL ticks dropped → tick_to_book=0 | Replace entire ws_connect_and_ingest_gateio with object parser |
| BUG-2 | 🔴 CRITICAL | main.rs, flat_book.rs | SNAPSHOT_START/END sentinels not handled → stale book on reconnect | Add clear() to FlatOrderBook, handle sentinels in builder loop |
| BUG-3 | 🔴 CRITICAL | main.rs | Python SHM signal queue never polled → Python signals silently ignored | Add SignalQueueConsumer, poll it every execution loop tick |
| BUG-4 | 🟠 HIGH | docker-compose.yml | Role "postgres" missing → Python DB crashes | Create init-db.sql with postgres superuser alias |
| BUG-5 | 🟠 HIGH | engine_config.toml | max_spread_bps=10 blocks all testnet snapshots; post_only=true blocks fills | Raise spread to 150, confirm post_only=false, lower depth thresholds |
| BUG-6 | 🟡 MEDIUM | main.rs | VPIN bucket 100k USDT → takes hours to warm up on testnet | Lower to 5k USDT |
| BUG-7 | 🟡 MEDIUM | strategy_engine.rs | MAX_SPREAD_BPS=200 + confidence thresholds too high for testnet | Raise to 500bps, lower confidence thresholds for IOC/GTC |
| BUG-8 | 🟡 MEDIUM | signal_queue.rs | `total_strategies` field may be missing in TradeIntent deserialization | Verify from_slot reads bytes 100..104 |
| ENH-1 | 💡 | main.rs | — | ATR-based per-asset SL/TP (BTC/ETH/SOL calibrated) |
| ENH-2 | 💡 | exit_evaluator.rs | — | Breakeven + 2R stop migration |
| ENH-3 | 💡 | strategy_engine.rs | — | Funding rate hard filter + contrarian boost |
| ENH-4 | 💡 | main.rs | — | 4-tier drawdown scaling (10%/6%/4%/2%) |
| ENH-5 | 💡 | main.rs | — | Daily trade count limiter (max 20/day) |
| CHK-1 | 🛡️ | main.rs | — | Duplicate position block per symbol |
| CHK-2 | 🛡️ | main.rs | — | Spread anomaly gate (5x rolling average) |
| CHK-3 | 🛡️ | strategy_engine.rs | — | Minimum $1k entry-side depth gate |
