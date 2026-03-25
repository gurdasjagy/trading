# Root Cause Analysis: Bot Errors After PR #23 Fixes

**Date:** 2026-03-25  
**Log file analysed:** `new.txt` (52 lines, timestamps 07:41:33 — 07:42:34 UTC)  
**Codebase state:** main branch with PR #23 merged

---

## Executive Summary

Three distinct root causes explain **all** observed failures. Two are bugs introduced by PR #23 itself (incomplete wiring), and one is a pre-existing architectural issue that PR #23 did not address.

| # | Symptom | Root Cause | Severity |
|---|---------|-----------|----------|
| 1 | Funding arb opportunities **always** rejected with "Margin monitor not yet initialized" even after balances are fetched | **Two separate `CrossVenueMarginMonitor` instances** — the one that gets refreshed is not the one the funding arb engine reads | **Critical** |
| 2 | Gate.io WS orders time out (5000ms) with zero ACK, ghost positions created | **Possible wrong WebSocket URL** (`ws.gate.com` vs documented `fx-ws.gateio.ws`) + **double `set_leverage()` call** wasting time | **Critical** |
| 3 | `set_leverage()` always fails on cross-margin first, then retries isolated | **Faulty cross-margin detection logic** — always defaults to cross mode even when account is isolated | **Medium** |

---

## Root Cause 1: Two Separate Margin Monitor Instances (Funding Arb Always Rejected)

### The Problem

Every single funding arb opportunity is rejected with:

```
[pre-trade] REJECTED BTC_USDT: Margin monitor not yet initialized for Bybit and/or Binance
```

This happens **even after** balances have been successfully fetched:

```
07:41:33.839  [margin] Gate.io balance: $941.14 (margin: 100.0%)
07:41:33.947  [margin] Bybit balance: $2944.30 (margin: 100.0%)
07:41:34.132  [margin] Binance balance: $4943.14 (margin: 100.0%)
07:41:34.692  Multi-exchange health: total_balance=$8828.58, exchanges=3
   ...18 seconds later...
07:41:52.647  REJECTED BTC_USDT: Margin monitor not yet initialized
07:41:52.647  REJECTED SOL_USDT: Margin monitor not yet initialized
```

### The Exact Cause

There are **two completely independent `CrossVenueMarginMonitor` instances** in the codebase. They never share data.

**Instance A** — lives inside `execution_router_loop` (`main.rs:2596`):

```rust
// main.rs:2596
let mut margin_monitor = multi_exchange::margin_monitor::CrossVenueMarginMonitor::with_defaults();
```

This instance gets `refresh_all()` called on it every health-check cycle (`main.rs:4543`):

```rust
// main.rs:4543
margin_monitor.refresh_all(&multi_gateways).await;
```

This is the instance that successfully fetches and logs the balances at 07:41:33.

**Instance B** — created fresh for the funding arb engine (`main.rs:5242-5246`):

```rust
// main.rs:5242-5246
let fab_margin_monitor = Arc::new(
    parking_lot::RwLock::new(
        multi_exchange::margin_monitor::CrossVenueMarginMonitor::with_defaults()
    )
);
```

This is a **brand-new, empty instance**. It is passed to `FundingArbEngine::run()` at `main.rs:5288`. **No code ever calls `refresh_all()` on this instance.** Its internal `health` HashMap is always empty.

The irony is that the comment directly above it says:

```rust
// main.rs:5224
// FIX: Share a single CrossVenueMarginMonitor instance so the engine
// sees the same margin health data as the rest of the system.
```

But the code immediately creates a **new** instance instead of sharing Instance A.

### Why `has_exchange_data()` Always Returns False

The startup gate added by PR #23 (`funding_arb_risk.rs:120-129`) calls:

```rust
if !margin_monitor.has_exchange_data(opp.short_exchange)
    || !margin_monitor.has_exchange_data(opp.long_exchange)
```

Since `fab_margin_monitor` (Instance B) has an empty `health` map, `has_exchange_data()` does:

```rust
self.health.get(&exchange)          // returns None (empty map)
    .map(|h| h.available_balance > 0.0 || h.total_equity > 0.0)
    .unwrap_or(false)               // always returns false
```

**Result:** Every opportunity is rejected, forever. The startup gate can never pass.

### Evidence from Logs

| Timestamp | Event | Instance |
|-----------|-------|----------|
| 07:41:33.839 | Gate.io balance fetched: $941.14 | Instance A (execution router) |
| 07:41:33.947 | Bybit balance fetched: $2944.30 | Instance A (execution router) |
| 07:41:34.132 | Binance balance fetched: $4943.14 | Instance A (execution router) |
| 07:41:52.647 | REJECTED: "not yet initialized" | Instance B (funding arb engine) |
| 07:42:22.646 | REJECTED: "not yet initialized" | Instance B (funding arb engine) |

The rejection happens because Instance B is queried, not Instance A.

### The Fix

Share Instance A with the funding arb engine. Replace `main.rs:5242-5246` — instead of creating a new `CrossVenueMarginMonitor`, wrap the existing `margin_monitor` (from line 2596) in an `Arc<RwLock<>>` and pass it to both the execution router loop and the funding arb engine.

---

## Root Cause 2: Gate.io WS Order Timeout (No ACK, Ghost Positions)

### The Problem

Manual trade orders sent via WebSocket get **zero response** from Gate.io — no ACK, no error, nothing. After 5000ms the code times out:

```
07:42:28.266  Timeout waiting for order r7 ACK (5000ms)
07:42:34.124  Timeout waiting for order r8 ACK (5000ms)
```

Reconciliation confirms the orders were never placed:

```
07:42:32.842  GHOST: Local tracking for BTC_USDT but REST shows no position
07:42:32.842  Removed ghost tracking entry: r7
07:42:32.842  Removed ghost tracking entry: r8
```

### Sub-Cause 2A: Potentially Wrong WebSocket URL

The code connects to:

```rust
// gateio_gateway.rs:65
const GATEIO_WS_URL: &str = "wss://ws.gate.com/v4/ws/futures/usdt";

// gateio_gateway.rs:70
const GATEIO_WS_TESTNET_URL: &str = "wss://ws-testnet.gate.com/v4/ws/futures/usdt";
```

But Gate.io's **official documentation** (https://www.gate.io/docs/developers/futures/ws/en/) specifies:

```
Real Trading:  wss://fx-ws.gateio.ws/v4/ws/usdt
TestNet:       wss://fx-ws-testnet.gateio.ws/v4/ws/usdt
```

The differences are:

| | Code Uses | Official Docs |
|---|-----------|---------------|
| **Domain** | `ws.gate.com` | `fx-ws.gateio.ws` |
| **Path** | `/v4/ws/futures/usdt` | `/v4/ws/usdt` |

Gate.io does have multiple domain aliases (`gate.io`, `gate.com`, `gateio.ws`) and a recent 2025 announcement used `ws-testnet.gate.io/v4/ws/futures/usdt`. However, the **official API documentation** consistently uses `fx-ws.gateio.ws/v4/ws/usdt` for futures. The 2023 announcement introducing WS order placement also uses `fx-ws.gateio.ws/v4/ws/usdt`.

**If `ws.gate.com/v4/ws/futures/usdt` is not a valid alias for the futures order-placement endpoint, the connection may succeed (accepting pings, subscriptions) but silently drop `futures.order_place` messages.** This would perfectly explain:

- WS connection appears healthy (pings work, `is_ready = true`)
- REST leverage calls work fine (they bypass WS entirely)
- `futures.order_place` messages vanish without any response
- No ACK, no error, no nothing — 100% silent

**This needs to be verified.** If switching to `wss://fx-ws.gateio.ws/v4/ws/usdt` resolves the timeouts, this is the primary cause.

### Sub-Cause 2B: Double `set_leverage()` Call

The manual trade path calls `set_leverage()` **twice** per order:

**First call** — explicit Step 2 in the execution router (`main.rs:2892-2895`):

```rust
// Step 2: Set leverage
if let Err(e) = gw.set_leverage(&sym_upper, manual_req.leverage as i32).await {
    warn!("[execution] Manual trade: failed to set leverage: {}", e);
}
```

**Second call** — inside `submit_order()` because `intent.leverage = Some(10)` (`gateio_gateway.rs:1931-1950`):

```rust
if let Some(target_leverage) = intent.leverage {
    if target_leverage > 0 && target_leverage <= 125 {
        let lev_result = self.set_leverage(&symbol, target_leverage).await;
```

The log timestamps confirm this double-call:

```
07:42:22.448  Cross-margin leverage failed (1st call — from main.rs:2893)
07:42:22.527  Leverage set to 10x isolated  (1st call succeeds)
    ↳ [500ms delay starts inside submit_order... but this is the OUTER call, not inside submit_order]
07:42:22.684  Cross-margin leverage failed (2nd call — from submit_order:1933)
07:42:22.764  Leverage set to 10x isolated  (2nd call succeeds)
    ↳ [500ms delay completes ~07:42:23.264]
    ↳ WS order r7 sent ~07:42:23.264
07:42:28.266  Timeout waiting for r7 ACK (5000ms)
```

**Wait — the first `set_leverage()` at main.rs:2893 happens BEFORE `submit_order()` is called.** The 500ms delay was placed inside `submit_order()` (after its internal `set_leverage` call). So the outer `set_leverage()` at main.rs:2893 does NOT have a 500ms delay after it. Instead, `submit_order()` is called immediately after it, which then calls `set_leverage()` AGAIN internally, and THAT one gets the 500ms delay.

The result: 4 REST API calls for leverage (2 cross-margin fails + 2 isolated-margin retries) when only 1 is needed, adding ~600ms of unnecessary latency before the WS order is even sent.

### Sub-Cause 2C: No Visibility Into WS Responses

The code's `handle_ws_message()` only looks for three patterns:

```rust
if text.contains("futures.order_place") { ... }
if text.contains("futures.order_cancel") { ... }
if text.contains("futures.orders") && text.contains("\"update\"") { ... }
```

Any other response (e.g., a generic error, an auth expiry notification, or a message in an unexpected format) is logged at `debug!` level only:

```rust
debug!("[gateio-ws] Unhandled message: {}", &text[..text.len().min(120)]);
```

Since the logs don't show any "Unhandled message" entries, either:
1. No response is received at all (supporting the wrong-URL theory), OR
2. The response is received but matches none of the patterns and is silently discarded at debug level (which may not be enabled)

**The fix should log ALL incoming WS messages at `info!` or `warn!` level during order submission windows**, so we can see exactly what Gate.io sends back (or doesn't send back).

---

## Root Cause 3: `set_leverage()` Always Tries Cross-Margin First, Always Fails

### The Problem

Every single `set_leverage()` call follows the same pattern:

```
Cross-margin leverage failed for BTC_USDT, retrying with isolated margin param
Leverage set to 10x for BTC_USDT (isolated margin)
```

This happens 100% of the time — cross-margin always fails, isolated always succeeds.

### The Exact Cause

The `set_leverage()` method (`gateio_gateway.rs:2130-2230`) tries to detect the margin mode by checking positions:

```rust
let is_cross = match self.rest_get("/futures/usdt/positions", "").await {
    Ok(positions_val) => {
        if let Some(positions) = positions_val.as_array() {
            positions.iter()
                .find(|p| p.get("contract").and_then(|v| v.as_str()) == Some(&normalized))
                .and_then(|p| p.get("mode").and_then(|v| v.as_str()))
                .map(|mode| mode == "dual" || mode == "single")
                .unwrap_or(true) // Default to cross margin (Gate.io default)
        } else {
            true // Default to cross margin
        }
    }
    Err(_) => true, // Default to cross margin on error
};
```

The logic has multiple flaws:

1. **`mode` field is the position mode (single/dual/both), NOT the margin mode.** Gate.io's position `mode` values are `single`, `dual`, and `both` — they describe whether hedging is enabled, not whether margin is cross or isolated. The margin mode is determined by `cross_leverage_limit` in the position data (if > 0, it's cross-margin).

2. **When no position exists yet (which is the case for a new manual trade), `find()` returns `None`, `.and_then()` returns `None`, `.map()` returns `None`, and `.unwrap_or(true)` defaults to cross-margin.** But the account may actually be configured for isolated margin.

3. **The `Err(_) => true` fallback also defaults to cross-margin.** So any REST error also results in a cross-margin attempt.

Since the account is configured for **isolated margin**, every call:
1. Sends `cross_leverage_limit=10` (wrong parameter for isolated mode)
2. Gate.io rejects it with a "cross_leverage_limit only for cross-margin" error
3. Code catches the error and retries with `leverage=10` (correct parameter)
4. This wastes ~150ms per `set_leverage()` call

With the double-call issue (Root Cause 2B), this means **4 wasted REST calls** (2 failed cross-margin attempts) per order.

---

## Timeline Reconstruction

```
07:41:33.839  [Instance A] Gate.io balance fetched: $941.14
07:41:33.947  [Instance A] Bybit balance fetched: $2944.30
07:41:34.132  [Instance A] Binance balance fetched: $4943.14
07:41:34.692  [Instance A] Total balance: $8828.58 across 3 exchanges
              ⚠ Instance B (funding arb engine) knows NOTHING about these balances

07:41:52.647  [Instance B] Found 2 opportunities → REJECTED (empty health map)

07:42:22.139  Manual trade: buy BTC_USDT 50 USDT @ 10x leverage
07:42:22.294  USDT→contracts: 500 USDT → 70 contracts
07:42:22.448  set_leverage #1 (main.rs:2893): cross-margin fails
07:42:22.527  set_leverage #1 retry: isolated succeeds (10x)
07:42:22.646  [Instance B] Found 2 opportunities → REJECTED again

07:42:22.684  set_leverage #2 (submit_order:1933): cross-margin fails AGAIN
07:42:22.764  set_leverage #2 retry: isolated succeeds AGAIN
              ↳ 500ms delay (gateio_gateway.rs:1942)
~07:42:23.264 WS order r7 sent via ws.gate.com
              ↳ No response received from Gate.io

07:42:28.266  r7 TIMEOUT (5000ms) → retryable error → retry attempt 2
07:42:28.542  set_leverage #3: cross-margin fails
07:42:28.621  set_leverage #3 retry: isolated succeeds
              ↳ 500ms delay
~07:42:29.121 WS order r8 sent
              ↳ No response received from Gate.io

07:42:32.842  Reconciliation: r7 and r8 are ghosts (no position on exchange)
07:42:34.124  r8 TIMEOUT (5000ms)
```

---

## Summary of Required Fixes

### Fix 1 (Critical): Share the margin monitor instance
- Wrap the `margin_monitor` from `main.rs:2596` in `Arc<RwLock<>>` from the start
- Pass that same `Arc` to both the execution router loop and the funding arb engine
- Remove the `fab_margin_monitor` creation at `main.rs:5242-5246`

### Fix 2 (Critical): Verify and correct the WebSocket URL
- Test with the official documented URL: `wss://fx-ws.gateio.ws/v4/ws/usdt`
- If the current `ws.gate.com` URL works for market data but not order placement, switch to the official URL
- Add `info!`-level logging for ALL incoming WS messages to diagnose what Gate.io actually sends back

### Fix 3 (Medium): Remove duplicate `set_leverage()` call
- Either remove the explicit `set_leverage()` call in the manual trade path (`main.rs:2893-2895`), OR
- Set `intent.leverage = None` so `submit_order()` doesn't call it internally
- Don't do both — pick one call site

### Fix 4 (Medium): Fix cross-margin detection in `set_leverage()`
- Stop checking the `mode` field (that's position mode, not margin mode)
- Instead, try isolated margin FIRST (since the account is configured for isolated), or
- Check `cross_leverage_limit` field in position data to detect actual margin mode, or
- Cache the margin mode after the first successful call and reuse it

---

## References

- Gate.io Futures WS API docs: https://www.gate.io/docs/developers/futures/ws/en/
- Gate.io WS order placement announcement: https://www.gate.io/article/32640
- PR #23 (previous fixes): https://github.com/jashanxjagy9/tradings/pull/23
- `main.rs:2596` — Instance A creation
- `main.rs:5242-5246` — Instance B creation (the bug)
- `main.rs:4543` — Instance A refresh (works)
- `main.rs:5288` — Instance B passed to engine (never refreshed)
- `funding_arb_risk.rs:120-129` — Startup gate that always rejects
- `gateio_gateway.rs:65` — WS URL constant
- `gateio_gateway.rs:1931-1950` — Internal `set_leverage()` inside `submit_order()`
- `gateio_gateway.rs:2130-2230` — Flawed cross-margin detection
- `main.rs:2892-2895` — Outer `set_leverage()` call (duplicate)
