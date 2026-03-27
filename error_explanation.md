# Root Cause Explanation of the 3 Errors

## 1. `futures.login failed — WS order placement will not work`

**Root cause:** The `futures.login` message was sent (our new fix), but Gate.io's response didn't match either of the two success patterns checked in the code:
- `"error":null`
- `"status":"success"`

This means either:
- **The API key/secret is invalid or has insufficient permissions** — Gate.io returned an error that was not caught by the `INVALID_KEY` check (e.g. `IP_NOT_ALLOWED`, `KEY_EXPIRED`, or a permission error). The code only breaks on `INVALID_KEY` or generic `"error":{`, but some error formats may slip through.
- **The login response format differs from what we expect** — Gate.io may return a success response with a slightly different JSON structure (e.g. `"result":{"status":"success"}` nested inside a result object instead of top-level). Since we're doing simple string matching (`txt.contains(...)`) instead of proper JSON parsing, a valid success response could be missed.
- **Timeout** — The 5-second deadline expired before any login response arrived (network latency, server delay). In that case `logged_in` stays `false` and the error is logged.

**Why manual trades still fail:** Since the login step fails (or times out), the WS session remains unauthenticated for the trading API. When `futures.order_place` is sent later, Gate.io rejects it with "Not login" because the session was never successfully logged in.

**What needs investigation:** Run the bot with `RUST_LOG=debug` to see the actual login response (`[gateio-ws] Login phase received: ...`). That will show exactly what Gate.io returned and why it didn't match the success check.

---

## 2. Funding arb rejections: "Minimum 1 contract = 1.0000 units = $68444.55 notional"

**Root cause: The `contract_multiplier` is wrong for SOL_USDT, ETH_USDT, and BTC_USDT in the funding arb context.** Here's what's happening:

The code at `funding_arb_risk.rs:219-231` checks: "Can you afford even 1 contract?" It calculates:
```
one_contract_qty = max(short_spec.contract_multiplier, long_spec.contract_multiplier)
one_contract_notional = one_contract_qty × price
one_contract_margin = one_contract_notional / leverage
```

For the rejections you see:
- `SOL_USDT: 1.0000 units = $68444.55` — This is **impossible**. 1 SOL ≈ $130, not $68k. The price being used ($68,444) is **BTC's price**, not SOL's.
- `ETH_USDT: 1.0000 units = $68426.15` — Same problem. 1 ETH ≈ $1,900, not $68k.
- `BTC_USDT: 1.0000 units = $68426.15` — This one makes sense if `contract_multiplier = 1.0`, meaning 1 contract = 1 whole BTC.

**There are TWO bugs here:**

### Bug A: Wrong price being used for SOL and ETH
The `get_approximate_price()` function is returning **BTC's price (~$68k)** for SOL_USDT and ETH_USDT. This means the global order book registry either doesn't have separate books for SOL/ETH, or the price lookup is cross-contaminated (returning the wrong symbol's price).

### Bug B: `contract_multiplier = 1.0` (wrong for Gate.io)
The `one_contract_qty` shows `1.0000` for all three symbols. On Gate.io:
- BTC_USDT: `quanto_multiplier = 0.0001` (1 contract = 0.0001 BTC ≈ $6.84)
- ETH_USDT: `quanto_multiplier = 0.01` (1 contract = 0.01 ETH ≈ $19)
- SOL_USDT: `quanto_multiplier = 0.01` or `0.1` depending on the contract

But the log shows `1.0000 units`, meaning `contract_multiplier = 1.0` for all — the **default fallback value**. This happens when `InstrumentManager` doesn't have specs loaded for these symbols on one of the two exchanges in the arb pair (`get_or_default()` returns `contract_multiplier: 1.0`).

**In summary:** The funding arb engine is using fallback contract specs (multiplier=1.0) and possibly wrong prices, causing it to think 1 contract = 1 whole BTC/ETH/SOL at $68k each. With your $2,943 balance and 2x leverage, it correctly rejects these because $34k margin > $2.9k balance. **The rejection logic is actually working correctly as a safety guard** — it's the input data (contract specs + prices) that's wrong.

---

## 3. Manual trade still fails: `WS_REJECT: Not login`

**Root cause: This is a direct consequence of Error #1.** The flow is:

1. WS connects ✅
2. `futures.login` is sent ✅
3. Login response is not recognized as success → `logged_in = false` ❌
4. Error logged: "futures.login failed — WS order placement will not work"
5. Subscription to `futures.orders` proceeds (for receiving updates) ✅
6. Manual trade comes in → `futures.order_place` is sent
7. Gate.io rejects: **"Not login"** because Step 3 failed ❌

The fix we implemented (sending `futures.login`) is the **correct architectural fix**. The remaining issue is that the login itself is failing — either due to credentials, response format mismatch, or timeout. The debug logs (`RUST_LOG=debug`) will show the exact response from Gate.io to pinpoint why.
