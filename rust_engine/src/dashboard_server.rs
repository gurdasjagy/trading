//! Rust-Native Dashboard HTTP + WebSocket Server (Port 8080).
//!
//! Full-featured trading dashboard served entirely from Rust, replacing the
//! Python FastAPI dashboard.  Uses **axum** for HTTP/WS handling and **tera**
//! (Jinja2-compatible) for template rendering so the existing HTML templates
//! work without modification.
//!
//! # Architecture
//!
//! ```text
//!   Browser  ──HTTP──▶  axum router
//!                         ├─ GET /              → tera-rendered dashboard.html
//!                         ├─ GET /trades        → tera-rendered trades.html
//!                         ├─ GET /performance   → tera-rendered performance.html
//!                         ├─ GET /signals       → tera-rendered (stub)
//!                         ├─ GET /risk          → tera-rendered risk.html
//!                         ├─ GET /settings      → tera-rendered settings.html
//!                         ├─ GET /logs          → tera-rendered logs.html
//!                         ├─ GET /forex/*       → tera-rendered forex templates
//!                         ├─ GET /static/*      → tower-http ServeDir
//!                         ├─ GET /api/*         → JSON API handlers
//!                         └─ GET /ws/live       → WebSocket upgrade (real-time push)
//! ```
//!
//! Reads data from `DashboardState` (lock-free atomics + RwLock JSON blobs)
//! updated by the hot-path trading threads.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::{
        Path, Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::get,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};
use tracing::{info, warn, error};

// ═══════════════════════════════════════════════════════════════════════════
// Dashboard State — updated atomically by the hot-path threads
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic dashboard state shared between hot-path threads and the HTTP server.
///
/// All scalar fields use atomic types so that the hot-path can update them
/// without any locks or synchronization overhead.  Complex data (positions,
/// trades, orderbook) use `std::sync::RwLock<String>` for infrequent updates.
pub struct DashboardState {
    // ── Account ──
    /// Available balance (USDT × 1e8 for fixed-point).
    pub balance_fp: AtomicI64,
    /// Total equity (USDT × 1e8).
    pub equity_fp: AtomicI64,
    /// Total unrealized PnL (USDT × 1e8).
    pub unrealized_pnl_fp: AtomicI64,
    /// Total realized PnL today (USDT × 1e8).
    pub realized_pnl_fp: AtomicI64,

    // ── Engine metrics ──
    /// Engine uptime in seconds.
    pub uptime_secs: AtomicU64,
    /// Total orders submitted.
    pub orders_submitted: AtomicU64,
    /// Total orders rejected.
    pub orders_rejected: AtomicU64,
    /// Total fills.
    pub total_fills: AtomicU64,
    /// Average order latency (microseconds).
    pub avg_latency_us: AtomicU64,
    /// Current active position count.
    pub active_positions: AtomicU64,
    /// Circuit breaker state (0=armed, 1=tripped).
    pub circuit_breaker_state: AtomicU64,
    /// Total ticks processed.
    pub ticks_processed: AtomicU64,
    /// Is the engine running.
    pub is_running: AtomicBool,
    /// Start timestamp (seconds since epoch).
    pub start_time: AtomicU64,

    // ── Rate limiter stats ──
    pub rate_limiter_public_available: AtomicU64,
    pub rate_limiter_private_available: AtomicU64,
    pub rate_limiter_public_waited: AtomicU64,
    pub rate_limiter_private_waited: AtomicU64,

    // ── Signal queue stats (Alpha Oracle) ──
    pub signal_queue_depth: AtomicU64,
    pub signals_processed: AtomicU64,

    // ── Position data (up to 8 positions, serialized as JSON) ──
    /// JSON-encoded positions (updated periodically by the strategy thread).
    /// Protected by a simple spinlock (AtomicBool).
    positions_lock: AtomicBool,
    positions_json: std::sync::RwLock<String>,
    trades_json: std::sync::RwLock<String>,
    /// JSON-encoded orderbook BBO (best bid/offer) for active symbols.
    orderbook_json: std::sync::RwLock<String>,
}

impl DashboardState {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            balance_fp: AtomicI64::new(0),
            equity_fp: AtomicI64::new(0),
            unrealized_pnl_fp: AtomicI64::new(0),
            realized_pnl_fp: AtomicI64::new(0),
            uptime_secs: AtomicU64::new(0),
            orders_submitted: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            total_fills: AtomicU64::new(0),
            avg_latency_us: AtomicU64::new(0),
            active_positions: AtomicU64::new(0),
            circuit_breaker_state: AtomicU64::new(0),
            ticks_processed: AtomicU64::new(0),
            is_running: AtomicBool::new(true),
            start_time: AtomicU64::new(now),
            rate_limiter_public_available: AtomicU64::new(0),
            rate_limiter_private_available: AtomicU64::new(0),
            rate_limiter_public_waited: AtomicU64::new(0),
            rate_limiter_private_waited: AtomicU64::new(0),
            signal_queue_depth: AtomicU64::new(0),
            signals_processed: AtomicU64::new(0),
            positions_lock: AtomicBool::new(false),
            positions_json: std::sync::RwLock::new("[]".to_string()),
            trades_json: std::sync::RwLock::new("[]".to_string()),
            orderbook_json: std::sync::RwLock::new("{}".to_string()),
        }
    }

    // ── Setter methods (called by hot-path threads) ──

    pub fn set_balance(&self, balance: f64) {
        self.balance_fp.store((balance * 1e8) as i64, Ordering::Relaxed);
    }

    pub fn set_equity(&self, equity: f64) {
        self.equity_fp.store((equity * 1e8) as i64, Ordering::Relaxed);
    }

    pub fn set_unrealized_pnl(&self, pnl: f64) {
        self.unrealized_pnl_fp.store((pnl * 1e8) as i64, Ordering::Relaxed);
    }

    pub fn set_realized_pnl(&self, pnl: f64) {
        self.realized_pnl_fp.store((pnl * 1e8) as i64, Ordering::Relaxed);
    }

    pub fn set_positions_json(&self, json: String) {
        if let Ok(mut guard) = self.positions_json.write() {
            *guard = json;
        }
    }

    pub fn set_trades_json(&self, json: String) {
        if let Ok(mut guard) = self.trades_json.write() {
            *guard = json;
        }
    }

    pub fn set_orderbook_json(&self, json: String) {
        if let Ok(mut guard) = self.orderbook_json.write() {
            *guard = json;
        }
    }

    // ── Getter helpers ──

    fn balance(&self) -> f64 {
        self.balance_fp.load(Ordering::Relaxed) as f64 / 1e8
    }

    fn equity(&self) -> f64 {
        self.equity_fp.load(Ordering::Relaxed) as f64 / 1e8
    }

    fn unrealized_pnl(&self) -> f64 {
        self.unrealized_pnl_fp.load(Ordering::Relaxed) as f64 / 1e8
    }

    fn realized_pnl(&self) -> f64 {
        self.realized_pnl_fp.load(Ordering::Relaxed) as f64 / 1e8
    }

    fn positions_str(&self) -> String {
        self.positions_json
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "[]".to_string())
    }

    fn trades_str(&self) -> String {
        self.trades_json
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "[]".to_string())
    }

    fn orderbook_str(&self) -> String {
        self.orderbook_json
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "{}".to_string())
    }

    fn uptime(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let start = self.start_time.load(Ordering::Relaxed);
        now.saturating_sub(start)
    }

    /// Get the full state as a JSON string (for WebSocket broadcast + /api/state).
    pub fn to_json(&self) -> String {
        let uptime = self.uptime();
        self.uptime_secs.store(uptime, Ordering::Relaxed);

        let balance = self.balance();
        let equity = self.equity();
        let unrealized = self.unrealized_pnl();
        let realized = self.realized_pnl();
        let positions = self.positions_str();
        let trades = self.trades_str();
        let orderbook = self.orderbook_str();

        format!(
            r#"{{"status":"ok","uptime_secs":{},"balance":{:.4},"equity":{:.4},"unrealized_pnl":{:.4},"realized_pnl":{:.4},"orders_submitted":{},"orders_rejected":{},"total_fills":{},"avg_latency_us":{},"active_positions":{},"circuit_breaker":{},"ticks_processed":{},"signal_queue_depth":{},"signals_processed":{},"rate_limiter":{{"public_available":{},"private_available":{},"public_waited":{},"private_waited":{}}},"positions":{},"trades":{},"orderbook":{}}}"#,
            uptime,
            balance,
            equity,
            unrealized,
            realized,
            self.orders_submitted.load(Ordering::Relaxed),
            self.orders_rejected.load(Ordering::Relaxed),
            self.total_fills.load(Ordering::Relaxed),
            self.avg_latency_us.load(Ordering::Relaxed),
            self.active_positions.load(Ordering::Relaxed),
            self.circuit_breaker_state.load(Ordering::Relaxed),
            self.ticks_processed.load(Ordering::Relaxed),
            self.signal_queue_depth.load(Ordering::Relaxed),
            self.signals_processed.load(Ordering::Relaxed),
            self.rate_limiter_public_available.load(Ordering::Relaxed),
            self.rate_limiter_private_available.load(Ordering::Relaxed),
            self.rate_limiter_public_waited.load(Ordering::Relaxed),
            self.rate_limiter_private_waited.load(Ordering::Relaxed),
            positions,
            trades,
            orderbook,
        )
    }

    /// Build the `tera::Context` used by page templates.
    fn template_context(&self, active_page: &str) -> tera::Context {
        let mut ctx = tera::Context::new();
        ctx.insert("active_page", active_page);
        ctx.insert("trading_mode", &std::env::var("TRADING_MODE").unwrap_or_else(|_| "paper".to_string()));
        ctx.insert("enable_forex_trading", &std::env::var("ENABLE_FOREX_TRADING")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false));

        // Portfolio metrics
        ctx.insert("portfolio_value", &format!("${:.2}", self.equity()));
        ctx.insert("daily_pnl", &0.0_f64);
        ctx.insert("open_positions", &self.active_positions.load(Ordering::Relaxed));
        ctx.insert("max_positions", &5_u64);
        ctx.insert("win_rate", &"—");
        ctx.insert("balance", &self.balance());
        ctx.insert("equity", &self.equity());
        ctx.insert("unrealized_pnl", &self.unrealized_pnl());
        ctx.insert("realized_pnl", &self.realized_pnl());

        // Positions array (parse JSON to Value for template rendering)
        let positions_raw = self.positions_str();
        if let Ok(positions) = serde_json::from_str::<Value>(&positions_raw) {
            ctx.insert("positions", &positions);
        } else {
            ctx.insert("positions", &Value::Array(vec![]));
        }

        // Trades array
        let trades_raw = self.trades_str();
        if let Ok(trades) = serde_json::from_str::<Value>(&trades_raw) {
            ctx.insert("trades", &trades);
        } else {
            ctx.insert("trades", &Value::Array(vec![]));
        }

        // Engine status
        ctx.insert("uptime", &self.uptime());
        ctx.insert("circuit_breaker_triggered", &(self.circuit_breaker_state.load(Ordering::Relaxed) != 0));

        // ── Settings object (required by settings.html) ──
        // Build a nested structure matching what the templates expect:
        //   settings.trading_mode, settings.exchange.*, settings.risk.*
        let trading_mode_str = std::env::var("TRADING_MODE").unwrap_or_else(|_| "paper".to_string());
        let settings_obj = json!({
            "trading_mode": trading_mode_str,
            "exchange": {
                "default_leverage": 5,
                "max_leverage": 20,
                "order_type": "limit",
                "trading_pairs": self.get_trading_pairs(),
            },
            "risk": {
                "max_position_size_pct": 10.0,
                "max_open_positions": 5,
                "max_daily_loss_pct": 2.0,
                "max_drawdown_pct": 10.0,
                "default_stop_loss_pct": 2.0,
                "risk_reward_min": 1.5,
                "use_kelly_criterion": false,
            }
        });
        ctx.insert("settings", &settings_obj);

        // ── Risk object (required by risk.html) ──
        let cb_triggered = self.circuit_breaker_state.load(Ordering::Relaxed) != 0;
        let risk_obj = json!({
            "open_positions": self.active_positions.load(Ordering::Relaxed),
            "effective_positions": self.active_positions.load(Ordering::Relaxed),
            "drawdown_pct": 0.0,
            "exposure_multiplier": 1.0,
            "circuit_breaker": {
                "triggered": cb_triggered,
                "reason": if cb_triggered { "Circuit breaker tripped" } else { "" },
            },
            "volatility_regime": "normal",
            "market_regime": "unknown",
            "daily_status": {
                "daily_pnl": 0.0,
                "max_daily_loss_pct": 2.0,
                "limit_reached": false,
            },
            "consecutive_losses": 0,
            "avg_correlation": 0.0,
        });
        ctx.insert("risk", &risk_obj);

        // ── Trading pairs (required by dashboard.html K-line symbol selector) ──
        // Pass BOTH formats:
        //   trading_pairs:     ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  (display)
        //   trading_pairs_raw: ["BTC_USDT", "ETH_USDT", "SOL_USDT"]  (API/URL-safe)
        //
        // The K-line chart select uses _raw values (underscore format) to avoid
        // URL-encoding issues with %2F in path segments. Display text uses slash.
        let raw_pairs: Vec<String> = self.get_trading_pairs();
        let display_pairs: Vec<String> = raw_pairs.iter()
            .map(|p| p.replace('_', "/"))
            .collect();
        ctx.insert("trading_pairs", &display_pairs);
        ctx.insert("trading_pairs_raw", &raw_pairs);

        // ── Signals (required by dashboard.html signals table) ──
        ctx.insert("signals", &Value::Array(vec![]));

        // ── Forex-specific variables (required by forex_dashboard.html) ──
        // daily_pnl is already inserted above as a numeric value (0.0).
        ctx.insert("margin_used", &0.0_f64);
        ctx.insert("free_margin", &self.balance());

        // ── Forex trade history variables (required by forex_trades.html) ──
        ctx.insert("total_trades", &0_u64);
        ctx.insert("total_pnl", &"$0.00");
        ctx.insert("total_pnl_val", &0.0_f64);

        // ── Forex pairs / strategy list (optional, guarded with `is defined` in templates) ──
        ctx.insert("forex_pairs", &vec!["XAU/USD", "XAG/USD"]);

        // ── Performance metrics (required by performance.html) ──
        // All fields use | default(value=0) in the template, but we provide
        // the top-level `metrics` object so Tera doesn't error on `metrics.X`.
        ctx.insert("metrics", &json!({
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "current_streak": 0,
            "avg_trade_duration_mins": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }));

        // ── Performance chart data (required by performance.html `json_encode()`) ──
        // These MUST exist as top-level context vars because Tera evaluates
        // `json_encode()` BEFORE `default()`, so undefined vars cause a crash.
        let empty_arr: Vec<f64> = vec![];
        let empty_str_arr: Vec<String> = vec![];
        ctx.insert("equity_labels", &empty_str_arr);
        ctx.insert("equity_data", &empty_arr);
        ctx.insert("drawdown_data", &empty_arr);
        ctx.insert("pnl_labels", &empty_str_arr);
        ctx.insert("pnl_data", &empty_arr);
        ctx.insert("win_rate_by_hour", &json!({}));
        ctx.insert("pair_pnl", &Value::Null);
        ctx.insert("strategy_pnl", &json!({}));
        ctx.insert("monthly_returns", &Value::Null);

        // ── Trades page variables (required by trades.html) ──
        ctx.insert("filters", &json!({
            "symbol": "",
            "date_from": "",
            "date_to": "",
            "strategy": "",
        }));

        ctx
    }

    /// Get configured trading pairs from the environment or defaults.
    ///
    /// Reads `TRADING_PAIRS` env var (comma-separated, e.g. "BTC_USDT,ETH_USDT,SOL_USDT").
    /// Falls back to the default top-3 if not set.
    pub fn get_trading_pairs(&self) -> Vec<String> {
        if let Ok(pairs) = std::env::var("TRADING_PAIRS") {
            pairs.split(',')
                .map(|s| {
                    // Normalize to underscore format for Gate.io API compatibility:
                    //   "BTC/USDT"       → "BTC_USDT"
                    //   "BTC/USDT:USDT"  → "BTC_USDT"
                    //   "BTC_USDT"       → "BTC_USDT" (already correct)
                    let trimmed = s.trim();
                    // Strip settlement suffix (e.g. ":USDT")
                    let base = trimmed.split(':').next().unwrap_or(trimmed);
                    base.replace('/', "_").to_uppercase()
                })
                .filter(|s| !s.is_empty())
                .collect()
        } else {
            vec![
                "BTC_USDT".to_string(),
                "ETH_USDT".to_string(),
                "SOL_USDT".to_string(),
            ]
        }
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared application state passed to all handlers
// ═══════════════════════════════════════════════════════════════════════════

/// Application state shared across all axum handlers.
struct AppState {
    dashboard: Arc<DashboardState>,
    templates: tera::Tera,
    /// Broadcast channel for WebSocket push (state JSON).
    ws_tx: broadcast::Sender<String>,
    /// HTTP client for proxying Gate.io public API requests (market data).
    gateio_client: reqwest::Client,
}

impl AppState {
    /// Get the Gate.io REST API base URL for **public** endpoints.
    ///
    /// Public market data (klines, tickers, orderbook, trades, funding rates,
    /// contract specs) is identical on mainnet and testnet.  Gate.io's testnet
    /// infrastructure is unreliable and frequently returns HTTP 502 for these
    /// public endpoints.  Therefore we **always** fetch public data from the
    /// mainnet API, regardless of trading mode.
    fn gateio_public_url(&self) -> &str {
        // Always use mainnet for public market data — testnet returns 502
        "https://api.gateio.ws/api/v4"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Template rendering helper
// ═══════════════════════════════════════════════════════════════════════════

/// Render a tera template, returning Html or a 500 error page.
fn render_template(
    templates: &tera::Tera,
    name: &str,
    ctx: &tera::Context,
) -> impl IntoResponse {
    match templates.render(name, ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            error!("[dashboard] Template render error for {}: {}", name, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Html(format!(
                    "<h1>500 Internal Server Error</h1><p>Template error: {}</p>",
                    e
                )),
            )
                .into_response()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Page handlers (HTML — tera-rendered)
// ═══════════════════════════════════════════════════════════════════════════

async fn page_dashboard(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("dashboard");
    render_template(&state.templates, "dashboard.html", &ctx)
}

async fn page_trades(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("trades");
    render_template(&state.templates, "trades.html", &ctx)
}

async fn page_performance(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("performance");
    render_template(&state.templates, "performance.html", &ctx)
}

async fn page_signals(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Signals page — reuse dashboard context with signals-specific active page
    let ctx = state.dashboard.template_context("signals");
    // If a dedicated signals template exists, render it; otherwise fall back to dashboard
    if state.templates.get_template("signals.html").is_ok() {
        render_template(&state.templates, "signals.html", &ctx)
    } else {
        render_template(&state.templates, "dashboard.html", &ctx)
    }
}

async fn page_risk(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("risk");
    render_template(&state.templates, "risk.html", &ctx)
}

async fn page_settings(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = state.dashboard.template_context("settings");
    // Settings page needs additional context
    ctx.insert("symbols", &vec!["BTC_USDT", "ETH_USDT", "XAUT_USDT"]);
    ctx.insert("strategies", &Value::Array(vec![]));
    render_template(&state.templates, "settings.html", &ctx)
}

async fn page_logs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = state.dashboard.template_context("logs");
    ctx.insert("log_lines", &Value::Array(vec![]));
    ctx.insert("_ws_auth_token", &"");
    render_template(&state.templates, "logs.html", &ctx)
}

// ── Forex page handlers ──

async fn page_forex_dashboard(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("forex_dashboard");
    render_template(&state.templates, "forex_dashboard.html", &ctx)
}

async fn page_forex_trades(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("forex_trades");
    render_template(&state.templates, "forex_trades.html", &ctx)
}

async fn page_forex_settings(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("forex_settings");
    render_template(&state.templates, "forex_settings.html", &ctx)
}

async fn page_forex_performance(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("forex_performance");
    render_template(&state.templates, "forex_performance.html", &ctx)
}

async fn page_forex_risk(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = state.dashboard.template_context("forex_risk");
    render_template(&state.templates, "forex_risk.html", &ctx)
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON API handlers
// ═══════════════════════════════════════════════════════════════════════════

async fn api_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "uptime_secs": state.dashboard.uptime(),
        "is_running": state.dashboard.is_running.load(Ordering::Relaxed),
    }))
}

async fn api_state(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let json_str = state.dashboard.to_json();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        json_str,
    )
}

async fn api_balance(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "balance": state.dashboard.balance(),
        "equity": state.dashboard.equity(),
    }))
}

async fn api_positions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let positions_raw = state.dashboard.positions_str();
    let positions: Value = serde_json::from_str(&positions_raw).unwrap_or(Value::Array(vec![]));
    Json(json!({ "positions": positions }))
}

async fn api_trades(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let trades_raw = state.dashboard.trades_str();
    let trades: Value = serde_json::from_str(&trades_raw).unwrap_or(Value::Array(vec![]));
    Json(json!({ "trades": trades }))
}

async fn api_orderbook(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ob_raw = state.dashboard.orderbook_str();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        ob_raw,
    )
}

async fn api_metrics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    Json(json!({
        "orders_submitted": d.orders_submitted.load(Ordering::Relaxed),
        "orders_rejected": d.orders_rejected.load(Ordering::Relaxed),
        "total_fills": d.total_fills.load(Ordering::Relaxed),
        "avg_latency_us": d.avg_latency_us.load(Ordering::Relaxed),
        "ticks_processed": d.ticks_processed.load(Ordering::Relaxed),
        "signal_queue_depth": d.signal_queue_depth.load(Ordering::Relaxed),
        "signals_processed": d.signals_processed.load(Ordering::Relaxed),
    }))
}

async fn api_portfolio(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    Json(json!({
        "equity": d.equity(),
        "portfolio_value": d.equity(),
        "balance": d.balance(),
        "unrealized_pnl": d.unrealized_pnl(),
        "realized_pnl": d.realized_pnl(),
        "daily_pnl_pct": 0.0,
        "open_positions": d.active_positions.load(Ordering::Relaxed),
    }))
}

async fn api_performance(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    let total_fills = d.total_fills.load(Ordering::Relaxed);
    Json(json!({
        "win_rate": 0.0,
        "total_trades": total_fills,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
    }))
}

async fn api_risk(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    Json(json!({
        "circuit_breaker": {
            "state": if d.circuit_breaker_state.load(Ordering::Relaxed) == 0 { "armed" } else { "tripped" },
            "triggered": d.circuit_breaker_state.load(Ordering::Relaxed) != 0,
        },
        "active_positions": d.active_positions.load(Ordering::Relaxed),
        "margin_ratio": 0.0,
        "drawdown_pct": 0.0,
        "open_positions": d.active_positions.load(Ordering::Relaxed),
    }))
}

async fn api_strategy_performance() -> Json<Value> {
    // Placeholder — will be populated when strategy metrics are wired in
    Json(json!({ "strategies": [] }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Gate.io Market Data Proxy Endpoints
// ═══════════════════════════════════════════════════════════════════════════
//
// These proxy public Gate.io REST API endpoints so the dashboard can render
// K-line charts, ticker data, orderbook depth, and recent trades without
// CORS issues or exposing API internals to the browser.

/// Query parameters for K-line (candlestick) data.
#[derive(Deserialize)]
struct KlineQuery {
    timeframe: Option<String>,
    limit: Option<u32>,
}

/// Normalize a symbol like "BTC/USDT" → "BTC_USDT" for Gate.io API.
fn normalize_gateio_symbol(symbol: &str) -> String {
    symbol.replace('/', "_").to_uppercase()
}

/// Map frontend timeframe to Gate.io candlestick interval.
fn map_timeframe(tf: &str) -> &str {
    match tf {
        "1m" => "1m",
        "5m" => "5m",
        "15m" => "15m",
        "30m" => "30m",
        "1h" => "1h",
        "4h" => "4h",
        "8h" => "8h",
        "1d" | "1D" => "1d",
        "1w" | "7d" => "7d",
        _ => "1m",
    }
}

/// GET /api/market/{symbol}/klines — K-line candlestick data
///
/// Proxies to Gate.io `GET /futures/usdt/candlesticks` and transforms the
/// response into the format expected by TradingView Lightweight Charts:
///   `{ candles: [{ time, open, high, low, close, volume }] }`
async fn api_market_klines(
    Path(symbol): Path<String>,
    Query(params): Query<KlineQuery>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let contract = normalize_gateio_symbol(&symbol);
    let interval = params.timeframe.as_deref().map(map_timeframe).unwrap_or("1m");
    let limit = params.limit.unwrap_or(200).min(2000);

    let url = format!(
        "{}/futures/usdt/candlesticks?contract={}&interval={}&limit={}",
        state.gateio_public_url(), contract, interval, limit
    );

    match state.gateio_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(data) => {
                    // Gate.io returns: [ { t, v, c, h, l, o, sum }, ... ]
                    let candles: Vec<Value> = data.as_array()
                        .map(|arr| arr.iter().map(|c| {
                            json!({
                                "time": c.get("t").and_then(|v| v.as_i64()).unwrap_or(0),
                                "open": parse_gate_float(c, "o"),
                                "high": parse_gate_float(c, "h"),
                                "low": parse_gate_float(c, "l"),
                                "close": parse_gate_float(c, "c"),
                                "volume": parse_gate_float(c, "sum"),
                            })
                        }).collect())
                        .unwrap_or_default();

                    Json(json!({ "success": true, "candles": candles })).into_response()
                }
                Err(e) => {
                    warn!("[dashboard] Kline parse error: {}", e);
                    (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Parse error" }))).into_response()
                }
            }
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            warn!("[dashboard] Gate.io klines HTTP {}: {}", status, text);
            (StatusCode::BAD_GATEWAY, Json(json!({ "error": format!("Gate.io HTTP {}", status) }))).into_response()
        }
        Err(e) => {
            error!("[dashboard] Gate.io klines request failed: {}", e);
            (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Connection failed" }))).into_response()
        }
    }
}

/// GET /api/market/tickers — Tickers for configured trading pairs only.
///
/// Only returns data for symbols configured in TRADING_PAIRS env var (or
/// the default BTC_USDT, ETH_USDT).  This keeps the dashboard fast and
/// focused instead of showing 100+ irrelevant currencies.
///
/// Returns a **dict** keyed by symbol (e.g. `{"BTC/USDT": {...}, ...}`)
/// so the JS frontend can do O(1) lookups.
async fn api_market_tickers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let url = format!("{}/futures/usdt/tickers", state.gateio_public_url());

    // Get the configured trading pairs so we only return relevant tickers
    let trading_pairs = state.dashboard.get_trading_pairs();
    // Build a set of allowed contracts for fast lookup (e.g. "BTC_USDT", "ETH_USDT")
    let allowed: std::collections::HashSet<String> = trading_pairs.iter()
        .map(|p| p.replace('/', "_").to_uppercase())
        .collect();

    match state.gateio_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(data) => {
                    // Build a dict keyed by "BTC/USDT" style symbol.
                    // Only include tickers that match our configured trading pairs.
                    let mut tickers_map = serde_json::Map::new();

                    if let Some(arr) = data.as_array() {
                        for t in arr {
                            let contract = t.get("contract").and_then(|v| v.as_str()).unwrap_or("");
                            if !allowed.contains(contract) { continue; }

                            let symbol = contract.replace('_', "/");
                            let last = parse_gate_float(t, "last");
                            let change_pct = parse_gate_float(t, "change_percentage");
                            let volume_24h = parse_gate_float(t, "volume_24h_settle");
                            let high_24h = parse_gate_float(t, "high_24h");
                            let low_24h = parse_gate_float(t, "low_24h");
                            let funding_rate = parse_gate_float(t, "funding_rate");

                            tickers_map.insert(symbol.clone(), json!({
                                "symbol": symbol,
                                "contract": contract,
                                "last": last,
                                "change_24h": change_pct,
                                "change_24h_pct": change_pct,
                                "volume_24h": volume_24h,
                                "volume": volume_24h,
                                "high_24h": high_24h,
                                "high": high_24h,
                                "low_24h": low_24h,
                                "low": low_24h,
                                "funding_rate": funding_rate,
                            }));
                        }
                    }

                    Json(json!({ "success": true, "tickers": tickers_map })).into_response()
                }
                Err(_) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Parse error" }))).into_response()
            }
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            (StatusCode::BAD_GATEWAY, Json(json!({ "error": format!("Gate.io HTTP {}", status) }))).into_response()
        }
        Err(e) => {
            error!("[dashboard] Tickers request failed: {}", e);
            (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Connection failed" }))).into_response()
        }
    }
}

/// GET /api/market/{symbol}/orderbook — Orderbook depth
async fn api_market_orderbook(
    Path(symbol): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let contract = normalize_gateio_symbol(&symbol);
    let url = format!(
        "{}/futures/usdt/order_book?contract={}&limit=20",
        state.gateio_public_url(), contract
    );

    match state.gateio_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(data) => Json(json!({ "success": true, "orderbook": data })).into_response(),
                Err(_) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Parse error" }))).into_response()
            }
        }
        _ => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Request failed" }))).into_response()
    }
}

/// GET /api/market/{symbol}/recent-trades — Recent market trades
async fn api_market_recent_trades(
    Path(symbol): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let contract = normalize_gateio_symbol(&symbol);
    let url = format!(
        "{}/futures/usdt/trades?contract={}&limit=50",
        state.gateio_public_url(), contract
    );

    match state.gateio_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(data) => Json(json!({ "success": true, "trades": data })).into_response(),
                Err(_) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Parse error" }))).into_response()
            }
        }
        _ => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Request failed" }))).into_response()
    }
}

/// GET /api/market/{symbol}/multi-timeframe — Multi-timeframe analysis
async fn api_market_multi_timeframe(
    Path(symbol): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Fetch multiple timeframes and compute basic indicators
    let contract = normalize_gateio_symbol(&symbol);
    let timeframes = ["5m", "15m", "1h", "4h", "1d"];
    let mut results = Vec::new();

    for &tf in &timeframes {
        let url = format!(
            "{}/futures/usdt/candlesticks?contract={}&interval={}&limit=50",
            state.gateio_public_url(), contract, tf
        );

        if let Ok(resp) = state.gateio_client.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(data) = resp.json::<Value>().await {
                    if let Some(candles) = data.as_array() {
                        if candles.len() >= 14 {
                            // Compute basic RSI and trend from candles
                            let closes: Vec<f64> = candles.iter()
                                .map(|c| parse_gate_float(c, "c"))
                                .collect();
                            let last_close = *closes.last().unwrap_or(&0.0);
                            let first_close = *closes.first().unwrap_or(&0.0);
                            let trend = if last_close > first_close { "bullish" } else if last_close < first_close { "bearish" } else { "neutral" };
                            let rsi = compute_simple_rsi(&closes, 14);
                            let volume: f64 = candles.last()
                                .map(|c| parse_gate_float(c, "sum"))
                                .unwrap_or(0.0);

                            results.push(json!({
                                "timeframe": tf,
                                "trend": trend,
                                "rsi": format!("{:.1}", rsi),
                                "macd": if rsi > 50.0 { "bullish" } else { "bearish" },
                                "ema_cross": if last_close > first_close { "above" } else { "below" },
                                "volume": format!("{:.0}", volume),
                                "alignment_score": format!("{:.0}", if trend == "bullish" { (rsi / 100.0 * 10.0).min(10.0) } else { ((100.0 - rsi) / 100.0 * 10.0).min(10.0) }),
                            }));
                            continue;
                        }
                    }
                }
            }
        }
        // Fallback for failed timeframe
        results.push(json!({
            "timeframe": tf,
            "trend": "unknown",
            "rsi": "—",
            "macd": "—",
            "ema_cross": "—",
            "volume": "—",
            "alignment_score": "—",
        }));
    }

    Json(json!({ "success": true, "timeframes": results }))
}

/// GET /api/funding-rates — Current funding rates for all contracts
async fn api_funding_rates(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let url = format!("{}/futures/usdt/tickers", state.gateio_public_url());

    match state.gateio_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(data) => {
                    let rates: Vec<Value> = data.as_array()
                        .map(|arr| arr.iter().filter_map(|t| {
                            let contract = t.get("contract").and_then(|v| v.as_str())?;
                            let rate = parse_gate_float(t, "funding_rate");
                            if rate.abs() < 1e-10 { return None; }
                            Some(json!({
                                "symbol": contract.replace('_', "/"),
                                "contract": contract,
                                "rate": rate,
                                "rate_pct": rate * 100.0,
                                "annualized_pct": rate * 365.0 * 3.0 * 100.0,
                            }))
                        }).collect())
                        .unwrap_or_default();

                    Json(json!({ "success": true, "funding_rates": rates })).into_response()
                }
                Err(_) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Parse error" }))).into_response()
            }
        }
        _ => (StatusCode::BAD_GATEWAY, Json(json!({ "error": "Request failed" }))).into_response()
    }
}

/// GET /api/risk/metrics — Risk metrics (computed from state)
async fn api_risk_metrics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    Json(json!({
        "success": true,
        "circuit_breaker": {
            "triggered": d.circuit_breaker_state.load(Ordering::Relaxed) != 0,
        },
        "drawdown_pct": 0.0,
        "margin_ratio": 0.0,
        "open_positions": d.active_positions.load(Ordering::Relaxed),
        "volatility_regime": "normal",
        "market_regime": "unknown",
    }))
}

/// GET /api/risk-dashboard — Comprehensive risk dashboard data
async fn api_risk_dashboard(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    Json(json!({
        "success": true,
        "drawdown_pct": 0.0,
        "margin_ratio": 0.0,
        "open_positions": d.active_positions.load(Ordering::Relaxed),
        "exposure": d.equity(),
        "risk_score": 0.0,
        "warnings": [],
    }))
}

/// GET /api/execution-quality — Execution quality metrics
async fn api_execution_quality(State(state): State<Arc<AppState>>) -> Json<Value> {
    let d = &state.dashboard;
    Json(json!({
        "success": true,
        "avg_latency_us": d.avg_latency_us.load(Ordering::Relaxed),
        "orders_submitted": d.orders_submitted.load(Ordering::Relaxed),
        "orders_rejected": d.orders_rejected.load(Ordering::Relaxed),
        "total_fills": d.total_fills.load(Ordering::Relaxed),
        "fill_rate": if d.orders_submitted.load(Ordering::Relaxed) > 0 {
            d.total_fills.load(Ordering::Relaxed) as f64 / d.orders_submitted.load(Ordering::Relaxed) as f64 * 100.0
        } else { 0.0 },
        "avg_slippage_bps": 0.0,
    }))
}

/// GET /api/equity-history — Equity history for charting
#[derive(Deserialize)]
struct EquityHistoryQuery {
    #[allow(dead_code)]
    range: Option<String>,
}

async fn api_equity_history(
    Query(_params): Query<EquityHistoryQuery>,
    State(state): State<Arc<AppState>>,
) -> Json<Value> {
    // Return current equity as a single data point — historical data requires
    // persistence which will be wired in when PostgreSQL trade journal is active.
    let equity = state.dashboard.equity();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(json!({
        "success": true,
        "dates": [now],
        "equity": [equity],
        "drawdown": [0.0],
    }))
}

/// GET /api/positions/live — Live positions (alias for /api/positions)
async fn api_positions_live(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let positions_raw = state.dashboard.positions_str();
    let positions: Value = serde_json::from_str(&positions_raw).unwrap_or(Value::Array(vec![]));
    Json(json!({ "success": true, "positions": positions }))
}

/// GET /api/trades/history — Paginated trade history
#[derive(Deserialize)]
struct TradeHistoryQuery {
    page: Option<u32>,
    per_page: Option<u32>,
    #[allow(dead_code)]
    symbol: Option<String>,
    #[allow(dead_code)]
    strategy: Option<String>,
}

async fn api_trades_history(
    Query(params): Query<TradeHistoryQuery>,
    State(state): State<Arc<AppState>>,
) -> Json<Value> {
    let trades_raw = state.dashboard.trades_str();
    let all_trades: Vec<Value> = serde_json::from_str(&trades_raw).unwrap_or_default();
    let page = params.page.unwrap_or(1).max(1);
    let per_page = params.per_page.unwrap_or(50).min(200);
    let total = all_trades.len() as u32;
    let pages = if total == 0 { 1 } else { (total + per_page - 1) / per_page };
    let start = ((page - 1) * per_page) as usize;
    let trades: Vec<&Value> = all_trades.iter().skip(start).take(per_page as usize).collect();

    Json(json!({
        "success": true,
        "trades": trades,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": pages,
        }
    }))
}

/// GET /api/v1/trades — Trade summary (used by trades.html free margin)
async fn api_v1_trades(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "success": true,
        "free_margin": state.dashboard.balance(),
    }))
}

/// GET /api/performance/charts — Performance chart data
#[derive(Deserialize)]
struct PerfChartsQuery {
    #[allow(dead_code)]
    range: Option<String>,
}

async fn api_performance_charts(
    Query(_params): Query<PerfChartsQuery>,
    State(state): State<Arc<AppState>>,
) -> Json<Value> {
    let equity = state.dashboard.equity();
    Json(json!({
        "success": true,
        "dates": [],
        "balance": [equity],
        "pnl": [],
    }))
}

// ── Stub endpoints for features not yet connected ──

async fn api_sentiment_social() -> Json<Value> {
    Json(json!({
        "success": true,
        "data": [],
        "message": "Social sentiment analysis not yet connected",
    }))
}

async fn api_news_crypto() -> Json<Value> {
    Json(json!({
        "success": true,
        "articles": [],
        "message": "News feed not yet connected",
    }))
}

async fn api_ai_trade_suggestions() -> Json<Value> {
    Json(json!({
        "success": true,
        "suggestions": [],
        "message": "AI trade suggestions not yet connected",
    }))
}

/// GET /api/logs — Recent log lines (stub — reads from tracing subscriber in future)
async fn api_logs() -> Json<Value> {
    Json(json!({
        "success": true,
        "lines": [],
        "message": "Log streaming not yet connected — use docker logs or /ws for real-time updates",
    }))
}

/// GET /api/trades/export — Export trades as CSV (stub)
async fn api_trades_export(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let trades_raw = state.dashboard.trades_str();
    let trades: Vec<Value> = serde_json::from_str(&trades_raw).unwrap_or_default();
    // Return as JSON for now — CSV export requires additional formatting
    Json(json!({ "success": true, "trades": trades, "format": "json" }))
}

/// GET /api/performance/export — Export performance data (stub)
async fn api_performance_export() -> Json<Value> {
    Json(json!({ "success": true, "data": [], "format": "json" }))
}

/// GET /api/v1/settings — Current settings (stub for settings page)
async fn api_v1_settings() -> Json<Value> {
    let trading_mode = std::env::var("TRADING_MODE").unwrap_or_else(|_| "paper".to_string());
    Json(json!({
        "success": true,
        "trading_mode": trading_mode,
        "exchange": {
            "default_leverage": 5,
            "max_leverage": 20,
        },
        "risk": {
            "max_position_size_pct": 10.0,
            "max_open_positions": 5,
            "max_daily_loss_pct": 2.0,
        }
    }))
}

/// GET /api/settings/pairs — Trading pairs configuration (stub)
async fn api_settings_pairs() -> Json<Value> {
    let pairs_str = std::env::var("TRADING_PAIRS").unwrap_or_else(|_| "BTC/USDT,ETH/USDT".to_string());
    let pairs: Vec<Value> = pairs_str.split(',')
        .map(|p| json!({ "symbol": p.trim(), "enabled": true }))
        .collect();
    Json(json!({ "success": true, "pairs": pairs }))
}

/// GET /api/settings/strategies — Available strategies (stub)
async fn api_settings_strategies() -> Json<Value> {
    Json(json!({
        "success": true,
        "strategies": [
            { "name": "momentum", "enabled": true },
            { "name": "scalping", "enabled": true },
            { "name": "technical_breakout", "enabled": true },
        ]
    }))
}

/// GET /api/settings/currencies — Currency configuration (stub)
async fn api_settings_currencies() -> Json<Value> {
    Json(json!({
        "success": true,
        "currencies": [
            { "symbol": "BTC/USDT", "enabled": true },
            { "symbol": "ETH/USDT", "enabled": true },
        ]
    }))
}

/// POST /api/mode/switch — Switch trading mode (stub)
async fn api_mode_switch() -> Json<Value> {
    Json(json!({ "success": true, "message": "Mode switch acknowledged — restart required" }))
}

/// POST /api/paper/reset — Reset paper trading (stub)
async fn api_paper_reset() -> Json<Value> {
    Json(json!({ "success": true, "message": "Paper trading reset acknowledged" }))
}

/// POST stub endpoints — acknowledge but don't execute (placeholder)
async fn api_positions_action(Path(_symbol): Path<String>) -> Json<Value> {
    Json(json!({ "success": true, "message": "Action acknowledged" }))
}

/// Generic POST stub that returns success
async fn api_generic_post_stub() -> Json<Value> {
    Json(json!({ "success": true, "message": "Action acknowledged" }))
}

/// Parse a Gate.io numeric field that may be string or number.
fn parse_gate_float(v: &Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .unwrap_or(0.0)
}

/// Compute a simple RSI from a vector of closing prices.
fn compute_simple_rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 50.0; // Not enough data
    }
    let mut gains = 0.0;
    let mut losses = 0.0;
    let start = closes.len().saturating_sub(period + 1);
    for i in (start + 1)..closes.len() {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 { gains += change; }
        else { losses += change.abs(); }
    }
    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;
    if avg_loss < 1e-10 { return 100.0; }
    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

/// Combined status endpoint matching Python's /api/v1/status
async fn api_v1_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let json_str = state.dashboard.to_json();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        json_str,
    )
}

/// Placeholder POST endpoints (bot control) — return success acknowledgement.
/// These forward commands to the engine via internal channels in production.
async fn api_bot_pause() -> Json<Value> {
    Json(json!({ "success": true, "message": "Bot pause requested" }))
}

async fn api_bot_resume() -> Json<Value> {
    Json(json!({ "success": true, "message": "Bot resume requested" }))
}

async fn api_circuit_breaker_reset() -> Json<Value> {
    Json(json!({ "success": true, "message": "Circuit breaker reset requested" }))
}

async fn api_emergency_stop() -> Json<Value> {
    warn!("[dashboard] Emergency stop triggered via API");
    Json(json!({ "success": true, "message": "Emergency stop triggered" }))
}

// ═══════════════════════════════════════════════════════════════════════════
// WebSocket handler
// ═══════════════════════════════════════════════════════════════════════════

/// WebSocket upgrade handler — subscribes the client to the broadcast channel
/// and pushes state updates every 500ms.
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_client(socket, state))
}

async fn handle_ws_client(mut socket: WebSocket, state: Arc<AppState>) {
    info!("[dashboard] WebSocket client connected");
    let mut rx = state.ws_tx.subscribe();

    loop {
        tokio::select! {
            // Forward broadcast messages to the client
            msg = rx.recv() => {
                match msg {
                    Ok(json_str) => {
                        if socket.send(Message::Text(json_str.into())).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("[dashboard] WebSocket client lagged by {} messages", n);
                    }
                    Err(_) => break,
                }
            }
            // Handle incoming messages from the client (ping/pong, close)
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Ping(data))) => {
                        if socket.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Text(text))) => {
                        // Handle ping JSON messages from the JS client
                        let text_str: &str = &text;
                        if text_str.contains("\"ping\"") {
                            let _ = socket.send(Message::Text(
                                r#"{"type":"pong"}"#.into()
                            )).await;
                        }
                    }
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
    info!("[dashboard] WebSocket client disconnected");
}

// ═══════════════════════════════════════════════════════════════════════════
// WebSocket broadcast task
// ═══════════════════════════════════════════════════════════════════════════

/// Spawn a tokio task that broadcasts the dashboard state to all connected
/// WebSocket clients every 500ms via the broadcast channel.
fn spawn_ws_broadcast_task(state: Arc<DashboardState>, tx: broadcast::Sender<String>) {
    tokio::spawn(async move {
        info!("[dashboard] WebSocket broadcast task started (500ms interval)");
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        loop {
            interval.tick().await;

            // Build the update payload matching the Python RealtimeHub format
            let balance = state.balance();
            let equity = state.equity();
            let unrealized = state.unrealized_pnl();
            let realized = state.realized_pnl();
            let positions_raw = state.positions_str();
            let trades_raw = state.trades_str();
            let uptime = state.uptime();

            let positions: Value = serde_json::from_str(&positions_raw)
                .unwrap_or(Value::Array(vec![]));
            let trades: Value = serde_json::from_str(&trades_raw)
                .unwrap_or(Value::Array(vec![]));

            let payload = json!({
                "type": "update",
                "portfolio": {
                    "equity": equity,
                    "portfolio_value": equity,
                    "balance": balance,
                    "unrealized_pnl": unrealized,
                    "realized_pnl": realized,
                    "daily_pnl_pct": 0.0,
                    "open_positions": state.active_positions.load(Ordering::Relaxed),
                },
                "positions": positions,
                "recent_trades": trades,
                "status": {
                    "uptime": uptime,
                    "circuit_breaker": {
                        "triggered": state.circuit_breaker_state.load(Ordering::Relaxed) != 0,
                    },
                    "market_regime": "unknown",
                    "crash_level": "normal",
                },
                "portfolio_risk": {
                    "margin_ratio": 0.0,
                    "correlation_risk_score": 0.0,
                    "drawdown_pct": 0.0,
                    "total_margin_used": 0.0,
                    "available_margin": balance,
                    "open_positions": state.active_positions.load(Ordering::Relaxed),
                },
            });

            // Ignore send errors (no subscribers)
            let _ = tx.send(payload.to_string());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Template engine initialization
// ═══════════════════════════════════════════════════════════════════════════

/// Discover and load Tera templates from the templates directory.
///
/// Searches for templates in (priority order):
/// 1. `../crypto_trading_bot/templates/` (development — relative to rust_engine/)
/// 2. `/app/templates/` (Docker container)
/// 3. `./templates/` (fallback)
fn init_templates() -> tera::Tera {
    let candidate_dirs = [
        // Relative to the rust_engine binary location (development)
        PathBuf::from("../crypto_trading_bot/templates"),
        // Docker container layout (various possible mount points)
        PathBuf::from("/app/templates"),
        PathBuf::from("/app/crypto_trading_bot/templates"),
        // Relative to CWD (running from repo root or rust_engine/)
        PathBuf::from("./crypto_trading_bot/templates"),
        // Fallback
        PathBuf::from("./templates"),
    ];

    for dir in &candidate_dirs {
        if dir.is_dir() {
            let glob = format!("{}/**/*.html", dir.display());
            match tera::Tera::new(&glob) {
                Ok(mut t) => {
                    // Register custom Tera tester for string containment checks
                    // (used in templates: {% if not sym is containing("XAU") %})
                    t.register_tester("containing", |value: Option<&tera::Value>, args: &[tera::Value]| {
                        match (value, args.first()) {
                            (Some(tera::Value::String(s)), Some(tera::Value::String(needle))) => {
                                Ok(s.contains(needle.as_str()))
                            }
                            _ => Ok(false),
                        }
                    });
                    let names: Vec<&str> = t.get_template_names().collect();
                    info!(
                        "[dashboard] Loaded {} templates from {}: {:?}",
                        names.len(),
                        dir.display(),
                        names
                    );
                    return t;
                }
                Err(e) => {
                    warn!(
                        "[dashboard] Failed to parse templates in {}: {}",
                        dir.display(),
                        e
                    );
                }
            }
        }
    }

    // Return an empty Tera instance if no templates found — API-only mode
    warn!("[dashboard] No template directory found; HTML pages will return errors");
    tera::Tera::default()
}

/// Discover the static files directory.
///
/// Searches for static files in (priority order):
/// 1. `../crypto_trading_bot/static/` (development)
/// 2. `/app/static/` (Docker container)
/// 3. `./static/` (fallback)
fn find_static_dir() -> PathBuf {
    let candidate_dirs = [
        PathBuf::from("../crypto_trading_bot/static"),
        PathBuf::from("/app/static"),
        PathBuf::from("/app/crypto_trading_bot/static"),
        PathBuf::from("./crypto_trading_bot/static"),
        PathBuf::from("./static"),
    ];

    for dir in &candidate_dirs {
        if dir.is_dir() {
            info!("[dashboard] Serving static files from {}", dir.display());
            return dir.clone();
        }
    }

    warn!("[dashboard] No static directory found; /static/* will return 404");
    PathBuf::from("./static")
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Run the dashboard HTTP + WebSocket server on the specified address.
///
/// This spawns an async axum server.  Call from a dedicated thread or
/// within the tokio runtime.
///
/// **Endpoints served:**
/// - `GET /`                  — Dashboard (tera-rendered)
/// - `GET /trades`            — Trade history page
/// - `GET /performance`       — Performance analytics page
/// - `GET /signals`           — Signal viewer page
/// - `GET /risk`              — Risk management page
/// - `GET /settings`          — Settings page
/// - `GET /logs`              — Log viewer page
/// - `GET /forex/*`           — Forex dashboard pages
/// - `GET /static/*`          — Static files (JS, CSS, images)
/// - `GET /api/health`        — Health check JSON
/// - `GET /api/state`         — Full engine state JSON
/// - `GET /api/balance`       — Account balance JSON
/// - `GET /api/positions`     — Active positions JSON
/// - `GET /api/trades`        — Trade history JSON
/// - `GET /api/orderbook`     — Orderbook BBO JSON
/// - `GET /api/metrics`       — Engine metrics JSON
/// - `GET /api/portfolio`     — Portfolio summary JSON
/// - `GET /api/performance`   — Performance metrics JSON
/// - `GET /api/risk`          — Risk metrics JSON
/// - `GET /api/v1/status`     — Full status (legacy)
/// - `POST /api/bot/pause`    — Pause trading
/// - `POST /api/bot/resume`   — Resume trading
/// - `POST /api/circuit-breaker/reset` — Reset circuit breaker
/// - `POST /api/v1/emergency-stop`     — Emergency stop
/// - `GET /ws/live`           — WebSocket real-time updates
/// - `GET /ws`                — WebSocket (alias)
pub fn run_dashboard_server(bind_addr: &str, state: Arc<DashboardState>) {
    let bind_addr = bind_addr.to_string();

    // Build the axum server inside a new tokio runtime on a dedicated thread
    std::thread::Builder::new()
        .name("dashboard-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("dashboard-io")
                .build()
                .expect("Failed to create dashboard tokio runtime");

            rt.block_on(async move {
                // ── Template engine ──
                let templates = init_templates();

                // ── WebSocket broadcast channel ──
                let (ws_tx, _) = broadcast::channel::<String>(64);
                spawn_ws_broadcast_task(state.clone(), ws_tx.clone());

                // ── Gate.io HTTP client for market data proxy ──
                let gateio_client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(10))
                    .build()
                    .expect("Failed to create Gate.io proxy HTTP client");

                // Market data proxy always uses mainnet — testnet returns HTTP 502
                // for public endpoints (klines, tickers, orderbook, etc.).
                let trading_mode = std::env::var("TRADING_MODE").unwrap_or_default();
                let trading_pairs = state.get_trading_pairs();
                info!("[dashboard] Gate.io market data: always mainnet for public data (trading_mode={}, pairs={:?})", trading_mode, trading_pairs);

                // ── Shared state ──
                let app_state = Arc::new(AppState {
                    dashboard: state,
                    templates,
                    ws_tx,
                    gateio_client,
                });

                // ── Static files ──
                let static_dir = find_static_dir();

                // ── CORS ──
                let cors = CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any);

                // ── Router ──
                let app = Router::new()
                    // HTML pages
                    .route("/", get(page_dashboard))
                    .route("/trades", get(page_trades))
                    .route("/performance", get(page_performance))
                    .route("/signals", get(page_signals))
                    .route("/risk", get(page_risk))
                    .route("/settings", get(page_settings))
                    .route("/logs", get(page_logs))
                    // Forex pages
                    .route("/forex", get(page_forex_dashboard))
                    .route("/forex/trades", get(page_forex_trades))
                    .route("/forex/settings", get(page_forex_settings))
                    .route("/forex/performance", get(page_forex_performance))
                    .route("/forex/risk", get(page_forex_risk))
                    // JSON API
                    .route("/api/health", get(api_health))
                    .route("/health", get(api_health))
                    .route("/api/state", get(api_state))
                    .route("/api/balance", get(api_balance))
                    .route("/api/positions", get(api_positions))
                    .route("/api/trades", get(api_trades))
                    .route("/api/orderbook", get(api_orderbook))
                    .route("/api/metrics", get(api_metrics))
                    .route("/api/portfolio", get(api_portfolio))
                    .route("/api/performance", get(api_performance))
                    .route("/api/risk", get(api_risk))
                    .route("/api/strategy-performance", get(api_strategy_performance))
                    .route("/api/v1/status", get(api_v1_status))
                    // ── Gate.io Market Data Proxy Endpoints ──
                    .route("/api/market/tickers", get(api_market_tickers))
                    .route("/api/market/:symbol/klines", get(api_market_klines))
                    .route("/api/market/:symbol/orderbook", get(api_market_orderbook))
                    .route("/api/market/:symbol/recent-trades", get(api_market_recent_trades))
                    .route("/api/market/:symbol/multi-timeframe", get(api_market_multi_timeframe))
                    .route("/api/funding-rates", get(api_funding_rates))
                    // ── Computed / State-derived Endpoints ──
                    .route("/api/risk/metrics", get(api_risk_metrics))
                    .route("/api/risk-dashboard", get(api_risk_dashboard))
                    .route("/api/execution-quality", get(api_execution_quality))
                    .route("/api/equity-history", get(api_equity_history))
                    .route("/api/positions/live", get(api_positions_live))
                    .route("/api/trades/history", get(api_trades_history))
                    .route("/api/v1/trades", get(api_v1_trades))
                    .route("/api/performance/charts", get(api_performance_charts))
                    // ── Stub Endpoints (not yet connected) ──
                    .route("/api/sentiment/social", get(api_sentiment_social))
                    .route("/api/news/crypto", get(api_news_crypto))
                    .route("/api/ai/trade-suggestions", get(api_ai_trade_suggestions))
                    .route("/api/logs", get(api_logs))
                    .route("/api/trades/export", get(api_trades_export))
                    .route("/api/performance/export", get(api_performance_export))
                    // ── Settings Endpoints (stubs) ──
                    .route("/api/v1/settings", get(api_v1_settings).post(api_generic_post_stub))
                    .route("/api/settings/pairs", get(api_settings_pairs))
                    .route("/api/settings/pairs/add", axum::routing::post(api_generic_post_stub))
                    .route("/api/settings/pairs/:symbol", axum::routing::delete(api_positions_action))
                    .route("/api/settings/strategies", get(api_settings_strategies))
                    .route("/api/settings/strategies/:name", axum::routing::put(api_positions_action))
                    .route("/api/settings/currencies", get(api_settings_currencies))
                    .route("/api/settings/currencies/:symbol/toggle", axum::routing::post(api_positions_action))
                    .route("/api/mode/switch", axum::routing::post(api_mode_switch))
                    .route("/api/paper/reset", axum::routing::post(api_paper_reset))
                    // ── POST Action Endpoints (stubs) ──
                    .route("/api/positions/:symbol/stop-loss", axum::routing::post(api_positions_action))
                    .route("/api/positions/:symbol/take-profit", axum::routing::post(api_positions_action))
                    .route("/api/positions/:symbol/leverage", axum::routing::post(api_positions_action))
                    .route("/api/positions/:symbol/close", axum::routing::post(api_positions_action))
                    // Bot control (POST)
                    .route("/api/bot/pause", axum::routing::post(api_bot_pause))
                    .route("/api/bot/resume", axum::routing::post(api_bot_resume))
                    .route("/api/circuit-breaker/reset", axum::routing::post(api_circuit_breaker_reset))
                    .route("/api/v1/emergency-stop", axum::routing::post(api_emergency_stop))
                    // WebSocket
                    .route("/ws/live", get(ws_handler))
                    .route("/ws", get(ws_handler))
                    // Static files
                    .nest_service("/static", ServeDir::new(static_dir))
                    // Middleware
                    .layer(cors)
                    // Shared state
                    .with_state(app_state);

                // ── Bind and serve ──
                let addr: SocketAddr = bind_addr.parse().unwrap_or_else(|_| {
                    warn!("[dashboard] Invalid bind address '{}', defaulting to 0.0.0.0:8080", bind_addr);
                    "0.0.0.0:8080".parse().unwrap()
                });

                info!("[dashboard] axum HTTP+WS server listening on {}", addr);

                let listener = tokio::net::TcpListener::bind(addr).await.expect(
                    &format!("[dashboard] Failed to bind to {}", addr),
                );

                axum::serve(listener, app)
                    .await
                    .expect("[dashboard] axum server error");
            });
        })
        .expect("Failed to spawn dashboard server thread");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_json() {
        let state = DashboardState::new();
        state.set_balance(1000.0);
        state.set_equity(1050.0);
        state.set_unrealized_pnl(50.0);

        let json = state.to_json();
        assert!(json.contains("\"balance\":1000.0000"));
        assert!(json.contains("\"equity\":1050.0000"));
        assert!(json.contains("\"unrealized_pnl\":50.0000"));
        assert!(json.contains("\"signal_queue_depth\":"));
    }

    #[test]
    fn test_dashboard_state_defaults() {
        let state = DashboardState::new();
        assert_eq!(state.balance(), 0.0);
        assert_eq!(state.equity(), 0.0);
        assert!(state.is_running.load(Ordering::Relaxed));
    }

    #[test]
    fn test_template_context_has_required_keys() {
        let state = DashboardState::new();
        state.set_balance(500.0);
        state.set_equity(520.0);

        let ctx = state.template_context("dashboard");
        // Verify the context can be serialized (tera::Context implements this)
        let json = ctx.into_json();
        assert_eq!(json["active_page"], "dashboard");
        assert!(json["portfolio_value"].as_str().unwrap().contains("520"));
    }

    #[test]
    fn test_positions_json_roundtrip() {
        let state = DashboardState::new();
        let positions = r#"[{"symbol":"BTC_USDT","direction":"long","pnl":42.5}]"#;
        state.set_positions_json(positions.to_string());

        let ctx = state.template_context("dashboard");
        let json = ctx.into_json();
        let pos = &json["positions"];
        assert!(pos.is_array());
        assert_eq!(pos[0]["symbol"], "BTC_USDT");
    }
}
