/**
 * Forex dashboard real-time updates.
 *
 * Connects to the existing /ws/live WebSocket, filters for forex-typed
 * messages, and drives the TradingView Lightweight Charts for XAU/USD and
 * XAG/USD as well as the live positions table and portfolio cards.
 */
class ForexDashboard {
    constructor() {
        this.ws = null;
        this.goldChart = null;
        this.silverChart = null;
        this.goldSeries = null;
        this.silverSeries = null;
        this._reconnectDelay = 2000;
        this._maxReconnectDelay = 30000;

        this.initCharts();
        this.initWebSocket();
        this.startSpreadPolling();
    }

    // ── WebSocket ────────────────────────────────────────────────────────

    initWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${proto}://${location.host}/ws/live`;
        try {
            this.ws = new WebSocket(wsUrl);
        } catch (e) {
            console.warn('ForexDashboard: WebSocket unavailable', e);
            return;
        }

        this.ws.addEventListener('open', () => {
            console.debug('ForexDashboard: WebSocket connected');
            this._reconnectDelay = 2000;
            this._updateConnectionIndicator(true);
        });

        this.ws.addEventListener('message', (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                this._handleMessage(msg);
            } catch (e) {
                /* ignore parse errors */
            }
        });

        this.ws.addEventListener('close', () => {
            this._updateConnectionIndicator(false);
            console.debug(`ForexDashboard: WS closed – reconnecting in ${this._reconnectDelay}ms`);
            setTimeout(() => this.initWebSocket(), this._reconnectDelay);
            this._reconnectDelay = Math.min(this._reconnectDelay * 2, this._maxReconnectDelay);
        });

        this.ws.addEventListener('error', () => {
            this._updateConnectionIndicator(false);
        });
    }

    _handleMessage(msg) {
        if (!msg || msg.type !== 'update') return;

        if (msg.portfolio) {
            // Re-use generic portfolio for forex when no dedicated key exists
            const forexPortfolio = msg.forex_portfolio || msg.portfolio;
            this.updatePortfolio(forexPortfolio);
        }

        if (msg.forex_positions) {
            this.updatePositions(msg.forex_positions);
        }

        // Feed latest OHLCV tick into charts when kline data is included
        if (msg.klines) {
            const xauKlines = msg.klines['XAU/USD'] || msg.klines['XAUUSD'];
            const xagKlines = msg.klines['XAG/USD'] || msg.klines['XAGUSD'];
            if (xauKlines && this.goldSeries) {
                this._appendKlineTick(this.goldSeries, xauKlines);
            }
            if (xagKlines && this.silverSeries) {
                this._appendKlineTick(this.silverSeries, xagKlines);
            }
        }
    }

    _updateConnectionIndicator(online) {
        const el = document.getElementById('forex-connection-indicator');
        if (!el) return;
        el.className = online
            ? 'status-dot status-dot--online'
            : 'status-dot status-dot--offline';
        el.title = online ? 'WebSocket connected' : 'WebSocket disconnected';
    }

    // ── TradingView Lightweight Charts ───────────────────────────────────

    initCharts() {
        this._initSingleChart('gold-chart', 'goldChart', 'goldSeries', '#ffd700');
        this._initSingleChart('silver-chart', 'silverChart', 'silverSeries', '#c0c0c0');

        // Load historical klines for both pairs
        this._fetchAndLoadKlines('XAU/USD', this.goldSeries);
        this._fetchAndLoadKlines('XAG/USD', this.silverSeries);
    }

    _initSingleChart(containerId, chartProp, seriesProp, color) {
        const container = document.getElementById(containerId);
        if (!container || typeof LightweightCharts === 'undefined') return;

        const chart = LightweightCharts.createChart(container, {
            layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
            grid: {
                vertLines: { color: 'rgba(255,255,255,0.05)' },
                horzLines: { color: 'rgba(255,255,255,0.05)' },
            },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
            timeScale: { borderColor: 'rgba(255,255,255,0.1)', timeVisible: true },
            width: container.clientWidth,
            height: container.clientHeight || 300,
        });

        const series = chart.addCandlestickSeries({
            upColor: '#20c997',
            downColor: '#dc3545',
            borderUpColor: '#20c997',
            borderDownColor: '#dc3545',
            wickUpColor: '#20c997',
            wickDownColor: '#dc3545',
        });

        this[chartProp] = chart;
        this[seriesProp] = series;

        // Resize on window resize
        window.addEventListener('resize', () => {
            chart.applyOptions({ width: container.clientWidth });
        });
    }

    async _fetchAndLoadKlines(symbol, series) {
        if (!series) return;
        try {
            const encoded = encodeURIComponent(symbol);
            const resp = await fetch(`/api/market/${encoded}/klines?interval=1h&limit=200`);
            if (!resp.ok) return;
            const data = await resp.json();
            const klines = data.klines || data;
            if (!Array.isArray(klines) || klines.length === 0) return;

            const candles = klines.map(k => ({
                time: Math.floor((k.timestamp || k.time || k[0]) / 1000),
                open: parseFloat(k.open || k[1]),
                high: parseFloat(k.high || k[2]),
                low: parseFloat(k.low || k[3]),
                close: parseFloat(k.close || k[4]),
            })).filter(c => c.time && c.open && c.high && c.low && c.close);

            candles.sort((a, b) => a.time - b.time);
            series.setData(candles);
        } catch (e) {
            console.debug('ForexDashboard: klines fetch error', e);
        }
    }

    _appendKlineTick(series, klineData) {
        if (!series || !klineData) return;
        const k = Array.isArray(klineData) ? klineData[klineData.length - 1] : klineData;
        if (!k) return;
        series.update({
            time: Math.floor((k.timestamp || k.time || k[0]) / 1000),
            open: parseFloat(k.open || k[1]),
            high: parseFloat(k.high || k[2]),
            low: parseFloat(k.low || k[3]),
            close: parseFloat(k.close || k[4]),
        });
    }

    // ── Positions table ──────────────────────────────────────────────────

    updatePositions(positions) {
        const tbody = document.getElementById('forex-positions-tbody');
        if (!tbody) return;

        if (!positions || positions.length === 0) {
            tbody.innerHTML = `
                <tr>
                  <td colspan="9" class="text-center text-muted py-4">
                    <i class="bi bi-inbox me-2"></i>No open forex positions
                  </td>
                </tr>`;
            return;
        }

        tbody.innerHTML = positions.map(pos => {
            const side = (pos.direction || pos.side || 'long').toLowerCase();
            const sideClass = side === 'long' ? 'text-success' : 'text-danger';
            const sideIcon = side === 'long' ? 'bi-arrow-up-circle-fill' : 'bi-arrow-down-circle-fill';

            const pipPnl = pos.pip_pnl ?? pos.pnl ?? 0;
            const pipClass = pipPnl >= 0 ? 'pip-pnl-positive' : 'pip-pnl-negative';
            const pipSign = pipPnl >= 0 ? '+' : '';

            const spread = pos.spread_pips != null ? pos.spread_pips.toFixed(1) : '—';
            const lotSize = pos.lot_size != null ? pos.lot_size.toFixed(2) : '—';
            const entryPrice = pos.entry_price != null ? pos.entry_price.toFixed(2) : '—';
            const currentPrice = pos.current_price != null ? pos.current_price.toFixed(2) : '—';
            const symbol = pos.symbol || '—';

            const pairClass = symbol.includes('XAU')
                ? 'pair-badge-xauusd'
                : symbol.includes('XAG')
                ? 'pair-badge-xagusd'
                : '';

            return `<tr>
              <td>
                <span class="pair-badge ${pairClass}">
                  <i class="bi bi-gem"></i>${symbol}
                </span>
              </td>
              <td><span class="${sideClass}"><i class="bi ${sideIcon} me-1"></i>${side.toUpperCase()}</span></td>
              <td>${lotSize}</td>
              <td>${entryPrice}</td>
              <td>${currentPrice}</td>
              <td class="${pipClass}">${pipSign}${pipPnl.toFixed(1)} pips</td>
              <td class="${pipClass}">${pipSign}$${(pos.pnl_usd ?? pos.pnl ?? 0).toFixed(2)}</td>
              <td><span class="spread-indicator">${spread} pips</span></td>
              <td>
                <button class="btn btn-xs btn-outline-danger btn-sm py-0 px-2"
                        onclick="closeForexPosition('${symbol}')">
                  Close
                </button>
              </td>
            </tr>`;
        }).join('');
    }

    // ── Portfolio cards ──────────────────────────────────────────────────

    updatePortfolio(portfolio) {
        if (!portfolio) return;
        this._setTextContent('forex-balance', this._fmt(portfolio.balance));
        this._setTextContent('forex-equity', this._fmt(portfolio.equity));
        this._setTextContent('forex-margin-used', this._fmt(portfolio.margin_used));
        this._setTextContent('forex-free-margin', this._fmt(portfolio.free_margin));
        const dailyPnl = portfolio.daily_pnl ?? 0;
        const el = document.getElementById('forex-daily-pnl');
        if (el) {
            el.textContent = (dailyPnl >= 0 ? '+' : '') + this._fmt(dailyPnl);
            el.className = dailyPnl >= 0 ? 'stat-value pip-pnl-positive' : 'stat-value pip-pnl-negative';
        }
    }

    // ── Spread polling ───────────────────────────────────────────────────

    startSpreadPolling() {
        this._pollSpreads();
        setInterval(() => this._pollSpreads(), 10000);
    }

    async _pollSpreads() {
        for (const symbol of ['XAU/USD', 'XAG/USD']) {
            try {
                const encoded = encodeURIComponent(symbol);
                const resp = await fetch(`/api/forex/spread/${encoded}`);
                if (!resp.ok) continue;
                const data = await resp.json();
                const elId = symbol.replace('/', '').toLowerCase() + '-spread';
                const el = document.getElementById(elId);
                if (el && data.spread_pips != null) {
                    el.textContent = data.spread_pips.toFixed(1) + ' pips';
                }
            } catch (e) { /* silent */ }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    _fmt(val) {
        if (val == null) return '—';
        return '$' + parseFloat(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    _setTextContent(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }
}

// ── Currency pair management ─────────────────────────────────────────────

async function addForexPair() {
    const input = document.getElementById('new-forex-pair');
    if (!input) return;
    const symbol = input.value.trim().toUpperCase();
    if (!symbol) return;

    try {
        const resp = await fetch('/api/forex/pairs/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol }),
        });
        const data = await resp.json();
        if (resp.ok) {
            input.value = '';
            await refreshForexPairs();
            showForexToast(`Added ${symbol}`, 'success');
        } else {
            showForexToast(data.detail || 'Failed to add pair', 'danger');
        }
    } catch (e) {
        showForexToast('Network error', 'danger');
    }
}

async function removeForexPair(symbol) {
    if (!confirm(`Remove ${symbol} from forex pairs?`)) return;
    try {
        const encoded = encodeURIComponent(symbol);
        const resp = await fetch(`/api/forex/pairs/${encoded}`, { method: 'DELETE' });
        if (resp.ok) {
            await refreshForexPairs();
            showForexToast(`Removed ${symbol}`, 'success');
        } else {
            const data = await resp.json();
            showForexToast(data.detail || 'Failed to remove pair', 'danger');
        }
    } catch (e) {
        showForexToast('Network error', 'danger');
    }
}

async function refreshForexPairs() {
    try {
        const resp = await fetch('/api/forex/pairs');
        if (!resp.ok) return;
        const pairs = await resp.json();
        const container = document.getElementById('forex-pairs-list');
        if (!container) return;
        if (!pairs || pairs.length === 0) {
            container.innerHTML = '<span class="text-muted small">No pairs configured</span>';
            return;
        }
        container.innerHTML = pairs.map(p => {
            const sym = typeof p === 'string' ? p : p.symbol;
            const enabled = typeof p === 'object' ? p.enabled !== false : true;
            const pairClass = sym.includes('XAU') ? 'pair-badge-xauusd' : sym.includes('XAG') ? 'pair-badge-xagusd' : '';
            return `<div class="d-flex align-items-center gap-2 mb-2">
              <span class="pair-badge ${pairClass}"><i class="bi bi-gem me-1"></i>${sym}</span>
              <div class="form-check form-switch mb-0">
                <input class="form-check-input" type="checkbox" ${enabled ? 'checked' : ''}
                       onchange="toggleForexPair('${sym}', this.checked)" title="Enable/disable">
              </div>
              <button class="btn btn-xs btn-outline-danger btn-sm py-0 px-2" onclick="removeForexPair('${sym}')">
                <i class="bi bi-x"></i>
              </button>
            </div>`;
        }).join('');
    } catch (e) {
        console.debug('refreshForexPairs error', e);
    }
}

async function toggleForexPair(symbol, enabled) {
    // No dedicated toggle API yet — inform user; pair management handled server-side
    showForexToast(`${symbol}: toggle (${enabled ? 'enabled' : 'disabled'}) – restart bot to apply`, 'info');
}

async function closeForexPosition(symbol) {
    if (!confirm(`Close forex position for ${symbol}?`)) return;
    try {
        const encoded = encodeURIComponent(symbol);
        const resp = await fetch(`/api/positions/${encoded}/close`, { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
            showForexToast(`Closed position for ${symbol}`, 'success');
        } else {
            showForexToast(data.detail || 'Failed to close position', 'danger');
        }
    } catch (e) {
        showForexToast('Network error', 'danger');
    }
}

function showForexToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const id = 'forex-toast-' + Date.now();
    const html = `<div id="${id}" class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
      <div class="d-flex">
        <div class="toast-body">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    </div>`;
    container.insertAdjacentHTML('beforeend', html);
    const toastEl = document.getElementById(id);
    const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });
    bsToast.show();
    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

// ── Initialise on page load ──────────────────────────────────────────────
let _forexDashboard = null;
document.addEventListener('DOMContentLoaded', () => {
    _forexDashboard = new ForexDashboard();
    refreshForexPairs();

    // Live clock
    function updateClock() {
        const el = document.getElementById('forex-current-time');
        if (el) el.textContent = new Date().toUTCString().replace('GMT', 'UTC');
    }
    updateClock();
    setInterval(updateClock, 1000);
});

// ── WebSocket forex data handler ─────────────────────────────────────────
/**
 * Subscribe to the dashboard WebSocket and update forex widgets on each push.
 * Call this from any page that embeds forex widgets and needs live updates.
 */
function subscribeForexWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    let ws;
    let retryDelay = 2000;

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type !== 'update') return;
                const forex = msg.forex;
                if (!forex) return;

                // Update session badge
                const sessEl = document.getElementById('forex-session-badge');
                if (sessEl && forex.session_name) {
                    sessEl.textContent = forex.session_name;
                }

                // Update session P&L
                const pnlEl = document.getElementById('forex-session-pnl-ws');
                if (pnlEl && forex.session_pnl !== undefined) {
                    const pnl = parseFloat(forex.session_pnl);
                    pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2);
                    pnlEl.className = pnl >= 0 ? 'text-success fw-bold' : 'text-danger fw-bold';
                }

                // Update recovery mode indicator
                const recovEl = document.getElementById('forex-recovery-badge');
                if (recovEl) {
                    if (forex.in_recovery_mode) {
                        recovEl.innerHTML = `<span class="badge bg-warning text-dark">Recovery L${forex.recovery_level||0}</span>`;
                    } else {
                        recovEl.innerHTML = `<span class="badge bg-success">Normal</span>`;
                    }
                }

                // Update open positions count
                const posCount = (forex.positions || []).length;
                const posEl = document.getElementById('forex-open-count-ws');
                if (posEl) posEl.textContent = posCount;

            } catch (e) { /* ignore parse errors */ }
        };
        ws.onclose = () => {
            setTimeout(connect, retryDelay);
            retryDelay = Math.min(retryDelay * 1.5, 30000);
        };
        ws.onerror = () => ws.close();
        ws.onopen = () => { retryDelay = 2000; };
    }

    connect();
    return { close: () => ws && ws.close() };
}

// ── Forex API helpers ────────────────────────────────────────────────────

/**
 * Load and render the spreads for all configured forex pairs.
 * Populates a table/container identified by `elementId`.
 */
async function loadForexSpreads(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    try {
        const r = await fetch('/api/forex/spreads');
        if (!r.ok) { el.innerHTML = `<span class="text-danger">Failed: ${r.status}</span>`; return; }
        const data = await r.json();
        const spreads = data.spreads || [];
        if (!spreads.length) { el.innerHTML = '<span class="text-muted">No spread data</span>'; return; }
        el.innerHTML = spreads.map(s => {
            const spreadHtml = s.spread_pips !== null
                ? `<span class="${s.spread_pips > 5 ? 'text-warning' : 'text-success'}">${s.spread_pips.toFixed(1)} pips</span>`
                : `<span class="text-muted">—</span>`;
            return `<div class="d-flex justify-content-between border-bottom py-1">
              <strong>${s.symbol}</strong>
              <div>${spreadHtml} &nbsp; B:${s.bid || '—'} / A:${s.ask || '—'}</div>
            </div>`;
        }).join('');
    } catch(e) { el.innerHTML = `<span class="text-danger">Error: ${e.message}</span>`; }
}

/**
 * Load and render session information and current trading sessions.
 * Populates a container identified by `elementId`.
 */
async function loadForexSessions(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    try {
        const r = await fetch('/api/forex/sessions');
        if (!r.ok) return;
        const data = await r.json();
        const sessions = data.sessions || {};
        const current = data.current_sessions || [];
        el.innerHTML = Object.entries(sessions).map(([name, info]) => {
            const active = info.active;
            const isCurrent = current.includes(name);
            const label = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            return `<div class="d-flex align-items-center justify-content-between mb-1">
              <span>${label}</span>
              <span class="badge ${active ? 'bg-success' : 'bg-secondary'}">${active ? 'OPEN' : 'CLOSED'}</span>
              <small class="text-muted ms-2">${info.hours}</small>
            </div>`;
        }).join('');
    } catch(e) { console.error('loadForexSessions error', e); }
}

/**
 * Load and render forex signals.
 * Populates a table body identified by `tbodyId`.
 */
async function loadForexSignals(tbodyId, limit = 20) {
    const tbody = document.getElementById(tbodyId);
    if (!tbody) return;
    try {
        const r = await fetch('/api/forex/signals');
        if (!r.ok) return;
        const data = await r.json();
        const signals = (data.signals || []).slice(0, limit);
        if (!signals.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No forex signals</td></tr>';
            return;
        }
        tbody.innerHTML = signals.map(s => {
            const dirHtml = (s.direction || s.side || '').toLowerCase() === 'long'
                ? '<span class="badge bg-success">LONG</span>'
                : '<span class="badge bg-danger">SHORT</span>';
            const confPct = s.confidence ? `${(s.confidence * 100).toFixed(0)}%` : '—';
            return `<tr>
              <td><strong>${s.symbol||'—'}</strong></td>
              <td>${dirHtml}</td>
              <td>${s.strategy||'—'}</td>
              <td>${confPct}</td>
              <td>${s.entry_price ? parseFloat(s.entry_price).toFixed(4) : '—'}</td>
              <td>${s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—'}</td>
            </tr>`;
        }).join('');
    } catch(e) { console.error('loadForexSignals error', e); }
}

/**
 * Modify the SL and/or TP for an open position via the dashboard API.
 *
 * @param {string} symbol  The forex pair.
 * @param {number|null} newSl  New stop-loss price (null to keep current).
 * @param {number|null} newTp  New take-profit price (null to keep current).
 */
async function modifyForexPosition(symbol, newSl, newTp) {
    const body = {};
    if (newSl !== null && newSl !== undefined) body.stop_loss = parseFloat(newSl);
    if (newTp !== null && newTp !== undefined) body.take_profit = parseFloat(newTp);
    if (!Object.keys(body).length) { showForexToast('No changes to submit', 'warning'); return; }
    try {
        const r = await fetch(`/api/forex/positions/${encodeURIComponent(symbol)}/modify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await r.json().catch(() => ({}));
        if (r.ok) {
            showForexToast(`Modified ${symbol}: SL=${data.stop_loss || '—'} TP=${data.take_profit || '—'}`, 'success');
        } else {
            showForexToast(data.detail || `Failed to modify ${symbol}`, 'danger');
        }
    } catch(e) {
        showForexToast('Network error modifying position', 'danger');
    }
}

/**
 * Load the ForexRiskManager metrics and render them into elements by ID.
 * Expected element IDs: forex-risk-recovery, forex-risk-margin, forex-risk-dd,
 *   forex-risk-consec-losses, forex-risk-session-pnl.
 */
async function loadForexRiskSummary() {
    try {
        const r = await fetch('/api/forex/risk');
        if (!r.ok) return;
        const d = await r.json();

        const idMap = {
            'forex-risk-recovery': d.in_recovery_mode ? `Level ${d.recovery_mode_level}` : 'OFF',
            'forex-risk-margin': (d.margin_level || '').toUpperCase(),
            'forex-risk-dd': d.max_drawdown_pct_seen !== undefined ? `${parseFloat(d.max_drawdown_pct_seen).toFixed(2)}%` : '—',
            'forex-risk-consec-losses': d.consecutive_losses ?? '—',
            'forex-risk-session-pnl': d.session_pnl !== undefined ? `$${parseFloat(d.session_pnl).toFixed(2)}` : '—',
        };
        for (const [id, val] of Object.entries(idMap)) {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        }
    } catch(e) { console.error('loadForexRiskSummary error', e); }
}
