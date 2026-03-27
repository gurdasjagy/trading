/**
 * main.js — Core dashboard JavaScript
 * Handles WebSocket connection, live metric updates, formatting helpers,
 * emergency stop, toast notifications, and auto-reconnect logic.
 */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────
const WS_PATH = '/ws/live';
// ISSUE 12 FIX: Exponential backoff constants (replaces fixed 3s delay)
const WS_RECONNECT_BASE_MS = 1000;   // Initial reconnect delay: 1 second
const WS_RECONNECT_MAX_MS  = 30000;  // Maximum reconnect delay: 30 seconds
const POLL_INTERVAL_MS = 10000;  // Polling is now just a fallback; hub pushes at 1 s

// ── Module state ─────────────────────────────────────────────────────────
let _ws = null;
let _reconnectTimer = null;
let _pollTimer = null;
// ISSUE 12 FIX: Backoff state
let _reconnectAttempts = 0;
let _pageUnloading = false;
let _tabHidden = false;

// Auth token for WebSocket (sha256 of username:password when auth is enabled).
// Populated by the server via a template variable if authentication is configured.
// Exposed as a global so logs.html can reuse it for its own WS connection.
/* global _ws_auth_token */
if (typeof _ws_auth_token === 'undefined') {
  window._ws_auth_token = '';
}

// ── WebSocket ─────────────────────────────────────────────────────────────

let _heartbeatTimer = null;
const HEARTBEAT_INTERVAL_MS = 30000; // Send ping every 30 seconds

/**
 * Open a WebSocket connection to WS_PATH and wire up handlers.
 * Automatically reconnects on disconnect.
 */
function connectWebSocket() {
  if (_ws && (_ws.readyState === WebSocket.CONNECTING || _ws.readyState === WebSocket.OPEN)) {
    return;
  }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  _ws = new WebSocket(`${proto}://${location.host}${WS_PATH}`);

  _ws.onopen = _onWsOpen;
  _ws.onmessage = _onWsMessage;
  _ws.onclose = _onWsClose;
  _ws.onerror = _onWsError;
}

function _onWsOpen() {
  _setConnectionIndicator(true);
  // ISSUE 12 FIX: Reset backoff counter on successful connection
  _reconnectAttempts = 0;
  if (_reconnectTimer) {
    clearTimeout(_reconnectTimer);
    _reconnectTimer = null;
  }
  // Start heartbeat ping
  _startHeartbeat();
}

function _startHeartbeat() {
  if (_heartbeatTimer) clearInterval(_heartbeatTimer);
  _heartbeatTimer = setInterval(() => {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      try {
        _ws.send(JSON.stringify({ type: 'ping' }));
      } catch (e) {
        // Failed to send ping — connection may be stale
      }
    }
  }, HEARTBEAT_INTERVAL_MS);
}

function _stopHeartbeat() {
  if (_heartbeatTimer) {
    clearInterval(_heartbeatTimer);
    _heartbeatTimer = null;
  }
}

function _onWsMessage(event) {
  try {
    const data = JSON.parse(event.data);
    _handleUpdate(data);
  } catch {
    // Ignore non-JSON frames
  }
}

function _onWsClose() {
  _setConnectionIndicator(false);
  _stopHeartbeat();
  // ISSUE 12 FIX: Don't reconnect if the page is being unloaded or tab is hidden
  if (_pageUnloading) return;
  if (_tabHidden) return; // Will reconnect when tab becomes visible again
  _scheduleReconnect();
}

/**
 * ISSUE 12 FIX: Schedule a reconnect with exponential backoff.
 * Delay sequence: 1s, 2s, 4s, 8s, 16s, 30s, 30s, ...
 */
function _scheduleReconnect() {
  if (_reconnectTimer) return; // Already scheduled
  const delay = Math.min(
    WS_RECONNECT_BASE_MS * Math.pow(2, _reconnectAttempts),
    WS_RECONNECT_MAX_MS
  );
  _reconnectAttempts++;
  console.log(`[ws] Reconnecting in ${delay}ms (attempt ${_reconnectAttempts})`);
  _reconnectTimer = setTimeout(connectWebSocket, delay);
}

function _onWsError() {
  _setConnectionIndicator(false);
  _stopHeartbeat();
}

/**
 * Dispatch an incoming WebSocket payload to the correct handler.
 * @param {Object} data - Parsed JSON payload from the server.
 */
function _handleUpdate(data) {
  if (!data || typeof data !== 'object') return;

  // Live push update from the server's _collect_live_data() or realtime hub
  if (data.type === 'update' || data.type === 'price_update' || data.type === 'position_update') {
    if (data.portfolio) _applyPortfolio(data.portfolio);
    if (data.positions) {
      _applyPositions(data.positions, data.funding_rates);
      // Dispatch a global event so other pages (e.g. trades.html) can react
      window.dispatchEvent(new CustomEvent('positions-updated', { detail: data.positions }));
    }
    if (data.open_orders) _applyOpenOrders(data.open_orders);
    if (data.status)    _applyStatus(data.status);
    if (data.recent_trades) _applyRecentTrades(data.recent_trades);
    if (data.portfolio_risk) _applyPortfolioHealth(data.portfolio_risk);
    const updatedEl = document.getElementById('last-updated');
    if (updatedEl) updatedEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    return;
  }

  // Order update — refresh open orders section
  if (data.type === 'order_update') {
    if (data.open_orders) _applyOpenOrders(data.open_orders);
    return;
  }

  // kline_update — forward to candlestick chart
  if (data.type === 'kline_update') {
    document.dispatchEvent(new CustomEvent('ws-kline', { detail: data }));
    return;
  }

  // ticker_update — forward to market overview
  if (data.type === 'ticker_update' && data.tickers) {
    document.dispatchEvent(new CustomEvent('ws-tickers', { detail: data.tickers }));
    return;
  }

  // positions_update — real-time position P&L from RealtimeHub._position_update_loop
  if (data.type === 'positions_update' && Array.isArray(data.data)) {
    if (typeof window.updatePositionsTable === 'function') window.updatePositionsTable(data.data);
    if (typeof window.updateTotalPnL === 'function') window.updateTotalPnL(data.data);
    return;
  }

  // Legacy metrics push
  if (data.type === 'metrics' || data.metrics) {
    updateDashboardMetrics(data.metrics || data);
  }

  if (data.type === 'alert' && data.message) {
    showToast(data.message, data.level || 'info');
  }

  // Log line — forwarded to log handler if present
  if (data.type === 'log' && typeof window._handleLogEntry === 'function') {
    window._handleLogEntry(data);
  }

  // Batched log lines — process all logs in the batch
  if (data.type === 'logs_batch' && data.logs && typeof window._handleLogEntry === 'function') {
    for (const logEntry of data.logs) {
      window._handleLogEntry(logEntry);
    }
  }
}

// ── HTTP API polling (fallback / complement to WebSocket) ────────────────

/**
 * Poll all live-data API endpoints and update the UI.
 */
async function pollApiData() {
  try {
    const [portfolio, posRes, perfRes] = await Promise.all([
      fetch('/api/portfolio').then(r => r.ok ? r.json() : null),
      fetch('/api/positions').then(r => r.ok ? r.json() : null),
      fetch('/api/performance').then(r => r.ok ? r.json() : null),
    ]);
    if (portfolio) _applyPortfolio(portfolio);
    if (posRes && posRes.positions) _applyPositions(posRes.positions);
    if (perfRes) _applyPerformance(perfRes);
    const updatedEl = document.getElementById('last-updated');
    if (updatedEl) updatedEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    // Silent — WebSocket takes priority
  }
}

/**
 * Start periodic polling every POLL_INTERVAL_MS milliseconds.
 */
function startPolling() {
  if (_pollTimer) return;
  _pollTimer = setInterval(pollApiData, POLL_INTERVAL_MS);
  // Run immediately
  pollApiData();
}

// ── Data renderers ───────────────────────────────────────────────────────

function _applyPortfolio(p) {
  if (!p) return;
  const equityEl = document.getElementById('metric-portfolio-value');
  if (equityEl) {
    const prevText = equityEl.textContent;
    const newText = formatCurrency(p.equity ?? p.portfolio_value);
    if (prevText !== newText) {
      const prevVal = parseFloat(prevText.replace(/[^0-9.\-]/g, ''));
      const newVal = parseFloat((p.equity ?? p.portfolio_value) || 0);
      const flashClass = newVal > prevVal ? 'flash-green' : newVal < prevVal ? 'flash-red' : '';
      equityEl.textContent = newText;
      if (flashClass) {
        equityEl.classList.add(flashClass);
        setTimeout(() => equityEl.classList.remove(flashClass), 600);
      }
    }
  }
  _setPnl('metric-daily-pnl', p.daily_pnl_pct, '%');
  _setPnl('metric-unrealized-pnl', p.unrealized_pnl, '');
  _setText('metric-open-positions', p.open_positions ?? '—');
}

function _applyStatus(s) {
  if (!s) return;
  const cbEl = document.getElementById('circuit-breaker-status');
  if (cbEl && s.circuit_breaker) {
    cbEl.textContent = s.circuit_breaker.triggered ? 'TRIGGERED' : 'OK';
    cbEl.className = s.circuit_breaker.triggered ? 'badge bg-danger' : 'badge bg-success';
  }
  const uptimeEl = document.getElementById('bot-uptime');
  if (uptimeEl && s.uptime != null) uptimeEl.textContent = formatDuration(s.uptime);

  // Regime indicator badge
  const regimeEl = document.getElementById('market-regime-badge');
  if (regimeEl && s.market_regime) {
    const regime = s.market_regime;
    const regimeColors = {
      trending_up: 'success', trending_down: 'danger',
      ranging: 'warning', crash: 'danger',
      low_volatility: 'secondary', high_volatility: 'warning',
      unknown: 'secondary'
    };
    const color = regimeColors[regime] || 'secondary';
    regimeEl.textContent = regime.replace(/_/g, ' ').toUpperCase();
    regimeEl.className = `badge bg-${color}`;
  }

  // Crash protection level badge
  const crashEl = document.getElementById('crash-level-badge');
  if (crashEl && s.crash_level) {
    const level = s.crash_level;
    const crashColors = {
      normal: 'success', yellow: 'warning', orange: 'warning',
      red: 'danger', black_swan: 'dark'
    };
    const color = crashColors[level] || 'secondary';
    crashEl.textContent = ('CRASH: ' + level.replace(/_/g, ' ')).toUpperCase();
    crashEl.className = `badge bg-${color}`;
    crashEl.style.display = level === 'normal' ? 'none' : 'inline-block';
  }
}

/**
 * Load and render the strategy performance table.
 * Called when the user navigates to /performance.
 */
async function loadStrategyPerformance() {
  try {
    const res = await fetch('/api/strategy-performance');
    if (!res.ok) return;
    const data = await res.json();
    _applyStrategyPerformanceTable(data.strategies || []);
  } catch {}
}

function _applyStrategyPerformanceTable(strategies) {
  const tbody = document.getElementById('strategy-perf-tbody');
  if (!tbody) return;
  if (!strategies.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No strategy data</td></tr>';
    return;
  }
  tbody.innerHTML = strategies.map(s => {
    const enabledBadge = s.enabled
      ? '<span class="badge bg-success">ON</span>'
      : '<span class="badge bg-secondary">OFF</span>';
    const wrClass = s.win_rate >= 0.5 ? 'text-success' : 'text-danger';
    const pfClass = s.profit_factor >= 1.0 ? 'text-success' : 'text-danger';
    return `<tr>
      <td class="fw-semibold">${escapeHtml(s.name)}</td>
      <td>${enabledBadge}</td>
      <td class="${wrClass}">${_fmt2(s.win_rate * 100)}%</td>
      <td class="${pfClass}">${_fmt2(s.profit_factor)}</td>
      <td>${_fmt2(s.sharpe)}</td>
      <td>${_fmt4(s.avg_profit)}</td>
      <td>${_fmt4(s.avg_loss)}</td>
      <td>${s.total_trades}</td>
    </tr>`;
  }).join('');
}

function _applyPerformance(perf) {
  if (!perf) return;
  const wr = perf.win_rate;
  if (wr != null) {
    // win_rate may be 0-100 (from trade_journal) or already a percentage
    _setText('metric-win-rate', formatPercentage(Math.abs(wr) > 1 ? wr / 100 : wr));
  }
}

function _applyPositions(positions, fundingRates) {
  const tbody = document.getElementById('positions-tbody');
  if (!tbody) return;
  if (!positions || !positions.length) {
    // colspan must match the number of <th> columns in the positions table header
    tbody.innerHTML = '<tr><td colspan="14" class="text-center text-muted py-3">No open positions</td></tr>';
    return;
  }

  // Build a funding rate map keyed by symbol for quick lookup
  const frMap = {};
  if (Array.isArray(fundingRates)) {
    fundingRates.forEach(fr => { if (fr && fr.symbol) frMap[fr.symbol] = fr; });
  } else if (typeof fundingRates === 'object' && fundingRates !== null) {
    // fundingRates may be an object mapping symbol → rate
    Object.entries(fundingRates).forEach(([sym, rate]) => { frMap[sym] = { rate_pct: rate * 100 }; });
  }

  tbody.innerHTML = positions.map(pos => {
    const dirClass = (pos.direction || '').toLowerCase() === 'long' ? 'success' : 'danger';
    const pnlClass = (pos.pnl || 0) >= 0 ? 'text-success' : 'text-danger';
    const roeClass = (pos.roe_pct || 0) >= 0 ? 'text-success' : 'text-danger';
    const sym = escapeHtml(pos.symbol || '');
    // Use JSON.stringify to safely encode symbol in data attributes to avoid XSS
    const symJson = escapeHtml(JSON.stringify(pos.symbol || ''));
    const liqPrice = pos.liquidation_price && pos.liquidation_price > 0
      ? '$' + _fmt4(pos.liquidation_price)
      : '—';
    const marginStr = (pos.margin != null && pos.margin > 0) ? '$' + _fmt2(pos.margin) : '—';
    const markPrice = pos.mark_price && pos.mark_price > 0
      ? '$' + _fmt4(pos.mark_price)
      : '$' + _fmt4(pos.current_price);
    const roeStr = pos.roe_pct != null
      ? (pos.roe_pct >= 0 ? '+' : '') + _fmt2(pos.roe_pct) + '%'
      : '—';

    // Calculate duration if timestamp is available
    let durationStr = '—';
    if (pos.timestamp && pos.timestamp > 0) {
      const openedMs = pos.timestamp;
      const nowMs = Date.now();
      const durationMins = Math.floor((nowMs - openedMs) / 60000);
      if (durationMins < 60) {
        durationStr = `${durationMins}m`;
      } else if (durationMins < 1440) {
        const hours = Math.floor(durationMins / 60);
        const mins = durationMins % 60;
        durationStr = `${hours}h ${mins}m`;
      } else {
        const days = Math.floor(durationMins / 1440);
        const hours = Math.floor((durationMins % 1440) / 60);
        durationStr = `${days}d ${hours}h`;
      }
    }

    // Funding rate badge
    let fundingBadge = '';
    const fr = frMap[pos.symbol] || (pos.funding_rate != null ? { rate_pct: pos.funding_rate * 100 } : null);
    if (fr && fr.rate_pct != null) {
      const frVal = parseFloat(fr.rate_pct);
      const frClass = frVal >= 0 ? 'text-success' : 'text-danger';
      const frSign = frVal >= 0 ? '+' : '';
      fundingBadge = `<small class="d-block ${frClass}" title="Funding rate">FR: ${frSign}${frVal.toFixed(4)}%</small>`;
    }

    // Break-even and trailing TP indicators
    const beLabel = pos.break_even_activated ? '<span class="badge bg-info me-1" title="Break-even SL active">BE</span>' : '';
    const ttLabel = pos.trailing_tp_active ? '<span class="badge bg-primary me-1" title="Trailing TP active">TTP</span>' : '';

    // Strategy/Mode badge
    let modeBadge = '<span class="badge bg-secondary">—</span>';
    if (pos.strategy === 'manual') {
      modeBadge = '<span class="badge bg-warning text-dark">Manual</span>';
    } else if (pos.strategy && pos.strategy !== '') {
      modeBadge = '<span class="badge bg-success">Auto</span>';
    }

    return `<tr>
      <td class="fw-semibold">${sym}${fundingBadge}</td>
      <td><span class="badge bg-${dirClass}">${escapeHtml((pos.direction || '').toUpperCase())}</span>${beLabel}${ttLabel}</td>
      <td class="small">$${_fmt2(pos.notional_size ?? pos.position_value ?? 0)}</td>
      <td>$${_fmt4(pos.entry_price)}</td>
      <td>${markPrice}</td>
      <td class="pnl-value ${pnlClass}">${pos.pnl >= 0 ? '+' : ''}${_fmt2(pos.pnl)}</td>
      <td class="${roeClass} small">${roeStr}</td>
      <td>${pos.leverage ?? 1}×</td>
      <td>${modeBadge}</td>
      <td class="small">${marginStr}</td>
      <td class="small text-muted">${liqPrice}</td>
      <td>${pos.stop_loss != null ? '$' + _fmt4(pos.stop_loss) : '—'}</td>
      <td>${pos.take_profit != null ? '$' + _fmt4(pos.take_profit) : '—'}</td>
      <td class="small text-muted">${durationStr}</td>
      <td>
        <div class="d-flex gap-1 flex-wrap">
          <button class="btn btn-danger btn-sm"
                  data-symbol="${symJson}"
                  onclick="closePosition(JSON.parse(this.dataset.symbol))"
                  title="Close full position">
            <i class="bi bi-x-circle me-1"></i>Close
          </button>
          <button class="btn btn-outline-warning btn-sm"
                  data-symbol="${symJson}"
                  onclick="reducePosition(JSON.parse(this.dataset.symbol), 50)"
                  title="Close 50% of position">
            50%
          </button>
        </div>
      </td>
    </tr>`;
  }).join('');

  // Update the real-time price ticker bar if present
  _applyPriceTicker(positions);
}

/**
 * Update the price ticker bar at the top of the page with latest prices.
 * @param {Array} positions
 */
function _applyPriceTicker(positions) {
  const bar = document.getElementById('price-ticker-bar');
  if (!bar || !positions || !positions.length) return;
  bar.innerHTML = positions.map(pos => {
    const pnlClass = (pos.pnl || 0) >= 0 ? 'text-success' : 'text-danger';
    const price = pos.mark_price && pos.mark_price > 0 ? pos.mark_price : pos.current_price;
    return `<span class="ticker-item me-4">
      <span class="fw-semibold">${escapeHtml(pos.symbol || '')}</span>
      <span class="ms-1">$${_fmt4(price)}</span>
      <span class="${pnlClass} ms-1 small">${(pos.pnl || 0) >= 0 ? '+' : ''}${_fmt2(pos.pnl)}</span>
    </span>`;
  }).join('');
}

/**
 * Render the open orders section.
 * @param {Array} orders
 */
function _applyOpenOrders(orders) {
  const tbody = document.getElementById('open-orders-tbody');
  if (!tbody) return;
  if (!orders || !orders.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-2">No open orders</td></tr>';
    return;
  }
  tbody.innerHTML = orders.map(o => {
    const typeLabel = (o.type || '').replace(/_/g, ' ').toUpperCase();
    const sideClass = (o.side || '').toLowerCase() === 'buy' ? 'success' : 'danger';
    const oIdJson = escapeHtml(JSON.stringify(o.id || ''));
    const oSymJson = escapeHtml(JSON.stringify(o.symbol || ''));
    return `<tr>
      <td class="fw-semibold small">${escapeHtml(o.symbol || '')}</td>
      <td><span class="badge bg-secondary">${escapeHtml(typeLabel)}</span></td>
      <td><span class="badge bg-${sideClass}">${escapeHtml((o.side || '').toUpperCase())}</span></td>
      <td>${o.price != null ? '$' + _fmt4(o.price) : '—'}</td>
      <td>${_fmt4(o.amount)}</td>
      <td>
        <button class="btn btn-outline-danger btn-sm"
                data-id="${oIdJson}" data-symbol="${oSymJson}"
                onclick="cancelOrder(JSON.parse(this.dataset.id), JSON.parse(this.dataset.symbol))"
                title="Cancel order">
          <i class="bi bi-x"></i>
        </button>
      </td>
    </tr>`;
  }).join('');
}

function _fmt2(v) { return (parseFloat(v) || 0).toFixed(2); }
function _fmt4(v) { return (parseFloat(v) || 0).toFixed(4); }

/**
 * Update the Portfolio Health section with margin, correlation, and risk metrics.
 * Renders into the element with id="portfolio-health-section" if present.
 * @param {Object} risk - Portfolio risk dict from /api/portfolio/risk
 */
function _applyPortfolioHealth(risk) {
  if (!risk) return;
  const section = document.getElementById('portfolio-health-section');
  if (!section) return;

  const marginRatio = parseFloat(risk.margin_ratio || 0);
  const marginPct = (marginRatio * 100).toFixed(1);
  const marginBarClass = marginRatio < 0.5 ? 'bg-success' : (marginRatio < 0.8 ? 'bg-warning' : 'bg-danger');
  const corrScore = parseFloat(risk.correlation_risk_score || 0);
  const corrPct = (corrScore * 100).toFixed(0);
  const corrBarClass = corrScore < 0.4 ? 'bg-success' : (corrScore < 0.7 ? 'bg-warning' : 'bg-danger');
  const drawdown = parseFloat(risk.drawdown_pct || 0);
  const marginUsed = parseFloat(risk.total_margin_used || 0).toFixed(2);
  const availMargin = parseFloat(risk.available_margin || 0).toFixed(2);

  section.innerHTML = `
    <div class="card border-0 shadow-sm mb-3">
      <div class="card-header bg-dark text-white py-2">
        <strong><i class="bi bi-shield-check me-1"></i>Portfolio Health</strong>
      </div>
      <div class="card-body py-2">
        <div class="row g-2 mb-2">
          <div class="col-6">
            <small class="text-muted">Margin Used</small>
            <div class="fw-semibold">$${marginUsed}</div>
          </div>
          <div class="col-6">
            <small class="text-muted">Available Margin</small>
            <div class="fw-semibold">$${availMargin}</div>
          </div>
        </div>
        <div class="mb-2">
          <div class="d-flex justify-content-between small">
            <span>Margin Ratio</span>
            <span class="${marginRatio >= 0.8 ? 'text-danger fw-bold' : ''}">${marginPct}%</span>
          </div>
          <div class="progress" style="height:6px">
            <div class="progress-bar ${marginBarClass}" style="width:${Math.min(marginPct, 100)}%"></div>
          </div>
        </div>
        <div class="mb-2">
          <div class="d-flex justify-content-between small">
            <span>Correlation Risk</span>
            <span>${corrPct}%</span>
          </div>
          <div class="progress" style="height:6px">
            <div class="progress-bar ${corrBarClass}" style="width:${Math.min(corrPct, 100)}%"></div>
          </div>
        </div>
        <div class="row g-2 small text-muted mt-1">
          <div class="col-6">Max Drawdown Today: <span class="text-danger">${_fmt2(drawdown)}%</span></div>
          <div class="col-6">Open Positions: <span class="fw-semibold">${risk.open_positions ?? 0}</span></div>
        </div>
      </div>
    </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * Update the trade history table when new trades arrive via WebSocket.
 * Only updates if the element exists on the page (trades.html).
 * @param {Array} trades - Most-recent trade objects.
 */
function _applyRecentTrades(trades) {
  const tbody = document.getElementById('trades-tbody');
  if (!tbody || !trades || !trades.length) return;

  // Prepend new rows for trades that aren't already in the table
  const existingIds = new Set(
    Array.from(tbody.querySelectorAll('tr[data-trade-id]')).map(r => r.dataset.tradeId)
  );

  const newRows = trades
    .filter(t => t.id && !existingIds.has(String(t.id)))
    .reverse()
    .map(t => {
      const pnlClass = parseFloat(t.pnl || 0) >= 0 ? 'text-success' : 'text-danger';
      const dirClass = (t.direction || '').toLowerCase() === 'long' ? 'success' : 'danger';
      return `<tr data-trade-id="${escapeHtml(String(t.id || ''))}">
        <td class="text-muted small">${escapeHtml(String(t.id || ''))}</td>
        <td class="fw-semibold">${escapeHtml(t.symbol || '')}</td>
        <td><span class="badge bg-${dirClass}">${escapeHtml((t.direction || '').toUpperCase())}</span></td>
        <td>$${_fmt4(t.entry_price)}</td>
        <td>${t.exit_price ? '$' + _fmt4(t.exit_price) : '<span class="text-muted">--</span>'}</td>
        <td class="${pnlClass}">${parseFloat(t.pnl || 0) >= 0 ? '+' : ''}${_fmt4(t.pnl)}</td>
        <td class="${pnlClass}">${parseFloat(t.pnl_pct || 0) >= 0 ? '+' : ''}${_fmt2(t.pnl_pct)}%</td>
        <td class="small text-muted">${escapeHtml(t.duration || '--')}</td>
        <td class="small text-muted">${escapeHtml(t.strategy || '--')}</td>
        <td><span class="badge ${t.status === 'closed' ? 'bg-secondary' : t.status === 'open' ? 'bg-success' : 'bg-warning text-dark'}">${escapeHtml((t.status || '').toUpperCase())}</span></td>
      </tr>`;
    });

  if (newRows.length > 0) {
    tbody.insertAdjacentHTML('afterbegin', newRows.join(''));
    // Remove any placeholder "no trades" row
    const placeholder = tbody.querySelector('td[colspan]');
    if (placeholder) placeholder.closest('tr').remove();
  }
}

// ── Dashboard metric updater (legacy) ────────────────────────────────────

/**
 * Update dashboard metric cards with fresh data.
 * @param {Object} metrics - Object with portfolio_value, daily_pnl, open_positions, win_rate.
 */
function updateDashboardMetrics(metrics) {
  if (!metrics) return;

  _setText('metric-portfolio-value', formatCurrency(metrics.portfolio_value));
  _setPnl('metric-daily-pnl', metrics.daily_pnl_pct, '%');
  _setText('metric-open-positions', metrics.open_positions ?? '—');
  _setText('metric-win-rate', metrics.win_rate != null ? formatPercentage(metrics.win_rate) : '—');

  const updatedEl = document.getElementById('last-updated');
  if (updatedEl) updatedEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

// ── Formatting helpers ───────────────────────────────────────────────────

/**
 * Format a number as a USD currency string.
 * @param {number|string} value
 * @returns {string}
 */
function formatCurrency(value) {
  const n = parseFloat(value);
  if (isNaN(n)) return '—';
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/**
 * Format a decimal ratio as a percentage string.
 * @param {number|string} value - e.g. 0.65 → "65.0%"
 * @returns {string}
 */
function formatPercentage(value) {
  const n = parseFloat(value);
  if (isNaN(n)) return '—';
  // If value looks like it's already a percentage (>1), don't multiply
  const pct = Math.abs(n) <= 1 ? n * 100 : n;
  return pct.toFixed(1) + '%';
}

/**
 * Format seconds into a human-readable duration string.
 * @param {number} seconds
 * @returns {string}
 */
function formatDuration(seconds) {
  if (seconds == null || isNaN(seconds)) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/**
 * Apply green/red colour class to all elements with class "pnl-value".
 */
function colourPnlValues() {
  document.querySelectorAll('.pnl-value').forEach(el => {
    const raw = el.textContent.replace(/[^0-9.\-+]/g, '');
    const n = parseFloat(raw);
    if (!isNaN(n)) {
      el.classList.toggle('positive', n >= 0);
      el.classList.toggle('negative', n < 0);
    }
  });
}

// ── Bot control actions ──────────────────────────────────────────────────

/**
 * Pause the trading bot.
 */
async function pauseBot() {
  try {
    const res = await fetch('/api/bot/pause', { method: 'POST' });
    const json = await res.json();
    if (json.success) {
      showToast('⏸ Bot paused.', 'warning');
    } else {
      showToast('Failed to pause: ' + (json.reason || json.error || 'Unknown error'), 'warning');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Resume the trading bot.
 */
async function resumeBot() {
  try {
    const res = await fetch('/api/bot/resume', { method: 'POST' });
    const json = await res.json();
    if (json.success) {
      showToast('▶ Bot resumed.', 'success');
    } else {
      showToast('Failed to resume: ' + (json.reason || json.error || 'Unknown error'), 'warning');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Reset the circuit breaker.
 */
async function resetCircuitBreaker() {
  const confirmed = confirm('Reset the circuit breaker and resume trading?');
  if (!confirmed) return;
  try {
    const res = await fetch('/api/circuit-breaker/reset', { method: 'POST' });
    const json = await res.json();
    if (json.success) {
      showToast('✅ Circuit breaker reset.', 'success');
    } else {
      showToast('Failed: ' + (json.reason || json.error || 'Unknown error'), 'warning');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

// ── Trade management actions ─────────────────────────────────────────────

/**
 * Open a manual trade via the /api/manual/trade endpoint.
 * @param {string} symbol  - e.g. "BTC/USDT"
 * @param {string} side    - "long" or "short"
 * @param {number} leverage - integer leverage (1–20)
 * @param {number} amountUsdt - notional USDT size
 */
async function openManualTrade(symbol, side, leverage, amountUsdt, exchange, stopLoss, takeProfit) {
  if (!symbol || !side || !leverage || !amountUsdt) {
    showToast('Please fill in all trade fields (symbol, side, leverage, amount).', 'warning');
    return;
  }
  const sizeUsdt = parseFloat(amountUsdt);
  if (isNaN(sizeUsdt) || sizeUsdt <= 0) {
    showToast('Please enter a valid USDT amount.', 'warning');
    return;
  }
  const lev = parseInt(leverage, 10);
  if (isNaN(lev) || lev < 1 || lev > 125) {
    showToast('Leverage must be between 1 and 125.', 'warning');
    return;
  }

  const exchangeLabel = exchange ? ` on ${exchange.toUpperCase()}` : '';
  const confirmed = confirm(
    `Open ${side.toUpperCase()} ${symbol} @ ${lev}x for $${sizeUsdt} USDT${exchangeLabel}?`
  );
  if (!confirmed) return;

  // Build payload — SL/TP are optional, only include if provided
  const payload = {
    symbol,
    side,
    leverage: lev,
    size_usdt: sizeUsdt,
  };
  if (exchange) payload.exchange = exchange;
  const sl = parseFloat(stopLoss);
  if (!isNaN(sl) && sl > 0) payload.stop_loss = sl;
  const tp = parseFloat(takeProfit);
  if (!isNaN(tp) && tp > 0) payload.take_profit = tp;

  try {
    const res = await fetch('/api/manual-trade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      showToast('Trade failed: ' + escapeHtml(text || `HTTP ${res.status}`), 'danger');
      return;
    }
    const json = await res.json();
    if (json.success) {
      showToast(`Trade opened: ${escapeHtml(symbol)} ${escapeHtml(side.toUpperCase())} @ ${lev}x`, 'success');
    } else {
      showToast('Trade failed: ' + escapeHtml(json.message || json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + escapeHtml(err.message), 'danger');
  }
}

/**
 * Close a position (full close) for the given symbol.
 * @param {string} symbol
 */
async function closePosition(symbol) {
  const confirmed = confirm(`Close position for ${symbol}?\n\nThis will execute a market order to close the full position.`);
  if (!confirmed) return;
  try {
    const res = await fetch(`/api/positions/${symbol}/close`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Position closed: ${escapeHtml(symbol)}`, 'success');
    } else {
      showToast('Close failed: ' + escapeHtml(json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + escapeHtml(err.message), 'danger');
  }
}

/**
 * Partially close a position for the given symbol by a percentage.
 * @param {string} symbol
 * @param {number} percentage - e.g. 50 means close 50%
 */
async function reducePosition(symbol, percentage) {
  const confirmed = confirm(`Close ${percentage}% of the ${symbol} position?`);
  if (!confirmed) return;
  try {
    const res = await fetch(`/api/positions/${symbol}/reduce`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ percentage }),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Reduced ${symbol} by ${percentage}%`, 'success');
    } else {
      showToast('Reduce failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Set (or update) the stop-loss for a position.
 * @param {string} symbol
 */
async function setStopLoss(symbol) {
  const symId = symbol.replace('/', '-');
  const input = document.getElementById(`sl-${symId}`);
  const price = input ? parseFloat(input.value) : NaN;
  if (!price || isNaN(price) || price <= 0) {
    showToast('Please enter a valid stop-loss price.', 'warning');
    return;
  }
  try {
    const res = await fetch(`/api/positions/${symbol}/stop-loss`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ price }),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Stop-loss set for ${symbol} @ $${price}`, 'success');
    } else {
      showToast('Set SL failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Set (or update) the take-profit for a position.
 * @param {string} symbol
 */
async function setTakeProfit(symbol) {
  const symId = symbol.replace('/', '-');
  const input = document.getElementById(`tp-${symId}`);
  const price = input ? parseFloat(input.value) : NaN;
  if (!price || isNaN(price) || price <= 0) {
    showToast('Please enter a valid take-profit price.', 'warning');
    return;
  }
  try {
    const res = await fetch(`/api/positions/${symbol}/take-profit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ price }),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Take-profit set for ${symbol} @ $${price}`, 'success');
    } else {
      showToast('Set TP failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Cancel all open orders for a symbol.
 * @param {string} symbol
 */
async function cancelSymbolOrders(symbol) {
  const confirmed = confirm(`Cancel all open orders for ${symbol}?`);
  if (!confirmed) return;
  try {
    const res = await fetch(`/api/positions/${symbol}/cancel-orders`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Cancelled ${json.cancelled ?? 'all'} orders for ${symbol}`, 'success');
    } else {
      showToast('Cancel failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Cancel a single order by ID.
 * @param {string} orderId
 * @param {string} symbol
 */
async function cancelOrder(orderId, symbol) {
  const confirmed = confirm(`Cancel order ${orderId}?`);
  if (!confirmed) return;
  try {
    const res = await fetch(`/api/orders/${encodeURIComponent(orderId)}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol }),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Order ${orderId} cancelled`, 'success');
    } else {
      showToast('Cancel failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Cancel all open orders across all symbols.
 */
async function cancelAllOrders() {
  const confirmed = confirm('Cancel ALL open orders across all symbols?\n\nThis cannot be undone.');
  if (!confirmed) return;
  try {
    const res = await fetch('/api/orders/cancel-all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Cancelled ${json.cancelled ?? 0} order(s)`, 'success');
    } else {
      showToast('Cancel all failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Modify leverage for an open position.
 * @param {string} symbol
 */
async function modifyLeverage(symbol) {
  const newLev = prompt(`Enter new leverage for ${symbol} (e.g. 10):`);
  if (newLev === null) return;
  const lev = parseInt(newLev, 10);
  if (isNaN(lev) || lev < 1) {
    showToast('Please enter a valid leverage value (>= 1).', 'warning');
    return;
  }
  try {
    const res = await fetch(`/api/positions/${symbol}/leverage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ leverage: lev }),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Leverage for ${symbol} set to ${lev}×`, 'success');
    } else {
      showToast('Modify leverage failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

/**
 * Add margin to an isolated position.
 * @param {string} symbol
 */
async function addMargin(symbol) {
  const amtStr = prompt(`Enter USDT amount to add as margin for ${symbol}:`);
  if (amtStr === null) return;
  const amt = parseFloat(amtStr);
  if (isNaN(amt) || amt <= 0) {
    showToast('Please enter a valid margin amount (> 0 USDT).', 'warning');
    return;
  }
  try {
    const res = await fetch(`/api/positions/${symbol}/add-margin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount: amt }),
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Added $${amt.toFixed(2)} margin to ${symbol}`, 'success');
    } else {
      showToast('Add margin failed: ' + (json.error || 'Unknown error'), 'danger');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

// ── Emergency stop ───────────────────────────────────────────────────────

/**
 * Prompt the user for confirmation then POST to /api/v1/emergency-stop.
 */
async function emergencyStop() {
  const confirmed = confirm(
    '⚠️  EMERGENCY STOP\n\n' +
    'This will trigger the circuit breaker and halt all trading.\n\n' +
    'Are you absolutely sure?'
  );
  if (!confirmed) return;

  try {
    const res = await fetch('/api/v1/emergency-stop', { method: 'POST' });
    const json = await res.json();
    if (json.success) {
      showToast('🚨 Emergency stop triggered. All trading halted.', 'danger');
    } else {
      showToast('Failed: ' + (json.reason || json.error || 'Unknown error'), 'warning');
    }
  } catch (err) {
    showToast('Request failed: ' + err.message, 'danger');
  }
}

// ── Toast notifications ──────────────────────────────────────────────────

/**
 * Display a Bootstrap 5 toast notification.
 * @param {string} message - The notification body text.
 * @param {string} level   - Bootstrap colour variant: info, success, warning, danger.
 */
function showToast(message, level = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const id = 'toast-' + Date.now();
  const bgClass = {
    info: 'bg-info', success: 'bg-success',
    warning: 'bg-warning', danger: 'bg-danger',
  }[level] || 'bg-secondary';

  const html = `
    <div id="${id}" class="toast align-items-center text-white ${bgClass} border-0"
         role="alert" aria-live="assertive" data-bs-delay="5000">
      <div class="d-flex">
        <div class="toast-body">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto"
                data-bs-dismiss="toast"></button>
      </div>
    </div>`;

  container.insertAdjacentHTML('beforeend', html);
  const el = document.getElementById(id);
  const toast = new bootstrap.Toast(el);
  toast.show();
  el.addEventListener('hidden.bs.toast', () => el.remove());
}

// ── Clock ─────────────────────────────────────────────────────────────────

function _updateClock() {
  const el = document.getElementById('current-time');
  if (el) el.textContent = new Date().toUTCString().replace(' GMT', ' UTC');
}

// ── Internal helpers ──────────────────────────────────────────────────────

function _setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? '—';
}

function _setPnl(id, value, suffix = '') {
  const el = document.getElementById(id);
  if (!el) return;
  const n = parseFloat(value);
  if (isNaN(n)) { el.textContent = '—'; return; }
  el.textContent = (n >= 0 ? '+' : '') + n.toFixed(2) + suffix;
  el.classList.toggle('positive', n >= 0);
  el.classList.toggle('negative', n < 0);
}

function _setConnectionIndicator(online) {
  const dot = document.getElementById('connection-indicator');
  if (!dot) return;
  dot.className = 'status-dot ' + (online ? 'status-dot--online' : 'status-dot--offline');
  dot.title = online ? 'WebSocket connected' : 'WebSocket disconnected';
}

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  _updateClock();
  setInterval(_updateClock, 1000);
  colourPnlValues();
  connectWebSocket();
  startPolling();

  // ISSUE 12 FIX: Pause WebSocket when tab is hidden, resume when visible.
  // Prevents unnecessary reconnect attempts and server load when the user
  // isn't looking at the dashboard.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      _tabHidden = true;
      // Close existing connection — server won't waste bandwidth sending
      // updates to a tab nobody is watching.
      if (_ws && _ws.readyState === WebSocket.OPEN) {
        _ws.close();
      }
      if (_reconnectTimer) {
        clearTimeout(_reconnectTimer);
        _reconnectTimer = null;
      }
    } else {
      _tabHidden = false;
      // Reset backoff and reconnect immediately when tab becomes visible
      _reconnectAttempts = 0;
      connectWebSocket();
    }
  });

  // ISSUE 12 FIX: Don't attempt reconnect when the page is being unloaded
  // (navigation away, tab close, browser close). Prevents console errors
  // and wasted reconnect attempts during teardown.
  window.addEventListener('beforeunload', () => {
    _pageUnloading = true;
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }
    if (_ws) {
      _ws.close();
    }
  });

  // Page-specific data loading
  const path = location.pathname;
  if (path === '/trades') {
    _loadTradesPage();
  } else if (path === '/performance') {
    _loadPerformancePage();
  } else if (path === '/risk') {
    _loadRiskPage();
  } else if (path === '/settings') {
    _loadSettingsPage();
  }
});

// ── Page-specific loaders ─────────────────────────────────────────────────

async function _loadTradesPage() {
  try {
    const res = await fetch('/api/trades');
    if (!res.ok) return;
    const data = await res.json();
    if (data.trades) _applyTradesTable(data.trades);
  } catch {}
}

async function _loadPerformancePage() {
  try {
    const res = await fetch('/api/performance');
    if (!res.ok) return;
    const data = await res.json();
    _applyPerformance(data);
  } catch {}
}

async function _loadRiskPage() {
  try {
    const res = await fetch('/api/risk');
    if (!res.ok) return;
    const data = await res.json();
    _applyRiskPage(data);
  } catch {}
}

async function _loadSettingsPage() {
  // Settings page is server-rendered; no additional fetch needed
}

function _applyTradesTable(trades) {
  const tbody = document.getElementById('trades-tbody');
  if (!tbody || !trades) return;
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted py-4">No trades found</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map((t, i) => {
    const pnl = parseFloat(t.pnl) || 0;
    const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
    const dirClass = (t.direction || '').toLowerCase() === 'long' ? 'success' : 'danger';
    return `<tr>
      <td class="text-muted small">${escapeHtml(String(t.id || i + 1))}</td>
      <td class="fw-semibold">${escapeHtml(t.symbol || '')}</td>
      <td><span class="badge bg-${dirClass}">${escapeHtml((t.direction || '').toUpperCase())}</span></td>
      <td>$${_fmt4(t.entry_price)}</td>
      <td>${t.exit_price ? '$' + _fmt4(t.exit_price) : '<span class="text-muted">—</span>'}</td>
      <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}${_fmt4(pnl)}</td>
      <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}${_fmt2(parseFloat(t.pnl_pct) || 0)}%</td>
      <td class="small text-muted">${escapeHtml(t.duration || '—')}</td>
      <td class="small text-muted">${escapeHtml(t.strategy || '—')}</td>
      <td><span class="badge bg-secondary">${escapeHtml((t.status || 'closed').toUpperCase())}</span></td>
    </tr>`;
  }).join('');
}

function _applyRiskPage(data) {
  const el = document.getElementById('risk-json-display');
  if (el) el.textContent = JSON.stringify(data, null, 2);
}

// ── Theme toggle ──────────────────────────────────────────────────────────

/**
 * Toggle between dark and light themes
 */
function toggleTheme() {
  const html = document.documentElement;
  const currentTheme = html.getAttribute('data-bs-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

  // Update theme
  html.setAttribute('data-bs-theme', newTheme);

  // Save preference to localStorage
  localStorage.setItem('theme', newTheme);

  // Update icon
  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.className = newTheme === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
  }

  // Update body background
  document.body.className = newTheme === 'dark' ? 'bg-primary-dark' : 'bg-light';

  showToast(`Switched to ${newTheme} theme`, 'success');
}

/**
 * Initialize theme from localStorage
 */
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  const html = document.documentElement;
  html.setAttribute('data-bs-theme', savedTheme);

  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.className = savedTheme === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
  }

  document.body.className = savedTheme === 'dark' ? 'bg-primary-dark' : 'bg-light';
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', initTheme);

