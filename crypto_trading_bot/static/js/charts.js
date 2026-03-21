/**
 * charts.js — Chart.js initialisation helpers for CryptoBot dashboard.
 * All charts use a consistent dark theme and responsive layout.
 */

'use strict';

// ── Shared chart defaults ────────────────────────────────────────────────

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: true,
  animation: { duration: 400 },
  plugins: {
    legend: {
      labels: { color: '#8b949e', font: { size: 11 } },
    },
    tooltip: {
      backgroundColor: '#1c2128',
      borderColor: '#30363d',
      borderWidth: 1,
      titleColor: '#e6edf3',
      bodyColor: '#8b949e',
      padding: 10,
    },
  },
  scales: {
    x: {
      ticks:  { color: '#6e7681', maxTicksLimit: 8 },
      grid:   { color: '#21262d' },
    },
    y: {
      ticks:  { color: '#6e7681' },
      grid:   { color: '#21262d' },
    },
  },
};

/** Merge extra options into CHART_DEFAULTS (shallow). */
function _mergeOpts(extra) {
  return Object.assign({}, CHART_DEFAULTS, extra, {
    plugins: Object.assign({}, CHART_DEFAULTS.plugins, extra.plugins || {}),
    scales:  Object.assign({}, CHART_DEFAULTS.scales,  extra.scales  || {}),
  });
}

// ── Equity curve (line) ──────────────────────────────────────────────────

/**
 * Initialise an equity-curve line chart on *canvasId*.
 *
 * @param {string}   canvasId - HTML canvas element id.
 * @param {string[]} labels   - X-axis labels (dates / timestamps).
 * @param {number[]} data     - Portfolio equity values.
 * @returns {Chart}
 */
function initEquityChart(canvasId, labels, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Equity (USDT)',
        data,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88, 166, 255, 0.08)',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.3,
      }],
    },
    options: _mergeOpts({
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => ' $' + parseFloat(ctx.raw).toLocaleString('en-US', {
              minimumFractionDigits: 2, maximumFractionDigits: 2,
            }),
          },
        },
      },
    }),
  });
}

// ── Daily P&L bar chart ──────────────────────────────────────────────────

/**
 * Initialise a daily P&L bar chart on *canvasId*.
 *
 * @param {string}   canvasId - HTML canvas element id.
 * @param {string[]} labels   - X-axis date labels.
 * @param {number[]} data     - Daily P&L values (positive or negative).
 * @returns {Chart}
 */
function initDailyPnlChart(canvasId, labels, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const colors = data.map(v => v >= 0 ? 'rgba(63, 185, 80, 0.75)' : 'rgba(248, 81, 73, 0.75)');
  const borderColors = data.map(v => v >= 0 ? '#3fb950' : '#f85149');

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Daily P&L (%)',
        data,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: _mergeOpts({
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => (ctx.raw >= 0 ? '+' : '') + ctx.raw.toFixed(2) + '%',
          },
        },
      },
    }),
  });
}

// ── Portfolio allocation doughnut ─────────────────────────────────────────

/**
 * Initialise a portfolio-allocation doughnut chart on *canvasId*.
 *
 * @param {string}   canvasId - HTML canvas element id.
 * @param {Object[]} data     - Array of { label, value } objects.
 * @returns {Chart}
 */
function initPositionsChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const palette = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149',
    '#bc8cff', '#39d353', '#ff7b72', '#ffa657',
  ];

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.map(d => d.label),
      datasets: [{
        data: data.map(d => d.value),
        backgroundColor: palette.slice(0, data.length),
        borderColor: '#1c2128',
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 400 },
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#8b949e', font: { size: 11 }, padding: 12 },
        },
        tooltip: {
          backgroundColor: '#1c2128',
          borderColor: '#30363d',
          borderWidth: 1,
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
          callbacks: {
            label: ctx => ` ${ctx.label}: $${parseFloat(ctx.raw).toLocaleString()}`,
          },
        },
      },
    },
  });
}

// ── Generic chart updater ────────────────────────────────────────────────

/**
 * Push new labels and data into an existing Chart.js instance
 * and re-render it.
 *
 * @param {Chart}    chart     - Existing Chart.js instance.
 * @param {string[]} newLabels - Replacement labels.
 * @param {number[]} newData   - Replacement data values.
 */
function updateChart(chart, newLabels, newData) {
  if (!chart) return;
  chart.data.labels = newLabels;
  if (chart.data.datasets.length > 0) {
    chart.data.datasets[0].data = newData;

    // Re-colour bars if it's a bar chart showing P&L
    if (chart.config.type === 'bar') {
      chart.data.datasets[0].backgroundColor =
        newData.map(v => v >= 0 ? 'rgba(63, 185, 80, 0.75)' : 'rgba(248, 81, 73, 0.75)');
      chart.data.datasets[0].borderColor =
        newData.map(v => v >= 0 ? '#3fb950' : '#f85149');
    }
  }
  chart.update('active');
}

// ── Equity curve with drawdown overlay ──────────────────────────────────

/**
 * Initialise an equity-curve line chart with an optional drawdown overlay.
 *
 * @param {string}   canvasId      - HTML canvas element id.
 * @param {string[]} labels        - X-axis labels.
 * @param {number[]} equityData    - Portfolio equity values.
 * @param {number[]} drawdownData  - Drawdown % values (same length, optional).
 * @returns {Chart}
 */
function initEquityDrawdownChart(canvasId, labels, equityData, drawdownData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const datasets = [
    {
      label: 'Equity (USDT)',
      data: equityData,
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88, 166, 255, 0.08)',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: true,
      tension: 0.3,
      yAxisID: 'yEquity',
    },
  ];

  if (drawdownData && drawdownData.length > 0) {
    datasets.push({
      label: 'Drawdown (%)',
      data: drawdownData,
      borderColor: 'rgba(248, 81, 73, 0.75)',
      backgroundColor: 'rgba(248, 81, 73, 0.10)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      tension: 0.3,
      yAxisID: 'yDrawdown',
    });
  }

  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: _mergeOpts({
      scales: {
        x: {
          ticks: { color: '#6e7681', maxTicksLimit: 8 },
          grid:  { color: '#21262d' },
        },
        yEquity: {
          type: 'linear',
          position: 'left',
          ticks: { color: '#6e7681' },
          grid:  { color: '#21262d' },
        },
        yDrawdown: {
          type: 'linear',
          position: 'right',
          reverse: true,
          ticks: { color: '#f85149', callback: v => v + '%' },
          grid:  { drawOnChartArea: false },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(ctx) {
              if (ctx.dataset.yAxisID === 'yDrawdown') {
                return ' Drawdown: ' + parseFloat(ctx.raw).toFixed(2) + '%';
              }
              return ' $' + parseFloat(ctx.raw).toLocaleString('en-US', {
                minimumFractionDigits: 2, maximumFractionDigits: 2,
              });
            },
          },
        },
      },
    }),
  });
}

// ── Win rate by hour bar chart ───────────────────────────────────────────

// ── Gold K-line chart (XAU/USDT) ────────────────────────────────────────

/**
 * Initialise an XAU/USDT gold futures candlestick chart using
 * TradingView Lightweight Charts.  Fetches data from the existing
 * /api/market/XAU%2FUSDT/klines endpoint.
 *
 * @param {string} containerId - HTML element id for the chart container.
 * @returns {{ chart: object, candleSeries: object } | null}
 */
function initGoldChart(containerId) {
  if (typeof LightweightCharts === 'undefined') return null;
  const container = document.getElementById(containerId);
  if (!container) return null;

  const chart = LightweightCharts.createChart(container, {
    width:  container.offsetWidth  || 800,
    height: container.offsetHeight || 400,
    layout: {
      background: { color: '#1a1a2e' },
      textColor:  '#e0e0e0',
    },
    grid: {
      vertLines: { color: '#2a2a3e' },
      horzLines: { color: '#2a2a3e' },
    },
    rightPriceScale: {
      borderColor: '#3a3a4e',
      scaleMargins: { top: 0.1, bottom: 0.2 },
    },
    timeScale: {
      borderColor:    '#3a3a4e',
      timeVisible:    true,
      secondsVisible: false,
    },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor:         '#00c853',
    downColor:       '#ff1744',
    borderUpColor:   '#00c853',
    borderDownColor: '#ff1744',
    wickUpColor:     '#00c853',
    wickDownColor:   '#ff1744',
  });

  fetchGoldKlines(candleSeries);

  return { chart, candleSeries };
}

/**
 * Fetch XAU/USDT klines and set them on *candleSeries*.
 *
 * @param {object} candleSeries - Lightweight Charts candlestick series.
 * @param {string} [timeframe]  - Candle interval (default '1m').
 * @param {number} [limit]      - Number of candles to fetch (default 200).
 */
function fetchGoldKlines(candleSeries, timeframe, limit) {
  if (!candleSeries) return;
  const tf  = timeframe || '1m';
  const lim = limit     || 200;
  fetch(`/api/market/${encodeURIComponent('XAU/USDT')}/klines?timeframe=${tf}&limit=${lim}`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data && data.candles && data.candles.length) {
        candleSeries.setData(data.candles);
      }
    })
    .catch(() => {});
}

/**
 * Initialise a win-rate-by-hour bar chart on *canvasId*.
 *
 * @param {string}   canvasId - HTML canvas element id.
 * @param {string[]} labels   - Hour labels (e.g. '0:00' … '23:00').
 * @param {number[]} data     - Win rate % for each hour.
 * @returns {Chart}
 */
function initWinRateByHourChart(canvasId, labels, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const colors = data.map(v => v >= 55 ? 'rgba(63, 185, 80, 0.75)' :
                                v >= 45 ? 'rgba(210, 153, 34, 0.75)' :
                                          'rgba(248, 81, 73, 0.75)');

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Win Rate (%)',
        data,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.75', '1')),
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: _mergeOpts({
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#6e7681', callback: v => v + '%' },
          grid:  { color: '#21262d' },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => ' ' + parseFloat(ctx.raw).toFixed(1) + '%',
          },
        },
      },
    }),
  });
}
