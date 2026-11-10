(function (root) {
  'use strict';

  function buildPurchasingPowerSeries(rows) {
    if (!Array.isArray(rows) || !rows.length) return [];
    const first = rows.find(row => Number.isFinite(row?.btc_usd) && row.btc_usd > 0 && Number.isFinite(row?.gold_usd) && row.gold_usd > 0);
    if (!first) return [];
    return rows.filter(row => /^\d{4}-\d{2}-\d{2}$/.test(row?.date) && Number.isFinite(row.btc_usd) && row.btc_usd > 0 && Number.isFinite(row.gold_usd) && row.gold_usd > 0)
      .map(row => ({
        date: row.date,
        btc_index: row.btc_usd / first.btc_usd * 100,
        gold_index: row.gold_usd / first.gold_usd * 100,
      }));
  }

  const supportedChartModes = new Set(['ratio', 'btc', 'gold', 'score', 'purchasingPower']);

  function chartModeFromSearch(search) {
    try {
      const selected = new URLSearchParams(search || '').get('chart');
      return supportedChartModes.has(selected) ? selected : 'ratio';
    } catch {
      return 'ratio';
    }
  }

  root.BTC2GMarketChartMath = { buildPurchasingPowerSeries, chartModeFromSearch };
  if (!root.document) return;

  const HISTORY_PATH = 'assets/btc_gold_history.json';
  const SIGNAL_PATH = 'assets/btc_signal.json';
  const modes = {
    ratio: { key: 'ratio', color: '#d4af37', label: 'BTC / Gold ratio', unit: ' oz t' },
    btc: { key: 'btc_usd', color: '#f7931a', label: 'Bitcoin price', unit: ' USD' },
    gold: { key: 'gold_usd', color: '#d4af37', label: 'Gold futures proxy', unit: ' USD/oz' },
    score: { key: 'model_score', color: '#10b981', label: 'Model score', unit: '/100' },
    purchasingPower: { label: 'Purchasing-power index', unit: ' index' },
  };
  let series = [];
  let mode = chartModeFromSearch(root.location?.search);
  let regimeAnalysis = null;

  function validHistory(data) {
    return data?.source === 'yfinance' && data?.tickers?.Bitcoin === 'BTC-USD' && data?.tickers?.Gold === 'GC=F' &&
      Array.isArray(data.series) && data.series.length > 1 && data.series.every(row =>
        /^\d{4}-\d{2}-\d{2}$/.test(row.date) && Number.isFinite(row.ratio) && row.ratio > 0 &&
        Number.isFinite(row.btc_usd) && row.btc_usd > 0 && Number.isFinite(row.gold_usd) && row.gold_usd > 0
      );
  }

  function formatValue(value, selectedMode = mode) {
    if (!Number.isFinite(value)) return '—';
    if (selectedMode === 'ratio') return `${value.toFixed(2)} oz t`;
    if (selectedMode === 'score') return `${Math.round(value)}/100`;
    if (selectedMode === 'purchasingPower') return `${value.toFixed(1)}`;
    return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: selectedMode === 'btc' ? 0 : 2 }).format(value);
  }

  function draw() {
    const canvas = document.getElementById('market-history-chart');
    if (!canvas || !series.length) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(rect.width));
    const height = Math.max(280, Math.round(rect.height));
    const dpr = Math.min(root.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const config = modes[mode];
    const indexed = mode === 'purchasingPower' ? buildPurchasingPowerSeries(series) : [];
    const plotSeries = mode === 'purchasingPower'
      ? [
          { label: 'BTC', color: '#f7931a', points: indexed.map(row => ({ date: row.date, value: row.btc_index })) },
          { label: 'Gold', color: '#d4af37', points: indexed.map(row => ({ date: row.date, value: row.gold_index })) },
        ]
      : [{ label: config.label, color: config.color, points: series.map(row => ({ date: row.date, value: row[config.key] })).filter(point => Number.isFinite(point.value)) }];
    const points = plotSeries[0]?.points || [];
    if (points.length < 2 || plotSeries.some(item => item.points.length !== points.length)) return;
    const values = plotSeries.flatMap(item => item.points.map(point => point.value));
    let min = Math.min(...values);
    let max = Math.max(...values);
    const padding = Math.max((max - min) * 0.1, max * 0.01, 1);
    min -= padding;
    max += padding;
    const left = 66, right = 20, top = mode === 'purchasingPower' ? 52 : 32, bottom = 44;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const x = index => left + index / (points.length - 1) * plotWidth;
    const y = value => top + (max - value) / (max - min) * plotHeight;

    if (mode === 'ratio' && regimeAnalysis?.assignments?.length === points.length) {
      const bandColors = { btc: '#f7931a18', gold: '#60a5fa18', neutral: '#94a3b80d' };
      let start = 0;
      while (start < points.length) {
        const regime = regimeAnalysis.assignments[start];
        let end = start;
        while (end + 1 < points.length && regimeAnalysis.assignments[end + 1] === regime) end += 1;
        if (regime && bandColors[regime]) {
          const from = start === 0 ? left : (x(start - 1) + x(start)) / 2;
          const to = end === points.length - 1 ? width - right : (x(end) + x(end + 1)) / 2;
          ctx.fillStyle = bandColors[regime];
          ctx.fillRect(from, top, to - from, plotHeight);
        }
        start = end + 1;
      }
    }

    ctx.strokeStyle = '#24344d';
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (let line = 0; line <= 4; line += 1) {
      const value = max - (max - min) * line / 4;
      const py = top + plotHeight * line / 4;
      ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(width - right, py); ctx.stroke();
      ctx.fillText(formatValue(value), left - 9, py + 4);
    }
    ctx.textAlign = 'left';
    ctx.fillText(points[0].date, left, height - 14);
    ctx.textAlign = 'right';
    ctx.fillText(points[points.length - 1].date, width - right, height - 14);

    plotSeries.forEach((item, seriesIndex) => {
      if (plotSeries.length === 1) {
        const gradient = ctx.createLinearGradient(0, top, 0, height - bottom);
        gradient.addColorStop(0, `${item.color}55`);
        gradient.addColorStop(1, `${item.color}00`);
        ctx.beginPath();
        item.points.forEach((point, index) => index ? ctx.lineTo(x(index), y(point.value)) : ctx.moveTo(x(index), y(point.value)));
        ctx.lineTo(x(item.points.length - 1), height - bottom); ctx.lineTo(left, height - bottom); ctx.closePath();
        ctx.fillStyle = gradient; ctx.fill();
      }
      ctx.beginPath();
      item.points.forEach((point, index) => index ? ctx.lineTo(x(index), y(point.value)) : ctx.moveTo(x(index), y(point.value)));
      ctx.strokeStyle = item.color; ctx.lineWidth = 2.5; ctx.stroke();
      const last = item.points[item.points.length - 1];
      ctx.beginPath(); ctx.arc(x(item.points.length - 1), y(last.value), 4, 0, Math.PI * 2); ctx.fillStyle = item.color; ctx.fill();
      ctx.fillStyle = item.color; ctx.font = '600 13px Inter, system-ui, sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(`${item.label}: ${formatValue(last.value)}`, left, 18 + seriesIndex * 20);
    });
  }

  function setMode(nextMode) {
    if (!modes[nextMode]) return;
    mode = nextMode;
    document.querySelectorAll('[data-chart-mode]').forEach(button => {
      const active = button.dataset.chartMode === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const note = document.getElementById('purchasing-power-note');
    if (note) note.hidden = mode !== 'purchasingPower';
    draw();
  }

  async function load(version = Date.now()) {
    const status = document.getElementById('market-history-status');
    status.textContent = 'Loading published history…';
    try {
      const [historyResponse, signalResponse] = await Promise.all([
        fetch(`${HISTORY_PATH}?v=${encodeURIComponent(version)}`, { cache: 'no-store' }),
        fetch(`${SIGNAL_PATH}?v=${encodeURIComponent(version)}`, { cache: 'no-store' }),
      ]);
      if (!historyResponse.ok || !signalResponse.ok) throw new Error('Published data request failed.');
      const history = await historyResponse.json();
      const signal = await signalResponse.json();
      if (!validHistory(history)) throw new Error('Published history could not be verified.');
      series = history.series;
      regimeAnalysis = root.BTC2GRegimes?.analyze(series) || null;
      if (regimeAnalysis) root.BTC2GRegimes.render(regimeAnalysis, document.documentElement.lang || 'en');
      status.textContent = `${history.source} · BTC-USD / GC=F · through ${history.updated_for_market_date} (UTC) · ${series.length} observations`;
      if (Number.isFinite(signal.btc_price_usd) && Number.isFinite(signal.gold_price_usd)) {
        document.getElementById('market-price-strip').textContent = `BTC: ${formatValue(signal.btc_price_usd, 'btc')} · Gold (GC=F): ${formatValue(signal.gold_price_usd, 'gold')} · Market date: ${signal.last_updated} UTC`;
      }
      draw();
    } catch (error) {
      series = [];
      regimeAnalysis = null;
      const detector = document.getElementById('regime-detector');
      if (detector) detector.hidden = true;
      status.textContent = 'Interactive history is temporarily unavailable. The published source files remain linked below.';
    }
  }

  document.querySelectorAll('[data-chart-mode]').forEach(button => button.addEventListener('click', () => setMode(button.dataset.chartMode)));
  root.addEventListener('resize', draw);
  root.BTC2GMarketChart = { load, setMode, validHistory, buildPurchasingPowerSeries, chartModeFromSearch };
  setMode(mode);
  load();
})(typeof window !== 'undefined' ? window : globalThis);
