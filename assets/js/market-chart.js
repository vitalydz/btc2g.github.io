(function (root) {
  'use strict';
  if (!root.document) return;

  const HISTORY_PATH = 'assets/btc_gold_history.json';
  const SIGNAL_PATH = 'assets/btc_signal.json';
  const modes = {
    ratio: { key: 'ratio', color: '#d4af37', label: 'BTC / Gold ratio', unit: ' oz t' },
    btc: { key: 'btc_usd', color: '#f7931a', label: 'Bitcoin price', unit: ' USD' },
    gold: { key: 'gold_usd', color: '#d4af37', label: 'Gold futures proxy', unit: ' USD/oz' },
    score: { key: 'model_score', color: '#10b981', label: 'Model score', unit: '/100' },
  };
  let series = [];
  let mode = 'ratio';
  let language = document.documentElement.lang || 'en', state = 'loading', historyMeta = null, currentPrices = null;
  const ui = () => root.BTC2GLanguageText.text(language);
  function modeLabel(selected = mode) { return root.BTC2GHomeV2Text.chartLabels(language)[['ratio','btc','gold','score'].indexOf(selected)]; }

  function validHistory(data) {
    return data?.source === 'yfinance' && data?.tickers?.Bitcoin === 'BTC-USD' && data?.tickers?.Gold === 'GC=F' &&
      Array.isArray(data.series) && data.series.length > 1 && data.series.every(row =>
        /^\d{4}-\d{2}-\d{2}$/.test(row.date) && Number.isFinite(row.ratio) && row.ratio > 0 &&
        Number.isFinite(row.btc_usd) && row.btc_usd > 0 && Number.isFinite(row.gold_usd) && row.gold_usd > 0
      );
  }

  function formatValue(value, selectedMode = mode) {
    if (!Number.isFinite(value)) return '—';
    if (selectedMode === 'ratio') return `${new Intl.NumberFormat(language, {maximumFractionDigits:2}).format(value)} oz t`;
    if (selectedMode === 'score') return `${Math.round(value)}/100`;
    return new Intl.NumberFormat(language, { style: 'currency', currency: 'USD', maximumFractionDigits: selectedMode === 'btc' ? 0 : 2 }).format(value);
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
    canvas.setAttribute('aria-label', modeLabel());
    const points = series.map(row => ({ date: row.date, value: row[config.key] })).filter(point => Number.isFinite(point.value));
    if (points.length < 2) return;
    const values = points.map(point => point.value);
    let min = Math.min(...values);
    let max = Math.max(...values);
    const padding = Math.max((max - min) * 0.1, max * 0.01, 1);
    min -= padding;
    max += padding;
    const left = 66, right = 20, top = 32, bottom = 44;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const x = index => left + index / (points.length - 1) * plotWidth;
    const y = value => top + (max - value) / (max - min) * plotHeight;

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

    const gradient = ctx.createLinearGradient(0, top, 0, height - bottom);
    gradient.addColorStop(0, `${config.color}55`);
    gradient.addColorStop(1, `${config.color}00`);
    ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(x(index), y(point.value)) : ctx.moveTo(x(index), y(point.value)));
    ctx.lineTo(x(points.length - 1), height - bottom); ctx.lineTo(left, height - bottom); ctx.closePath();
    ctx.fillStyle = gradient; ctx.fill();
    ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(x(index), y(point.value)) : ctx.moveTo(x(index), y(point.value)));
    ctx.strokeStyle = config.color; ctx.lineWidth = 2.5; ctx.stroke();

    const last = points[points.length - 1];
    ctx.beginPath(); ctx.arc(x(points.length - 1), y(last.value), 4, 0, Math.PI * 2); ctx.fillStyle = config.color; ctx.fill();
    ctx.fillStyle = '#f8fafc'; ctx.font = '600 13px Inter, system-ui, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(`${modeLabel()}: ${formatValue(last.value)}`, left, 18);
  }

  function setMode(nextMode) {
    if (!modes[nextMode]) return;
    mode = nextMode;
    document.querySelectorAll('[data-chart-mode]').forEach(button => {
      const active = button.dataset.chartMode === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    draw();
  }

  function renderMetadata() {
    const status = document.getElementById('market-history-status');
    status.textContent = state === 'loading' ? ui().historyLoading : state === 'error' ? ui().historyError
      : `${historyMeta.source} · BTC-USD / GC=F · ${ui().through} ${historyMeta.updated_for_market_date} (UTC) · ${series.length} ${ui().observations}`;
    if (currentPrices && Number.isFinite(currentPrices.btc_price_usd) && Number.isFinite(currentPrices.gold_price_usd)) {
      document.getElementById('market-price-strip').textContent = `BTC: ${formatValue(currentPrices.btc_price_usd, 'btc')} · ${ui().gold} (GC=F): ${formatValue(currentPrices.gold_price_usd, 'gold')} · ${ui().marketDate}: ${currentPrices.last_updated} UTC`;
    } else {
      document.getElementById('market-price-strip').textContent = state === 'loading' ? ui().historyLoading : ui().unavailable;
    }
  }
  function setLanguage(value) {
    language = value;
    renderMetadata();
    draw();
  }

  async function load(version = Date.now()) {
    const status = document.getElementById('market-history-status');
    state = 'loading';
    renderMetadata();
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
      historyMeta = history; currentPrices = signal; state = 'loaded';
      renderMetadata();
      draw();
    } catch (error) {
      series = [];
      state = 'error'; currentPrices = null;
      renderMetadata();
    }
  }

  document.querySelectorAll('[data-chart-mode]').forEach(button => button.addEventListener('click', () => setMode(button.dataset.chartMode)));
  root.addEventListener('resize', draw);
  root.BTC2GMarketChart = { load, setMode, setLanguage, validHistory };
  load();
})(typeof window !== 'undefined' ? window : globalThis);
