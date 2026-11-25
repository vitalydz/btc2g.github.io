(function (root) {
  'use strict';

  function dataStatus(signal, meta, now = new Date()) {
    if (!signal || typeof signal.ratio !== 'number' || !Number.isFinite(signal.ratio) || signal.ratio <= 0) return 'unavailable';
    if (meta?.status !== 'ok' || meta?.source !== 'yfinance' || meta?.tickers?.Bitcoin !== 'BTC-USD' || meta?.tickers?.Gold !== 'GC=F') return 'unavailable';
    const value = signal.last_updated;
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return 'unavailable';
    const date = new Date(`${value}T00:00:00Z`);
    if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== value) return 'unavailable';
    const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
    const age = (today - date.getTime()) / 86400000;
    return !Number.isFinite(age) || age < 0 ? 'unavailable' : age > 3 ? 'stale' : 'fresh';
  }

  function convert(value, direction, signal, meta, now = new Date()) {
    const status = dataStatus(signal, meta, now);
    if (status !== 'fresh') return { status };
    const input = String(value).trim();
    if (!/^(?:\d+(?:[.,]\d+)?|[.,]\d+)$/.test(input) || input.length > 40) return { status: 'invalid' };
    const amount = Number(input.replace(',', '.'));
    if (!Number.isFinite(amount) || amount < 0 || !['btc', 'gold'].includes(direction)) return { status: 'invalid' };
    const result = direction === 'btc' ? amount * signal.ratio : amount / signal.ratio;
    if (!Number.isFinite(result) || (amount > 0 && result === 0)) return { status: 'invalid' };
    return { status: 'ok', result, unit: direction === 'btc' ? 'oz t' : 'BTC' };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { dataStatus, convert };
  if (!root.document) return;
  const doc = root.document;
  const byId = id => doc.getElementById(id);
  let signal = null;
  let meta = null;
  let language = 'en';
  let shareText = '';
  const words = () => root.BTC2GAudienceText[language] || root.BTC2GAudienceText.en;
  const text = (id, value) => { byId(id).textContent = value; };

  function render() {
    const t = words();
    text('site-purpose', t.intro);
    text('calculator-jump', t.calculator);
    text('save-entry-title', t.saveTitle);
    text('save-entry-note', t.saveNote);
    text('save-entry-link', t.saveAction);
    text('chart-method', t.chart);
    text('signal-confidence-label', t.score);
    text('signal-updated-label', t.marketDate);
    text('insight-update-title', t.generated);
    text('calculator-title', t.calculator);
    text('calculator-note', t.note);
    text('calculator-amount-label', t.amount);
    text('calculator-direction-label', t.direction);
    text('calculator-btc-option', `BTC → ${t.gold}`);
    text('calculator-gold-option', `${t.gold} → BTC`);
    const status = dataStatus(signal, meta);
    const result = convert(byId('calculator-amount').value, byId('calculator-direction').value, signal, meta);
    byId('calculator-amount').disabled = status !== 'fresh';
    byId('calculator-direction').disabled = status !== 'fresh';
    byId('calculator-amount').setAttribute('aria-invalid', String(result.status === 'invalid'));
    text('calculator-source', ['fresh', 'stale'].includes(status)
      ? `yfinance · BTC-USD / GC=F · ${signal.last_updated} (UTC)${status === 'fresh' ? ` · 1 BTC ≈ ${signal.ratio} oz t` : ''}`
      : t.unavailable);
    if (result.status === 'ok') {
      const digits = result.unit === 'BTC' ? 8 : 6;
      const minimum = 10 ** -digits;
      const formatter = new Intl.NumberFormat(language, { maximumFractionDigits: digits });
      const amount = result.result > 0 && result.result < minimum
        ? `< ${formatter.format(minimum)}` : formatter.format(result.result);
      shareText = `BTC2G: ${byId('calculator-amount').value} ${byId('calculator-direction').value === 'btc' ? 'BTC' : 'oz t'} ≈ ${amount} ${result.unit} · ${signal.last_updated} UTC · yfinance BTC-USD / GC=F`;
      text('calculator-result', `≈ ${amount} ${result.unit}`);
      const reciprocal = 1 / signal.ratio;
      const kilograms = signal.ratio * 0.0311034768;
      text('calculator-equivalents', `1 BTC = ${signal.ratio.toFixed(3)} oz t = ${kilograms.toFixed(3)} kg gold · 1 oz t gold = ${reciprocal.toFixed(8)} BTC · BTC ${new Intl.NumberFormat(language, {style:'currency',currency:'USD',maximumFractionDigits:0}).format(signal.btc_price_usd)} · Gold ${new Intl.NumberFormat(language, {style:'currency',currency:'USD',maximumFractionDigits:2}).format(signal.gold_price_usd)}/oz`);
    } else {
      shareText = '';
      text('calculator-result', result.status === 'invalid' ? t.invalid : t.unavailable);
      text('calculator-equivalents', '');
    }
  }

  root.BTC2GTools = {
    update(nextSignal, nextMeta, nextLanguage) {
      signal = nextSignal;
      meta = nextMeta;
      language = nextLanguage || language;
      render();
    },
  };
  byId('calculator-amount').addEventListener('input', render);
  byId('calculator-direction').addEventListener('change', render);
  byId('calculator-copy').addEventListener('click', async () => {
    if (!shareText) return;
    try {
      await navigator.clipboard.writeText(shareText);
      text('calculator-action-status', language === 'ru' ? 'Результат скопирован.' : 'Result copied.');
    } catch (error) {
      text('calculator-action-status', shareText);
    }
  });
  byId('calculator-share').addEventListener('click', async () => {
    if (!shareText) return;
    if (navigator.share) {
      try { await navigator.share({title:'BTC2G calculator', text:shareText, url:location.href}); } catch (error) {}
    } else {
      try { await navigator.clipboard.writeText(shareText); text('calculator-action-status', language === 'ru' ? 'Результат скопирован.' : 'Result copied.'); } catch (error) { text('calculator-action-status', shareText); }
    }
  });
  // Recheck age on return to a long-lived page and at UTC date boundaries.
  doc.addEventListener('visibilitychange', () => { if (!doc.hidden) render(); });
  root.setInterval(render, 60000);
})(typeof window !== 'undefined' ? window : globalThis);
