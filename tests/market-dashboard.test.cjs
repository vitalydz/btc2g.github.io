const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');

test('published history is finite, sourced, ordered, and matches latest signal', () => {
  const history = JSON.parse(read('assets/btc_gold_history.json'));
  const signal = JSON.parse(read('assets/btc_signal.json'));
  assert.equal(history.source, 'yfinance');
  assert.deepEqual(history.tickers, {Gold: 'GC=F', Bitcoin: 'BTC-USD'});
  assert.ok(history.series.length >= 365);
  let previous = '';
  for (const row of history.series) {
    assert.match(row.date, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(row.date > previous);
    previous = row.date;
    for (const key of ['ratio', 'btc_usd', 'gold_usd', 'model_score']) assert.ok(Number.isFinite(row[key]));
    assert.ok(row.ratio > 0 && row.btc_usd > 0 && row.gold_usd > 0);
  }
  const latest = history.series.at(-1);
  assert.equal(latest.date, signal.last_updated);
  assert.equal(Number(latest.ratio.toFixed(3)), signal.ratio);
  assert.equal(latest.btc_usd, signal.btc_price_usd);
  assert.equal(latest.gold_usd, signal.gold_price_usd);
  assert.ok(Number.isFinite(signal.ratio_change_7d_pct));
});

test('homepage is data-first without fabricated audience totals', () => {
  const html = read('index.html');
  for (const mode of ['ratio', 'btc', 'gold', 'score']) assert.match(html, new RegExp(`data-chart-mode="${mode}"`));
  assert.match(html, /assets\/btc_gold_history\.json/);
  assert.match(html, /id="ratio-change"/);
  assert.match(html, /id="subscribe"/);
  assert.match(html, /id="hero-account-link"[^>]+https:\/\/btc2g-accounts\.btc2g\.workers\.dev\//);
  assert.match(html, /setTextById\('hero-account-link', text\.navAccount\)/);
  assert.match(html, /<details class="forecast-static" open>/);
  assert.match(html, /id="forecastChart"[^>]+assets\/btc_gold_forecast\.png/);
  assert.ok(html.indexOf('id="forecastChart"') < html.indexOf('class="signal-card"'));
  assert.ok(html.indexOf('class="signal-card"') < html.indexOf('</details>'));
  assert.ok(html.indexOf('class="signal-card"') < html.indexOf('id="subscribe"'));
  assert.match(html, /id="subscribe-thanks"[^>]+hidden/);
  assert.match(html, /id="email-preview-ratio"/);
  assert.match(html, /confirmedSubscribers\?\.status === 'available'/);
  assert.doesNotMatch(html, /1[\s ]247/);
  assert.doesNotMatch(html, /380\+|380 readers|380 читателей/i);
  assert.doesNotMatch(html, /id="research-support-title"/);
});

test('daily workflows publish chart history and Premium records explicit preferences', () => {
  for (const workflow of ['.github/workflows/update-chart.yml', '.github/workflows/update-graph.yml']) {
    assert.match(read(workflow), /assets\/btc_gold_history\.json/);
  }
  const premium = read('premium.html');
  assert.match(premium, /name="weekly_digest"/);
  assert.match(premium, /first-release-list/);
  assert.match(read('assets/js/premium-v1.js'), /12 months|12 месяцев/);
  assert.match(read('assets/js/premium-v1.js'), /CSV/);
});

test('three existing insight articles end with a functional signup form', () => {
  for (const file of ['bitcoin-vs-gold.html', 'bitcoin-gold-ratio.html', 'bitcoin-vs-gold-forecast-2026.html']) {
    const html = read(file);
    assert.match(html, /class="article-signup-form"/);
    assert.match(html, /assets\/js\/article-signup\.js/);
    assert.match(html, /assets\/js\/subscription-request\.js/);
  }
});
