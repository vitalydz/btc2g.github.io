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
  for (const mode of ['ratio', 'btc', 'gold', 'score', 'purchasingPower']) assert.match(html, new RegExp(`data-chart-mode="${mode}"`));
  assert.match(html, /id="purchasing-power-note"[^>]+hidden/);
  assert.match(html, /assets\/btc_gold_history\.json/);
  assert.match(html, /id="ratio-change"/);
  assert.match(html, /id="subscribe"/);
  assert.match(html, /id="hero-account-link"[^>]+https:\/\/btc2g-accounts\.btc2g\.workers\.dev\//);
  assert.match(html, /id="hero-account-link" class="hero-button hero-button-primary"/);
  assert.match(html, /id="hero-subscribe-link" class="hero-button"/);
  assert.match(html, /setTextById\('hero-account-link', createAccountLabels\[lang\] \|\| createAccountLabels\.en\)/);
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

test('purchasing-power comparison indexes both assets to the same starting value', () => {
  const chartPath = path.join(root, 'assets/js/market-chart.js');
  delete require.cache[require.resolve(chartPath)];
  require(chartPath);
  const build = globalThis.BTC2GMarketChartMath.buildPurchasingPowerSeries;
  const indexed = build([
    {date: '2026-01-01', btc_usd: 100, gold_usd: 20},
    {date: '2026-01-02', btc_usd: 125, gold_usd: 22},
  ]);
  assert.deepEqual(indexed[0], {date: '2026-01-01', btc_index: 100, gold_index: 100});
  assert.equal(indexed[1].date, '2026-01-02');
  assert.equal(indexed[1].btc_index, 125);
  assert.ok(Math.abs(indexed[1].gold_index - 110) < 1e-10);
});

test('chart deep links select only supported dashboard modes', () => {
  const chartPath = path.join(root, 'assets/js/market-chart.js');
  delete require.cache[require.resolve(chartPath)];
  require(chartPath);
  const select = globalThis.BTC2GMarketChartMath.chartModeFromSearch;
  assert.equal(select('?chart=purchasingPower'), 'purchasingPower');
  assert.equal(select('?chart=gold'), 'gold');
  assert.equal(select('?chart=unknown'), 'ratio');
  assert.equal(select('not a query'), 'ratio');
});

test('chart selection builds a shareable URL without dropping campaign parameters', () => {
  const chartPath = path.join(root, 'assets/js/market-chart.js');
  delete require.cache[require.resolve(chartPath)];
  require(chartPath);
  const update = globalThis.BTC2GMarketChartMath.chartSearchWithMode;
  assert.equal(update('?utm_source=telegram', 'purchasingPower'), '?utm_source=telegram&chart=purchasingPower');
  assert.equal(update('?chart=gold&utm_source=reddit', 'btc'), '?chart=btc&utm_source=reddit');
  assert.equal(update('?chart=gold&utm_source=reddit', 'ratio'), '?utm_source=reddit');
  assert.equal(update('?chart=gold', 'unknown'), '?chart=gold');
});

test('all localized homepages load the purchasing-power release assets', () => {
  const languages = ['ru', 'he', 'de', 'fr', 'es', 'zh', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'tr', 'pl', 'nl', 'uk', 'id', 'vi'];
  for (const language of languages) {
    const html = read(`${language}/index.html`);
    assert.match(html, /assets\/css\/index\.css\?v=20261110-1/);
    assert.match(html, /assets\/js\/home-v2-text\.js\?v=20261110-1/);
    assert.match(html, /assets\/js\/market-chart\.js\?v=20261110-3/);
  }
  const languageSource = read('assets/js/home-v2-text.js');
  assert.match(languageSource, /ensurePurchasingPowerElements/);
  assert.match(languageSource, /purchasingPowerLabels/);
  assert.match(languageSource, /purchasingPowerNotes/);
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
    assert.match(html, /data-article-account[^>]+btc2g-accounts\.btc2g\.workers\.dev[^>]*>Create a free BTC2G account<\/a>/);
    assert.match(html, /class="related-guides"/);
    assert.equal((html.match(/class="related-guides"[\s\S]*?<\/nav>/) || [''])[0].match(/<a /g)?.length, 2);
    assert.match(html, /assets\/js\/article-signup\.js/);
    assert.match(html, /assets\/js\/subscription-request\.js/);
    assert.match(html, /assets\/js\/insight-article-language\.js/);
  }
  const language = read('assets/js/insight-article-language.js');
  assert.match(language, /url\.searchParams\.set\('lang', lang\)/);
  assert.match(language, /'Create a free BTC2G account': 'Создать бесплатный аккаунт BTC2G'/);
  assert.match(language, /'Related BTC2G guides': 'Связанные материалы BTC2G'/);
});

test('insight links preserve the selected language', () => {
  const html = read('index.html');
  assert.match(html, /bitcoin-vs-gold\.html\?lang=' \+ encodeURIComponent\(lang\)/);
  assert.match(html, /bitcoin-gold-ratio\.html\?lang=' \+ encodeURIComponent\(lang\)/);
  assert.match(html, /bitcoin-vs-gold-forecast-2026\.html\?lang=' \+ encodeURIComponent\(lang\)/);
  assert.match(read('assets/js/insight-article-language.js'), /const russian/);
});
