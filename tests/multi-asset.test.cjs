const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const dashboard = require('../assets/js/multi-asset-dashboard');

const root = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'assets/multi_asset_strength.json'), 'utf8'));
const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const workflow = fs.readFileSync(path.join(root, '.github/workflows/update-chart.yml'), 'utf8');

test('published multi-asset data is finite, sourced, aligned, and current internally', () => {
  assert.equal(dashboard.validData(data), true);
  assert.deepEqual(data.tickers, { Bitcoin: 'BTC-USD', Gold: 'GC=F', Silver: 'SI=F', SPX: '^GSPC' });
  assert.equal(data.price_basis, 'Latest completed common daily close; no forward-filled quotes');
  assert.equal(data.series.at(-1).date, data.updated_for_market_date);
  for (const row of data.series) {
    assert.ok(Math.abs(row.btc_gold - row.btc_spx / row.gold_spx) <= Math.max(0.00002, row.btc_gold * 0.000002));
  }
});

test('dashboard exposes all four comparisons and identifies a real strongest Bitcoin pair', () => {
  assert.deepEqual(dashboard.pairs, ['btc_gold', 'btc_silver', 'btc_spx', 'gold_spx']);
  assert.ok(['btc_gold', 'btc_silver', 'btc_spx'].includes(data.current.strongest_btc_30d));
  const expected = ['btc_gold', 'btc_silver', 'btc_spx'].sort((a, b) => data.current.pairs[b].change_30d_pct - data.current.pairs[a].change_30d_pct)[0];
  assert.equal(data.current.strongest_btc_30d, expected);
  for (const id of ['relative-strength', 'relative-strength-leader', 'relative-strength-cards', 'relative-strength-source']) {
    assert.match(homepage, new RegExp(`id="${id}"`));
  }
});

test('multi-asset UI has complete text for all 19 languages and responsive layout', () => {
  assert.equal(Object.keys(dashboard.translations).length, 19);
  for (const [language, values] of Object.entries(dashboard.translations)) {
    assert.equal(values.length, 8, `${language} must define every dashboard label`);
    assert.ok(values.every(value => typeof value === 'string' && value.trim()), `${language} labels must be non-empty`);
  }
  const css = fs.readFileSync(path.join(root, 'assets/css/index.css'), 'utf8');
  assert.match(css, /\.relative-strength-cards\s*\{[^}]*grid-template-columns:repeat\(4/);
  assert.match(css, /@media\(max-width:460px\)\{\.relative-strength-cards\{grid-template-columns:1fr\}/);
});

test('scheduled and manual market workflows publish the multi-asset artifact', () => {
  assert.match(workflow, /python scripts\/update_multi_asset\.py/);
  assert.match(workflow, /assets\/multi_asset_strength\.json/);
  const manual = fs.readFileSync(path.join(root, '.github/workflows/update-graph.yml'), 'utf8');
  assert.match(manual, /python scripts\/update_multi_asset\.py/);
  assert.match(manual, /assets\/multi_asset_strength\.json/);
});
