const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const detector = require('../assets/js/regime-detector');

const root = path.join(__dirname, '..');
const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const chart = fs.readFileSync(path.join(root, 'assets/js/market-chart.js'), 'utf8');
const css = fs.readFileSync(path.join(root, 'assets/css/index.css'), 'utf8');

test('classifies regimes from a transparent moving-average rule', () => {
  const ratios = [100, 100, 100, 120, 120, 80, 80, 100];
  const series = ratios.map((ratio, index) => ({ date: `2026-01-${String(index + 1).padStart(2, '0')}`, ratio }));
  const result = detector.analyze(series, { windowSize: 3, threshold: 0.05 });
  assert.deepEqual(result.assignments, [null, null, 'neutral', 'btc', 'btc', 'gold', 'gold', 'btc']);
  assert.equal(result.current, 'btc');
  assert.equal(result.stats.btc.observations, 3);
  assert.equal(result.stats.btc.typicalRunObservations, 1.5);
  assert.equal(result.stats.gold.typicalRunObservations, 2);
  assert.ok(Math.abs(result.stats.btc.averageDailyChangePct - 15) < 1e-9);
  assert.ok(Math.abs(result.stats.gold.averageDailyChangePct + 16.6666666667) < 1e-8);
});

test('rejects short, unsafe, or unordered history instead of inventing regimes', () => {
  assert.equal(detector.analyze([{ date: '2026-01-01', ratio: 20 }]), null);
  assert.equal(detector.analyze([
    { date: '2026-01-02', ratio: 20 }, { date: '2026-01-01', ratio: 21 }
  ], { windowSize: 2 }), null);
  assert.equal(detector.analyze([
    { date: '2026-01-01', ratio: 20 }, { date: '2026-01-02', ratio: Infinity }
  ], { windowSize: 2 }), null);
});

test('homepage exposes responsive regime analytics backed by published chart history', () => {
  for (const id of ['regime-detector', 'regime-current', 'regime-summary', 'regime-method', 'regime-premium-link']) {
    assert.match(homepage, new RegExp(`id="${id}"`));
  }
  assert.match(homepage, /regime-detector\.js[\s\S]*market-chart\.js/);
  assert.match(chart, /BTC2GRegimes\?\.analyze\(series\)/);
  assert.match(chart, /mode === 'ratio'/);
  assert.match(chart, /fillRect\(from, top, to - from, plotHeight\)/);
  assert.match(css, /\.regime-summary\s*\{[^}]*grid-template-columns:repeat\(3/);
  assert.match(css, /@media\(max-width:760px\)[\s\S]*\.regime-summary\{grid-template-columns:1fr\}/);
});

test('regime detector includes complete UI text for all 19 site languages', () => {
  assert.equal(Object.keys(detector.labels).length, 19);
  for (const [language, values] of Object.entries(detector.labels)) {
    assert.equal(values.length, 10, `${language} must define every regime label`);
    assert.ok(values.every(value => typeof value === 'string' && value.trim()), `${language} labels must be non-empty`);
  }
});
