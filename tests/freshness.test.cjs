const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const code = html.match(/    function getSignalFreshness\([^]*?\n    \}/)[0];
const context = vm.createContext({});
vm.runInContext(code, context);
const classify = context.getSignalFreshness;
assert.equal(classify('2026-10-01', new Date('2026-10-04T23:59:59Z')), 'fresh');
assert.equal(classify('2026-10-01', new Date('2026-10-05T00:00:00Z')), 'delayed');
assert.equal(classify('2026-10-05', new Date('2026-10-05T23:59:59Z')), 'fresh');
assert.equal(classify('2026-10-06', new Date('2026-10-05T00:00:00Z')), 'unknown');
for (const date of [null, undefined, '', 'nonsense', '2026-02-30', '2026-13-01', '2026-2-01', '2026-10-01T00:00:00Z', 0]) {
  assert.equal(classify(date, new Date('2026-10-05T00:00:00Z')), 'unknown');
}
assert.equal(classify('2024-02-29', new Date('2024-03-03T00:00:00Z')), 'fresh');
assert.equal(classify('2023-02-29', new Date('2024-03-03T00:00:00Z')), 'unknown');
console.log('PASS: UTC freshness boundary, missing, malformed, impossible and future dates.');

// Exercise the real rendering functions with only the DOM and translations stubbed.
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {
    textContent: '', hidden: true, classList: {toggle() {}},
    setAttribute() {}, removeAttribute() {},
  });
  return elements.get(id);
};
Object.assign(context, {
  document: {getElementById: element}, currentLang: 'en', currentMetaData: {},
  getCurrentText: () => ({signalUnavailable: 'Unavailable', signalLabels: {HOLD: 'HOLD', BUY: 'BUY', SELL: 'SELL'}}),
  setTextById: (id, text) => { element(id).textContent = text; },
  localizeSignalNote: note => note,
  formatLocalizedDate: date => date,
  LOCALE_BY_LANGUAGE: {en: 'en-US'},
});
for (const name of ['normalizeSignal', 'renderFreshness', 'setSignalUnavailable', 'renderRatio', 'renderSignal']) {
  vm.runInContext(html.match(new RegExp(`    function ${name}\\([^]*?\\n    \\}`))[0], context);
}
const valid = {signal: 'HOLD', confidence: 0.56, ratio: 20.192, last_updated: '2000-01-01', note: ''};
context.currentSignalData = valid;
context.renderSignal();
assert.match(element('data-freshness').textContent, /Data delayed/);
assert.equal(element('data-freshness').hidden, false);
assert.equal(element('signal-value').textContent, 'HOLD');
context.currentSignalData = {...valid, last_updated: '2026-02-30'};
context.renderSignal();
assert.match(element('data-freshness').textContent, /freshness unavailable/);
assert.equal(element('signal-updated').textContent, '--');
for (const data of [null, {...valid, confidence: NaN}, {...valid, confidence: Infinity}, {...valid, confidence: -0.1}, {...valid, confidence: 1.1}, {...valid, ratio: 0}, {...valid, ratio: Infinity}, {...valid, ratio: '20'}]) {
  context.currentSignalData = data;
  context.renderSignal();
  assert.equal(element('signal-value').textContent, 'Unavailable');
  assert.match(element('data-freshness').textContent, /Data unavailable/);
  assert.equal(element('ratio-value').textContent, 'Ratio temporarily unavailable');
}
console.log('PASS: delayed/unknown date rendering and invalid signal rejection.');
