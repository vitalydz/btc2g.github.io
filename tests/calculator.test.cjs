const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { dataStatus, convert } = require('../assets/js/ratio-tools.js');
const now = new Date('2026-10-27T22:00:00Z');
const signal = { ratio: 20, last_updated: '2026-10-26' };
const meta = { status: 'ok', source: 'yfinance', tickers: { Bitcoin: 'BTC-USD', Gold: 'GC=F' } };
const calculate = (value, direction = 'btc', s = signal, m = meta) => convert(value, direction, s, m, now);
assert.deepEqual(calculate('2'), { status: 'ok', result: 40, unit: 'oz t' });
assert.deepEqual(calculate('40', 'gold'), { status: 'ok', result: 2, unit: 'BTC' });
assert.equal(calculate('0').result, 0);
assert.equal(calculate('0,5').result, 10);
assert.equal(calculate('.5').result, 10);
for (const value of ['', ' ', '-1', 'NaN', 'Infinity', '1e309', '1,000.00', '<script>', '0x20', '9'.repeat(41)]) {
  assert.equal(calculate(value).status, 'invalid', value);
}
for (const ratio of [0, -1, NaN, Infinity, '20', null]) {
  assert.equal(calculate('1', 'btc', { ...signal, ratio }).status, 'unavailable');
}
assert.equal(calculate('1', 'wrong').status, 'invalid');
assert.equal(calculate('99', 'btc', { ...signal, ratio: Number.MAX_VALUE }).status, 'invalid');
assert.equal(calculate('0.000000000000000000000000000001', 'gold', { ...signal, ratio: Number.MAX_VALUE }).status, 'invalid');
for (const date of ['2026-10-28', '2026-02-30', '', null, '2026-10-26T12:00:00Z']) {
  assert.equal(dataStatus({ ...signal, last_updated: date }, meta, now), 'unavailable');
}
assert.equal(dataStatus({ ...signal, last_updated: '2026-10-24' }, meta, now), 'fresh');
assert.equal(dataStatus({ ...signal, last_updated: '2026-10-23' }, meta, now), 'stale');
assert.equal(dataStatus({ ...signal, last_updated: '2026-10-01' }, { ...meta, last_updated_utc: now.toISOString() }, now), 'stale', 'Regenerating JSON must not make an old market close fresh');
assert.equal(dataStatus({ ratio: 20, last_updated: '2024-02-29' }, meta, new Date('2024-03-03T23:59:59Z')), 'fresh');
for (const metadata of [null, {}, { ...meta, status: 'error' }, { ...meta, tickers: { Bitcoin: 'BTC-USD', Gold: 'XAUUSD' } }]) {
  assert.equal(calculate('1', 'btc', signal, metadata).status, 'unavailable');
}

// Exercise rendering and data refresh through the actual browser adapter.
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, { textContent: '', value: '', disabled: true, setAttribute() {}, addEventListener() {} });
  return elements.get(id);
};
element('calculator-amount').value = '1';
element('calculator-direction').value = 'btc';
class Clock extends Date { constructor(...args) { super(...(args.length ? args : ['2026-10-27T22:00:00Z'])); } }
const window = { document: { getElementById: element, addEventListener() {} }, setInterval() {} };
const context = vm.createContext({ window, Date: Clock, Intl });
const root = path.join(__dirname, '..');
vm.runInContext(fs.readFileSync(path.join(root, 'assets/js/audience-text.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(root, 'assets/js/ratio-tools.js'), 'utf8'), context);
window.BTC2GTools.update(signal, meta, 'en');
assert.equal(element('calculator-result').textContent, '≈ 20 oz t');
assert.equal(element('calculator-amount').disabled, false);
window.BTC2GTools.update({ ...signal, last_updated: '2026-10-01' }, meta, 'en');
assert.equal(element('calculator-amount').disabled, true);
assert.match(element('calculator-result').textContent, /Conversion unavailable/);
assert.doesNotMatch(element('calculator-result').textContent, /≈/);
assert.match(element('calculator-source').textContent, /2026-10-01/);
assert.doesNotMatch(element('calculator-source').textContent, /1 BTC ≈/);
window.BTC2GTools.update(null, null, 'ru');
assert.match(element('calculator-result').textContent, /недоступен/);
window.BTC2GTools.update(signal, meta, 'he');
assert.equal(element('calculator-amount').disabled, false);
assert.match(element('creator-bio').textContent, /vitlyd/);
element('calculator-amount').value = '.000000001';
window.BTC2GTools.update(signal, meta, 'en');
assert.match(element('calculator-result').textContent, /< 0.000001 oz t/);
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const languages = [...html.matchAll(/<option value="([a-z]{2})">/g)].map(match => match[1]);
assert.equal(languages.length, 19);
for (const language of languages) {
  assert.ok(window.BTC2GAudienceText[language], language);
  for (const value of Object.values(window.BTC2GAudienceText[language])) assert.ok(value.length > 0, language);
}
console.log('PASS: bidirectional units, invalid inputs, UTC freshness, source validation, rendering, recovery and 19 languages.');
