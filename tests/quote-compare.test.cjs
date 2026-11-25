const test = require('node:test');
const assert = require('node:assert/strict');
const { units, compare } = require('../assets/js/quote-compare.js');

test('compare final delivered BTC for equal total payment', () => {
  const result = compare('50', '0.00045', '0.00046');
  assert.equal(result.status, 'ready');
  assert.equal(result.winner, 'Topper');
  assert.equal(result.differenceSats, 1000);
  assert.ok(Math.abs(result.firstRate - 111111.11111111111) < 1e-8);
  assert.ok(Math.abs(result.moreBTCPercent - 2.2222222222222223) < 1e-12);
  assert.equal(compare('50', '0.00046', '0.00045').winner, 'MoonPay');
  assert.equal(compare('50', '0.00045', '0.00045').winner, null);
});

test('empty or invalid quotes never produce a provider recommendation', () => {
  for (const invalid of ['', ' ', '0', '-1', '1e-4', 'NaN', 'Infinity', '0x10', '1,000.00', '<script>', '0.000000001', '999999999999.99999999']) {
    assert.equal(compare('50', invalid, '0.001').status, 'incomplete', invalid);
    assert.equal(compare('50', '0.001', invalid).status, 'incomplete', invalid);
  }
  for (const invalid of ['', '0', '-1', '50.001', 'Infinity']) {
    assert.equal(compare(invalid, '0.001', '0.002').status, 'incomplete');
  }
});

test('integer satoshis and decimal commas preserve precision', () => {
  assert.equal(units('0,00000001', 8), 1);
  assert.equal(units('50,25', 2), 5025);
  assert.equal(compare('50.25', '0.00000001', '0.00000002').differenceSats, 1);
  assert.equal(compare('100', '0.0009', '0.00092').differenceSats, 2000);
});
