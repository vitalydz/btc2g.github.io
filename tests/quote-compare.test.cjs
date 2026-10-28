const test = require('node:test');
const assert = require('node:assert/strict');
const { units, compare, comparePurchases } = require('../assets/js/quote-compare.js');

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

test('two independent quotes versus one combined quote never assume fixed fees', () => {
  const q = (amount, btc, currency = 'USD') => ({ amount, btc, currency });
  const first = q('50', '0.00045');
  const second = q('50', '0.00044');
  const result = comparePurchases(first, second, q('100', '0.00091'));
  assert.equal(result.status, 'ready');
  assert.equal(result.separateSats, 89000);
  assert.equal(result.combinedSats, 91000);
  assert.equal(result.differenceSats, 2000);
  assert.equal(comparePurchases(first, second, q('100', '0.00088')).differenceSats, -1000);
  assert.equal(comparePurchases(first, second, q('100', '0.00089')).differenceSats, 0);
});

test('purchase scenarios require equal totals and matching currencies', () => {
  const q = (amount, btc, currency = 'USD') => ({ amount, btc, currency });
  assert.equal(comparePurchases(q('50', '0.001'), q('50', '0.001'), q('99.99', '0.002')).status, 'amount-mismatch');
  assert.equal(comparePurchases(q('50', '0.001'), q('50', '0.001', 'EUR'), q('100', '0.002')).status, 'currency-mismatch');
  assert.equal(comparePurchases(q('0.10', '0.00000001'), q('0.20', '0.00000001'), q('0.30', '0.00000003')).differenceSats, 1);
  assert.equal(comparePurchases(null, q('50', '0.001'), q('100', '0.002')).status, 'incomplete');
  assert.equal(comparePurchases(q('50', ''), q('50', '0.001'), q('100', '0.002')).status, 'incomplete');
  assert.equal(comparePurchases(q('50', '90000000'), q('50', '90000000'), q('100', '1')).status, 'incomplete');
});
