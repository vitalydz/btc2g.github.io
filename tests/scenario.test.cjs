const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const { KEYS, translations, validateMarketData, calculateScenario } = require('../assets/js/scenario-tools.js');

const root = path.resolve(__dirname, '..');
const page = fs.readFileSync(path.join(root, 'scenario.html'), 'utf8');
const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

test('scenario calculation uses published prices and preserves arithmetic identities', () => {
  const market = validateMarketData({btc_price_usd:80000,gold_price_usd:4000,ratio:20,last_updated:'2026-10-28'});
  assert.deepEqual(market, {btc:80000,gold:4000,ratio:20,date:'2026-10-28'});
  const result = calculateScenario(market, 30, 10, 30);
  assert.equal(result.futureBtc, 104000);
  assert.equal(result.futureGold, 4400);
  assert.ok(Math.abs(result.futureRatio - 23.6363636364) < 1e-9);
  assert.ok(Math.abs(result.ratioChangePct - 18.1818181818) < 1e-9);
  assert.equal(result.requiredBtc, 132000);
  assert.equal(calculateScenario(market, 0, 0, 20).futureRatio, 20);
});

test('scenario calculation rejects missing, inconsistent, or unsafe market data', () => {
  for (const data of [null,{}, {btc_price_usd:0,gold_price_usd:4000,ratio:20,last_updated:'2026-10-28'}, {btc_price_usd:80000,gold_price_usd:4000,ratio:30,last_updated:'2026-10-28'}, {btc_price_usd:80000,gold_price_usd:4000,ratio:20,last_updated:'bad'}, {btc_price_usd:80000,gold_price_usd:4000,ratio:20,last_updated:'2026-02-30'}]) {
    assert.equal(validateMarketData(data), null);
  }
  const market = {btc:80000,gold:4000,ratio:20};
  for (const values of [[-100,0,30],[0,-100,30],[0,0,0],[NaN,0,30],[0,Infinity,30]]) assert.equal(calculateScenario(market,...values), null);
});

test('scenario page exposes accessible controls, safety disclosure, and share actions', () => {
  for (const id of ['btc-change','gold-change','target-ratio','future-ratio','required-btc','copy-scenario','share-scenario','data-status']) assert.match(page, new RegExp(`id="${id}"`));
  assert.match(page, /Nothing entered here is stored/);
  assert.match(page, /id="scenario-account-link"[^>]+btc2g-accounts\.btc2g\.workers\.dev/);
  assert.match(page, /Scenario, not forecast/);
  assert.match(page, /not predictions, financial advice, or guarantees/);
  assert.match(page, /yfinance BTC-USD and GC=F/);
  assert.match(page, /<script src="assets\/js\/traffic\.js" defer><\/script>/);
  assert.match(homepage, /href="scenario\.html"/);
  assert.match(homepage, /BTC \/ Gold scenario/);
});

test('scenario and homepage entry provide complete text for all 19 languages', () => {
  assert.equal(Object.keys(translations).length, 19);
  for (const [lang, text] of Object.entries(translations)) {
    assert.deepEqual(Object.keys(text), KEYS, lang);
    for (const key of KEYS) assert.ok(text[key].trim().length > 0, `${lang}.${key}`);
  }
  const nodes = new Map();
  const context = { document: { getElementById(id) { if (!nodes.has(id)) nodes.set(id,{textContent:'',href:''}); return nodes.get(id); } } };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root,'assets/js/scenario-entry.js'),'utf8'),context);
  assert.equal(Object.keys(context.BTC2GScenarioEntry.values).length,19);
  for (const [lang,row] of Object.entries(context.BTC2GScenarioEntry.values)) {
    assert.equal(row.length,4,lang);
    assert.ok(row.every(value=>typeof value==='string'&&value.length>0),lang);
  }
  context.BTC2GScenarioEntry.setLanguage('ru');
  assert.equal(nodes.get('scenario-entry-link').href,'scenario.html?lang=ru');
});
