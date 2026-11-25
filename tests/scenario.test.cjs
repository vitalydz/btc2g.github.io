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
  for (const id of ['btc-change','gold-change','target-ratio','future-ratio','hypothetical-label','scenario-market-date','ratio-formula','required-btc','scenario-result-account-link','copy-scenario','share-scenario','data-status']) assert.match(page, new RegExp(`id="${id}"`));
  assert.match(page, /Nothing entered here is stored/);
  assert.match(page, /id="scenario-account-link"[^>]+btc2g-accounts\.btc2g\.workers\.dev/);
  assert.match(page, /id="scenario-result-account-link"[^>]+btc2g-accounts\.btc2g\.workers\.dev/);
  assert.match(page, /Scenario, not forecast/);
  assert.match(page, /not predictions, financial advice, or guarantees/);
  assert.match(page, /yfinance BTC-USD and GC=F/);
  assert.match(page, /scenario\.css\?v=20261103-3/);
  assert.match(page, /scenario-tools\.js\?v=20261103-6/);
  assert.match(fs.readFileSync(path.join(root, 'assets/css/scenario.css'), 'utf8'), /border:2px dashed #8a6f24/);
  assert.match(fs.readFileSync(path.join(root, 'assets/css/scenario.css'), 'utf8'), /@media\(max-width:700px\).*\.scenario-header\{align-items:stretch;flex-direction:column\}.*html\[dir=rtl\] \.scenario-header nav\{[^}]*flex-direction:column-reverse/s);
  assert.match(page, /<script src="assets\/js\/traffic\.js\?v=\d{8}-\d+" defer><\/script>/);
  assert.match(homepage, /href="scenario\.html"/);
  assert.match(homepage, /BTC \/ Gold scenario/);
});

test('scenario and homepage entry provide complete text for all 19 languages', () => {
  const freeMarkers = {en:'free',ru:'бесплатный',he:'חינם',de:'Kostenloses',fr:'gratuit',es:'gratis',zh:'免费',it:'gratuito',pt:'grátis',ar:'مجاني',hi:'मुफ़्त',ja:'無料',ko:'무료',tr:'Ücretsiz',pl:'bezpłatne',nl:'Gratis',uk:'безкоштовний',id:'gratis',vi:'miễn phí'};
  assert.equal(Object.keys(translations).length, 19);
  for (const [lang, text] of Object.entries(translations)) {
    assert.deepEqual(Object.keys(text), KEYS, lang);
    for (const key of KEYS) assert.ok(text[key].trim().length > 0, `${lang}.${key}`);
    assert.ok(text.account.includes(freeMarkers[lang]), `${lang}.account is explicitly free`);
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

test('scenario account links preserve all 19 selected languages', () => {
  for (const lang of ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi']) {
    const file = lang === 'en' ? path.join(root, 'scenario.html') : path.join(root, lang, 'scenario.html');
    const html = fs.readFileSync(file, 'utf8');
    const accountLanguages = [...html.matchAll(/href="https:\/\/btc2g-accounts\.btc2g\.workers\.dev\/\?lang=([a-z]{2})"/g)].map(match => match[1]);
    assert.deepEqual(accountLanguages, [lang, lang], lang);
  }
});
