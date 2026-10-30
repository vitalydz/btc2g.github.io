const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const quotes = require('../assets/js/quote-compare.js');
const read = path => fs.readFileSync(path, 'utf8');

function harness({query = '', saved = null, browser = 'en-US', denied = false, hrefs = []} = {}) {
  const elements = new Map(), storage = new Map(saved ? [['lang', saved]] : []);
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      id, value: '', textContent: '', disabled: false, checked: false, dataset: {}, children: [], events: {}, attributes: {},
      addEventListener(event, handler) { this.events[event] = handler; },
      setAttribute(key, value) { this.attributes[key] = value; },
      getAttribute(key) { return this.attributes[key]; },
      replaceChildren(...children) { this.children = children; },
      classList: {toggle() {}},
    });
    return elements.get(id);
  };
  const anchors = hrefs.map((href, index) => {const el = element('anchor' + index); el.attributes.href = href; return el;});
  const selected = new Map();
  const queryElement = selector => { if (!selected.has(selector)) selected.set(selector, element('query:' + selector)); return selected.get(selector); };
  const labels = [...read('save.html').matchAll(/data-text="([^"]+)"/g)].map((m, i) => {
    const el = element('label' + i); el.dataset.text = m[1]; return el;
  });
  const statuses = ['title','pending','disclaimer','home'].map(key => {const el = element('status-' + key);el.dataset.status = key;return el;});
  const document = {
    documentElement: {}, getElementById: element, querySelector: queryElement, title: '', createElement: tag => ({tag, value: '', textContent: ''}),
    querySelectorAll(selector) {
      if (selector === 'a[href]') return anchors;
      if (selector === '[data-text]') return labels;
      if (selector === '[data-status]') return statuses;
      return [];
    }
  };
  const location = {href: 'https://btc2g.com/save.html' + query, origin: 'https://btc2g.com', search: query, hash: ''};
  const context = vm.createContext({window: {BTC2GQuotes: quotes}, document, location, URL, URLSearchParams, navigator: {language: browser}, Intl,
    localStorage: {
      getItem(key) {if (denied) throw Error('SecurityError');return storage.get(key) ?? null;},
      setItem(key,value) {if (denied) throw Error('SecurityError');storage.set(key,value);}
    },
    history: {replaceState(_, __, url) {location.href = String(url);location.search = new URL(url).search;}},
    setTimeout, clearTimeout, AbortController
  });
  const run = path => vm.runInContext(read(path), context, {filename: path});
  run('assets/js/languages.js');
  return {context, run, element, elements, selected, query: queryElement, labels, statuses, anchors, document, storage, api: context.window.BTC2GLanguages};
}

test('language precedence, regional codes, prototype names and denied storage', () => {
  assert.equal(harness({query:'?lang=AR-sa',saved:'ru',browser:'de-DE',denied:true}).api.requested(), 'ar');
  assert.equal(harness({query:'?lang=unsupported',saved:'he'}).api.requested(), 'he');
  assert.equal(harness({browser:'zh-Hant'}).api.requested(), 'zh');
  assert.equal(harness({browser:'fr-CA',denied:true}).api.requested(), 'fr');
  assert.equal(harness({query:'?lang=__proto__',browser:'xx'}).api.requested(), 'en');
  const h = harness({denied:true});
  for (const lang of Object.keys(h.api.names)) {
    assert.doesNotThrow(() => h.api.apply(lang));
    assert.equal(h.document.documentElement.dir, ['he','ar'].includes(lang) ? 'rtl' : 'ltr');
    assert.ok(h.api.menu(lang, false));assert.ok(h.api.menu(lang, true));
  }
});

test('navigation keeps language and hashes, leaving providers, mail and APIs untouched', () => {
  const hrefs = ['index.html#calculator','save.html?lang=en','privacy.html','#forecast',
    'https://www.topperpay.com/','https://www.moonpay.com/buy/btc','mailto:support@btc2g.com',
    'https://btc2g-accounts.btc2g.workers.dev/','/api/stats'];
  const h = harness({hrefs});h.api.links('ar');
  assert.equal(h.anchors[0].attributes.href, '/index.html?lang=ar#calculator');
  assert.equal(h.anchors[1].attributes.href, '/save.html?lang=ar');
  assert.equal(h.anchors[2].attributes.href, '/privacy.html?lang=ar');
  for (const index of [3,4,5,6,8]) assert.equal(h.anchors[index].attributes.href, hrefs[index]);
  assert.equal(h.anchors[7].attributes.href, 'https://btc2g-accounts.btc2g.workers.dev/?lang=ar');
});

test('SAVE renders complete real dictionaries and valid/invalid results in all 19 languages', () => {
  const h = harness({query:'?lang=ar',denied:true});
  h.run('assets/js/ui-text.js');h.run('assets/js/save-text.js');
  h.element('save-amount').value = '50';h.element('save-currency').value = 'USD';
  h.run('assets/js/save.js');
  const select = h.element('save-language');assert.equal([...read('save.html').matchAll(/option value="([a-z]{2})"/g)].length, 19);
  for (const lang of Object.keys(h.api.names)) {
    select.events.change({target:{value:lang}});
    assert.equal(h.document.documentElement.lang, lang);
    for (const label of h.labels) assert.ok(typeof label.textContent === 'string' && label.textContent.trim(), lang + ':' + label.dataset.text);
    assert.ok(!h.labels.some(label => label.textContent.includes('undefined')));
    h.element('save-moonpay').value = '0.00045';h.element('save-topper').value = '0.00046';h.element('save-confirm').checked = true;
    h.element('save-compare').events.click();
    const result = h.element('save-result').textContent;
    assert.match(result, /Topper/, lang);
    assert.doesNotMatch(result, /undefined|\{provider\}|\{sats\}|\{percent\}/, lang);
    h.element('save-topper').value = '-1';h.element('save-compare').events.click();
    assert.doesNotMatch(h.element('save-result').textContent, /Topper/);
    assert.ok(h.element('save-result').textContent.trim());
    h.element('save-currency').events.change();
    assert.equal(h.element('save-moonpay').value, '');assert.equal(h.element('save-confirm').checked, false);
  }
});

test('mobile numeral normalization preserves integer-satoshi math and rejects grouping', () => {
  const {api} = harness();
  for (const value of ['٠٫٠٠٠٤٦','۰٫۰۰۰۴۶','०.०००४६','０.０００４６']) assert.equal(api.decimal(value), '0.00046');
  const result = quotes.compare(api.decimal('٥٠'), api.decimal('٠٫٠٠٠٤٥'), api.decimal('٠٫٠٠٠٤٦'));
  assert.equal(result.status, 'ready');assert.equal(result.differenceSats, 1000);
  assert.equal(quotes.compare(api.decimal('٥٠'), api.decimal('١٬٠٠٠٫٠٠'), '0.01').status, 'incomplete');
  assert.equal(quotes.compare('50', api.decimal('٠٫٠٠٠٠٠٠٠٠١'), '0.01').status, 'incomplete');
});

test('buy page supports all 19 locales without overwriting provider hashes', () => {
  const h = harness({denied:true,query:'?lang=ar'});h.run('assets/js/buy-locales.js');
  const html = read('buy.html'), code = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
  vm.runInContext(code, h.context);
  for (const lang of Object.keys(h.api.names)) {
    h.context.window.setLang(lang);
    assert.equal(h.document.documentElement.lang, lang);
    for (const id of ['title','topper-title','buy-btn','topper-btn','provider-notice','back-home','footer-text','save-link']) {
      assert.ok(typeof h.element(id).textContent === 'string' && h.element(id).textContent.trim(), lang + ':' + id);
    }
    assert.equal(h.element('save-link').href, 'save.html?lang=' + lang);
    assert.doesNotMatch(h.element('buy-btn').textContent, /undefined/);
  }
});

test('Premium full renderer receives all 19 localized labels and storage denial is safe', () => {
  for (const lang of Object.keys(harness().api.names)) {
    const h = harness({query:'?lang=' + lang,denied:true});
    h.document.createTextNode = value => ({textContent:value});
    h.context.location.assign = value => { h.context.nextLocation = value; };
    h.run('assets/js/ui-text.js');h.run('assets/js/subscription-request.js');h.run('assets/js/premium-language.js');h.run('assets/js/premium-v1.js');
    assert.equal(h.document.documentElement.lang,lang);
    assert.equal(h.element('premium-language').attributes['aria-label'],h.context.window.BTC2GLanguageText.text(lang).language);
    assert.equal(h.query('.hero h1').textContent,h.context.window.BTC2GPremiumText.heroTitle);
    assert.equal(h.element('early-access').dir,['he','ar'].includes(lang)?'rtl':'ltr');
  }
});

test('payment status is localized and never treats a static URL as activation', () => {
  const h = harness();h.run('assets/js/payment-status-locales.js');
  for (const lang of Object.keys(h.api.names)) {
    h.element('status-language').events.change({target:{value:lang}});
    assert.equal(h.document.documentElement.lang,lang);
    for (const el of h.statuses) assert.ok(el.textContent?.trim(),lang);
  }
  h.element('status-language').events.change({target:{value:'en'}});
  assert.match(h.statuses[1].textContent,/does not confirm payment/);
  assert.doesNotMatch(read('premium-success.html'), /<h1[^>]*>Premium subscription activated/);
});

test('HTML advertises the same supported options and all scripts parse', () => {
  const expected = Object.keys(harness().api.names).sort();
  for (const file of ['index.html','buy.html','save.html']) {
    const html = read(file), options = [...html.matchAll(/<option value="([a-z]{2})">/g)].map(m=>m[1]);
    assert.deepEqual(options.sort(),expected,file);
    for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) if(match[1].trim()) new vm.Script(match[1]);
  }
});

test('all financial-data warnings and legacy modal labels have 19 real translations', () => {
  const h = harness();h.run('assets/js/ui-text.js');h.run('assets/js/legacy-premium-text.js');
  const html = read('index.html');
  const start = html.indexOf('    const texts = {'), end = html.indexOf('    function setLang(lang)');
  h.context.STRIPE_CHECKOUT_URL = 'https://YOUR-VERCEL-PROJECT.vercel.app/api/stripe/create-checkout-session';
  vm.runInContext(html.slice(start,end),h.context);
  vm.runInContext(html.match(/    function getUIWords\([^]*?\n    \}/)[0],h.context);
  vm.runInContext(html.match(/    function renderFreshness\([^]*?\n    \}/)[0],h.context);
  for(const lang of Object.keys(h.api.names)) {
    h.context.currentLang = lang;
    const labels = h.context.getTextForLanguage(lang), ui = h.context.window.BTC2GLanguageText.text(lang);
    for(const key of ['premiumModalTitle','premiumModalIntro','premiumBreakdown','premiumZones','premiumEarly','premiumWeekly','premiumEmailPlaceholder','premiumSubmit','closeModal']) assert.ok(labels[key]?.trim(),lang+':'+key);
    assert.equal(labels.premiumPrice,ui.soon);
    for(const status of ['delayed','unknown','unavailable']) {
      h.context.renderFreshness(status);assert.equal(h.element('data-freshness').textContent,ui[status]);
      assert.notEqual(ui[status],ui.soon);
    }
    if(lang!=='en') assert.notEqual(ui.delayed,h.context.window.BTC2GLanguageText.text('en').delayed);
  }
});

test('Research normalizes unsafe/region languages and translates interface without changing provider address', () => {
  for(const lang of ['he-IL','ar','__proto__','constructor']) {
    const h = harness({query:'?lang='+lang,denied:true});h.context.window.location = h.context.location;
    h.run('assets/js/ui-text.js');h.run('assets/js/research-support.js');
    const expected = h.api.normalize(lang)||'en';
    assert.equal(h.document.documentElement.lang,expected);
    assert.equal(h.element('research-title').textContent,h.context.window.BTC2GResearch.translations[expected].title);
    assert.equal(h.context.window.BTC2GResearch.ADDRESS,'TKWzm9cXYEM5hqsDboj9Uh2NDLZUgBzaN4');
  }
});

test('graph loaded/error states localize without refetching when language changes', async () => {
  const h = harness();h.context.window.document = h.document;h.context.window.addEventListener = () => {};
  const painted = [];
  const ctx = new Proxy({createLinearGradient(){return {addColorStop(){}}},fillText(text){painted.push(text);}}, {get(target,key){return key in target ? target[key] : () => {};}});
  Object.assign(h.element('market-history-chart'),{getBoundingClientRect:()=>({width:390,height:300}),getContext:()=>ctx});
  let calls=0, failed=false;
  h.context.fetch=async path=>{calls++;if(failed)throw Error('Network error');return {ok:true,json:async()=>path.startsWith('assets/btc_signal')
    ? {btc_price_usd:80000,gold_price_usd:4000,last_updated:'2026-10-29'}
    : {source:'yfinance',tickers:{Bitcoin:'BTC-USD',Gold:'GC=F'},updated_for_market_date:'2026-10-29',series:[{date:'2026-10-28',ratio:20,btc_usd:80000,gold_usd:4000,model_score:50},{date:'2026-10-29',ratio:21,btc_usd:84000,gold_usd:4000,model_score:55}]}};};
  h.run('assets/js/ui-text.js');h.run('assets/js/home-v2-text.js');h.run('assets/js/market-chart.js');
  await new Promise(resolve=>setImmediate(resolve));
  const graph = h.context.window.BTC2GMarketChart;
  const initialCalls=calls;
  for(const lang of Object.keys(h.api.names)) {
    graph.setLanguage(lang);h.context.window.BTC2GHomeV2Text.setLanguage(lang);
    const ui=h.context.window.BTC2GLanguageText.text(lang);
    assert.equal(h.query('#chart-method').textContent,ui.chartMethod);
    assert.ok(h.element('market-history-status').textContent.includes(ui.observations));
    assert.equal(h.element('market-history-chart').attributes['aria-label'],h.context.window.BTC2GHomeV2Text.chartLabels(lang)[0]);
    assert.equal(calls,initialCalls);
  }
  failed=true;await graph.load();const failedCalls=calls;
  for(const lang of Object.keys(h.api.names)) {
    graph.setLanguage(lang);
    assert.equal(h.element('market-history-status').textContent,h.context.window.BTC2GLanguageText.text(lang).historyError);
    assert.equal(calls,failedCalls);
  }
  assert.ok(painted.length>0);
});

test('scenario selection updates the URL, localized navigation and account link in every language', async () => {
  const h=harness({query:'?lang=ru',denied:true});
  h.context.location.href='https://btc2g.com/scenario.html?lang=ru';
  const back=h.element('back-link'),account=h.element('scenario-account-link');
  back.attributes.href='index.html';account.attributes.href='https://btc2g-accounts.btc2g.workers.dev/';h.anchors.push(back,account);
  let initialize;h.document.addEventListener=(_,fn)=>{initialize=fn;};h.context.AbortSignal=AbortSignal;
  Object.assign(h.context.window,{document:h.document,location:h.context.location,navigator:h.context.navigator,
    fetch:async()=>({ok:true,json:async()=>({btc_price_usd:80000,gold_price_usd:4000,ratio:20,last_updated:'2026-10-29'})})});
  h.run('assets/js/ui-text.js');h.run('assets/js/scenario-tools.js');initialize();
  await new Promise(resolve=>setImmediate(resolve));
  for(const lang of Object.keys(h.api.names)) {
    h.element('scenario-language').events.change({target:{value:lang}});
    assert.equal(new URL(h.context.location.href).searchParams.get('lang'),lang);
    assert.equal(h.api.requested(),lang);
    assert.equal(h.element('scenario-language').attributes['aria-label'],h.context.window.BTC2GLanguageText.text(lang).language);
    assert.equal(new URL(account.attributes.href).searchParams.get('lang'),lang);
    assert.ok(back.attributes.href.includes('lang='+lang));
  }
});
