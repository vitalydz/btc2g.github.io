const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const researchPage = fs.readFileSync(path.join(root, 'research.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'assets/js/research-support.js'), 'utf8');
const address = 'TKWzm9cXYEM5hqsDboj9Uh2NDLZUgBzaN4';

test('research page links all three official records and homepage stays market-focused', () => {
  for (const record of ['17603069', '17748957', '18002304']) {
    assert.match(researchPage, new RegExp(`https://zenodo\\.org/records/${record}`));
  }
  assert.match(researchPage, /timet-document-2-figure-no-logo\.png/);
  assert.equal((researchPage.match(/rel="noopener noreferrer"/g) || []).length >= 5, true);
  assert.doesNotMatch(homepage, /id="research-support-title"/);
  assert.match(homepage, /href="research\.html"/);
});

test('support block states the exact asset, network, address, and safety boundaries', () => {
  assert.match(researchPage, /Tether \(USDT\) · TRON network \(TRC-20\)/);
  assert.match(researchPage, new RegExp(address));
  assert.match(researchPage, new RegExp(`https://tronscan\\.org/#/address/${address}`));
  assert.match(researchPage, /Crypto transfers are irreversible/);
  assert.match(researchPage, /not an investment or purchase/);
  assert.match(researchPage, /no ownership, financial return, tax benefit, delivery date, or guarantee/);
  assert.doesNotMatch(researchPage, /connect wallet|guaranteed return|tax-deductible/i);
});

test('copy control uses the exact address and has all 19 translations', async () => {
  const nodes = new Map();
  const document = {
    documentElement: {},
    getElementById(id) {
      if (id === 'research-language') return null;
      if (!nodes.has(id)) nodes.set(id, { textContent: '', addEventListener() {} });
      return nodes.get(id);
    },
  };
  let copied = '';
  const context = {
    document,
    navigator: { clipboard: { async writeText(value) { copied = value; } } },
    URLSearchParams,
    window: { location: {search: ''} },
  };
  vm.createContext(context);
  vm.runInContext(script, context);
  assert.equal(context.window.BTC2GResearch.ADDRESS, address);
  assert.equal(Object.keys(context.window.BTC2GResearch.translations).length, 19);
  for (const text of Object.values(context.window.BTC2GResearch.translations)) {
    for (const key of ['title', 'intro', 'support', 'copy', 'copied', 'failed', 'warning', 'disclaimer']) {
      assert.equal(typeof text[key], 'string');
      assert.ok(text[key].length > 0);
    }
  }
  await context.window.BTC2GResearch.copyAddress();
  assert.equal(copied, address);
  assert.equal(nodes.get('research-copy-status').textContent, 'Address copied');
});
