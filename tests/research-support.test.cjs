const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'assets/js/research-support.js'), 'utf8');
const address = 'TKWzm9cXYEM5hqsDboj9Uh2NDLZUgBzaN4';

test('homepage links all three official research records safely', () => {
  for (const record of ['17603069', '17748957', '18002304']) {
    assert.match(homepage, new RegExp(`https://zenodo\\.org/records/${record}`));
  }
  assert.match(homepage, /timet-document-2-figure-no-logo\.png/);
  assert.equal((homepage.match(/rel="noopener noreferrer"/g) || []).length >= 5, true);
});

test('support block states the exact asset, network, address, and safety boundaries', () => {
  assert.match(homepage, /Tether \(USDT\) · TRON network \(TRC-20\)/);
  assert.match(homepage, new RegExp(address));
  assert.match(homepage, new RegExp(`https://tronscan\\.org/#/address/${address}`));
  assert.match(homepage, /Crypto transfers are irreversible/);
  assert.match(homepage, /not an investment or purchase/);
  assert.match(homepage, /no ownership, financial return, tax benefit, delivery date, or guarantee/);
  assert.doesNotMatch(homepage, /connect wallet|guaranteed return|tax-deductible/i);
});

test('copy control uses the exact address and has all 19 translations', async () => {
  const nodes = new Map();
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, { textContent: '', addEventListener() {} });
      return nodes.get(id);
    },
  };
  let copied = '';
  const context = {
    document,
    navigator: { clipboard: { async writeText(value) { copied = value; } } },
    window: {},
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
