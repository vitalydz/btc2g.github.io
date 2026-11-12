const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const llms = readFileSync(path.join(__dirname, '..', 'llms.txt'), 'utf8');

test('llms.txt summarizes only public BTC2G tools and their limitations', () => {
  for (const url of [
    'https://btc2g.com/',
    'https://btc2g.com/scenario.html',
    'https://btc2g.com/save.html',
    'https://btc2g.com/research.html',
    'https://btc2g.com/about.html',
    'https://btc2g.com/support.html',
    'https://btc2g.com/terms.html',
    'https://btc2g.com/privacy.html',
    'https://btc2g.com/sitemap.xml'
  ]) assert.match(llms, new RegExp(url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(llms, /not investment, financial, trading, tax, legal, or accounting advice/);
  assert.match(llms, /scenarios, not guaranteed future results/);
  assert.match(llms, /Premium is in development/);
  assert.match(llms, /submissions and voting are not open/);
  assert.match(llms, /server-rendered pages in 19 languages/);
  assert.doesNotMatch(llms, /guaranteed (?:profit|return|ranking|indexing)/i);
});
