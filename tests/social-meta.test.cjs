const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const homepage = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

test('homepage social preview names BTC2G and describes its image', () => {
  assert.match(homepage, /property="og:site_name" content="BTC2G"/);
  assert.match(homepage, /property="og:image:type" content="image\/png"/);
  assert.match(homepage, /property="og:image:width" content="2800"/);
  assert.match(homepage, /property="og:image:height" content="1400"/);
  assert.match(homepage, /property="og:image:alt" content="[^"]*Bitcoin-to-gold ratio[^"]*"/);
  assert.match(homepage, /name="twitter:image:alt" content="[^"]*Bitcoin-to-gold ratio[^"]*"/);
});

test('homepage social preview declares all supported locale variants', () => {
  assert.match(homepage, /property="og:locale" content="en_US"/);
  const alternateLocales = [...homepage.matchAll(/property="og:locale:alternate" content="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(alternateLocales, ['ru_RU','he_IL','de_DE','fr_FR','es_ES','zh_CN','it_IT','pt_PT','ar_SA','hi_IN','ja_JP','ko_KR','tr_TR','pl_PL','nl_NL','uk_UA','id_ID','vi_VN']);
});

test('homepage publishes valid multilingual WebSite structured data', () => {
  const match = homepage.match(/<script type="application\/ld\+json">\s*([\s\S]*?)\s*<\/script>/);
  assert.ok(match);
  const data = JSON.parse(match[1]);
  assert.equal(data['@context'], 'https://schema.org');
  assert.equal(data['@type'], 'WebSite');
  assert.equal(data.name, 'BTC2G');
  assert.equal(data.url, 'https://btc2g.com/');
  assert.equal(data.inLanguage.length, 19);
});

test('homepage allows large search-result image previews', () => {
  assert.match(homepage, /name="robots" content="index,follow,max-image-preview:large"/);
});
