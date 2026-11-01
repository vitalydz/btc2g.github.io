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
