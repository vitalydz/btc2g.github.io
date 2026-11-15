const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const readme = readFileSync(path.join(__dirname, '..', 'README.md'), 'utf8');

test('README leads with the current public product and honest limitations', () => {
  assert.match(readme, /https:\/\/btc2g\.com\/\?chart=purchasingPower#forecast/);
  assert.match(readme, /https:\/\/btc2g-accounts\.btc2g\.workers\.dev\//);
  assert.match(readme, /email-verified account.*submit ideas privately/);
  assert.match(readme, /supports 19 languages/);
  assert.match(readme, /not adjusted for consumer-price inflation/);
  assert.match(readme, /not financial advice/);
  assert.match(readme, /does not custody assets or process payments/);
  assert.match(readme, /node --test \$testFiles/);
  assert.match(readme, /requires a Node release that provides `node:sqlite`/);
  assert.match(readme, /Historical implementation notes/);
  assert.doesNotMatch(readme, /currently has no `test` or `build` scripts/);
});
