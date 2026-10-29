const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..');
const read = file => readFileSync(path.join(root, file), 'utf8');

test('Terms describe current accounts, passkeys, providers, forecasts, and support boundaries', () => {
  const terms = read('terms.html');
  assert.match(terms, /Version 2026-10-29\.3/);
  assert.match(terms, /No Financial Advice/);
  assert.match(terms, /Administrator passkeys/);
  assert.match(terms, /Joining an email list or verifying an account email does not purchase or activate Premium/);
  assert.match(terms, /BTC2G does not custody funds or crypto assets/);
  assert.match(terms, /no response time is guaranteed/i);
  assert.match(terms, /Never email a password, passkey/);
  assert.match(terms, /support\.html/);
  assert.match(terms, /Private Idea Submissions/);
  assert.match(terms, /not automatically published, entered into a contest, or opened for voting/);
});

test('Support routes requests without asking users to disclose secrets', () => {
  const support = read('support.html');
  for (const subject of ['BTC2G%20Account', 'BTC2G%20Privacy', 'BTC2G%20Security', 'BTC2G%20Feedback']) {
    assert.match(support, new RegExp(subject));
  }
  assert.match(support, /contact the provider named on the receipt first/);
  assert.match(support, /Do not send<\/strong> passwords, passkeys/);
  assert.match(support, /no response time is guaranteed/i);
  assert.doesNotMatch(support, /24\/7|within \d+ (hours|days)/i);
});

test('Support is discoverable and new account acceptance uses the current Terms version', () => {
  const homepage = read('index.html');
  const sitemap = read('sitemap.xml');
  const workerConfig = read('cloudflare/auth/wrangler.jsonc');
  assert.match(homepage, /id="support-link"/);
  assert.equal((homepage.match(/support:\s*'/g) || []).length, 19);
  assert.match(sitemap, /https:\/\/btc2g\.com\/support\.html/);
  assert.match(workerConfig, /"TERMS_VERSION": "2026-10-29\.3"/);
});
