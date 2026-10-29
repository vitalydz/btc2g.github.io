import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../public/', import.meta.url);
test('preserves hidden authenticated state and offers useful public tools', async () => {
  const [html, script, css] = await Promise.all(['index.html','account.js','account.css'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(script, /form\.hidden = true/);
  assert.match(css, /\[hidden\]\s*\{\s*display\s*:\s*none\s*!important;\s*\}/);
  assert.match(html, /https:\/\/btc2g.com\/save.html/);
  assert.match(html, /https:\/\/btc2g.com\/index.html#calculator/);
});

test('registration fields explain their accepted format before submission', async () => {
  const [html, script] = await Promise.all(['index.html', 'account.js'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(html, /aria-describedby="username-hint"/);
  assert.match(html, /id="username-hint">3–32 characters/);
  assert.match(html, /aria-describedby="password-hint"/);
  assert.match(html, /id="password-hint">Use at least 12 characters/);
  assert.match(script, /Enter the password for this account/);
});

test('homepage form receipt does not claim confirmation and Premium CTA uses existing form', async () => {
  const source = await readFile(new URL('../../../index.html', import.meta.url), 'utf8');
  assert.equal((source.match(/subscribeSuccess:/g) || []).length, 19);
  assert.doesNotMatch(source, /successEventName: 'email_subscribe'/);
  assert.match(source, /successEventName: 'subscription_request_received'/);
  assert.match(source, /window.location.href = 'premium.html\?lang='/);
});

test('Product Hunt attribution is forwarded only through the account registration flow', async () => {
  const [traffic, account] = await Promise.all([
    readFile(new URL('../../../assets/js/traffic.js', import.meta.url), 'utf8'),
    readFile(new URL('account.js', root), 'utf8')
  ]);
  assert.match(traffic, /document\.referrer/);
  assert.match(traffic, /sessionStorage\.setItem\(storageKey, data\.attribution\)/);
  assert.match(traffic, /url\.searchParams\.set\('attribution'/);
  assert.match(account, /attribution: mode === 'register'/);
});
