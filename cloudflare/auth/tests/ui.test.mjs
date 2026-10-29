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

test('homepage form receipt does not claim confirmation and Premium CTA uses existing form', async () => {
  const source = await readFile(new URL('../../../index.html', import.meta.url), 'utf8');
  assert.equal((source.match(/subscribeSuccess:/g) || []).length, 19);
  assert.doesNotMatch(source, /successEventName: 'email_subscribe'/);
  assert.match(source, /successEventName: 'subscription_request_received'/);
  assert.match(source, /window.location.href = 'premium.html#early-access'/);
});
