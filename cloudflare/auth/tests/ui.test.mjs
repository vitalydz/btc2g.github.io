import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const publicRoot = new URL('../public/', import.meta.url);

test('authenticated state hides the registration form despite form display rules', async () => {
  const [html, script, css] = await Promise.all([
    readFile(new URL('index.html', publicRoot), 'utf8'),
    readFile(new URL('account.js', publicRoot), 'utf8'),
    readFile(new URL('account.css', publicRoot), 'utf8')
  ]);

  assert.match(html, /<div id="signed-in" hidden>/);
  assert.match(script, /form\.hidden = true/);
  assert.match(script, /signedIn\.hidden = false/);
  assert.match(css, /\[hidden\]\s*\{\s*display\s*:\s*none\s*!important;\s*\}/);
});
