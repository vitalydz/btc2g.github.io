import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../../../assets/js/traffic.js', import.meta.url), 'utf8');
function browser({ response, failure, hostname = 'btc2g.com', protocol = 'https:', webdriver = false } = {}) {
  const calls = [], events = [];
  const window = { dispatchEvent: event => events.push(event) };
  const context = vm.createContext({
    window, navigator: { webdriver },
    location: { hostname, protocol, pathname: '/save.html' },
    CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    async fetch(url, options) {
      calls.push({ url, options });
      if (failure) throw new Error('Offline');
      return response || { status: 202, json: async () => ({ ok: true }) };
    }
  });
  return { calls, events, run: () => vm.runInContext(source, context) };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('collector uses only first-party endpoint without cookies, referrer or query data', async () => {
  const ui = browser(); ui.run(); await settle();
  assert.equal(ui.calls.length, 1);
  const { url, options } = ui.calls[0];
  assert.equal(url, '/api/visit');
  assert.equal(options.mode, 'same-origin');
  assert.equal(options.credentials, 'omit');
  assert.equal(options.cache, 'no-store');
  assert.equal(options.referrerPolicy, 'no-referrer');
  assert.equal(options.keepalive, true);
  assert.equal(options.method, 'POST');
  assert.deepEqual(JSON.parse(options.body), { path: '/save.html' });
  assert.equal(ui.events[0].type, 'btc2g:traffic');
  assert.equal(ui.events[0].detail.status, 'recorded');
});

test('duplicate script execution sends only one visit per document', async () => {
  const ui = browser(); ui.run(); ui.run(); await settle();
  assert.equal(ui.calls.length, 1);
});

test('non-production, file and automated previews do not write to counters', async () => {
  for (const options of [{ hostname: 'localhost' }, { hostname: 'evil.example' },
    { protocol: 'file:' }, { protocol: 'http:' }, { webdriver: true }]) {
    const ui = browser(options); ui.run(); await settle();
    assert.equal(ui.calls.length, 0);
    assert.equal(ui.events.length, 0);
  }
});

test('network failures are unavailable and never automatically retried', async () => {
  const ui = browser({ failure: true }); ui.run(); await settle(); ui.run(); await settle();
  assert.equal(ui.calls.length, 1);
  assert.equal(ui.events[0].detail.status, 'unavailable');
});

test('only acknowledged 202 JSON responses count as recorded', async () => {
  for (const response of [
    { status: 403, json: async () => ({ error: 'Denied' }) },
    { status: 503, json: async () => ({ error: 'Unavailable' }) },
    { status: 200, json: async () => ({ ok: true }) },
    { status: 202, json: async () => ({ ok: false }) },
    { status: 202, json: async () => { throw new SyntaxError('HTML from Pages'); } }
  ]) {
    const ui = browser({ response }); ui.run(); await settle();
    assert.equal(ui.events.length, 1);
    assert.equal(ui.events[0].detail.status, 'unavailable');
    assert.equal(ui.calls.length, 1);
  }
});

test('production routes keep Pages and account mutations outside their scope', () => {
  const config = JSON.parse(readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8'));
  assert.deepEqual(config.routes, [
    { pattern: 'https://btc2g.com/api/visit*', zone_name: 'btc2g.com' },
    { pattern: 'https://btc2g.com/api/stats*', zone_name: 'btc2g.com' }
  ]);
  assert.equal(config.d1_databases[0].database_id, 'c84b5fe0-b86e-4a79-bfaf-097d424e0a87');
});
