const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const button = {disabled: false, setAttribute() {}, removeAttribute() {}};
const pending = [];
const requests = [];
const rendered = [];
let chartVersion;
const context = vm.createContext({
  Date, AbortSignal, console: {log() {}},
  SIGNAL_PATH: 'signal.json', META_PATH: 'meta.json',
  currentSignalData: {ratio: 10}, currentMetaData: {last_updated_utc: 'old'},
  document: {getElementById: () => button},
  fetch: (url, options) => {
    requests.push({url, options});
    return new Promise((resolve, reject) => pending.push({resolve, reject}));
  },
  setChartSrc: version => { chartVersion = version; },
  renderSignal: () => rendered.push(context.currentSignalData),
  renderMeta: () => rendered.push(context.currentMetaData),
});
for (const name of ['loadSignal', 'loadMeta', 'refreshForecast']) {
  vm.runInContext(html.match(new RegExp(`    async function ${name}\\([^]*?\\n    \\}`))[0], context);
}
const response = data => ({ok: true, json: async () => data});
(async () => {
  const refresh = context.refreshForecast();
  assert.equal(button.disabled, true);
  await context.refreshForecast();
  assert.equal(requests.length, 2, 'Repeated clicks must not overlap refreshes');
  assert.deepEqual(requests.map(r => r.url), [`signal.json?v=${chartVersion}`, `meta.json?v=${chartVersion}`]);
  for (const {options} of requests) {
    assert.equal(options.cache, 'no-store');
    assert.ok(options.signal instanceof AbortSignal);
  }
  pending[1].resolve(response({last_updated_utc: '2026-10-03'}));
  pending[0].resolve(response({ratio: 20.301}));
  await refresh;
  assert.equal(context.currentSignalData.ratio, 20.301);
  assert.equal(context.currentMetaData.last_updated_utc, '2026-10-03');
  assert.equal(button.disabled, false);
  assert.equal(rendered.at(-1), context.currentMetaData);

  const failed = context.refreshForecast();
  pending[2].reject(new Error('Network timeout'));
  pending[3].resolve({ok: false, status: 503});
  await failed;
  assert.equal(context.currentSignalData, null, 'Failed refresh must clear the previous signal');
  assert.equal(context.currentMetaData, null);
  assert.equal(button.disabled, false, 'A failed refresh must allow retry');

  const retry = context.refreshForecast();
  pending[4].resolve(response({ratio: 21}));
  pending[5].resolve(response({last_updated_utc: '2026-10-04'}));
  await retry;
  assert.equal(context.currentSignalData.ratio, 21);
  console.log('PASS: full forecast refresh, shared version, duplicate clicks, failures and retry.');
})().catch(error => { console.error(error); process.exitCode = 1; });
