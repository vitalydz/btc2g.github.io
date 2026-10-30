import assert from 'node:assert/strict';
import test from 'node:test';
import { marketHealth } from '../src/market-health.js';

const now = new Date('2026-10-30T12:00:00Z');
function source({ marketDate = '2026-10-29', updatedAt = '2026-10-30T08:00:00Z', conclusion = 'success', status = 'completed' } = {}) {
  return async url => ({ ok: true, json: async () => url.includes('api.github.com')
    ? { workflow_runs: [{ status, conclusion, created_at: '2026-10-30T09:00:00Z' }] }
    : { market_date: marketDate, last_updated_utc: updatedAt } });
}

test('admin diagnostics distinguish successful refresh, failed workflow, stale and unavailable sources', async () => {
  const healthy = await marketHealth(source(), now);
  assert.equal(healthy.status, 'ok');
  assert.equal(healthy.marketDate, '2026-10-29');
  assert.deepEqual(healthy.issues, []);
  const failed = await marketHealth(source({ conclusion: 'failure' }), now);
  assert.equal(failed.status, 'attention');
  assert.match(failed.issues.join(' '), /workflow ended with failure/);
  const stale = await marketHealth(source({ marketDate: '2026-10-20', updatedAt: '2026-10-28T08:00:00Z' }), now);
  assert.equal(stale.issues.length, 2);
  const unavailable = await marketHealth(async () => { throw new Error('source detail'); }, now);
  assert.equal(unavailable.issues.length, 2);
  assert.doesNotMatch(JSON.stringify(unavailable), /source detail/);
  assert.equal((await marketHealth(source({ status: 'in_progress', conclusion: null }), now)).status, 'ok');
});
