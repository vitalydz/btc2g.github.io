import assert from 'node:assert/strict';
import test from 'node:test';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import worker from '../src/worker.js';
import { jerusalemDay } from '../src/core.js';
import { confirmedSubscriptions, registrationRatio, trafficMetrics } from '../src/metrics.js';
import { formatCount, measurementNote, percentage } from '../public/metrics-view.js';

// Run the real Worker SQL against SQLite; all visits/accounts are local fixtures.
function database(t) {
  const db = new DatabaseSync(':memory:');
  t.after(() => db.close());
  for (const file of ['0001_initial.sql', '0002_traffic_metrics.sql']) {
    db.exec(readFileSync(new URL(`../migrations/${file}`, import.meta.url), 'utf8'));
  }
  return {
    db,
    env: {
      PUBLIC_SITE_ORIGIN: 'https://btc2g.com', IP_HASH_SECRET: 'local-test-only',
      DB: {
        prepare(sql) {
          const statement = db.prepare(sql);
          let args = [];
          return {
            bind(...values) { args = values; return this; },
            async first() { return statement.get(...args) || null; },
            async all() { return { results: statement.all(...args) }; },
            async run() { return statement.run(...args); }
          };
        },
        async batch(statements) {
          db.exec('BEGIN');
          try {
            const rows = await Promise.all(statements.map(s => s.run()));
            db.exec('COMMIT'); return rows;
          } catch (error) { db.exec('ROLLBACK'); throw error; }
        }
      }
    }
  };
}

function admin(db) {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(`INSERT INTO users (id, username, username_key, password_hash,
    password_salt, password_iterations, role, created_at, created_day,
    accepted_terms_at, terms_version) VALUES ('admin', 'Admin', 'admin', '', '', 1, 'admin', ?, ?, ?, 'test')`)
    .run(now, jerusalemDay(), now);
  const token = 'local-admin-session';
  db.prepare('INSERT INTO sessions VALUES (?, ?, ?, ?)')
    .run(createHash('sha256').update(token).digest('base64url'), 'admin', now, now + 3600);
  return { Cookie: `btc2g_session=${token}` };
}

test('unknown metrics and subscription source remain null, never zero', () => {
  assert.equal(trafficMetrics({ first_seen_at: null }).visitorsToday, null);
  assert.equal(trafficMetrics(null).status, 'unavailable');
  assert.equal(confirmedSubscriptions().total, null);
  assert.equal(registrationRatio(3, 0), null);
  assert.equal(registrationRatio(3, null), null);
  assert.equal(registrationRatio(3, 2), 1.5);
  for (const value of [null, undefined, NaN, -1, '0']) assert.equal(formatCount(value), 'Data unavailable');
  assert.equal(formatCount(0), '0');
  assert.equal(percentage(null), 'Data unavailable');
});

test('measurement notes require a timestamp and Jerusalem reporting day', () => {
  assert.throws(() => measurementNote({}), /measurement period/);
  assert.throws(() => measurementNote({ day: '2026-10-29', timeZone: 'Asia/Jerusalem', measuredAt: 'bad' }), /timestamp/);
  assert.match(measurementNote({ day: '2026-10-29', timeZone: 'Asia/Jerusalem', measuredAt: '2026-10-29T05:00:00Z', source: 'Cloudflare D1' }), /Cloudflare D1.*2026-10-29.*Asia\/Jerusalem/);
});

test('public stats exclude admin and distinguish no observations from measured zero', async t => {
  const { env, db } = database(t);
  admin(db);
  const result = await worker.fetch(new Request('https://metrics.test/api/stats'), env);
  assert.equal(result.status, 200);
  const data = await result.json();
  assert.equal(data.totalAccounts, 0);
  assert.equal(data.traffic.visitorsToday, null);
  assert.equal(data.confirmedSubscribers.total, null);
  assert.equal(data.confirmedSubscribers.status, 'unavailable');
  assert.equal(data.timeZone, 'Asia/Jerusalem');
  assert.ok(Number.isFinite(Date.parse(data.measuredAt)));
  assert.equal(result.headers.get('cache-control'), 'no-store');
});

test('repeated page visits deduplicate daily visitor estimate but count each view', async t => {
  const { env, db } = database(t);
  const headers = admin(db);
  for (const path of ['/', '/save.html', '/save.html']) {
    const result = await worker.fetch(new Request('https://metrics.test/api/visit', {
      method: 'POST', headers: { Origin: env.PUBLIC_SITE_ORIGIN,
        'Content-Type': 'application/json', 'CF-Connecting-IP': '192.0.2.1', 'User-Agent': 'local-test' },
      body: JSON.stringify({ path })
    }), env);
    assert.equal(result.status, 202);
  }
  const data = await (await worker.fetch(new Request('https://metrics.test/api/stats'), env)).json();
  assert.equal(data.traffic.visitorsToday, 1);
  assert.equal(data.traffic.viewsToday, 3);
  assert.equal(data.totalAccounts, 0);
  assert.equal(data.confirmedSubscribers.total, null);
  const report = await (await worker.fetch(new Request('https://metrics.test/api/admin/stats?days=7', { headers }), env)).json();
  assert.equal(report.summary.visitorsToday, 1);
  assert.equal(report.daily[0].views, 3);
  assert.equal(report.daily[1].views, null);
  assert.equal(report.daily[1].visitors, null);
  assert.equal(report.topPaths[0].path, '/save.html');
  assert.equal(report.topPaths[0].views, 2);
});

test('empty historical traffic and zero denominators are unavailable', async t => {
  const { env, db } = database(t);
  const headers = admin(db);
  const result = await worker.fetch(new Request('https://metrics.test/api/admin/stats', { headers }), env);
  const report = await result.json();
  assert.equal(result.status, 200);
  assert.equal(report.summary.conversionToday, null);
  assert.equal(report.summary.visitorsToday, null);
  assert.ok(report.daily.every(row => row.visitors === null && row.views === null));
});

test('traffic storage failure preserves known account counts without fabricated traffic', async t => {
  const { env, db } = database(t);
  db.exec('DROP TABLE daily_visitors');
  const data = await (await worker.fetch(new Request('https://metrics.test/api/stats'), env)).json();
  assert.equal(data.totalAccounts, 0);
  assert.equal(data.traffic.status, 'unavailable');
  assert.equal(data.traffic.viewsToday, null);
});

test('admin details stay protected and untrusted origins cannot record visits', async t => {
  const { env } = database(t);
  const privateResult = await worker.fetch(new Request('https://metrics.test/api/admin/stats'), env);
  assert.equal(privateResult.status, 403);
  const blocked = await worker.fetch(new Request('https://metrics.test/api/visit', {
    method: 'POST', headers: { Origin: 'https://other.test', 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: '/' })
  }), env);
  assert.equal(blocked.status, 403);
});
