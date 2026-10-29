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
  for (const file of ['0001_initial.sql', '0002_traffic_metrics.sql', '0003_acquisition_attribution.sql',
    '0004_email_verification.sql', '0005_login_rate_limits.sql', '0006_admin_passkeys.sql']) {
    db.exec(readFileSync(new URL(`../migrations/${file}`, import.meta.url), 'utf8'));
  }
  return {
    db,
    env: {
      APP_ORIGIN: 'https://metrics.test', PUBLIC_SITE_ORIGIN: 'https://btc2g.com', IP_HASH_SECRET: 'local-test-only',
      WEBAUTHN_RP_ID: 'metrics.test', WEBAUTHN_RP_NAME: 'BTC2G Test',
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
  db.prepare('INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)')
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

test('attributes a same-day Product Hunt registration with a signed browser-bound token', async t => {
  const { env, db } = database(t);
  let sent;
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    sent = JSON.parse(options.body);
    return new Response(JSON.stringify({ id: 'local-email' }), { status: 200 });
  });
  const configured = {
    ...env,
    TERMS_VERSION: 'test',
    RESEND_API_KEY: 'local-test-key',
    EMAIL_FROM: 'BTC2G <accounts@btc2g.com>'
  };
  const commonHeaders = {
    'CF-Connecting-IP': '192.0.2.44', 'User-Agent': 'producthunt-local-test'
  };
  const visit = await worker.fetch(new Request('https://metrics.test/api/visit', {
    method: 'POST', headers: { ...commonHeaders, Origin: env.PUBLIC_SITE_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: '/', referrer: 'https://www.producthunt.com/posts/btc2g' })
  }), configured);
  const { attribution } = await visit.json();
  assert.equal(visit.status, 202);
  assert.match(attribution, /^\d{4}-\d{2}-\d{2}\.producthunt\./);

  const registration = await worker.fetch(new Request('https://metrics.test/api/register', {
    method: 'POST', headers: { ...commonHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'ProductHunter', email: 'hunter@example.com', password: 'LongEnoughPass9', acceptedTerms: true, attribution })
  }), configured);
  assert.equal(registration.status, 201);
  assert.equal(db.prepare("SELECT acquisition_source FROM users WHERE username_key = 'producthunter'").get().acquisition_source, 'producthunt');
  const duplicatePending = await worker.fetch(new Request('https://metrics.test/api/register', {
    method: 'POST', headers: { ...commonHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'AnotherHunter', email: 'HUNTER@example.com', password: 'LongEnoughPass9', acceptedTerms: true })
  }), configured);
  assert.equal(duplicatePending.status, 409);

  const adminHeaders = admin(db);
  let report = await (await worker.fetch(new Request('https://metrics.test/api/admin/stats', { headers: adminHeaders }), configured)).json();
  assert.equal(report.summary.productHuntVisitorsToday, 1);
  assert.equal(report.summary.productHuntRegisteredToday, 0);
  assert.equal(report.summary.pendingEmailTotal, 1);
  const pendingPublic = await (await worker.fetch(new Request('https://metrics.test/api/stats'), configured)).json();
  assert.equal(pendingPublic.totalAccounts, 0);

  const token = new URL(sent.text.match(/https:\/\/\S+/)[0]).searchParams.get('token');
  const confirmed = await worker.fetch(new Request('https://metrics.test/api/email-verification/confirm', {
    method: 'POST',
    headers: {
      Cookie: registration.headers.get('set-cookie'),
      Origin: configured.APP_ORIGIN,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ token })
  }), configured);
  assert.equal(confirmed.status, 200);

  report = await (await worker.fetch(new Request('https://metrics.test/api/admin/stats', { headers: adminHeaders }), configured)).json();
  assert.equal(report.summary.productHuntRegisteredToday, 1);
  assert.equal(report.summary.productHuntConversionToday, 1);
  assert.equal(report.summary.pendingEmailTotal, 0);
  const completedPublic = await (await worker.fetch(new Request('https://metrics.test/api/stats'), configured)).json();
  assert.equal(completedPublic.totalAccounts, 1);
  assert.equal(completedPublic.newToday, 1);
});

test('empty historical traffic and zero denominators are unavailable', async t => {
  const { env, db } = database(t);
  const headers = admin(db);
  const result = await worker.fetch(new Request('https://metrics.test/api/admin/stats', { headers }), env);
  const report = await result.json();
  assert.equal(result.status, 200);
  assert.equal(report.summary.conversionToday, null);
  assert.equal(report.summary.visitorsToday, null);
  assert.equal(report.summary.productHuntVisitorsToday, null);
  assert.equal(report.summary.productHuntConversionToday, null);
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

test('repeated password failures trigger a temporary login lock without revealing usernames', async t => {
  const { env, db } = database(t);
  admin(db);
  const attempt = (username, ip = '192.0.2.90') => worker.fetch(new Request('https://metrics.test/api/login', {
    method: 'POST',
    headers: { Origin: env.APP_ORIGIN, 'Content-Type': 'application/json', 'CF-Connecting-IP': ip },
    body: JSON.stringify({ username, password: 'DefinitelyWrong9' })
  }), env);

  for (let count = 1; count < 5; count += 1) {
    const response = await attempt('Admin');
    assert.equal(response.status, 401);
    assert.equal((await response.json()).error, 'Invalid username or password.');
  }
  const locked = await attempt('Admin');
  assert.equal(locked.status, 429);
  assert.equal(locked.headers.get('retry-after'), '900');
  assert.match((await locked.json()).error, /15 minutes/);

  const stillLocked = await attempt('Admin');
  assert.equal(stillLocked.status, 429);
  const distributedAttempt = await attempt('Admin', '192.0.2.91');
  assert.equal(distributedAttempt.status, 401);

  const unknown = await attempt('NotARealUser', '192.0.2.92');
  assert.equal(unknown.status, 401);
  assert.equal((await unknown.json()).error, 'Invalid username or password.');
  assert.ok(db.prepare('SELECT COUNT(*) AS count FROM login_rate_limits').get().count >= 7);
});

test('login fails closed when the rate-limit secret is missing', async t => {
  const { env, db } = database(t);
  admin(db);
  delete env.IP_HASH_SECRET;
  const response = await worker.fetch(new Request('https://metrics.test/api/login', {
    method: 'POST', headers: { Origin: env.APP_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'Admin', password: 'anything' })
  }), env);
  assert.equal(response.status, 503);
});

test('an enrolled administrator receives no session until passkey verification', async t => {
  const { env, db } = database(t);
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ id: 'local-email' }), { status: 200 }));
  const configured = {
    ...env, TERMS_VERSION: 'test', RESEND_API_KEY: 'local-test-key',
    EMAIL_FROM: 'BTC2G <accounts@btc2g.com>'
  };
  const registration = await worker.fetch(new Request('https://metrics.test/api/register', {
    method: 'POST',
    headers: { Origin: env.APP_ORIGIN, 'Content-Type': 'application/json', 'CF-Connecting-IP': '192.0.2.120' },
    body: JSON.stringify({
      username: 'SecureAdmin', email: 'secure-admin@example.com',
      password: 'LongEnoughPass9', acceptedTerms: true
    })
  }), configured);
  assert.equal(registration.status, 201);
  const sessionCookie = registration.headers.get('set-cookie');
  const user = db.prepare("SELECT id FROM users WHERE username_key = 'secureadmin'").get();
  db.prepare("UPDATE users SET role = 'admin' WHERE id = ?").run(user.id);
  const setup = await worker.fetch(new Request('https://metrics.test/api/passkeys/registration/options', {
    method: 'POST',
    headers: {
      Cookie: sessionCookie, Origin: env.APP_ORIGIN,
      'Content-Type': 'application/json', 'CF-Connecting-IP': '192.0.2.120'
    },
    body: JSON.stringify({ password: 'LongEnoughPass9' })
  }), configured);
  assert.equal(setup.status, 200);
  const setupData = await setup.json();
  assert.equal(setupData.options.rp.id, 'metrics.test');
  assert.equal(setupData.options.authenticatorSelection.userVerification, 'required');
  const rejectedSetup = await worker.fetch(new Request('https://metrics.test/api/passkeys/registration/verify', {
    method: 'POST', headers: { Cookie: sessionCookie, Origin: env.APP_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ challengeId: setupData.challengeId, credential: { id: 'invalid' } })
  }), configured);
  assert.equal(rejectedSetup.status, 400);
  assert.equal(db.prepare("SELECT COUNT(*) AS count FROM webauthn_challenges WHERE purpose = 'registration'").get().count, 0);
  db.prepare(`INSERT INTO passkeys
    (credential_id, user_id, public_key, counter, transports, display_name, created_at)
    VALUES (?, ?, ?, 0, '[]', 'Local test passkey', ?)`)
    .run('local-credential-id', user.id, new Uint8Array([1, 2, 3]), Math.floor(Date.now() / 1000));

  const login = await worker.fetch(new Request('https://metrics.test/api/login', {
    method: 'POST',
    headers: { Origin: env.APP_ORIGIN, 'Content-Type': 'application/json', 'CF-Connecting-IP': '192.0.2.120' },
    body: JSON.stringify({ username: 'SecureAdmin', password: 'LongEnoughPass9' })
  }), configured);
  assert.equal(login.status, 200);
  assert.equal(login.headers.get('set-cookie'), null);
  const data = await login.json();
  assert.equal(data.passkeyRequired, true);
  assert.equal(data.options.userVerification, 'required');
  assert.deepEqual(data.options.allowCredentials.map(item => item.id), ['local-credential-id']);
  assert.equal(db.prepare("SELECT COUNT(*) AS count FROM webauthn_challenges WHERE purpose = 'authentication'").get().count, 1);
});

test('signed-in users can verify an email without creating marketing consent', async t => {
  const { env, db } = database(t);
  const headers = admin(db);
  let sent;
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    sent = JSON.parse(options.body);
    return new Response(JSON.stringify({ id: 'local-email' }), { status: 200 });
  });
  const configured = {
    ...env,
    RESEND_API_KEY: 'local-test-key',
    EMAIL_FROM: 'BTC2G <accounts@btc2g.com>'
  };
  const requested = await worker.fetch(new Request('https://metrics.test/api/email-verification/request', {
    method: 'POST',
    headers: { ...headers, Origin: configured.APP_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'Owner@Example.com' })
  }), configured);
  assert.equal(requested.status, 200);
  assert.deepEqual(sent.to, ['Owner@Example.com']);
  assert.equal(sent.reply_to, 'support@btc2g.com');
  assert.match(sent.text, /does not subscribe you to marketing or activate Premium/);
  const token = new URL(sent.text.match(/https:\/\/\S+/)[0]).searchParams.get('token');
  assert.ok(token);

  const unsigned = await worker.fetch(new Request('https://metrics.test/api/email-verification/confirm', {
    method: 'POST', headers: { Origin: configured.APP_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ token })
  }), configured);
  assert.equal(unsigned.status, 401);

  const confirmed = await worker.fetch(new Request('https://metrics.test/api/email-verification/confirm', {
    method: 'POST',
    headers: { ...headers, Origin: configured.APP_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ token })
  }), configured);
  assert.equal(confirmed.status, 200);
  const account = db.prepare("SELECT email, email_key, email_verified_at FROM users WHERE id = 'admin'").get();
  assert.equal(account.email, 'Owner@Example.com');
  assert.equal(account.email_key, 'owner@example.com');
  assert.ok(account.email_verified_at > 0);
  assert.equal(db.prepare('SELECT COUNT(*) AS count FROM email_verification_tokens').get().count, 0);
  const report = await (await worker.fetch(new Request('https://metrics.test/api/admin/stats', { headers }), configured)).json();
  assert.equal(report.summary.verifiedEmailTotal, 0);
  assert.equal(report.confirmedSubscribers.total, null);
});

test('email verification fails closed when delivery is not configured', async t => {
  const { env, db } = database(t);
  const response = await worker.fetch(new Request('https://metrics.test/api/email-verification/request', {
    method: 'POST',
    headers: { ...admin(db), Origin: env.APP_ORIGIN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'owner@example.com' })
  }), env);
  assert.equal(response.status, 503);
  assert.equal(db.prepare('SELECT COUNT(*) AS count FROM email_verification_tokens').get().count, 0);
});
