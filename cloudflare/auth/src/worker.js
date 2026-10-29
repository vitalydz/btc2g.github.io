import {
  PASSWORD_ITERATIONS,
  SESSION_MAX_AGE_SECONDS,
  clearSessionCookie,
  jerusalemDay,
  normalizePagePath,
  normalizeUsername,
  parseCookies,
  publicStats,
  sessionCookie,
  validateRegistration
} from './core.js';

const encoder = new TextEncoder();

function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers }
  });
}

function corsHeaders(origin, env) {
  if (!origin || origin !== env.PUBLIC_SITE_ORIGIN) return {};
  return {
    'access-control-allow-origin': origin,
    'access-control-allow-methods': 'POST, OPTIONS',
    'access-control-allow-headers': 'content-type',
    'access-control-max-age': '86400',
    vary: 'Origin'
  };
}

function bytesToBase64Url(bytes) {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}

function randomToken(byteLength = 32) {
  return bytesToBase64Url(crypto.getRandomValues(new Uint8Array(byteLength)));
}

async function sha256(value) {
  return bytesToBase64Url(new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(value))));
}

async function hmacSha256(secret, value) {
  const key = await crypto.subtle.importKey(
    'raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']
  );
  return bytesToBase64Url(new Uint8Array(await crypto.subtle.sign('HMAC', key, encoder.encode(value))));
}

async function passwordHash(password, salt, iterations = PASSWORD_ITERATIONS) {
  // Workers cap one PBKDF2 operation at 100k iterations. Chain supported
  // rounds so the stored work factor remains 210k without exceeding the API cap.
  let material = encoder.encode(password);
  let remaining = iterations;
  let round = 0;
  while (remaining > 0) {
    const roundIterations = Math.min(remaining, 100000);
    const key = await crypto.subtle.importKey('raw', material, 'PBKDF2', false, ['deriveBits']);
    const bits = await crypto.subtle.deriveBits(
      { name: 'PBKDF2', hash: 'SHA-256', salt: encoder.encode(`${salt}:${round}`), iterations: roundIterations },
      key,
      256
    );
    material = new Uint8Array(bits);
    remaining -= roundIterations;
    round += 1;
  }
  return bytesToBase64Url(material);
}

function constantTimeEqual(left, right) {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

function sameOrigin(request, env) {
  const origin = request.headers.get('Origin');
  return !origin || origin === env.APP_ORIGIN || origin === new URL(request.url).origin;
}

async function body(request) {
  const type = request.headers.get('content-type') || '';
  if (!type.includes('application/json')) throw new Error('Content-Type must be application/json.');
  return request.json();
}

async function enforceRegistrationLimit(request, env) {
  if (!env.IP_HASH_SECRET) throw new Error('Registration is temporarily unavailable.');
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const ipHash = await sha256(`${env.IP_HASH_SECRET}:${ip}`);
  const now = Math.floor(Date.now() / 1000);
  const windowStart = Math.floor(now / 3600) * 3600;
  const result = await env.DB.prepare(`
    INSERT INTO registration_attempts (ip_hash, window_start, attempt_count)
    VALUES (?, ?, 1)
    ON CONFLICT(ip_hash, window_start)
    DO UPDATE SET attempt_count = attempt_count + 1
    RETURNING attempt_count
  `).bind(ipHash, windowStart).first();
  const limit = Number(env.REGISTRATION_LIMIT_PER_HOUR || 5);
  return Number(result?.attempt_count || 1) <= limit;
}

async function createSession(env, userId) {
  const token = randomToken();
  const tokenHash = await sha256(token);
  const now = Math.floor(Date.now() / 1000);
  await env.DB.prepare('INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)')
    .bind(tokenHash, userId, now, now + SESSION_MAX_AGE_SECONDS).run();
  return token;
}

async function register(request, env, { admin = false } = {}) {
  let input;
  try { input = await body(request); } catch (error) { return json({ error: error.message }, 400); }
  const validation = validateRegistration(input, { allowReserved: admin });
  if (!validation.ok) return json({ error: validation.errors[0], errors: validation.errors }, 400);
  if (admin && validation.usernameKey !== 'codex_admin') return json({ error: 'Bootstrap username must be Codex_admin.' }, 400);

  if (!admin) {
    try {
      if (!(await enforceRegistrationLimit(request, env))) {
        return json({ error: 'Registration limit reached for this network. Try again later.' }, 429);
      }
    } catch {
      return json({ error: 'Registration is temporarily unavailable.' }, 503);
    }
  }

  const salt = randomToken(16);
  const hash = await passwordHash(input.password, salt);
  const now = Math.floor(Date.now() / 1000);
  const id = crypto.randomUUID();
  try {
    await env.DB.prepare(`
      INSERT INTO users
        (id, username, username_key, password_hash, password_salt, password_iterations, role,
         created_at, created_day, accepted_terms_at, terms_version)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      id, validation.username, validation.usernameKey, hash, salt, PASSWORD_ITERATIONS,
      admin ? 'admin' : 'user', now, jerusalemDay(now * 1000), now, env.TERMS_VERSION
    ).run();
  } catch (error) {
    if (String(error).toLowerCase().includes('unique')) return json({ error: 'That username is already registered.' }, 409);
    throw error;
  }
  const token = await createSession(env, id);
  return json(
    { user: { username: validation.username, role: admin ? 'admin' : 'user' } },
    201,
    { 'set-cookie': sessionCookie(token) }
  );
}

async function bootstrapAdmin(request, env) {
  if (!env.BOOTSTRAP_TOKEN) return json({ error: 'Admin bootstrap is disabled.' }, 404);
  if (request.headers.get('Authorization') !== `Bearer ${env.BOOTSTRAP_TOKEN}`) {
    return json({ error: 'Unauthorized.' }, 401);
  }
  const existing = await env.DB.prepare("SELECT id FROM users WHERE role = 'admin' LIMIT 1").first();
  if (existing) return json({ error: 'An administrator already exists.' }, 409);
  return register(request, env, { admin: true });
}

async function login(request, env) {
  let input;
  try { input = await body(request); } catch (error) { return json({ error: error.message }, 400); }
  const user = await env.DB.prepare(`
    SELECT id, username, role, password_hash, password_salt, password_iterations
    FROM users WHERE username_key = ? AND disabled_at IS NULL
  `).bind(normalizeUsername(input?.username)).first();
  if (!user) return json({ error: 'Invalid username or password.' }, 401);
  const hash = await passwordHash(String(input?.password || ''), user.password_salt, user.password_iterations);
  if (!constantTimeEqual(hash, user.password_hash)) return json({ error: 'Invalid username or password.' }, 401);
  const token = await createSession(env, user.id);
  return json({ user: { username: user.username, role: user.role } }, 200, { 'set-cookie': sessionCookie(token) });
}

async function currentUser(request, env) {
  const token = parseCookies(request.headers.get('Cookie')).btc2g_session;
  if (!token) return null;
  const tokenHash = await sha256(token);
  const now = Math.floor(Date.now() / 1000);
  return env.DB.prepare(`
    SELECT users.username, users.role
    FROM sessions JOIN users ON users.id = sessions.user_id
    WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.disabled_at IS NULL
  `).bind(tokenHash, now).first();
}

async function logout(request, env) {
  const token = parseCookies(request.headers.get('Cookie')).btc2g_session;
  if (token) await env.DB.prepare('DELETE FROM sessions WHERE token_hash = ?').bind(await sha256(token)).run();
  return json({ ok: true }, 200, { 'set-cookie': clearSessionCookie() });
}

async function stats(env) {
  const day = jerusalemDay();
  const row = await env.DB.prepare(`
    SELECT COUNT(*) AS total_accounts,
           SUM(CASE WHEN created_day = ? THEN 1 ELSE 0 END) AS new_today
    FROM users WHERE disabled_at IS NULL AND role = 'user'
  `).bind(day).first();
  return json({ ...publicStats(row?.total_accounts, row?.new_today), day, timeZone: 'Asia/Jerusalem' });
}

async function recordVisit(request, env) {
  const origin = request.headers.get('Origin');
  const cors = corsHeaders(origin, env);
  if (!cors['access-control-allow-origin']) return json({ error: 'Origin not allowed.' }, 403);
  if (!env.IP_HASH_SECRET) return json({ error: 'Metrics are temporarily unavailable.' }, 503, cors);

  let input;
  try { input = await body(request); } catch (error) { return json({ error: error.message }, 400, cors); }
  const pagePath = normalizePagePath(input?.path);
  if (!pagePath) return json({ error: 'Invalid page path.' }, 400, cors);

  const day = jerusalemDay();
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const userAgent = (request.headers.get('User-Agent') || 'unknown').slice(0, 256);
  const visitorHash = await hmacSha256(env.IP_HASH_SECRET, `${day}:${ip}:${userAgent}`);
  const now = Math.floor(Date.now() / 1000);
  await env.DB.batch([
    env.DB.prepare(`
      INSERT INTO daily_visitors
        (day, visitor_hash, first_path, last_path, first_seen_at, last_seen_at, views)
      VALUES (?, ?, ?, ?, ?, ?, 1)
      ON CONFLICT(day, visitor_hash) DO UPDATE SET
        last_path = excluded.last_path,
        last_seen_at = excluded.last_seen_at,
        views = daily_visitors.views + 1
    `).bind(day, visitorHash, pagePath, pagePath, now, now),
    env.DB.prepare(`
      INSERT INTO daily_pageviews (day, page_path, views) VALUES (?, ?, 1)
      ON CONFLICT(day, page_path) DO UPDATE SET views = daily_pageviews.views + 1
    `).bind(day, pagePath)
  ]);
  return json({ ok: true }, 202, cors);
}

async function adminStats(request, env) {
  const user = await currentUser(request, env);
  if (!user || user.role !== 'admin') return json({ error: 'Administrator access required.' }, 403);
  const url = new URL(request.url);
  const requestedDays = Number.parseInt(url.searchParams.get('days') || '7', 10);
  const days = [7, 30, 90].includes(requestedDays) ? requestedDays : 7;
  const today = jerusalemDay();

  const [summary, dailyResult, pathsResult, recentResult] = await Promise.all([
    env.DB.prepare(`
      SELECT
        (SELECT COUNT(*) FROM daily_visitors WHERE day = ?) AS visitors_today,
        (SELECT COALESCE(SUM(views), 0) FROM daily_pageviews WHERE day = ?) AS views_today,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL) AS registered_total,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL AND created_day = ?) AS registered_today
    `).bind(today, today, today).first(),
    env.DB.prepare(`
      WITH RECURSIVE dates(day) AS (
        SELECT date(?, '-' || ? || ' day')
        UNION ALL SELECT date(day, '+1 day') FROM dates WHERE day < ?
      ), traffic AS (
        SELECT day, COUNT(*) AS visitors, COALESCE(SUM(views), 0) AS views
        FROM daily_visitors GROUP BY day
      ), registrations AS (
        SELECT created_day AS day, COUNT(*) AS registrations
        FROM users WHERE role = 'user' AND disabled_at IS NULL GROUP BY created_day
      )
      SELECT dates.day,
             COALESCE(traffic.visitors, 0) AS visitors,
             COALESCE(traffic.views, 0) AS views,
             COALESCE(registrations.registrations, 0) AS registrations
      FROM dates
      LEFT JOIN traffic ON traffic.day = dates.day
      LEFT JOIN registrations ON registrations.day = dates.day
      ORDER BY dates.day DESC
    `).bind(today, days - 1, today).all(),
    env.DB.prepare(`
      SELECT page_path AS path, SUM(views) AS views
      FROM daily_pageviews
      WHERE day >= date(?, '-' || ? || ' day')
      GROUP BY page_path ORDER BY views DESC, page_path ASC LIMIT 10
    `).bind(today, days - 1).all(),
    env.DB.prepare(`
      SELECT username, created_at, created_day
      FROM users WHERE role = 'user' AND disabled_at IS NULL
      ORDER BY created_at DESC LIMIT 20
    `).all()
  ]);

  const visitorsToday = Number(summary?.visitors_today || 0);
  const registeredToday = Number(summary?.registered_today || 0);
  return json({
    rangeDays: days,
    day: today,
    timeZone: 'Asia/Jerusalem',
    summary: {
      visitorsToday,
      viewsToday: Number(summary?.views_today || 0),
      registeredToday,
      registeredTotal: Number(summary?.registered_total || 0),
      conversionToday: visitorsToday ? registeredToday / visitorsToday : 0
    },
    daily: dailyResult.results || [],
    topPaths: pathsResult.results || [],
    recentRegistrations: recentResult.results || [],
    methodology: 'Unique visitors are daily salted hashes of IP + user agent. Raw IP addresses are not stored.'
  });
}

async function api(request, env) {
  const url = new URL(request.url);
  if (request.method === 'GET' && url.pathname === '/api/stats') return stats(env);
  if (request.method === 'GET' && url.pathname === '/api/admin/stats') return adminStats(request, env);
  if (request.method === 'GET' && url.pathname === '/api/me') {
    const user = await currentUser(request, env);
    return user ? json({ user }) : json({ user: null }, 401);
  }
  if (request.method === 'POST') {
    if (url.pathname === '/api/visit') return recordVisit(request, env);
    if (!sameOrigin(request, env)) return json({ error: 'Origin not allowed.' }, 403);
    if (url.pathname === '/api/register') return register(request, env);
    if (url.pathname === '/api/login') return login(request, env);
    if (url.pathname === '/api/logout') return logout(request, env);
    if (url.pathname === '/api/bootstrap-admin') return bootstrapAdmin(request, env);
  }
  if (request.method === 'OPTIONS' && url.pathname === '/api/visit') {
    const headers = corsHeaders(request.headers.get('Origin'), env);
    return headers['access-control-allow-origin']
      ? new Response(null, { status: 204, headers })
      : json({ error: 'Origin not allowed.' }, 403);
  }
  return json({ error: 'Not found.' }, 404);
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      if (url.pathname.startsWith('/api/')) return await api(request, env);
      return env.ASSETS.fetch(request);
    } catch (error) {
      console.error('Unhandled request error', error);
      return json({ error: 'Unexpected server error.' }, 500);
    }
  },

  async scheduled(_event, env) {
    const now = Math.floor(Date.now() / 1000);
    const retentionDays = Math.max(30, Number(env.METRICS_RETENTION_DAYS || 400));
    await env.DB.batch([
      env.DB.prepare('DELETE FROM sessions WHERE expires_at <= ?').bind(now),
      env.DB.prepare('DELETE FROM registration_attempts WHERE window_start < ?').bind(now - 86400),
      env.DB.prepare("DELETE FROM daily_visitors WHERE day < date('now', '-' || ? || ' day')").bind(retentionDays),
      env.DB.prepare("DELETE FROM daily_pageviews WHERE day < date('now', '-' || ? || ' day')").bind(retentionDays)
    ]);
  }
};
