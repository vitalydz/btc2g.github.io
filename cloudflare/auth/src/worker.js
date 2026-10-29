import {
  EMAIL_VERIFICATION_MAX_AGE_SECONDS,
  PASSWORD_ITERATIONS,
  SESSION_MAX_AGE_SECONDS,
  acquisitionSource,
  clearSessionCookie,
  jerusalemDay,
  normalizeEmail,
  normalizePagePath,
  normalizeUsername,
  parseCookies,
  publicStats,
  sessionCookie,
  validateEmail,
  validateRegistration
} from './core.js';
import { confirmedSubscriptions, registrationRatio, trafficMetrics } from './metrics.js';

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

async function requestVisitorHash(request, env, day) {
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const userAgent = (request.headers.get('User-Agent') || 'unknown').slice(0, 256);
  return hmacSha256(env.IP_HASH_SECRET, `${day}:${ip}:${userAgent}`);
}

async function createAttributionToken(env, day, visitorHash, source) {
  const signature = await hmacSha256(env.IP_HASH_SECRET, `${day}:${visitorHash}:${source}`);
  return `${day}.${source}.${signature}`;
}

async function verifiedAttributionSource(request, env, token) {
  if (!env.IP_HASH_SECRET || typeof token !== 'string') return null;
  const [day, source, signature, extra] = token.split('.');
  if (extra !== undefined || day !== jerusalemDay() || source !== 'producthunt' || !signature) return null;
  const visitorHash = await requestVisitorHash(request, env, day);
  const expected = await hmacSha256(env.IP_HASH_SECRET, `${day}:${visitorHash}:${source}`);
  return constantTimeEqual(signature, expected) ? source : null;
}

function sameOrigin(request, env) {
  const origin = request.headers.get('Origin');
  return !origin || origin === env.APP_ORIGIN || origin === new URL(request.url).origin;
}

function userView(user) {
  return {
    username: user.username,
    role: user.role,
    email: user.email || null,
    emailVerified: Boolean(user.email_verified_at)
  };
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
    const duplicateEmail = await env.DB.prepare(`
      SELECT id FROM users WHERE email_key = ? AND disabled_at IS NULL LIMIT 1
    `).bind(validation.emailKey).first();
    if (duplicateEmail) return json({ error: 'That email is already attached to an account.' }, 409);
    const pendingEmail = await env.DB.prepare(`
      SELECT user_id FROM email_verification_tokens
      WHERE email_key = ? AND expires_at > ? LIMIT 1
    `).bind(validation.emailKey, Math.floor(Date.now() / 1000)).first();
    if (pendingEmail) return json({ error: 'That email is awaiting verification for another account.' }, 409);
  }

  const salt = randomToken(16);
  const hash = await passwordHash(input.password, salt);
  const now = Math.floor(Date.now() / 1000);
  const id = crypto.randomUUID();
  const source = admin ? null : await verifiedAttributionSource(request, env, input?.attribution);
  try {
    await env.DB.prepare(`
      INSERT INTO users
        (id, username, username_key, password_hash, password_salt, password_iterations, role,
         created_at, created_day, accepted_terms_at, terms_version, acquisition_source)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      id, validation.username, validation.usernameKey, hash, salt, PASSWORD_ITERATIONS,
      admin ? 'admin' : 'user', now, jerusalemDay(now * 1000), now, env.TERMS_VERSION, source
    ).run();
  } catch (error) {
    if (String(error).toLowerCase().includes('unique')) return json({ error: 'That username is already registered.' }, 409);
    throw error;
  }
  const token = await createSession(env, id);
  let verificationEmailSent = false;
  if (!admin) {
    try {
      await issueEmailVerification(env, { id, username: validation.username }, {
        ok: true, email: validation.email, emailKey: validation.emailKey
      }, { checkDuplicate: false });
      verificationEmailSent = true;
    } catch {
      // Keep the pending account accessible so the owner can sign in and retry delivery.
    }
  }
  return json(
    {
      user: {
        username: validation.username,
        role: admin ? 'admin' : 'user',
        email: null,
        emailVerified: false,
        pendingEmail: admin ? null : validation.email
      },
      verificationEmailSent
    },
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
    SELECT id, username, role, email, email_verified_at,
           password_hash, password_salt, password_iterations
    FROM users WHERE username_key = ? AND disabled_at IS NULL
  `).bind(normalizeUsername(input?.username)).first();
  if (!user) return json({ error: 'Invalid username or password.' }, 401);
  const hash = await passwordHash(String(input?.password || ''), user.password_salt, user.password_iterations);
  if (!constantTimeEqual(hash, user.password_hash)) return json({ error: 'Invalid username or password.' }, 401);
  const token = await createSession(env, user.id);
  return json({ user: userView(user) }, 200, { 'set-cookie': sessionCookie(token) });
}

async function currentUser(request, env) {
  const token = parseCookies(request.headers.get('Cookie')).btc2g_session;
  if (!token) return null;
  const tokenHash = await sha256(token);
  const now = Math.floor(Date.now() / 1000);
  return env.DB.prepare(`
    SELECT users.id, users.username, users.role, users.email, users.email_verified_at
    FROM sessions JOIN users ON users.id = sessions.user_id
    WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.disabled_at IS NULL
  `).bind(tokenHash, now).first();
}

async function sendVerificationEmail(env, user, email, token) {
  if (!env.RESEND_API_KEY || !env.EMAIL_FROM || !env.APP_ORIGIN) {
    throw new Error('Email delivery is not configured.');
  }
  const link = new URL('/verify-email.html', env.APP_ORIGIN);
  link.searchParams.set('token', token);
  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      authorization: `Bearer ${env.RESEND_API_KEY}`,
      'content-type': 'application/json',
      'idempotency-key': `btc2g-email-verification/${await sha256(token)}`
    },
    body: JSON.stringify({
      from: env.EMAIL_FROM,
      to: [email],
      subject: 'Verify your BTC2G email',
      text: `Hello ${user.username},\n\nConfirm this email for your BTC2G account:\n${link}\n\nThe link expires in 30 minutes. This does not subscribe you to marketing or activate Premium.`,
      html: `<p>Hello ${user.username},</p><p>Confirm this email for your BTC2G account:</p><p><a href="${link}">Verify email</a></p><p>The link expires in 30 minutes. This does not subscribe you to marketing or activate Premium.</p>`
    })
  });
  if (!response.ok) throw new Error('Email delivery failed.');
}

async function issueEmailVerification(env, user, validation, { checkDuplicate = true } = {}) {
  if (checkDuplicate) {
    const duplicate = await env.DB.prepare(`
      SELECT id FROM users WHERE email_key = ? AND id <> ? AND disabled_at IS NULL LIMIT 1
    `).bind(validation.emailKey, user.id).first();
    if (duplicate) {
      const error = new Error('That email is already attached to an account.');
      error.status = 409;
      throw error;
    }
    const pendingDuplicate = await env.DB.prepare(`
      SELECT user_id FROM email_verification_tokens
      WHERE email_key = ? AND user_id <> ? AND expires_at > ? LIMIT 1
    `).bind(validation.emailKey, user.id, Math.floor(Date.now() / 1000)).first();
    if (pendingDuplicate) {
      const error = new Error('That email is awaiting verification for another account.');
      error.status = 409;
      throw error;
    }
  }

  const now = Math.floor(Date.now() / 1000);
  const recent = await env.DB.prepare(`
    SELECT COUNT(*) AS count FROM email_verification_tokens
    WHERE user_id = ? AND created_at >= ?
  `).bind(user.id, now - 3600).first();
  if (Number(recent?.count || 0) >= 5) {
    const error = new Error('Too many verification emails. Try again later.');
    error.status = 429;
    throw error;
  }

  const token = randomToken();
  const tokenHash = await sha256(token);
  await env.DB.prepare(`
    INSERT INTO email_verification_tokens
      (token_hash, user_id, email, email_key, created_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).bind(tokenHash, user.id, validation.email, validation.emailKey, now,
    now + EMAIL_VERIFICATION_MAX_AGE_SECONDS).run();
  try {
    await sendVerificationEmail(env, user, validation.email, token);
  } catch {
    await env.DB.prepare('DELETE FROM email_verification_tokens WHERE token_hash = ?').bind(tokenHash).run();
    const error = new Error('Verification email could not be sent. Try again later.');
    error.status = 503;
    throw error;
  }
  return { expiresIn: EMAIL_VERIFICATION_MAX_AGE_SECONDS };
}

async function requestEmailVerification(request, env) {
  const user = await currentUser(request, env);
  if (!user) return json({ error: 'Sign in before adding an email.' }, 401);
  let input;
  try { input = await body(request); } catch (error) { return json({ error: error.message }, 400); }
  const validation = validateEmail(input?.email);
  if (!validation.ok) return json({ error: 'Enter a valid email address.' }, 400);

  try {
    const result = await issueEmailVerification(env, user, validation);
    return json({ ok: true, ...result });
  } catch (error) {
    return json({ error: error.message }, error.status || 503);
  }
}

async function confirmEmailVerification(request, env) {
  const user = await currentUser(request, env);
  if (!user) return json({ error: 'Sign in to confirm this email.' }, 401);
  let input;
  try { input = await body(request); } catch (error) { return json({ error: error.message }, 400); }
  if (typeof input?.token !== 'string' || input.token.length < 32 || input.token.length > 128) {
    return json({ error: 'Invalid verification link.' }, 400);
  }
  const tokenHash = await sha256(input.token);
  const now = Math.floor(Date.now() / 1000);
  const pending = await env.DB.prepare(`
    SELECT email, email_key FROM email_verification_tokens
    WHERE token_hash = ? AND user_id = ? AND expires_at > ?
  `).bind(tokenHash, user.id, now).first();
  if (!pending) return json({ error: 'This verification link is invalid or expired.' }, 400);

  try {
    await env.DB.batch([
      env.DB.prepare(`
        UPDATE users SET email = ?, email_key = ?, email_verified_at = ?, email_verified_day = ?
        WHERE id = ? AND disabled_at IS NULL
      `).bind(pending.email, normalizeEmail(pending.email_key), now, jerusalemDay(now * 1000), user.id),
      env.DB.prepare('DELETE FROM email_verification_tokens WHERE user_id = ?').bind(user.id)
    ]);
  } catch (error) {
    if (String(error).toLowerCase().includes('unique')) {
      return json({ error: 'That email is already attached to an account.' }, 409);
    }
    throw error;
  }
  return json({ user: { ...userView(user), email: pending.email, emailVerified: true } });
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
           SUM(CASE WHEN email_verified_day = ? THEN 1 ELSE 0 END) AS new_today
    FROM users
    WHERE disabled_at IS NULL AND role = 'user' AND email_verified_at IS NOT NULL
  `).bind(day).first();
  let traffic;
  try {
    const trafficRow = await env.DB.prepare(`
      SELECT
        (SELECT MIN(first_seen_at) FROM daily_visitors) AS first_seen_at,
        (SELECT COUNT(*) FROM daily_visitors WHERE day = ?) AS visitors_today,
        (SELECT COALESCE(SUM(views), 0) FROM daily_pageviews WHERE day = ?) AS views_today
    `).bind(day, day).first();
    traffic = trafficMetrics(trafficRow);
  } catch {
    traffic = { status: 'unavailable', visitorsToday: null, viewsToday: null,
      firstRecordedVisitAt: null, reason: 'Traffic storage is unavailable.' };
  }
  return json({
    ...publicStats(row?.total_accounts, row?.new_today), day, timeZone: 'Asia/Jerusalem',
    measuredAt: new Date().toISOString(), source: 'Cloudflare D1',
    traffic, confirmedSubscribers: confirmedSubscriptions()
  });
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
  const source = acquisitionSource(input?.referrer);
  const visitorHash = await requestVisitorHash(request, env, day);
  const now = Math.floor(Date.now() / 1000);
  await env.DB.batch([
    env.DB.prepare(`
      INSERT INTO daily_visitors
        (day, visitor_hash, first_path, last_path, first_seen_at, last_seen_at, views, acquisition_source)
      VALUES (?, ?, ?, ?, ?, ?, 1, ?)
      ON CONFLICT(day, visitor_hash) DO UPDATE SET
        last_path = excluded.last_path,
        last_seen_at = excluded.last_seen_at,
        acquisition_source = COALESCE(daily_visitors.acquisition_source, excluded.acquisition_source),
        views = daily_visitors.views + 1
    `).bind(day, visitorHash, pagePath, pagePath, now, now, source),
    env.DB.prepare(`
      INSERT INTO daily_pageviews (day, page_path, views) VALUES (?, ?, 1)
      ON CONFLICT(day, page_path) DO UPDATE SET views = daily_pageviews.views + 1
    `).bind(day, pagePath)
  ]);
  const attribution = source ? await createAttributionToken(env, day, visitorHash, source) : null;
  return json({ ok: true, attribution }, 202, cors);
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
        (SELECT MIN(first_seen_at) FROM daily_visitors) AS first_seen_at,
        (SELECT COUNT(*) FROM daily_visitors WHERE day = ?) AS visitors_today,
        (SELECT COALESCE(SUM(views), 0) FROM daily_pageviews WHERE day = ?) AS views_today,
        (SELECT COUNT(*) FROM daily_visitors WHERE day = ? AND acquisition_source = 'producthunt') AS producthunt_visitors_today,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL AND email_verified_at IS NOT NULL) AS registered_total,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL AND email_verified_day = ?) AS registered_today,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL AND email_verified_at IS NOT NULL) AS verified_email_total,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL AND email_verified_at IS NULL) AS pending_email_total,
        (SELECT COUNT(*) FROM users WHERE role = 'user' AND disabled_at IS NULL AND email_verified_day = ? AND acquisition_source = 'producthunt') AS producthunt_registered_today
    `).bind(today, today, today, today, today).first(),
    env.DB.prepare(`
      WITH RECURSIVE dates(day) AS (
        SELECT date(?, '-' || ? || ' day')
        UNION ALL SELECT date(day, '+1 day') FROM dates WHERE day < ?
      ), coverage AS (
        SELECT MIN(day) AS first_day FROM daily_visitors
      ), traffic AS (
        SELECT day, COUNT(*) AS visitors, COALESCE(SUM(views), 0) AS views,
               SUM(CASE WHEN acquisition_source = 'producthunt' THEN 1 ELSE 0 END) AS producthunt_visitors
        FROM daily_visitors GROUP BY day
      ), registrations AS (
        SELECT email_verified_day AS day, COUNT(*) AS registrations,
               SUM(CASE WHEN acquisition_source = 'producthunt' THEN 1 ELSE 0 END) AS producthunt_registrations
        FROM users
        WHERE role = 'user' AND disabled_at IS NULL AND email_verified_at IS NOT NULL
        GROUP BY email_verified_day
      )
      SELECT dates.day,
             CASE WHEN dates.day >= coverage.first_day THEN COALESCE(traffic.visitors, 0) ELSE NULL END AS visitors,
             CASE WHEN dates.day >= coverage.first_day THEN COALESCE(traffic.views, 0) ELSE NULL END AS views,
             COALESCE(registrations.registrations, 0) AS registrations,
             CASE WHEN dates.day >= coverage.first_day THEN COALESCE(traffic.producthunt_visitors, 0) ELSE NULL END AS producthunt_visitors,
             COALESCE(registrations.producthunt_registrations, 0) AS producthunt_registrations
      FROM dates
      CROSS JOIN coverage
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
      SELECT username, email_verified_at AS created_at, email_verified_day AS created_day, 1 AS email_verified
      FROM users
      WHERE role = 'user' AND disabled_at IS NULL AND email_verified_at IS NOT NULL
      ORDER BY email_verified_at DESC LIMIT 20
    `).all()
  ]);

  const traffic = trafficMetrics(summary);
  const visitorsToday = traffic.visitorsToday;
  const registeredToday = Number(summary?.registered_today || 0);
  const productHuntVisitorsToday = traffic.status === 'available'
    ? Number(summary?.producthunt_visitors_today || 0) : null;
  const productHuntRegisteredToday = Number(summary?.producthunt_registered_today || 0);
  return json({
    rangeDays: days,
    day: today,
    timeZone: 'Asia/Jerusalem',
    measuredAt: new Date().toISOString(),
    source: 'Cloudflare D1',
    traffic,
    confirmedSubscribers: confirmedSubscriptions(),
    summary: {
      visitorsToday,
      viewsToday: traffic.viewsToday,
      registeredToday,
      registeredTotal: Number(summary?.registered_total || 0),
      verifiedEmailTotal: Number(summary?.verified_email_total || 0),
      pendingEmailTotal: Number(summary?.pending_email_total || 0),
      conversionToday: registrationRatio(registeredToday, visitorsToday),
      productHuntVisitorsToday,
      productHuntRegisteredToday,
      productHuntConversionToday: registrationRatio(productHuntRegisteredToday, productHuntVisitorsToday)
    },
    daily: dailyResult.results || [],
    topPaths: pathsResult.results || [],
    recentRegistrations: recentResult.results || [],
    methodology: 'Visitors are daily IP + browser hash estimates, not verified people. Raw IP addresses are not stored. A registration is counted only after the account email is verified. Product Hunt is detected from the browser-reported referrer; a registration is attributed only when a signed same-day token matches the same browser/network estimate. Other registration/visitor ratios are not attributed conversion. Days before the earliest retained visit have no traffic data. Verified account emails are not marketing subscriptions.'
  });
}

async function api(request, env) {
  const url = new URL(request.url);
  if (request.method === 'GET' && url.pathname === '/api/stats') return stats(env);
  if (request.method === 'GET' && url.pathname === '/api/admin/stats') return adminStats(request, env);
  if (request.method === 'GET' && url.pathname === '/api/me') {
    const user = await currentUser(request, env);
    return user ? json({ user: userView(user) }) : json({ user: null }, 401);
  }
  if (request.method === 'POST') {
    if (url.pathname === '/api/visit') return recordVisit(request, env);
    if (!sameOrigin(request, env)) return json({ error: 'Origin not allowed.' }, 403);
    if (url.pathname === '/api/register') return register(request, env);
    if (url.pathname === '/api/login') return login(request, env);
    if (url.pathname === '/api/logout') return logout(request, env);
    if (url.pathname === '/api/email-verification/request') return requestEmailVerification(request, env);
    if (url.pathname === '/api/email-verification/confirm') return confirmEmailVerification(request, env);
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
      env.DB.prepare('DELETE FROM email_verification_tokens WHERE expires_at <= ?').bind(now),
      env.DB.prepare("DELETE FROM daily_visitors WHERE day < date('now', '-' || ? || ' day')").bind(retentionDays),
      env.DB.prepare("DELETE FROM daily_pageviews WHERE day < date('now', '-' || ? || ' day')").bind(retentionDays)
    ]);
  }
};
