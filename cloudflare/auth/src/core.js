export const USERNAME_PATTERN = /^[A-Za-z][A-Za-z0-9_]{2,31}$/;
export const PASSWORD_ITERATIONS = 210000;
export const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;
export const EMAIL_VERIFICATION_MAX_AGE_SECONDS = 60 * 30;

const RESERVED_USERNAMES = new Set(['admin', 'administrator', 'codex_admin', 'btc2g']);

export function normalizeUsername(username) {
  return String(username || '').trim().toLocaleLowerCase('en-US');
}

export function normalizeEmail(email) {
  return String(email || '').trim().toLocaleLowerCase('en-US');
}

export function validateEmail(email) {
  const value = String(email || '').trim();
  const normalized = normalizeEmail(value);
  const valid = value.length <= 254
    && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
    && !value.includes('..');
  return { ok: valid, email: value, emailKey: normalized };
}

export function validateRegistration(input, { allowReserved = false } = {}) {
  const username = String(input?.username || '').trim();
  const password = String(input?.password || '');
  const acceptedTerms = input?.acceptedTerms === true;
  const emailValidation = validateEmail(input?.email);
  const errors = [];

  if (!USERNAME_PATTERN.test(username)) {
    errors.push('Username must be 3–32 characters and use only letters, numbers, or underscores.');
  }
  if (!allowReserved && RESERVED_USERNAMES.has(normalizeUsername(username))) {
    errors.push('That username is reserved.');
  }
  if (password.length < 12 || password.length > 128) {
    errors.push('Password must be between 12 and 128 characters.');
  }
  if (!/[a-z]/.test(password) || !/[A-Z]/.test(password) || !/[0-9]/.test(password)) {
    errors.push('Password must include uppercase, lowercase, and a number.');
  }
  if (!acceptedTerms) {
    errors.push('You must accept the Terms and Privacy Policy.');
  }
  if (!allowReserved && !emailValidation.ok) {
    errors.push('A valid email address is required.');
  }

  return {
    ok: errors.length === 0,
    errors,
    username,
    usernameKey: normalizeUsername(username),
    email: emailValidation.email,
    emailKey: emailValidation.emailKey
  };
}

export function jerusalemDay(timestampMs = Date.now()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Jerusalem',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).formatToParts(new Date(timestampMs));
  const value = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return `${value.year}-${value.month}-${value.day}`;
}

export function sessionCookie(token, maxAge = SESSION_MAX_AGE_SECONDS) {
  return `btc2g_session=${token}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`;
}

export function clearSessionCookie() {
  return 'btc2g_session=; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=0';
}

export function parseCookies(header = '') {
  return Object.fromEntries(String(header || '').split(';').map(part => part.trim()).filter(Boolean).map(part => {
    const separator = part.indexOf('=');
    return separator === -1 ? [part, ''] : [part.slice(0, separator), part.slice(separator + 1)];
  }));
}

export function publicStats(totalAccounts, newToday, goal = 100) {
  const total = Number(totalAccounts) || 0;
  const today = Number(newToday) || 0;
  return {
    totalAccounts: total,
    newToday: today,
    dailyGoal: goal,
    remainingToday: Math.max(0, goal - today),
    countingMethod: 'Email-verified D1 account rows; no estimated or fabricated users'
  };
}

export function normalizePagePath(value) {
  const path = String(value || '').trim();
  if (!path.startsWith('/') || path.startsWith('//') || path.length > 256) return null;
  return path.split(/[?#]/, 1)[0] || '/';
}

export function acquisitionSource(referrer) {
  try {
    const url = new URL(String(referrer || ''));
    if (url.protocol !== 'https:') return null;
    const host = url.hostname.toLowerCase();
    return host === 'producthunt.com' || host.endsWith('.producthunt.com') ? 'producthunt' : null;
  } catch {
    return null;
  }
}
