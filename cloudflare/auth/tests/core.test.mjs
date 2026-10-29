import assert from 'node:assert/strict';
import test from 'node:test';
import {
  clearSessionCookie,
  jerusalemDay,
  normalizePagePath,
  normalizeUsername,
  parseCookies,
  publicStats,
  sessionCookie,
  validateRegistration
} from '../src/core.js';

test('normalizes usernames without changing the display value', () => {
  assert.equal(normalizeUsername('  Codex_Admin '), 'codex_admin');
});

test('validates public registration and protects reserved usernames', () => {
  const valid = validateRegistration({ username: 'Satoshi_21', password: 'LongEnoughPass9', acceptedTerms: true });
  assert.equal(valid.ok, true);
  assert.equal(valid.usernameKey, 'satoshi_21');

  const reserved = validateRegistration({ username: 'Codex_admin', password: 'LongEnoughPass9', acceptedTerms: true });
  assert.equal(reserved.ok, false);
  assert.match(reserved.errors.join(' '), /reserved/);
});

test('requires strong-enough passwords and explicit terms acceptance', () => {
  const result = validateRegistration({ username: 'valid_user', password: 'weak', acceptedTerms: false });
  assert.equal(result.ok, false);
  assert.equal(result.errors.length, 3);
});

test('calculates the registration day in Asia/Jerusalem', () => {
  assert.equal(jerusalemDay(Date.parse('2026-10-28T22:30:00Z')), '2026-10-29');
});

test('serializes and parses session cookies safely', () => {
  assert.match(sessionCookie('abc'), /^btc2g_session=abc; HttpOnly; Secure; SameSite=Lax/);
  assert.match(clearSessionCookie(), /Max-Age=0/);
  assert.deepEqual(parseCookies('a=1; btc2g_session=abc.def; empty='), {
    a: '1', btc2g_session: 'abc.def', empty: ''
  });
  assert.deepEqual(parseCookies(null), {});
});

test('reports only stored account totals and a non-negative goal remainder', () => {
  assert.deepEqual(publicStats(7, 3), {
    totalAccounts: 7,
    newToday: 3,
    dailyGoal: 100,
    remainingToday: 97,
    countingMethod: 'D1 account rows; no estimated or fabricated users'
  });
  assert.equal(publicStats(140, 120).remainingToday, 0);
});

test('accepts only bounded same-site page paths for traffic metrics', () => {
  assert.equal(normalizePagePath('/bitcoin-gold-ratio.html?campaign=x'), '/bitcoin-gold-ratio.html');
  assert.equal(normalizePagePath('/'), '/');
  assert.equal(normalizePagePath('https://example.com/'), null);
  assert.equal(normalizePagePath('//example.com'), null);
  assert.equal(normalizePagePath(`/${'x'.repeat(300)}`), null);
});
