import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const root = new URL('../public/', import.meta.url);
test('preserves hidden authenticated state and offers useful public tools', async () => {
  const [html, script, css] = await Promise.all(['index.html','account.js','account.css'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(script, /form\.hidden = true/);
  assert.match(css, /\[hidden\]\s*\{\s*display\s*:\s*none\s*!important;\s*\}/);
  assert.match(html, /https:\/\/btc2g.com\/save.html/);
  assert.match(html, /https:\/\/btc2g.com\/index.html#calculator/);
});

test('unavailable subscriber totals stay out of the public account summary', async () => {
  const [html, script, css] = await Promise.all(['index.html', 'account.js', 'account.css'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(html, /id="subscribers-card" hidden/);
  assert.match(script, /confirmedSubscribers\?\.status === 'available'/);
  assert.match(script, /subscribersCard\.hidden = !Number\.isInteger\(subscriberTotal\)/);
  assert.match(css, /:has\(#subscribers-card\[hidden\]\)/);
});

test('administrator dashboard distinguishes accounts from confirmed newsletter subscribers', async () => {
  const adminHtml = await readFile(new URL('admin/index.html', root), 'utf8');
  assert.match(adminHtml, /Administrator accounts and verified account emails do not automatically join the newsletter/);
  assert.match(adminHtml, /https:\/\/btc2g\.com\/#subscribe/);
  assert.match(adminHtml, /confirm the email separately/);
});

test('registration fields explain their accepted format before submission', async () => {
  const [html, script] = await Promise.all(['index.html', 'account.js'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(html, /aria-describedby="username-hint"/);
  assert.match(html, /id="username-hint">3–32 characters/);
  assert.match(html, /aria-describedby="password-hint"/);
  assert.match(html, /id="password-hint">Use at least 12 characters/);
  assert.match(html, /id="registration-email"[^>]+type="email"[^>]+required/);
  assert.match(html, /Registration is complete only after you verify this address/);
  assert.match(script, /email: mode === 'register'/);
  assert.match(script, /accountText\('loginPasswordHint'\)/);
});

test('account page localizes its primary UI in all 19 site languages', async () => {
  const [html, languageScript, homepage] = await Promise.all([
    readFile(new URL('index.html', root), 'utf8'),
    readFile(new URL('account-language.js', root), 'utf8'),
    readFile(new URL('../../../index.html', import.meta.url), 'utf8')
  ]);
  const options = [...html.matchAll(/<option value="([a-z]{2})"/g)].map(match => match[1]);
  assert.equal(options.length, 19);
  const executable = languageScript
    .replaceAll('export let ', 'let ')
    .replaceAll('export function ', 'function ')
    .replace(/export \{[^}]+\};/, '') + '\nwindow.__i18n = { languages, dictionaries };';
  const context = { window:{}, location:{search:'?lang=ru'}, localStorage:{getItem(){return'';}}, navigator:{language:'en'}, URLSearchParams };
  vm.createContext(context);
  vm.runInContext(executable, context);
  assert.equal(context.window.__i18n.languages.length, 19);
  for (const lang of options) {
    const dictionary = context.window.__i18n.dictionaries[lang];
    assert.ok(dictionary, lang);
    assert.ok(Object.values(dictionary).every(value => typeof value === 'string' && value.trim()), lang);
  }
  assert.equal(context.window.__i18n.dictionaries.ru.title, 'Ваш аккаунт BTC2G');
  assert.match(homepage, /btc2g-accounts\.btc2g\.workers\.dev\/\?lang=' \+ encodeURIComponent\(lang\)/);
});

test('password can be revealed without changing registration or sign-in submission', async () => {
  const [html, script, css] = await Promise.all(['index.html', 'account.js', 'account.css'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(html, /id="toggle-password"[^>]+aria-controls="password"[^>]+aria-pressed="false"/);
  assert.match(script, /password\.type = visible \? 'password' : 'text'/);
  assert.match(script, /togglePassword\.setAttribute\('aria-pressed', String\(!visible\)\)/);
  assert.match(script, /password\.focus\(\)/);
  assert.match(css, /\.password-control button\s*\{/);
});

test('registration flow explains that email verification completes the account', async () => {
  const [accountPage, accountCss] = await Promise.all(['index.html', 'account.css'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(accountPage, /class="registration-steps" aria-label="Registration steps"/);
  assert.match(accountPage, /<strong>Create account<\/strong>/);
  assert.match(accountPage, /<strong>Check your inbox<\/strong>/);
  assert.match(accountPage, /Open the verification link within 30 minutes/);
  assert.match(accountPage, /<strong>Email verified<\/strong>/);
  assert.match(accountPage, /Your registration is complete/);
  assert.match(accountCss, /\.registration-steps\s*\{/);
  assert.match(accountCss, /@media \(max-width:560px\)[^{]*\{[^}]*[\s\S]*\.registration-steps\{grid-template-columns:1fr\}/);
});

test('homepage form receipt does not claim confirmation and Premium CTA uses existing form', async () => {
  const source = await readFile(new URL('../../../index.html', import.meta.url), 'utf8');
  assert.equal((source.match(/subscribeSuccess:/g) || []).length, 19);
  assert.doesNotMatch(source, /successEventName: 'email_subscribe'/);
  assert.match(source, /successEventName: 'subscription_request_received'/);
  assert.match(source, /window.location.href = 'premium.html\?v=20261029-21&lang='/);
});

test('Product Hunt attribution is forwarded only through the account registration flow', async () => {
  const [traffic, account] = await Promise.all([
    readFile(new URL('../../../assets/js/traffic.js', import.meta.url), 'utf8'),
    readFile(new URL('account.js', root), 'utf8')
  ]);
  assert.match(traffic, /document\.referrer/);
  assert.match(traffic, /sessionStorage\.setItem\(storageKey, data\.attribution\)/);
  assert.match(traffic, /url\.searchParams\.set\('attribution'/);
  assert.match(account, /attribution: mode === 'register'/);
});

test('signed-in account UI separates email verification from Premium and marketing', async () => {
  const [html, script, verification] = await Promise.all([
    readFile(new URL('index.html', root), 'utf8'),
    readFile(new URL('account.js', root), 'utf8'),
    readFile(new URL('verify-email.html', root), 'utf8')
  ]);
  assert.match(html, /id="email-form"/);
  assert.match(html, /does not subscribe you to marketing or activate Premium/);
  assert.match(html, /https:\/\/btc2g\.com\/#subscribe/);
  assert.match(html, /Join the separate confirmed newsletter and choose the Sunday digest/);
  assert.match(script, /\/api\/email-verification\/request/);
  assert.match(verification, /id="confirm-email"/);
  assert.match(verification, /no-referrer/);
});

test('administrator UI supports passkey enrollment and second-factor sign-in', async () => {
  const [html, adminScript, accountScript, client] = await Promise.all([
    readFile(new URL('admin/index.html', root), 'utf8'),
    readFile(new URL('admin/admin.js', root), 'utf8'),
    readFile(new URL('account.js', root), 'utf8'),
    readFile(new URL('passkey-client.js', root), 'utf8')
  ]);
  assert.match(html, /id="add-passkey"/);
  assert.match(html, /both the password and Windows Hello/);
  assert.match(adminScript, /passkeys\/registration\/options/);
  assert.match(adminScript, /passkeys\/authenticate/);
  assert.match(accountScript, /passkeys\/authenticate/);
  assert.match(client, /navigator\.credentials\.create/);
  assert.match(client, /navigator\.credentials\.get/);
});

test('idea submission is private, email-gated, and available to administrators for review', async () => {
  const [html, accountScript, adminHtml, adminScript] = await Promise.all([
    readFile(new URL('index.html', root), 'utf8'),
    readFile(new URL('account.js', root), 'utf8'),
    readFile(new URL('admin/index.html', root), 'utf8'),
    readFile(new URL('admin/admin.js', root), 'utf8')
  ]);
  assert.match(html, /visible only to you and BTC2G administrators/);
  assert.match(html, /id="idea-form"/);
  assert.match(accountScript, /user\.emailVerified/);
  assert.match(accountScript, /\/api\/ideas\/mine/);
  assert.match(accountScript, /Idempotency-Key/);
  assert.match(adminHtml, /PRIVATE · ADMIN ONLY/);
  assert.match(adminScript, /\/api\/admin\/ideas/);
  assert.doesNotMatch(html, /public idea list/i);
});

test('administrator UI shows the configured application version', async () => {
  const [html, adminScript, worker, config, version] = await Promise.all([
    readFile(new URL('admin/index.html', root), 'utf8'),
    readFile(new URL('admin/admin.js', root), 'utf8'),
    readFile(new URL('../src/worker.js', root), 'utf8'),
    readFile(new URL('../wrangler.jsonc', root), 'utf8'),
    readFile(new URL('../../../VERSION', import.meta.url), 'utf8')
  ]);
  const expected = version.trim();
  assert.match(html, /id="app-version"/);
  assert.match(adminScript, /data\.appVersion/);
  assert.match(worker, /appVersion: env\.APP_VERSION/);
  assert.match(config, new RegExp(`"APP_VERSION": "${expected.replaceAll('.', '\\.') }"`));
});
