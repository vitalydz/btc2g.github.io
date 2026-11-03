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
  assert.match(html, /aria-describedby="username-hint username-privacy"/);
  assert.match(html, /id="username-hint">3–32 characters\. Start with a Latin letter/);
  assert.match(html, /id="username-privacy">Your username is used only for sign-in and is not displayed publicly/);
  assert.match(html, /aria-describedby="password-hint"/);
  assert.match(html, /id="password-hint">Use at least 12 characters/);
  assert.match(html, /id="registration-email"[^>]+type="email"[^>]+required/);
  assert.match(html, /Registration is complete only after you verify this address/);
  assert.match(script, /email: mode === 'register'/);
  assert.match(script, /accountText\('loginPasswordHint'\)/);
  assert.match(script, /accountText\(data\.verificationEmailSent \? 'verificationSent' : 'verificationFailed'\)/);
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
    .replace(/export \{[^}]+\};/, '') + '\nwindow.__i18n = { languages, dictionaries, registrationStepRows, termsRows, extras };';
  const context = { window:{}, location:{search:'?lang=ru'}, localStorage:{getItem(){return'';}}, navigator:{language:'en'}, URLSearchParams };
  vm.createContext(context);
  vm.runInContext(executable, context);
  assert.equal(context.window.__i18n.languages.length, 19);
  for (const lang of options) {
    const dictionary = context.window.__i18n.dictionaries[lang];
    assert.ok(dictionary, lang);
    assert.ok(Object.values(dictionary).every(value => typeof value === 'string' && value.trim()), lang);
    assert.ok(dictionary.trust, lang);
    assert.ok(dictionary.usernamePrivacy, lang);
    assert.ok(dictionary.usernameHint, lang);
    assert.ok(dictionary.emailHint, lang);
    assert.ok(dictionary.verificationSent, lang);
    assert.ok(dictionary.verificationFailed, lang);
    assert.equal(context.window.__i18n.registrationStepRows[lang].length, 7, lang);
    assert.ok(context.window.__i18n.registrationStepRows[lang].every(value => typeof value === 'string' && value.trim()), lang);
    assert.equal(context.window.__i18n.termsRows[lang].length, 4, lang);
    assert.ok(context.window.__i18n.termsRows[lang].every(value => typeof value === 'string' && value.trim()), lang);
    assert.ok(context.window.__i18n.extras[lang].dataUnavailable.trim(), lang);
    assert.ok(context.window.__i18n.extras[lang].statsUnavailable.trim(), lang);
  }
  assert.equal(context.window.__i18n.dictionaries.ru.title, 'Ваш аккаунт BTC2G');
  assert.match(context.window.__i18n.dictionaries.ru.trust, /не подписывает/);
  assert.match(html, /id="account-trust">Email verification does not subscribe you to marketing/);
  assert.match(languageScript, /trust:'account-trust'/);
  assert.match(languageScript, /usernamePrivacy:'username-privacy'/);
  assert.match(languageScript, /usernameHint:'username-hint'/);
  assert.match(languageScript, /emailHint:'registration-email-hint'/);
  assert.match(languageScript, /verificationSent:verificationSentRows\[lang\]/);
  assert.match(languageScript, /verificationFailed:verificationFailedRows\[lang\]/);
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

test('switching registration mode moves focus to the first account field', async () => {
  const account = await readFile(new URL('account.js', root), 'utf8');
  assert.match(account, /const usernameInput = document\.querySelector\('#username'\)/);
  assert.match(account, /usernameInput\.focus\(\)/);
});

test('registration flow explains that email verification completes the account', async () => {
  const [accountPage, accountCss, accountLanguage] = await Promise.all(['index.html', 'account.css', 'account-language.js'].map(path => readFile(new URL(path, root), 'utf8')));
  assert.match(accountPage, /class="registration-steps" aria-label="Registration steps"/);
  assert.match(accountPage, /id="step-create-title">Create account<\/strong>/);
  assert.match(accountPage, /id="submit-button"[^>]*>Create free account<\/button>/);
  assert.match(accountLanguage, /const freeAccountRows = \{en:'Create free account',ru:'Создать бесплатный аккаунт'/);
  assert.match(accountPage, /id="account-title">Create your free BTC2G account<\/h1>/);
  assert.match(accountLanguage, /const freeTitleRows = \{en:'Create your free BTC2G account',ru:'Создайте бесплатный аккаунт BTC2G'/);
  assert.match(accountPage, /id="step-inbox-title">Check your inbox<\/strong>/);
  assert.match(accountPage, /Open the verification link within 30 minutes/);
  assert.match(accountPage, /id="step-verified-title">Email verified<\/strong>/);
  assert.match(accountPage, /Your registration is complete/);
  assert.match(accountLanguage, /const registrationStepRows = \{/);
  assert.match(accountLanguage, /he:\['שלבי ההרשמה'/);
  assert.match(accountLanguage, /registrationStepRows\[accountLanguage\] \|\| registrationStepRows\.en/);
  assert.match(accountPage, /id="terms-prefix">I accept the<\/span>/);
  assert.match(accountLanguage, /const termsRows = \{/);
  assert.match(accountLanguage, /termsRows\[accountLanguage\] \|\| termsRows\.en/);
  assert.match(accountCss, /\.registration-steps\s*\{/);
  assert.match(accountCss, /@media \(max-width:560px\)[^{]*\{[^}]*[\s\S]*\.registration-steps\{grid-template-columns:1fr\}/);
});

test('registration form appears before public progress metrics', async () => {
  const accountPage = await readFile(new URL('index.html', root), 'utf8');
  assert.ok(accountPage.indexOf('id="account-card"') < accountPage.indexOf('class="stats"'));
  assert.ok(accountPage.indexOf('id="account-form"') < accountPage.indexOf('id="today-count"'));
});

test('homepage form receipt does not claim confirmation and Premium CTA uses existing form', async () => {
  const source = await readFile(new URL('../../../index.html', import.meta.url), 'utf8');
  assert.equal((source.match(/subscribeSuccess:/g) || []).length, 19);
  assert.doesNotMatch(source, /successEventName: 'email_subscribe'/);
  assert.match(source, /successEventName: 'subscription_request_received'/);
  assert.match(source, /window.location.href = 'premium.html\?v=20261029-21&lang='/);
});

test('approved acquisition attribution is forwarded only through the account registration flow', async () => {
  const [traffic, account] = await Promise.all([
    readFile(new URL('../../../assets/js/traffic.js', import.meta.url), 'utf8'),
    readFile(new URL('account.js', root), 'utf8')
  ]);
  assert.match(traffic, /document\.referrer/);
  assert.match(traffic, /utm_source.*=== 'telegram'/s);
  assert.match(traffic, /campaignSource/);
  assert.match(traffic, /sessionStorage\.setItem\(storageKey, data\.attribution\)/);
  assert.match(traffic, /url\.searchParams\.set\('attribution'/);
  assert.match(account, /attribution: mode === 'register'/);
  assert.match(account, /language: accountLanguage/);
  assert.match(account, /form\.setAttribute\('aria-busy', 'true'\)/);
  assert.match(account, /form\.removeAttribute\('aria-busy'\)/);
  assert.match(account, /if \(submitButton\.disabled\) return/);
  assert.match(account, /emailForm\.setAttribute\('aria-busy', 'true'\)/);
  assert.match(account, /emailForm\.removeAttribute\('aria-busy'\)/);
  assert.match(account, /if \(emailSubmit\.disabled\) return/);
});

test('signed-in account UI separates email verification from Premium and marketing', async () => {
  const [html, script, verification, verificationScript] = await Promise.all([
    readFile(new URL('index.html', root), 'utf8'),
    readFile(new URL('account.js', root), 'utf8'),
    readFile(new URL('verify-email.html', root), 'utf8'),
    readFile(new URL('verify-email.js', root), 'utf8')
  ]);
  assert.match(html, /id="email-form"/);
  assert.match(html, /id="username"[^>]+autocapitalize="none"[^>]+spellcheck="false"/);
  assert.match(html, /id="username"[^>]+enterkeyhint="next"/);
  assert.match(html, /id="registration-email"[^>]+inputmode="email"/);
  assert.match(html, /id="registration-email"[^>]+enterkeyhint="next"/);
  assert.match(html, /id="account-email"[^>]+aria-describedby="account-email-hint"/);
  assert.match(html, /id="account-email"[^>]+enterkeyhint="send"/);
  assert.match(html, /id="account-email-hint"/);
  assert.match(html, /id="password"[^>]+passwordrules="minlength: 12; maxlength: 128;"/);
  assert.match(html, /id="password"[^>]+enterkeyhint="go"/);
  assert.match(html, /does not subscribe you to marketing or activate Premium/);
  assert.match(html, /https:\/\/btc2g\.com\/#subscribe/);
  assert.match(html, /Join the separate confirmed newsletter and choose the Sunday digest/);
  assert.match(script, /\/api\/email-verification\/request/);
  assert.match(verification, /id="confirm-email"/);
  assert.match(verification, /no-referrer/);
  assert.match(verificationScript, /import \{ accountLanguage \}/);
  for (const language of ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi']) {
    assert.match(verificationScript, new RegExp(`\\b${language}:\\[`));
  }
  assert.match(verificationScript, /document\.documentElement\.dir = \['he','ar'\]/);
  assert.match(verificationScript, /accountUrl = `\/\?lang=\$\{encodeURIComponent\(accountLanguage\)\}`/);
  assert.match(verificationScript, /querySelector\('#verify-return'\)\.href = accountUrl/);
  assert.match(verificationScript, /error\.status === 400 \? text\[6\] : error\.status === 401 \? text\[8\] : error\.message/);
  assert.match(verificationScript, /querySelector\('#verify-return'\)\.focus\(\)/);
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
  assert.match(config, /"run_worker_first": true/);
});

test('account statistics refresh prevents duplicate requests and announces loading', async () => {
  const account = await readFile(new URL('../public/account.js', import.meta.url), 'utf8');
  assert.match(account, /if \(refreshStatsButton\.disabled\) return/);
  assert.match(account, /refreshStatsButton\.setAttribute\('aria-busy', 'true'\)/);
  assert.match(account, /refreshStatsButton\.removeAttribute\('aria-busy'\)/);
});
