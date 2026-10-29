const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const root = path.join(__dirname, '..');
const helper = fs.readFileSync(path.join(root, 'assets/js/subscription-request.js'), 'utf8');
const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const premium = fs.readFileSync(path.join(root, 'premium.html'), 'utf8');

function harness(page, response) {
  const timers = new Map();
  const calls = [];
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: 'person@example.com', disabled: false, textContent: '', lang: 'ru',
      attributes: {}, classList: { toggle() {} },
      setAttribute(key, value) { this.attributes[key] = value; },
      removeAttribute(key) { delete this.attributes[key]; },
      focus() { this.focused = true; }
    });
    return elements.get(id);
  };
  const text = {
    subscribeEmpty: 'empty', subscribeInvalid: 'invalid', subscribeSending: 'sending',
    subscribeError: 'error', subscribeSuccess: 'received', subscribeButton: 'Subscribe'
  };
  const context = vm.createContext({
    window: {}, document: { getElementById: element }, AbortController,
    console: { log() {} }, currentLang: 'ru', SUBSCRIPTION_ENDPOINT: 'https://example.invalid/never-requested',
    getCurrentText: () => text, trackEvent() {},
    setTimeout(callback, delay) { assert.equal(delay, 15000); const id = {}; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, options) { calls.push({url, options}); return response(options); },
    FormData: class { constructor(form) { this.form = form; } }
  });
  vm.runInContext(helper, context);
  context.window.BTC2GPremiumText = { invalid: 'invalid', sending: 'sending', error: 'error', received: 'received', button: 'Join' };
  const source = page === 'home' ? homepage : premium;
  const end = page === 'home' ? '    function attachSubscriptionForm' : '    function getSharePayload';
  vm.runInContext(source.slice(source.indexOf('    function isValidEmail'), source.indexOf(end)), context);
  const email = element(page === 'home' ? 'subscriber-email' : 'early-access-email');
  const button = element(page === 'home' ? 'subscribe-button' : 'early-access-button');
  const status = element(page === 'home' ? 'subscribe-message' : 'form-status');
  const form = { reset() { email.value = ''; } };
  const submit = () => page === 'home'
    ? context.submitEmailSubscription({ emailInputId: 'subscriber-email', buttonId: 'subscribe-button', messageId: 'subscribe-message', buttonTextKey: 'subscribeButton', successTextKey: 'subscribeSuccess', showThankYou: true, source: 'test' })
    : context.submitEarlyAccess({ preventDefault() {}, currentTarget: form });
  return { context, timers, calls, email, button, status, submit };
}

for (const page of ['home', 'premium']) {
  test(`${page}: timeout restores the form, retains email and never retries`, async () => {
    const h = harness(page, () => new Promise(() => {}));
    const pending = h.submit();
    assert.equal(h.button.disabled, true);
    await h.submit(); // A repeated event during submission must not create a second request.
    assert.equal(h.calls.length, 1);
    [...h.timers.values()][0]();
    await pending;
    assert.equal(h.calls[0].options.signal.aborted, true);
    assert.equal(h.button.disabled, false);
    assert.equal(h.email.value, 'person@example.com');
    assert.match(h.status.textContent, /Не удалось подтвердить получение заявки/);
    assert.equal(h.timers.size, 0);
    assert.equal(h.calls.length, 1);
  });

  test(`${page}: success clears email and timeout; late timeout cannot replace receipt`, async () => {
    const h = harness(page, async () => ({ok: true}));
    await h.submit();
    assert.equal(h.email.value, '');
    assert.equal(h.status.textContent, 'received');
    if (page === 'home') assert.equal(h.context.document.getElementById('subscribe-thanks').hidden, false);
    assert.equal(h.button.disabled, false);
    assert.equal(h.timers.size, 0);
  });

  test(`${page}: HTTP and network failures restore the form without losing email`, async () => {
    for (const response of [async () => ({ok: false, status: 503}), async () => { throw new TypeError('Network unavailable'); }]) {
      const h = harness(page, response);
      await h.submit();
      assert.equal(h.button.disabled, false);
      assert.equal(h.email.value, 'person@example.com');
      assert.equal(h.status.textContent, 'error');
      assert.equal(h.timers.size, 0);
      assert.equal(h.calls.length, 1);
    }
  });

  test(`${page}: invalid email is marked and focused without sending`, async () => {
    const h = harness(page, () => { throw new Error('Must not send'); });
    h.email.value = 'invalid';
    await h.submit();
    assert.equal(h.email.attributes['aria-invalid'], 'true');
    assert.equal(h.email.focused, true);
    assert.equal(h.calls.length, 0);
  });
}

test('homepage email label and timeout message cover every offered language', () => {
  const h = harness('home', async () => ({ok: true}));
  assert.match(homepage, /<label id="subscriber-email-label" for="subscriber-email">Email address<\/label>/);
  assert.match(homepage, /'subscriber-email-label'\).textContent = window.BTC2GSubscription.text\(currentLang\).email/);
  for (const [, language] of homepage.matchAll(/<option value="([a-z]{2})">/g)) {
    const text = h.context.window.BTC2GSubscription.text(language);
    assert.ok(text.email);
    assert.ok(text.uncertain);
    if (language !== 'en') assert.notEqual(text.email, 'Email address');
  }
});
