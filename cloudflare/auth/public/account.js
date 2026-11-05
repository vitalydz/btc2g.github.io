import { formatCount, measurementNote, request } from './metrics-view.js';
import { authenticatePasskey } from './passkey-client.js';
import { accountLanguage, applyAccountLanguage, accountText } from './account-language.js';

const form = document.querySelector('#account-form');
const message = document.querySelector('#message');
const submitButton = document.querySelector('#submit-button');
const termsRow = document.querySelector('#terms-row');
const termsCheckbox = document.querySelector('#accepted-terms');
const usernameInput = document.querySelector('#username');
const registrationEmailRow = document.querySelector('#registration-email-row');
const registrationEmail = document.querySelector('#registration-email');
const password = document.querySelector('#password');
const registrationPasswordPattern = password.getAttribute('pattern');
const registrationPasswordRules = password.getAttribute('passwordrules');
const togglePassword = document.querySelector('#toggle-password');
const passwordHint = document.querySelector('#password-hint');
const signedIn = document.querySelector('#signed-in');
const emailForm = document.querySelector('#email-form');
const emailInput = document.querySelector('#account-email');
const emailSubmit = document.querySelector('#email-submit');
const emailMessage = document.querySelector('#email-message');
const ideaPanel = document.querySelector('#idea-panel');
const ideaForm = document.querySelector('#idea-form');
const ideaSubmit = document.querySelector('#idea-submit');
const ideaMessage = document.querySelector('#idea-message');
const ideaAccess = document.querySelector('#idea-access');
const ideaList = document.querySelector('#idea-list');
const refreshStatsButton = document.querySelector('#refresh-stats');
let mode = 'register';
let ideaSubmissionKey = '';

applyAccountLanguage();

togglePassword.addEventListener('click', () => {
  const visible = password.type === 'text';
  password.type = visible ? 'password' : 'text';
  togglePassword.textContent = visible ? accountText('show') : accountText('hide');
  togglePassword.setAttribute('aria-pressed', String(!visible));
  password.focus();
});

function setMessage(text, success = false) {
  message.textContent = text;
  message.classList.toggle('success', success);
}

async function refreshStats() {
  if (refreshStatsButton.disabled) return;
  refreshStatsButton.disabled = true;
  refreshStatsButton.setAttribute('aria-busy', 'true');
  const subscribersCard = document.querySelector('#subscribers-card');
  try {
    const data = await request('/api/stats');
    const note = measurementNote(data);
    document.querySelector('#today-count').textContent = formatCount(data.newToday);
    document.querySelector('#total-count').textContent = formatCount(data.totalAccounts);
    document.querySelector('#visitors-count').textContent = formatCount(data.traffic?.visitorsToday);
    document.querySelector('#views-count').textContent = formatCount(data.traffic?.viewsToday);
    const subscriberTotal = data.confirmedSubscribers?.status === 'available'
      ? data.confirmedSubscribers.total : null;
    subscribersCard.hidden = !Number.isInteger(subscriberTotal) || subscriberTotal < 0;
    document.querySelector('#subscribers-count').textContent = subscribersCard.hidden ? '—' : formatCount(subscriberTotal);
    document.querySelector('#stats-note').textContent = accountLanguage === 'ru'
      ? `${data.source} · Отчётный день: ${data.day} (${data.timeZone}) · Обновлено: ${new Date(data.measuredAt).toLocaleString('ru-RU')}. Посетители — суточная оценка браузеров/IP, а не подтверждённые люди. Аккаунты не являются email-подписками.`
      : `${note} Visitors are daily browser/IP estimates, not verified people. Accounts are not email subscriptions.`;
  } catch {
    subscribersCard.hidden = true;
    document.querySelectorAll('.stats [data-count]').forEach(item => { item.textContent = accountText('dataUnavailable'); });
    document.querySelector('#stats-note').textContent = accountText('statsUnavailable');
  } finally {
    refreshStatsButton.disabled = false;
    refreshStatsButton.removeAttribute('aria-busy');
  }
}

refreshStatsButton.addEventListener('click', refreshStats);

function renderIdeas(ideas) {
  if (!ideas.length) {
    const empty = document.createElement('li');
    empty.textContent = accountLanguage === 'ru' ? 'Идей пока нет.' : 'No ideas submitted yet.';
    ideaList.replaceChildren(empty);
    return;
  }
  ideaList.replaceChildren(...ideas.map(idea => {
    const item = document.createElement('li');
    const heading = document.createElement('strong');
    heading.textContent = idea.title;
    const status = document.createElement('span');
    status.textContent = idea.status;
    const date = document.createElement('small');
    date.textContent = new Date(idea.createdAt * 1000).toLocaleDateString();
    item.append(heading, status, date);
    if (idea.moderationNote) {
      const note = document.createElement('p');
      note.textContent = `${accountLanguage === 'ru' ? 'Комментарий BTC2G' : 'BTC2G note'}: ${idea.moderationNote}`;
      item.append(note);
    }
    return item;
  }));
}

async function loadMyIdeas() {
  const data = await request('/api/ideas/mine');
  renderIdeas(data.ideas);
}

function showUser(user) {
  form.hidden = true;
  document.querySelector('.tabs').hidden = true;
  signedIn.hidden = false;
  document.querySelector('#signed-in-name').textContent = user.username;
  document.querySelector('#admin-link').hidden = user.role !== 'admin';
  const displayedEmail = user.email || user.pendingEmail;
  document.querySelector('#email-status').textContent = user.emailVerified && user.email
    ? `${accountText('emailVerified')}: ${user.email}`
    : displayedEmail
      ? `${accountText('emailPending')}: ${displayedEmail}`
      : accountText('emailMissing');
  if (displayedEmail) emailInput.value = displayedEmail;
  const isAdmin = user.role === 'admin';
  ideaPanel.hidden = false;
  if (user.role === 'user' || isAdmin) {
    ideaForm.hidden = !isAdmin && !user.emailVerified;
    ideaAccess.textContent = isAdmin
      ? accountLanguage === 'ru' ? 'Администраторы могут отправлять идеи без ограничений. Откройте панель администратора выше, чтобы просмотреть идеи всех пользователей.' : 'Administrators can submit unlimited ideas. Open the admin dashboard above to review all users’ ideas.'
      : user.emailVerified
      ? accountLanguage === 'ru' ? 'Ваш подтверждённый аккаунт может отправлять идеи приватно.' : 'Your verified account can submit ideas privately.'
      : accountLanguage === 'ru' ? 'Подтвердите email выше, чтобы открыть приватную отправку идей.' : 'Verify your email above to unlock private idea submission.';
    loadMyIdeas().catch(error => { ideaMessage.textContent = error.message; });
  }
}

ideaForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (ideaSubmit.disabled) return;
  ideaSubmit.disabled = true;
  ideaMessage.textContent = '';
  ideaMessage.classList.remove('success');
  const data = new FormData(ideaForm);
  if (!ideaSubmissionKey) ideaSubmissionKey = crypto.randomUUID();
  try {
    await request('/api/ideas', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'Idempotency-Key': ideaSubmissionKey },
      body: JSON.stringify({
        title: data.get('title'), audience: data.get('audience'),
        problem: data.get('problem'), solution: data.get('solution'),
        reviewConsent: data.get('reviewConsent') === 'on'
      })
    });
    ideaSubmissionKey = crypto.randomUUID();
    ideaForm.reset();
    ideaMessage.textContent = accountLanguage === 'ru' ? 'Идея отправлена приватно и не отображается на публичном сайте.' : 'Idea submitted privately. It is not visible on the public site.';
    ideaMessage.classList.add('success');
    try { await loadMyIdeas(); } catch {
      ideaMessage.textContent += accountLanguage === 'ru' ? ' Не удалось обновить список. Перезагрузите страницу, чтобы увидеть сохранённые идеи.' : ' Your list could not be refreshed. Reload the page to view saved ideas.';
    }
  } catch (error) {
    ideaMessage.textContent = error.status === 409
      ? accountLanguage === 'ru' ? 'Вы уже отправили 3 идеи. Они показаны ниже.' : 'You have already submitted 3 ideas. View your submissions below.'
      : !error.status || error.status >= 500
        ? accountLanguage === 'ru' ? 'Не удалось сохранить идею. Текст сохранён в форме; повторите попытку.' : 'Your idea could not be saved. Your text is preserved; please try again.'
        : error.message;
  } finally { ideaSubmit.disabled = false; }
});

emailForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (emailSubmit.disabled) return;
  emailSubmit.disabled = true;
  emailForm.setAttribute('aria-busy', 'true');
  emailMessage.textContent = '';
  emailMessage.classList.remove('success');
  try {
    await request('/api/email-verification/request', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: emailInput.value, language: accountLanguage })
    });
    emailMessage.textContent = accountLanguage === 'ru' ? 'Письмо подтверждения отправлено. Ссылка действует 30 минут.' : 'Verification email sent. The link expires in 30 minutes.';
    emailMessage.classList.add('success');
  } catch (error) {
    emailMessage.textContent = error.message;
  } finally {
    emailSubmit.disabled = false;
    emailForm.removeAttribute('aria-busy');
  }
});

const modeTabs = [...document.querySelectorAll('.tab')];
modeTabs.forEach(tab => tab.addEventListener('click', () => {
  mode = tab.dataset.mode;
  const signingIn = mode === 'login';
  if (typeof history !== 'undefined' && typeof location !== 'undefined') {
    const url = new URL(location.href);
    if (signingIn) url.searchParams.set('mode', 'login');
    else url.searchParams.delete('mode');
    history.replaceState(null, '', url);
  }
  modeTabs.forEach(item => {
    const active = item === tab;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
  });
  form.setAttribute('aria-labelledby', tab.id);
  termsRow.hidden = signingIn;
  document.querySelector('.registration-steps').hidden = signingIn;
  document.querySelector('#account-trust').hidden = signingIn;
  document.querySelector('#account-benefit').hidden = signingIn;
  document.querySelector('#account-title').textContent = accountText(signingIn ? 'loginTitle' : 'title');
  document.querySelector('#account-intro').textContent = accountText(signingIn ? 'loginIntro' : 'intro');
  document.querySelector('#registration-support').textContent = accountText(signingIn ? 'loginSupport' : 'registrationSupport');
  document.title = accountText(signingIn ? 'loginTitle' : 'title');
  termsCheckbox.required = mode === 'register';
  registrationEmailRow.hidden = signingIn;
  registrationEmail.required = mode === 'register';
  password.type = 'password';
  togglePassword.setAttribute('aria-pressed', 'false');
  togglePassword.textContent = accountText('show');
  password.autocomplete = mode === 'register' ? 'new-password' : 'current-password';
  if (mode === 'register') {
    password.setAttribute('pattern', registrationPasswordPattern);
    password.setAttribute('passwordrules', registrationPasswordRules);
  } else {
    password.removeAttribute('pattern');
    password.removeAttribute('passwordrules');
  }
  passwordHint.textContent = mode === 'register' ? accountText('passwordHint') : accountText('loginPasswordHint');
  submitButton.textContent = mode === 'register' ? accountText('create') : accountText('signIn');
  setMessage('');
  usernameInput.focus();
}));

document.querySelector('#account-tabs').addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const activeIndex = modeTabs.findIndex(tab => tab.getAttribute('aria-selected') === 'true');
  const rtlDirection = document.documentElement.dir === 'rtl' ? -1 : 1;
  const nextIndex = event.key === 'Home' ? 0
    : event.key === 'End' ? modeTabs.length - 1
    : (activeIndex + (event.key === 'ArrowRight' ? rtlDirection : -rtlDirection) + modeTabs.length) % modeTabs.length;
  modeTabs[nextIndex].click();
  modeTabs[nextIndex].focus();
});

if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('mode') === 'login') {
  document.querySelector('#login-tab').click();
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (submitButton.disabled) return;
  submitButton.disabled = true;
  form.setAttribute('aria-busy', 'true');
  setMessage('');
  try {
    const payload = {
      username: form.username.value,
      email: mode === 'register' ? registrationEmail.value : undefined,
      password: form.password.value,
      acceptedTerms: mode === 'register' ? termsCheckbox.checked : undefined,
      attribution: mode === 'register' ? new URLSearchParams(location.search).get('attribution') : undefined,
      language: accountLanguage
    };
    let data = await request(`/api/${mode}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (data.passkeyRequired) {
      setMessage(accountLanguage === 'ru' ? 'Подтвердите вход администратора с помощью passkey.' : 'Confirm this administrator sign-in with your passkey.');
      const credential = await authenticatePasskey(data.options);
      data = await request('/api/passkeys/authenticate', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ challengeId: data.challengeId, credential })
      });
    }
    showUser(data.user);
    if (mode === 'register') {
      emailMessage.textContent = accountText(data.verificationEmailSent ? 'verificationSent' : 'verificationFailed');
      emailMessage.classList.toggle('success', data.verificationEmailSent);
    }
    await refreshStats();
  } catch (error) {
    setMessage(error.message);
  } finally {
    submitButton.disabled = false;
    form.removeAttribute('aria-busy');
  }
});

document.querySelector('#logout-button').addEventListener('click', async () => {
  await request('/api/logout', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  location.reload();
});

refreshStats();
request('/api/me').then(data => data.user && showUser(data.user)).catch(() => {});

