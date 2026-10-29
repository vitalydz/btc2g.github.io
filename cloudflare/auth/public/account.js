import { formatCount, measurementNote, request } from './metrics-view.js';

const form = document.querySelector('#account-form');
const message = document.querySelector('#message');
const submitButton = document.querySelector('#submit-button');
const termsRow = document.querySelector('#terms-row');
const termsCheckbox = document.querySelector('#accepted-terms');
const password = document.querySelector('#password');
const passwordHint = document.querySelector('#password-hint');
const signedIn = document.querySelector('#signed-in');
let mode = 'register';

function setMessage(text, success = false) {
  message.textContent = text;
  message.classList.toggle('success', success);
}

async function refreshStats() {
  try {
    const data = await request('/api/stats');
    const note = measurementNote(data);
    document.querySelector('#today-count').textContent = formatCount(data.newToday);
    document.querySelector('#total-count').textContent = formatCount(data.totalAccounts);
    document.querySelector('#visitors-count').textContent = formatCount(data.traffic?.visitorsToday);
    document.querySelector('#views-count').textContent = formatCount(data.traffic?.viewsToday);
    document.querySelector('#subscribers-count').textContent = formatCount(data.confirmedSubscribers?.total);
    document.querySelector('#stats-note').textContent = `${note} Visitors are daily browser/IP estimates, not verified people. Accounts are not email subscriptions.`;
  } catch {
    document.querySelectorAll('.stats [data-count]').forEach(item => { item.textContent = 'Data unavailable'; });
    document.querySelector('#stats-note').textContent = 'Data unavailable. The statistics service could not be reached. Retry below.';
  }
}

document.querySelector('#refresh-stats').addEventListener('click', refreshStats);

function showUser(user) {
  form.hidden = true;
  document.querySelector('.tabs').hidden = true;
  signedIn.hidden = false;
  document.querySelector('#signed-in-name').textContent = user.username;
  document.querySelector('#admin-link').hidden = user.role !== 'admin';
}

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  mode = tab.dataset.mode;
  document.querySelectorAll('.tab').forEach(item => {
    const active = item === tab;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
  });
  termsRow.hidden = mode === 'login';
  termsCheckbox.required = mode === 'register';
  password.autocomplete = mode === 'register' ? 'new-password' : 'current-password';
  passwordHint.textContent = mode === 'register' ? 'Use at least 12 characters.' : 'Enter the password for this account.';
  submitButton.textContent = mode === 'register' ? 'Create account' : 'Sign in';
  setMessage('');
}));

form.addEventListener('submit', async event => {
  event.preventDefault();
  submitButton.disabled = true;
  setMessage('');
  try {
    const payload = {
      username: form.username.value,
      password: form.password.value,
      acceptedTerms: mode === 'register' ? termsCheckbox.checked : undefined,
      attribution: mode === 'register' ? new URLSearchParams(location.search).get('attribution') : undefined
    };
    const data = await request(`/api/${mode}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    setMessage(mode === 'register' ? 'Account created.' : 'Signed in.', true);
    showUser(data.user);
    await refreshStats();
  } catch (error) {
    setMessage(error.message);
  } finally {
    submitButton.disabled = false;
  }
});

document.querySelector('#logout-button').addEventListener('click', async () => {
  await request('/api/logout', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  location.reload();
});

refreshStats();
request('/api/me').then(data => data.user && showUser(data.user)).catch(() => {});
