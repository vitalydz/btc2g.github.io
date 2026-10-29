const form = document.querySelector('#account-form');
const message = document.querySelector('#message');
const submitButton = document.querySelector('#submit-button');
const termsRow = document.querySelector('#terms-row');
const termsCheckbox = document.querySelector('#accepted-terms');
const password = document.querySelector('#password');
const signedIn = document.querySelector('#signed-in');
let mode = 'register';

function setMessage(text, success = false) {
  message.textContent = text;
  message.classList.toggle('success', success);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}

async function refreshStats() {
  try {
    const data = await request('/api/stats');
    document.querySelector('#today-count').textContent = data.newToday.toLocaleString();
    document.querySelector('#remaining-count').textContent = data.remainingToday.toLocaleString();
    document.querySelector('#total-count').textContent = data.totalAccounts.toLocaleString();
    document.querySelector('#stats-note').textContent = `Live D1 rows for ${data.day} (${data.timeZone}); no estimated users.`;
  } catch {
    document.querySelector('#stats-note').textContent = 'Live totals are temporarily unavailable.';
  }
}

function showUser(user) {
  form.hidden = true;
  document.querySelector('.tabs').hidden = true;
  signedIn.hidden = false;
  document.querySelector('#signed-in-name').textContent = user.username;
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
      acceptedTerms: mode === 'register' ? termsCheckbox.checked : undefined
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
