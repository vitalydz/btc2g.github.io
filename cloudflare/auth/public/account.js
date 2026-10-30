import { formatCount, measurementNote, request } from './metrics-view.js';
import { authenticatePasskey } from './passkey-client.js';

const form = document.querySelector('#account-form');
const message = document.querySelector('#message');
const submitButton = document.querySelector('#submit-button');
const termsRow = document.querySelector('#terms-row');
const termsCheckbox = document.querySelector('#accepted-terms');
const registrationEmailRow = document.querySelector('#registration-email-row');
const registrationEmail = document.querySelector('#registration-email');
const password = document.querySelector('#password');
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
let mode = 'register';
let ideaSubmissionKey = '';

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

function renderIdeas(ideas) {
  if (!ideas.length) {
    const empty = document.createElement('li');
    empty.textContent = 'No ideas submitted yet.';
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
      note.textContent = `BTC2G note: ${idea.moderationNote}`;
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
    ? `Verified: ${user.email}`
    : displayedEmail ? `Verification pending for ${displayedEmail}.` : 'No verified email is attached.';
  if (displayedEmail) emailInput.value = displayedEmail;
  ideaPanel.hidden = user.role !== 'user';
  if (user.role === 'user') {
    ideaForm.hidden = !user.emailVerified;
    ideaAccess.textContent = user.emailVerified
      ? 'Your verified account can submit ideas privately.'
      : 'Verify your email above to unlock private idea submission.';
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
    ideaMessage.textContent = 'Idea submitted privately. It is not visible on the public site.';
    ideaMessage.classList.add('success');
    try { await loadMyIdeas(); } catch {
      ideaMessage.textContent += ' Your list could not be refreshed. Reload the page to view saved ideas.';
    }
  } catch (error) {
    ideaMessage.textContent = error.status === 409
      ? 'You have already submitted 3 ideas. View your submissions below.'
      : !error.status || error.status >= 500
        ? 'Your idea could not be saved. Your text is preserved; please try again.'
        : error.message;
  } finally { ideaSubmit.disabled = false; }
});

emailForm.addEventListener('submit', async event => {
  event.preventDefault();
  emailSubmit.disabled = true;
  emailMessage.textContent = '';
  emailMessage.classList.remove('success');
  try {
    await request('/api/email-verification/request', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: emailInput.value })
    });
    emailMessage.textContent = 'Verification email sent. The link expires in 30 minutes.';
    emailMessage.classList.add('success');
  } catch (error) {
    emailMessage.textContent = error.message;
  } finally {
    emailSubmit.disabled = false;
  }
});

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  mode = tab.dataset.mode;
  document.querySelectorAll('.tab').forEach(item => {
    const active = item === tab;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
  });
  termsRow.hidden = mode === 'login';
  termsCheckbox.required = mode === 'register';
  registrationEmailRow.hidden = mode === 'login';
  registrationEmail.required = mode === 'register';
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
      email: mode === 'register' ? registrationEmail.value : undefined,
      password: form.password.value,
      acceptedTerms: mode === 'register' ? termsCheckbox.checked : undefined,
      attribution: mode === 'register' ? new URLSearchParams(location.search).get('attribution') : undefined
    };
    let data = await request(`/api/${mode}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (data.passkeyRequired) {
      setMessage('Confirm this administrator sign-in with your passkey.');
      const credential = await authenticatePasskey(data.options);
      data = await request('/api/passkeys/authenticate', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ challengeId: data.challengeId, credential })
      });
    }
    showUser(data.user);
    if (mode === 'register') {
      emailMessage.textContent = data.verificationEmailSent
        ? 'Verification email sent. Open the link within 30 minutes to complete registration.'
        : 'Your account is pending. We could not send the verification email; use the button above to retry.';
      emailMessage.classList.toggle('success', data.verificationEmailSent);
    }
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

