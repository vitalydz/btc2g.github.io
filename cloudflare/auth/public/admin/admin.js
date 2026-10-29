import { formatCount, measurementNote, percentage, request } from '../metrics-view.js';
import { authenticatePasskey, createPasskey } from '../passkey-client.js';

const loginPanel = document.querySelector('#login-panel');
const dashboard = document.querySelector('#dashboard');
const loginMessage = document.querySelector('#login-message');
let requestVersion = 0;

function showAdmin(user) {
  loginPanel.hidden = true;
  dashboard.hidden = false;
  document.querySelector('#admin-name').textContent = user.username;
  document.querySelector('#logout').hidden = false;
}

function renderPasskeys(passkeys) {
  const list = document.querySelector('#passkey-list');
  if (!passkeys.length) {
    const item = document.createElement('li');
    item.textContent = 'No passkey enrolled — administrator sign-in is still password-only.';
    list.replaceChildren(item);
    return;
  }
  list.replaceChildren(...passkeys.map(passkey => {
    const item = document.createElement('li');
    item.textContent = `${passkey.name}${passkey.backedUp ? ' · synced' : ''}`;
    const date = document.createElement('span');
    date.textContent = new Date(passkey.createdAt * 1000).toLocaleDateString();
    item.append(date);
    return item;
  }));
}

async function loadPasskeys() {
  const data = await request('/api/passkeys');
  renderPasskeys(data.passkeys);
}

function renderPrivateIdeas(ideas) {
  const queue = document.querySelector('#idea-queue');
  if (!ideas.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No private ideas have been submitted.';
    queue.replaceChildren(empty);
    return;
  }
  queue.replaceChildren(...ideas.map(idea => {
    const card = document.createElement('article');
    card.className = 'idea-card';
    const heading = document.createElement('h3');
    heading.textContent = idea.title;
    const meta = document.createElement('p');
    meta.className = 'idea-meta';
    meta.textContent = `${idea.username} · ${new Date(idea.createdAt * 1000).toLocaleString()} · ${idea.status}`;
    const audience = document.createElement('p');
    audience.textContent = `Users: ${idea.audience}`;
    const problem = document.createElement('p');
    problem.textContent = `Problem: ${idea.problem}`;
    const solution = document.createElement('p');
    solution.textContent = `Proposed solution: ${idea.solution}`;
    const note = document.createElement('textarea');
    note.maxLength = 500;
    note.rows = 3;
    note.placeholder = 'Private note to the author (required when declining)';
    note.value = idea.moderationNote || '';
    const actions = document.createElement('div');
    actions.className = 'idea-actions';
    for (const [status, label] of [['reviewing', 'Mark reviewing'], ['accepted', 'Accept privately'], ['declined', 'Decline']]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.addEventListener('click', () => reviewIdea(idea.id, status, note.value, button));
      actions.append(button);
    }
    card.append(heading, meta, audience, problem, solution, note, actions);
    return card;
  }));
}

async function loadPrivateIdeas() {
  const message = document.querySelector('#ideas-message');
  message.textContent = 'Loading private ideas…';
  try {
    const data = await request('/api/admin/ideas');
    renderPrivateIdeas(data.ideas);
    message.textContent = `${data.ideas.length} private submission${data.ideas.length === 1 ? '' : 's'}.`;
  } catch (error) {
    document.querySelector('#idea-queue').replaceChildren();
    message.textContent = error.message;
  }
}

async function reviewIdea(ideaId, status, note, button) {
  button.disabled = true;
  const message = document.querySelector('#ideas-message');
  try {
    await request('/api/admin/ideas/review', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ideaId, status, note })
    });
    await loadPrivateIdeas();
  } catch (error) { message.textContent = error.message; }
  finally { button.disabled = false; }
}

async function loadStats() {
  const version = ++requestVersion;
  const status = document.querySelector('#stats-status');
  status.textContent = 'Loading statistics…';
  try {
    const data = await request(`/api/admin/stats?days=${document.querySelector('#range').value}`);
    if (version !== requestVersion) return;
    const note = measurementNote(data);
    document.querySelector('#visitors-today').textContent = formatCount(data.summary.visitorsToday);
    document.querySelector('#views-today').textContent = formatCount(data.summary.viewsToday);
    document.querySelector('#registered-today').textContent = formatCount(data.summary.registeredToday);
    document.querySelector('#registered-total').textContent = formatCount(data.summary.registeredTotal);
    document.querySelector('#verified-email-total').textContent = formatCount(data.summary.verifiedEmailTotal);
    document.querySelector('#pending-email-total').textContent = formatCount(data.summary.pendingEmailTotal);
    document.querySelector('#confirmed-subscribers').textContent = formatCount(data.confirmedSubscribers?.total);
    document.querySelector('#conversion-today').textContent = percentage(data.summary.conversionToday);
    document.querySelector('#producthunt-visitors').textContent = formatCount(data.summary.productHuntVisitorsToday);
    document.querySelector('#producthunt-registrations').textContent = formatCount(data.summary.productHuntRegisteredToday);
    document.querySelector('#producthunt-conversion').textContent = percentage(data.summary.productHuntConversionToday);
    document.querySelector('#daily-body').replaceChildren(...data.daily.map(row => {
      const tr = document.createElement('tr');
      const conversion = Number.isFinite(row.visitors) && row.visitors > 0 ? row.registrations / row.visitors : null;
      const productHuntConversion = Number.isFinite(row.producthunt_visitors) && row.producthunt_visitors > 0
        ? row.producthunt_registrations / row.producthunt_visitors : null;
      [row.day, formatCount(row.visitors), formatCount(row.views), formatCount(row.registrations), percentage(conversion),
        formatCount(row.producthunt_visitors), formatCount(row.producthunt_registrations), percentage(productHuntConversion)].forEach(value => {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      });
      return tr;
    }));
    document.querySelector('#top-paths').replaceChildren(...data.topPaths.map(row => {
      const li = document.createElement('li'); li.textContent = row.path;
      const span = document.createElement('span'); span.textContent = formatCount(row.views); li.append(span); return li;
    }));
    document.querySelector('#recent-users').replaceChildren(...data.recentRegistrations.map(row => {
      const li = document.createElement('li'); li.textContent = `${row.username}${row.email_verified ? ' · email verified' : ''}`;
      const span = document.createElement('span'); span.textContent = row.created_day; li.append(span); return li;
    }));
    document.querySelector('#methodology').textContent = data.methodology;
    status.textContent = note;
  } catch (error) {
    if (version !== requestVersion) return;
    document.querySelectorAll('.cards [data-count]').forEach(item => { item.textContent = 'Data unavailable'; });
    document.querySelector('#daily-body').replaceChildren();
    document.querySelector('#top-paths').replaceChildren();
    document.querySelector('#recent-users').replaceChildren();
    document.querySelector('#methodology').textContent = '';
    status.textContent = `Data unavailable. ${error.message} Use Refresh statistics to retry.`;
  }
}

document.querySelector('#login-form').addEventListener('submit', async event => {
  event.preventDefault(); loginMessage.textContent = '';
  const form = event.currentTarget;
  try {
    let data = await request('/api/login', { method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({ username:form.username.value, password:form.password.value }) });
    if (data.passkeyRequired) {
      loginMessage.textContent = 'Confirm sign-in with your passkey.';
      const credential = await authenticatePasskey(data.options);
      data = await request('/api/passkeys/authenticate', {
        method:'POST', headers:{'content-type':'application/json'},
        body:JSON.stringify({ challengeId:data.challengeId, credential })
      });
    }
    if (data.user.role !== 'admin') {
      await request('/api/logout', { method:'POST', headers:{'content-type':'application/json'}, body:'{}' });
      throw new Error('Administrator access required.');
    }
    showAdmin(data.user); await Promise.all([loadStats(), loadPasskeys(), loadPrivateIdeas()]);
  } catch (error) { loginMessage.textContent = error.message; }
});

document.querySelector('#range').addEventListener('change', loadStats);
document.querySelector('#refresh-stats').addEventListener('click', loadStats);
document.querySelector('#refresh-ideas').addEventListener('click', loadPrivateIdeas);
document.querySelector('#add-passkey').addEventListener('click', async event => {
  const button = event.currentTarget;
  const message = document.querySelector('#passkey-message');
  const password = document.querySelector('#passkey-password');
  button.disabled = true; message.textContent = '';
  try {
    const setup = await request('/api/passkeys/registration/options', {
      method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({ password:password.value })
    });
    const credential = await createPasskey(setup.options);
    const result = await request('/api/passkeys/registration/verify', {
      method:'POST', headers:{'content-type':'application/json'},
      body:JSON.stringify({ challengeId:setup.challengeId, credential })
    });
    renderPasskeys(result.passkeys);
    password.value = '';
    message.textContent = 'Passkey added. Future administrator sign-ins now require it.';
    message.classList.add('success');
  } catch (error) {
    message.textContent = error.message;
    message.classList.remove('success');
  } finally { button.disabled = false; }
});
document.querySelector('#logout').addEventListener('click', async () => {
  await request('/api/logout', { method:'POST', headers:{'content-type':'application/json'}, body:'{}' }); location.reload();
});

request('/api/me').then(async data => {
  if (data.user?.role !== 'admin') return;
  showAdmin(data.user); await Promise.all([loadStats(), loadPasskeys(), loadPrivateIdeas()]);
}).catch(() => {});
