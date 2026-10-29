import { formatCount, measurementNote, percentage, request } from '../metrics-view.js';

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
      const li = document.createElement('li'); li.textContent = row.username;
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
    const data = await request('/api/login', { method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({ username:form.username.value, password:form.password.value }) });
    if (data.user.role !== 'admin') {
      await request('/api/logout', { method:'POST', headers:{'content-type':'application/json'}, body:'{}' });
      throw new Error('Administrator access required.');
    }
    showAdmin(data.user); await loadStats();
  } catch (error) { loginMessage.textContent = error.message; }
});

document.querySelector('#range').addEventListener('change', loadStats);
document.querySelector('#refresh-stats').addEventListener('click', loadStats);
document.querySelector('#logout').addEventListener('click', async () => {
  await request('/api/logout', { method:'POST', headers:{'content-type':'application/json'}, body:'{}' }); location.reload();
});

request('/api/me').then(async data => {
  if (data.user?.role !== 'admin') return;
  showAdmin(data.user); await loadStats();
}).catch(() => {});
