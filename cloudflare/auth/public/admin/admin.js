const loginPanel = document.querySelector('#login-panel');
const dashboard = document.querySelector('#dashboard');
const loginMessage = document.querySelector('#login-message');
const number = new Intl.NumberFormat();

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}

function percentage(value) {
  return new Intl.NumberFormat(undefined, { style:'percent', maximumFractionDigits:1 }).format(value || 0);
}

function showAdmin(user) {
  loginPanel.hidden = true;
  dashboard.hidden = false;
  document.querySelector('#admin-name').textContent = user.username;
  document.querySelector('#logout').hidden = false;
}

async function loadStats() {
  const data = await request(`/api/admin/stats?days=${document.querySelector('#range').value}`);
  document.querySelector('#visitors-today').textContent = number.format(data.summary.visitorsToday);
  document.querySelector('#views-today').textContent = number.format(data.summary.viewsToday);
  document.querySelector('#registered-today').textContent = number.format(data.summary.registeredToday);
  document.querySelector('#registered-total').textContent = number.format(data.summary.registeredTotal);
  document.querySelector('#conversion-today').textContent = percentage(data.summary.conversionToday);
  document.querySelector('#daily-body').replaceChildren(...data.daily.map(row => {
    const tr = document.createElement('tr');
    const conversion = Number(row.visitors) ? Number(row.registrations) / Number(row.visitors) : 0;
    [row.day, number.format(row.visitors), number.format(row.views), number.format(row.registrations), percentage(conversion)].forEach(value => {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    });
    return tr;
  }));
  document.querySelector('#top-paths').replaceChildren(...data.topPaths.map(row => {
    const li = document.createElement('li'); li.textContent = row.path;
    const span = document.createElement('span'); span.textContent = number.format(row.views); li.append(span); return li;
  }));
  document.querySelector('#recent-users').replaceChildren(...data.recentRegistrations.map(row => {
    const li = document.createElement('li'); li.textContent = row.username;
    const span = document.createElement('span'); span.textContent = row.created_day; li.append(span); return li;
  }));
  document.querySelector('#methodology').textContent = `${data.methodology} Reporting day: ${data.day} (${data.timeZone}).`;
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
document.querySelector('#logout').addEventListener('click', async () => {
  await request('/api/logout', { method:'POST', headers:{'content-type':'application/json'}, body:'{}' }); location.reload();
});

request('/api/me').then(async data => {
  if (data.user?.role !== 'admin') return;
  showAdmin(data.user); await loadStats();
}).catch(() => {});
