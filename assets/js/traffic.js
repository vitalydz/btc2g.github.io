(() => {
  const endpoint = 'https://btc2g-accounts.btc2g.workers.dev/api/visit';
  if (!endpoint || navigator.webdriver || location.protocol === 'file:') return;
  fetch(endpoint, {
    method: 'POST',
    mode: 'cors',
    keepalive: true,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: location.pathname })
  }).catch(() => {});
})();
