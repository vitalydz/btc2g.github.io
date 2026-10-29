(() => {
  // Filled with the verified workers.dev URL during production deployment.
  const endpoint = '';
  if (!endpoint || navigator.webdriver || location.protocol === 'file:') return;
  fetch(endpoint, {
    method: 'POST',
    mode: 'cors',
    keepalive: true,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: location.pathname })
  }).catch(() => {});
})();
