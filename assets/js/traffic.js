(() => {
  // One attempt per document, using only BTC2G's own origin. No third-party
  // collector, fallback host, cookies, persistent browser ID or retry queue.
  if (navigator.webdriver || location.protocol !== 'https:' ||
      location.hostname !== 'btc2g.com' || window.btc2gTrafficStarted) return;
  window.btc2gTrafficStarted = true;
  const report = status => {
    window.dispatchEvent(new CustomEvent('btc2g:traffic', { detail: { status } }));
  };
  fetch('/api/visit', {
    method: 'POST',
    mode: 'same-origin',
    credentials: 'omit',
    cache: 'no-store',
    referrerPolicy: 'no-referrer',
    keepalive: true,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: location.pathname })
  }).then(async response => {
    if (response.status !== 202 || (await response.json()).ok !== true) {
      throw new Error('Visit was not acknowledged.');
    }
    report('recorded');
  }).catch(() => report('unavailable'));
})();
