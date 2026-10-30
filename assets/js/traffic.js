(() => {
  const endpoint = 'https://btc2g-accounts.btc2g.workers.dev/api/visit';
  const accountOrigin = 'https://btc2g-accounts.btc2g.workers.dev/';
  const storageKey = 'btc2g_attribution';
  if (!endpoint || navigator.webdriver || location.protocol === 'file:') return;

  const campaignSource = new URLSearchParams(location.search).get('utm_source')?.trim().toLowerCase() === 'telegram'
    ? 'telegram' : null;

  function storedAttribution() {
    try { return sessionStorage.getItem(storageKey); } catch { return null; }
  }

  function decorateAccountLinks(attribution) {
    if (!attribution) return;
    document.querySelectorAll(`a[href^="${accountOrigin}"]`).forEach(link => {
      const url = new URL(link.href);
      url.searchParams.set('attribution', attribution);
      link.href = url.toString();
    });
  }

  if (!campaignSource) decorateAccountLinks(storedAttribution());
  fetch(endpoint, {
    method: 'POST',
    mode: 'cors',
    keepalive: true,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: location.pathname, referrer: document.referrer, campaignSource })
  }).then(response => response.ok ? response.json() : null).then(data => {
    if (!data?.attribution) return;
    try { sessionStorage.setItem(storageKey, data.attribution); } catch {}
    decorateAccountLinks(data.attribution);
  }).catch(() => {});
})();
