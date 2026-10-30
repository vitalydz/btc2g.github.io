const WORKFLOW = 'https://api.github.com/repos/vitalydz/btc2g.github.io/actions/workflows/update-chart.yml/runs?per_page=1';

export async function marketHealth(fetcher = fetch, now = new Date()) {
  const issues = [];
  async function read(url) {
    const response = await fetcher(url, {
      headers: { 'User-Agent': 'BTC2G-market-health', Accept: 'application/json' },
      cache: 'no-store', signal: AbortSignal.timeout(10000)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
  let marketDate = null;
  let updatedAt = null;
  let latestRun = null;
  await Promise.all([
    (async () => {
      try {
        const meta = await read('https://btc2g.com/assets/btc_gold_forecast_meta.json');
        marketDate = meta.market_date || null;
        updatedAt = meta.last_updated_utc || null;
        const updated = Date.parse(updatedAt);
        if (!Number.isFinite(updated) || updated > now.getTime() || now - updated > 86400000) {
          issues.push('Published market data has not been refreshed within 24 hours.');
        }
        const date = typeof marketDate === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(marketDate)
          ? Date.parse(`${marketDate}T00:00:00Z`) : NaN;
        const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
        if (!Number.isFinite(date) || new Date(date).toISOString().slice(0, 10) !== marketDate || date > today || today - date > 4 * 86400000) {
          issues.push('Market date is missing, invalid, or more than four days old.');
        }
      } catch {
        issues.push('Could not read published market metadata.');
      }
    })(),
    (async () => {
      try {
        const data = await read(WORKFLOW);
        const run = data.workflow_runs?.[0];
        if (!run) throw new Error('No workflow run');
        latestRun = { status: run.status, conclusion: run.conclusion, createdAt: run.created_at };
        if (run.status === 'completed' && run.conclusion !== 'success') issues.push(`Market update workflow ended with ${run.conclusion || 'unknown status'}.`);
        const started = Date.parse(run.created_at);
        if (!Number.isFinite(started) || now - started > 86400000) issues.push('No market update workflow started within 24 hours.');
      } catch {
        issues.push('Could not check the market update workflow.');
      }
    })()
  ]);
  return { status: issues.length ? 'attention' : 'ok', marketDate, updatedAt, latestRun, issues };
}
