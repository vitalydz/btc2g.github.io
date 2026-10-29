const number = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 });

export function formatCount(value) {
  return Number.isInteger(value) && value >= 0 ? number.format(value) : 'Data unavailable';
}

export function percentage(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? percent.format(value) : 'Data unavailable';
}

export function measurementNote(data) {
  if (!data.measuredAt || !data.day || data.timeZone !== 'Asia/Jerusalem') {
    throw new Error('Statistics response is missing its measurement period.');
  }
  const date = new Date(data.measuredAt);
  if (!Number.isFinite(date.getTime())) throw new Error('Statistics timestamp is invalid.');
  const checked = new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium', timeStyle: 'medium', timeZone: data.timeZone
  }).format(date);
  return `${data.source || 'Cloudflare D1'} · Reporting day: ${data.day} (${data.timeZone}) · Updated: ${checked}.`;
}

export async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(path, { ...options, cache: 'no-store', signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  } finally {
    clearTimeout(timer);
  }
}
