import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { formatCount, measurementNote, percentage } from '../public/metrics-view.js';

function page(htmlPath) {
  const html = readFileSync(new URL(htmlPath, import.meta.url), 'utf8');
  const nodes = new Map();
  function node() {
    return { textContent: '', value: '7', hidden: false, children: [], listeners: {},
      addEventListener(name, handler) { this.listeners[name] = handler; },
      replaceChildren(...children) { this.children = children; },
      append(child) { this.children.push(child); } };
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(`#${match[1]}`, node());
  const counts = [...html.matchAll(/id="([^"]+)" data-count/g)].map(match => nodes.get(`#${match[1]}`));
  return {
    nodes,
    document: {
      querySelector(selector) {
        assert.ok(nodes.has(selector), `HTML must contain ${selector}`);
        return nodes.get(selector);
      },
      querySelectorAll(selector) {
        if (selector.endsWith('[data-count]')) return counts;
        if (selector === '.tab') return [];
        throw new Error(`Unexpected selector: ${selector}`);
      },
      createElement: node
    }
  };
}

const snapshot = {
  appVersion: '2026.10.29.1',
  source: 'Cloudflare D1', measuredAt: '2026-10-29T06:00:00Z',
  day: '2026-10-29', timeZone: 'Asia/Jerusalem',
  newToday: 2, totalAccounts: 3,
  traffic: { visitorsToday: 4, viewsToday: 9 },
  confirmedSubscribers: { total: null },
  summary: { visitorsToday: 4, viewsToday: 9, registeredToday: 2, registeredTotal: 3, verifiedEmailTotal: 3, pendingEmailTotal: 1, conversionToday: .5,
    productHuntVisitorsToday: 2, productHuntRegisteredToday: 1, productHuntConversionToday: .5,
    scenarioVisitorsToday: 1, scenarioRegisteredToday: 1, scenarioConversionToday: 1 },
  daily: [], topPaths: [], recentRegistrations: [], methodology: 'Local test data'
};

test('account statistics recover from failure and never leave stale figures visible', async () => {
  const ui = page('../public/index.html');
  let failure = false;
  const source = readFileSync(new URL('../public/account.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
  const context = vm.createContext({ document: ui.document, formatCount, measurementNote,
    request: async path => {
      if (path === '/api/me') return { user: null };
      if (failure) throw new Error('Service unavailable');
      return snapshot;
    }
  });
  vm.runInContext(source, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui.nodes.get('#views-count').textContent, '9');
  assert.equal(ui.nodes.get('#subscribers-count').textContent, 'Data unavailable');
  failure = true;
  await context.refreshStats();
  for (const id of ['#today-count', '#total-count', '#visitors-count', '#views-count']) {
    assert.equal(ui.nodes.get(id).textContent, 'Data unavailable');
  }
  assert.match(ui.nodes.get('#stats-note').textContent, /could not be reached/);
  failure = false;
  await context.refreshStats();
  assert.equal(ui.nodes.get('#views-count').textContent, '9');
  assert.match(ui.nodes.get('#stats-note').textContent, /2026-10-29.*Asia\/Jerusalem/);
});

test('admin refresh clears stale totals and private lists on failure', async () => {
  const ui = page('../public/admin/index.html');
  let failure = false;
  const source = readFileSync(new URL('../public/admin/admin.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
  const context = vm.createContext({ document: ui.document, formatCount, percentage, measurementNote,
    authenticatePasskey: async () => ({}), createPasskey: async () => ({}),
    request: async path => {
      if (path === '/api/me') return { user: null };
      if (failure) throw new Error('Request failed (403).');
      return snapshot;
    }
  });
  vm.runInContext(source, context);
  await context.loadStats();
  assert.equal(ui.nodes.get('#visitors-today').textContent, '4');
  assert.equal(ui.nodes.get('#confirmed-subscribers').textContent, 'Data unavailable');
  assert.equal(ui.nodes.get('#producthunt-registrations').textContent, '1');
  assert.equal(ui.nodes.get('#scenario-visitors').textContent, '1');
  assert.equal(ui.nodes.get('#scenario-registrations').textContent, '1');
  assert.equal(ui.nodes.get('#scenario-conversion').textContent, '100%');
  assert.equal(ui.nodes.get('#pending-email-total').textContent, '1');
  assert.equal(ui.nodes.get('#app-version').textContent, '2026.10.29.1');
  ui.nodes.get('#recent-users').children = ['old private list'];
  failure = true;
  await context.loadStats();
  assert.equal(ui.nodes.get('#visitors-today').textContent, 'Data unavailable');
  assert.equal(ui.nodes.get('#recent-users').children.length, 0);
  assert.equal(ui.nodes.get('#app-version').textContent, 'Unavailable');
  assert.match(ui.nodes.get('#stats-status').textContent, /403/);
});

test('out-of-order admin responses cannot overwrite a newer reporting range', async () => {
  const ui = page('../public/admin/index.html');
  const pending = [];
  const source = readFileSync(new URL('../public/admin/admin.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
  const context = vm.createContext({ document: ui.document, formatCount, percentage, measurementNote,
    authenticatePasskey: async () => ({}), createPasskey: async () => ({}),
    request: path => path === '/api/me' ? Promise.resolve({ user: null }) : new Promise(resolve => pending.push(resolve))
  });
  vm.runInContext(source, context);
  const oldRequest = context.loadStats();
  const newRequest = context.loadStats();
  pending[1](snapshot);
  await newRequest;
  pending[0]({ ...snapshot, summary: { ...snapshot.summary, visitorsToday: 999 } });
  await oldRequest;
  assert.equal(ui.nodes.get('#visitors-today').textContent, '4');
});


test('idea form preserves text and key on failure, deduplicates retry, and reports refresh separately', async () => {
  const ui = page('../public/index.html');
  let resetCount = 0;
  let keyCount = 0;
  let failure = Object.assign(new Error('internal database detail'), { status: 500 });
  let refreshFailure = false;
  const keys = [];
  const ideaForm = ui.nodes.get('#idea-form');
  ideaForm.reset = () => { resetCount += 1; };
  const message = ui.nodes.get('#idea-message');
  message.classList = { remove() {}, add() {} };
  const source = readFileSync(new URL('../public/account.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
  const context = vm.createContext({ document: ui.document, formatCount, measurementNote,
    crypto: { randomUUID: () => `local-key-${++keyCount}` },
    FormData: class { get(name) { return name === 'reviewConsent' ? 'on' : 'Preserved idea text'; } },
    request: async (path, options) => {
      if (path === '/api/me') return { user: null };
      if (path === '/api/stats') return snapshot;
      if (path === '/api/ideas') {
        keys.push(options.headers['Idempotency-Key']);
        if (failure) throw failure;
        return { duplicate: true, idea: { id: 'saved-idea' } };
      }
      if (path === '/api/ideas/mine') {
        if (refreshFailure) throw new Error('List unavailable');
        return { ideas: [] };
      }
      throw new Error(`Unexpected path: ${path}`);
    }
  });
  vm.runInContext(source, context);
  const submit = () => ideaForm.listeners.submit({ preventDefault() {} });
  for (const error of [failure, new TypeError('Failed to fetch'), Object.assign(new Error('limit'), { status: 409 })]) {
    failure = error;
    await submit();
    assert.equal(resetCount, 0);
    assert.equal(ui.nodes.get('#idea-submit').disabled, false);
    assert.doesNotMatch(message.textContent, /internal database detail|Failed to fetch/);
  }
  assert.match(message.textContent, /already submitted 3 ideas.*below/);
  assert.equal(new Set(keys).size, 1);
  failure = null;
  refreshFailure = true;
  await submit();
  assert.equal(new Set(keys).size, 1);
  assert.equal(resetCount, 1);
  assert.match(message.textContent, /Idea submitted privately.*list could not be refreshed/);
  refreshFailure = false;
  await submit();
  assert.notEqual(keys.at(-1), keys[0]);
  assert.equal(resetCount, 2);
  ui.nodes.set('.tabs', { hidden: false });
  context.showUser({ username: 'Admin', role: 'admin', emailVerified: false });
  assert.equal(ui.nodes.get('#idea-panel').hidden, false);
  assert.equal(ui.nodes.get('#idea-form').hidden, false);
  assert.equal(ui.nodes.get('#admin-link').hidden, false);
  assert.match(ui.nodes.get('#idea-access').textContent, /unlimited ideas.*admin dashboard/);
});
