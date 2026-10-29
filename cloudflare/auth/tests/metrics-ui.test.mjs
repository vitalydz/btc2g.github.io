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
  source: 'Cloudflare D1', measuredAt: '2026-10-29T06:00:00Z',
  day: '2026-10-29', timeZone: 'Asia/Jerusalem',
  newToday: 2, totalAccounts: 3,
  traffic: { visitorsToday: 4, viewsToday: 9 },
  confirmedSubscribers: { total: null },
  summary: { visitorsToday: 4, viewsToday: 9, registeredToday: 2, registeredTotal: 3, verifiedEmailTotal: 3, pendingEmailTotal: 1, conversionToday: .5,
    productHuntVisitorsToday: 2, productHuntRegisteredToday: 1, productHuntConversionToday: .5 },
  daily: [], topPaths: [], recentRegistrations: [], methodology: 'Local test data'
};

test('account statistics recover from failure and never leave stale figures visible', async () => {
  const ui = page('../public/index.html');
  let failure = false;
  const source = readFileSync(new URL('../public/account.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/, '');
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
  const source = readFileSync(new URL('../public/admin/admin.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/, '');
  const context = vm.createContext({ document: ui.document, formatCount, percentage, measurementNote,
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
  assert.equal(ui.nodes.get('#pending-email-total').textContent, '1');
  ui.nodes.get('#recent-users').children = ['old private list'];
  failure = true;
  await context.loadStats();
  assert.equal(ui.nodes.get('#visitors-today').textContent, 'Data unavailable');
  assert.equal(ui.nodes.get('#recent-users').children.length, 0);
  assert.match(ui.nodes.get('#stats-status').textContent, /403/);
});

test('out-of-order admin responses cannot overwrite a newer reporting range', async () => {
  const ui = page('../public/admin/index.html');
  const pending = [];
  const source = readFileSync(new URL('../public/admin/admin.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/, '');
  const context = vm.createContext({ document: ui.document, formatCount, percentage, measurementNote,
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
