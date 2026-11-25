const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..');

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    if (['.git', 'node_modules', 'tmp'].includes(entry.name)) return [];
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(target) : [target];
  });
}

test('internal navigation never advertises the duplicate /index.html URL', () => {
  const offenders = [];
  for (const file of walk(root).filter(file => file.endsWith('.html'))) {
    const html = fs.readFileSync(file, 'utf8');
    for (const match of html.matchAll(/href=["']([^"']*index\.html(?:[?#][^"']*)?)["']/gi)) {
      offenders.push(`${path.relative(root, file)}: ${match[1]}`);
    }
  }
  assert.deepEqual(offenders, []);
});

test('language-aware scripts build canonical home URLs without /index.html', () => {
  for (const file of ['assets/js/save.js', 'assets/js/research-support.js']) {
    assert.doesNotMatch(fs.readFileSync(path.join(root, file), 'utf8'), /index\.html/);
  }
});
