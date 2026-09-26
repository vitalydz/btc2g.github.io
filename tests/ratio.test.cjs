const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
let scriptCount = 0;
for (const file of fs.readdirSync(root).filter(name => name.endsWith('.html'))) {
  const html = fs.readFileSync(path.join(root, file), 'utf8');
  for (const [i, script] of [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)].entries()) {
    if (/\bsrc\s*=|application\/ld\+json/i.test(script[1]) || !script[2].trim()) continue;
    new vm.Script(script[2], {filename: `${file}:script${i + 1}`});
    scriptCount++;
  }
}
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const renderRatio = html.match(/    function renderRatio\(\) \{[\s\S]*?\n    \}/)[0];
const output = {};
const context = vm.createContext({
  currentSignalData: null,
  currentLang: 'en',
  LOCALE_BY_LANGUAGE: {en: 'en-US', ru: 'ru-RU'},
  setTextById: (id, value) => { output[id] = value; },
});
vm.runInContext(renderRatio, context);
for (const ratio of [undefined, null, 0, -1, NaN, Infinity, '40']) {
  context.currentSignalData = {ratio};
  vm.runInContext('renderRatio()', context);
  assert.equal(output['ratio-value'], 'Ratio temporarily unavailable');
  assert.equal(output['insight-ratio-value'], output['ratio-value']);
}
context.currentSignalData = {ratio: 40.1234};
vm.runInContext('renderRatio()', context);
assert.equal(output['ratio-value'], '40.123 oz');
context.currentLang = 'ru';
vm.runInContext('renderRatio()', context);
assert.equal(output['ratio-value'], '40,123 oz');
context.currentSignalData = null;
vm.runInContext('renderRatio()', context);
assert.equal(output['ratio-value'], 'Данные временно недоступны');
console.log(`PASS: ${scriptCount} JavaScript blocks; ratio formatting and invalid/missing data.`);
