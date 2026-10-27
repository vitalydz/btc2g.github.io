const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const sandbox={window:{}};vm.runInNewContext(fs.readFileSync('assets/js/contest-text.js','utf8'),sandbox);
const texts=sandbox.window.BTC2GContestText;
test('19 locales cover every public UI key without empty fallbacks',()=>{
 assert.equal(Object.keys(texts).length,19);
 const keys=new Set([...fs.readFileSync('contest.html','utf8').matchAll(/data-t="([^"]+)"/g)].map(m=>m[1]));
 ['p_draft','p_submissions','p_voting','p_runoff','p_completed','demo','pending','approved','rejected','confirmed'].forEach(k=>keys.add(k));
 for(const [lang,row] of Object.entries(texts)) for(const key of keys)assert.ok(typeof row[key]==='string'&&row[key].trim(),`${lang}:${key}`);
});
test('DOM renderer treats every submitted value as text, not markup',()=>{
 const ui=fs.readFileSync('assets/js/contest.js','utf8');
 assert.ok(!/innerHTML|insertAdjacentHTML|document\.write/.test(ui));
 assert.match(ui,/n\.textContent=value/);new vm.Script(ui);
});
test('static page starts closed and has no fake totals or local vote persistence',()=>{
 const html=fs.readFileSync('contest.html','utf8'),ui=fs.readFileSync('assets/js/contest.js','utf8');
 assert.match(html,/id="send"[^>]*disabled/);assert.match(html,/Contest in preparation/);
 assert.ok(!ui.includes('localStorage.setItem(\'vote'));
 assert.match(ui,/Idempotency-Key/);assert.match(ui,/textContent=t\(state\?'demo':'blocked'\)/);
});
