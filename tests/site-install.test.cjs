const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const {parseHTML}=require('linkedom');
const code=fs.readFileSync(path.join(__dirname,'../home-screen/site-install.js'),'utf8');
function setup(options={}) {
  const {document,Event}=parseHTML(`<html lang="${options.lang||'ru'}"><head><title>Test</title></head><body><footer></footer></body></html>`);
  const listeners={};
  const create=document.createElement.bind(document);
  document.createElement=tag=>{const e=create(tag);if(tag==='dialog'){e.showModal=()=>{e.open=true;e.setAttribute('open','');};e.close=()=>{e.open=false;e.removeAttribute('open');};}return e;};
  const context={document,navigator:{userAgent:options.ios?'iPhone':'Android',platform:'',standalone:!!options.standalone},matchMedia:()=>({matches:!!options.standalone}),addEventListener:(type,fn)=>listeners[type]=fn};
  vm.runInNewContext(code,context);
  return {document,Event,listeners,button:()=>document.querySelector('[data-site-install]')};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
test('no automatic prompt; a native prompt runs only after a click',async()=>{
  const s=setup();let calls=0,prevented=0;
  s.listeners.beforeinstallprompt({preventDefault(){prevented++;},prompt:async()=>calls++,userChoice:Promise.resolve({outcome:'accepted'})});
  assert.equal(calls,0);assert.equal(prevented,1);
  s.button().click();await tick();assert.equal(calls,1);
  assert.notEqual(s.button().hidden,true,'acceptance alone is not proof of installation');
  s.listeners.appinstalled();assert.equal(s.button().hidden,true);
});
test('dismissed native prompt stays retryable with manual instructions',async()=>{
  const s=setup();let calls=0;
  s.listeners.beforeinstallprompt({preventDefault(){},prompt:async()=>calls++,userChoice:Promise.resolve({outcome:'dismissed'})});
  s.button().click();await tick();s.button().click();await tick();
  assert.equal(calls,1);assert.equal(s.document.querySelector('dialog').open,true);
  assert.notEqual(s.button().hidden,true);
});
test('iPhone gets localized Safari steps and close works',async()=>{
  const s=setup({ios:true,lang:'he'});s.button().click();await tick();
  const dialog=s.document.querySelector('dialog');
  assert.match(dialog.textContent,/Safari/);assert.match(dialog.textContent,/שיתוף/);
  dialog.querySelector('button').click();assert.equal(dialog.open,false);
});
test('standalone mode has no install button or prevented browser event',()=>{
  const s=setup({standalone:true});assert.equal(s.button(),null);
  s.listeners.beforeinstallprompt({preventDefault(){throw Error('must not intercept');}});
});
test('unsupported browser has manual instructions and a single button',async()=>{
  const s=setup();s.button().click();await tick();
  assert.match(s.document.querySelector('dialog p').textContent,/меню браузера/);
  assert.equal(s.document.querySelectorAll('[data-site-install]').length,1);
});
test('manifest stays on the public site and has separate maskable icons',()=>{
  const m=JSON.parse(fs.readFileSync(path.join(__dirname,'../manifest.json'),'utf8'));
  assert.equal(m.id,'/');assert.equal(m.scope,'/');assert.equal(m.start_url,'/?source=shortcut');
  assert.equal(m.name,'BTC2G');assert.equal(m.display,'standalone');
  for(const size of ['192x192','512x512'])assert.ok(m.icons.some(i=>i.sizes===size&&i.purpose==='any'));
  assert.ok(m.icons.some(i=>i.purpose==='maskable'));
  for(const icon of m.icons)assert.ok(fs.existsSync(path.join(__dirname,'..',icon.src)));
});
