const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.join(__dirname, '..');
const read = p => fs.readFileSync(path.join(root, p), 'utf8');

test('traffic sends only approved campaign labels and forwards the server-signed token', async () => {
  for (const source of ['reddit','linkedin','directory','producthunt','telegram','private@example.com']) {
    let payload; const stored = new Map(); const link = {href:'https://btc2g-accounts.btc2g.workers.dev/'};
    const context = {navigator:{webdriver:false}, location:{protocol:'https:',pathname:'/ru/scenario.html',search:'?utm_source='+encodeURIComponent(source)}, URL, URLSearchParams,
      sessionStorage:{getItem:k=>stored.get(k),setItem:(k,v)=>stored.set(k,v)},
      document:{referrer:'',querySelectorAll:()=>[link]},
      fetch:async (_url,options)=>{payload=JSON.parse(options.body); return {ok:true,json:async()=>({attribution:'signed-local-token'})};}};
    vm.runInNewContext(read('assets/js/traffic.js'),context);
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(payload.campaignSource, source.includes('@') ? null : source);
    assert.equal(payload.path,'/ru/scenario.html');
    assert.equal(new URL(link.href).searchParams.get('attribution'),'signed-local-token');
  }
});

function articleHarness(checked, implementation) {
  let handler; const calls=[]; const input={value:'reader@example.com',focus(){}};
  const consent={checked,focus(){this.focused=true;}}; const button={disabled:false}; const status={textContent:''};
  const form={querySelector:q=>q==='input[type=email]'?input:q==='input[name=consent]'?consent:button,
    addEventListener:(_event,callback)=>handler=callback,
    parentElement:{querySelector:()=>status},reset(){input.value='';consent.checked=false;}};
  vm.runInNewContext(read('assets/js/article-signup.js'),{document:{querySelectorAll:()=>[form]},window:{BTC2GSubscription:{
    subscribe:async(...args)=>{calls.push(args);return implementation?.();},text:()=>({uncertain:'Receipt unknown; do not automatically retry.'})}}});
  return {submit:()=>handler({preventDefault(){}}),calls,input,consent,button,status};
}

test('articles require explicit consent and retain email on missing consent', async () => {
  const ui=articleHarness(false);await ui.submit();assert.equal(ui.calls.length,0);assert.equal(ui.input.value,'reader@example.com');assert.equal(ui.consent.focused,true);
});

test('articles use the confirmed-subscription request service, never claim account registration', async () => {
  const ui=articleHarness(true);await ui.submit();assert.deepEqual(ui.calls,[['reader@example.com',true,'en']]);assert.equal(ui.input.value,'');assert.match(ui.status.textContent,/only after confirmation/);assert.match(ui.status.textContent,/does not create a BTC2G account/);assert.equal(ui.button.disabled,false);
  for(const page of ['bitcoin-gold-ratio.html','bitcoin-vs-gold.html','bitcoin-vs-gold-forecast-2026.html']) {
    const html=read(page);assert.match(html,/name="consent" required/);assert.match(html,/href="privacy.html"/);assert.match(html,/data-article-account/);assert.ok(html.indexOf('subscription-request.js')<html.indexOf('article-signup.js'));
  }
  assert.doesNotMatch(read('assets/js/article-signup.js'),/formspree/i);
});

test('article failure retains inputs and does not retry or show confirmation', async () => {
  const ui=articleHarness(true,()=>{throw new Error('network');});await ui.submit();assert.equal(ui.calls.length,1);assert.equal(ui.input.value,'reader@example.com');assert.equal(ui.consent.checked,true);assert.equal(ui.button.disabled,false);assert.match(ui.status.textContent,/unknown/);
});
