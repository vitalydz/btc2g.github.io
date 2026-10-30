const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require.resolve('../assets/js/digest-preferences'),'utf8');
function setup(mode='accepted'){
 const nodes=new Map(),calls=[];const el=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',checked:false,disabled:false,addEventListener(name,fn){this[name]=fn;}});return nodes.get(id);};
 const context={window:{},document:{getElementById:el},URL,AbortSignal,fetch:async(url,options={})=>{
  calls.push({url,options});
  if(url==='assets/audience-config.json')return{ok:true,json:async()=>({apiOrigin:mode==='unconfigured'?null:'https://audience.example.com'})};
  if(mode==='timeout')throw Error('TimeoutError');
  return{ok:true,json:async()=>mode==='unexpected'?{saved:true}:{status:'request_received',confirmed:false}};
 }};
 vm.runInNewContext(source,context);context.window.BTC2GDigest.setEmail('person@example.com');context.window.BTC2GDigest.setLanguage('ru');
 return{el,calls,context,submit:()=>el('digest-preferences-form').submit({preventDefault(){}})};
}
test('restored digest form uses its service and does not claim saved before email confirmation',async()=>{
 const h=setup();h.el('weekly-digest-choice').checked=true;await h.submit();
 assert.equal(h.calls.length,2);assert.equal(h.calls[1].url,'https://audience.example.com/api/digest-request');
 assert.equal(JSON.parse(h.calls[1].options.body).weeklyDigest,true);assert.equal(JSON.parse(h.calls[1].options.body).lang,'ru');
 assert.match(h.el('digest-preferences-status').textContent,/Подтвердите настройку/);assert.doesNotMatch(h.el('digest-preferences-status').textContent,/сохранена|сохранены/);
 assert.equal(h.el('save-digest-preferences').disabled,false);
});
test('unconfigured service, timeout or unexpected reply preserves choice and recovers controls without retry',async()=>{
 for(const mode of ['unconfigured','timeout','unexpected']){
  const h=setup(mode);h.el('weekly-digest-choice').checked=true;await h.submit();
  assert.match(h.el('digest-preferences-status').textContent,/Не удалось подтвердить/);assert.equal(h.el('weekly-digest-choice').checked,true);assert.equal(h.el('save-digest-preferences').disabled,false);assert.equal(h.el('weekly-digest-choice').disabled,false);
  assert.equal(h.calls.length,mode==='unconfigured'?1:2);
 }
});
test('unchecked is the default and concurrent submission is suppressed',async()=>{
 const h=setup();assert.equal(h.el('weekly-digest-choice').checked,false);
 await Promise.all([h.submit(),h.submit()]);assert.equal(h.calls.length,2);assert.equal(JSON.parse(h.calls[1].options.body).weeklyDigest,false);
});
test('all offered languages have digest labels and honest confirmation messages',()=>{
 const h=setup();const home=fs.readFileSync(require.resolve('../index.html'),'utf8');
 for(const [,lang]of home.matchAll(/<option value="([a-z]{2})">/g)){
  assert.equal(h.context.window.BTC2GDigest.translations[lang].length,6);assert.ok(h.context.window.BTC2GDigest.translations[lang].every(Boolean));
  h.context.window.BTC2GDigest.setLanguage(lang);assert.ok(h.el('digest-preferences-help').textContent);
 }
 assert.match(home,/BTC2GDigest\?\.setEmail\(email\)/);
 assert.doesNotMatch(home,/weekly_digest:.*Formspree|Настройки сохранены\./);
});
