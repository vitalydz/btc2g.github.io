const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const home=fs.readFileSync(require.resolve('../index.html'),'utf8');
test('receipt explicitly avoids confirmation claims and exposes no unimplemented digest form',()=>{
 assert.doesNotMatch(home,/You’re in\.|You will receive an email only when the model signal changes\.|Настройки сохранены\.|id="digest-preferences-form"/);
 assert.match(home,/subscribe-thanks-title', text.subscribeSuccess/);
 assert.match(home,/subscribe-thanks-text', window.BTC2GSubscription.text\(lang\).receiptNote/);
 const context={window:{}};vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/subscription-request'),'utf8'),context);
 for(const [,lang] of home.matchAll(/<option value="([a-z]{2})">/g))assert.ok(context.window.BTC2GSubscription.text(lang).receiptNote,lang);
});
test('receipt replaces the form only after the handler reports a successful request',()=>{
 const nodes=new Map();const node=id=>{if(!nodes.has(id))nodes.set(id,{hidden:false});return nodes.get(id);};
 const start=home.indexOf('    function showSubscriptionThanks('),end=home.indexOf('    function setFormMessage(',start);
 const context={document:{getElementById:node}};vm.runInNewContext(home.slice(start,end),context);
 context.showSubscriptionThanks('unused@example.com');
 assert.equal(node('subscribe-form').hidden,true);assert.equal(node('subscribe-thanks').hidden,false);
});
