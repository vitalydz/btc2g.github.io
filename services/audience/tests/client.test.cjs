const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
test('staged client is inactive by default, requires consent, and suppresses concurrent visit calls',async()=>{
 let calls=0;const context={window:{},URL,AbortSignal,navigator:{webdriver:false},location:{pathname:'/save.html',protocol:'https:'},fetch:async()=>{calls++;return{ok:true,json:async()=>({status:'recorded'})};}};
 vm.runInNewContext(fs.readFileSync(__dirname+'/../client/audience-client.js','utf8'),context);
 assert.equal(calls,0);
 assert.throws(()=>context.window.createBTC2GAudienceClient('http://insecure.example'));
 const client=context.window.createBTC2GAudienceClient('https://api.example');
 await assert.rejects(client.subscribe('person@example.com',false),/consent/);assert.equal(calls,0);
 await Promise.all([client.recordVisit(),client.recordVisit()]);assert.equal(calls,1);
});
