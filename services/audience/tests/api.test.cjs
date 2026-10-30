const test=require('node:test');
const assert=require('node:assert/strict');
const {createHandler,dayAt,digest}=require('../lib/server');
const env={SITE_ORIGIN:'https://btc2g.com',API_ORIGIN:'https://audience.example.com',VISITOR_HASH_SECRET:'s'.repeat(40),STATS_TOKEN:'secret',RESEND_API_KEY:'fake-never-sent',MAIL_FROM:'updates@example.com'};
const email={email:'person@example.com',consent:true,consentVersion:'2026-10-29'};
async function call(route,{method='POST',body={},origin='https://btc2g.com',url='/api/'+route,headers={},query=async()=>({rows:[]}),send=async()=>{throw Error('No real mail allowed');},config=env}={}) {
 const res={headers:{},setHeader(k,v){this.headers[k]=v;},end(body){this.body=body;}};
 await createHandler(route,{env:config,db:{query},send,now:()=>new Date('2026-10-29T22:30:00Z'),random:()=> 'a'.repeat(64)})({method,body,url,headers:{origin,'user-agent':'ordinary browser','x-forwarded-for':'192.0.2.1',...headers}},res);
 return res;
}
test('Israel day is independent of UTC midnight',()=>assert.equal(dayAt(new Date('2026-10-29T22:30:00Z')),'2026-10-30'));
test('rejects foreign origin, missing consent and oversized body before storage or mail',async()=>{
 const query=()=>{throw Error('Must not query');};
 for(const options of [{origin:'https://evil.example'},{body:{...email,consent:false}},{body:{email:'x'.repeat(3000)}}]){
  const r=await call('subscribe',{query,...options}); assert.ok([400,403,413].includes(r.statusCode));
 }
});
test('subscribe stores hashes, sends one confirmation and reports no confirmation',async()=>{
 const queries=[],mail=[];
 const r=await call('subscribe',{body:email,query:async(sql,args)=>{queries.push({sql,args});return{rows:sql.includes('rate_limits')?[{attempts:1}]:[{email:email.email}]};},send:async(url,options)=>{mail.push({url,...options});return{ok:true};}});
 assert.equal(r.statusCode,202);assert.equal(JSON.parse(r.body).confirmed,false);assert.equal(mail.length,1);
 assert.equal(queries[1].args[2],digest('a'.repeat(64)));assert.doesNotMatch(queries[1].args[2],/^a{64}$/);
 const content=JSON.parse(mail[0].body);assert.match(content.text,/press Confirm subscription/);assert.match(content.text,/unsubscribe/);
 assert.equal(r.body.includes(email.email),false);
});
test('repeat or already active subscription never sends another email or exposes membership',async()=>{
 let n=0; const r=await call('subscribe',{body:email,query:async()=>({rows:++n===1?[{attempts:1}]:[]})});
 assert.equal(r.statusCode,429);assert.equal(JSON.parse(r.body).confirmed,false);
});
test('persistent rate limit stops outgoing email',async()=>{
 const r=await call('subscribe',{body:email,query:async()=>({rows:[{attempts:6}]})});assert.equal(r.statusCode,429);
});
test('GET confirmation and unsubscribe never mutate the database',async()=>{
 for(const route of ['confirm','unsubscribe']){
  const r=await call(route,{method:'GET',url:'/api/'+route+'?token='+'a'.repeat(64),query:async sql=>{assert.match(sql,/^SELECT language/);return {rows:[]};}});
  assert.equal(r.statusCode,200);assert.match(r.body,/<form method="post">/);assert.equal(r.headers['Referrer-Policy'],'no-referrer');
 }
});
test('confirmation requires an unexpired unused token and does not expose email',async()=>{
 let sqlText;
 const r=await call('confirm',{body:{token:'a'.repeat(64)},query:async(sql,args)=>{sqlText=sql;assert.equal(args[0],digest('a'.repeat(64)));return{rows:[{email:email.email}]};}});
 assert.equal(r.statusCode,200);assert.match(sqlText,/expires_at>now\(\)/);assert.match(sqlText,/token_hash=NULL/);assert.equal(r.body.includes(email.email),false);
 const reused=await call('confirm',{body:{token:'a'.repeat(64)}});assert.equal(reused.statusCode,400);
});
test('invalid token cannot inject HTML or reach storage',async()=>{
 const r=await call('confirm',{method:'GET',url:'/api/confirm?token=%22%3E%3Cscript%3E'});assert.equal(r.statusCode,400);assert.doesNotMatch(r.body,/<script>/);
});
test('unsubscribe consumes pending confirmation and removes active status',async()=>{
 let sqlText;const r=await call('unsubscribe',{body:{token:'a'.repeat(64)},query:async(sql)=>{sqlText=sql;return{rows:[{}]};}});
 assert.equal(r.statusCode,200);assert.match(sqlText,/unsubscribed_at=coalesce/);assert.match(sqlText,/token_hash=NULL/);
});
test('stats are authenticated, Jerusalem dated and unavailable traffic is null',async()=>{
 assert.equal((await call('stats',{method:'GET'})).statusCode,401);
 let n=0;const r=await call('stats',{method:'GET',headers:{authorization:'Bearer secret'},query:async()=>({rows:++n===1?[{since:null}]:[{active:'0',confirmed_today:'0'}]})});
 assert.equal(r.statusCode,200);const stats=JSON.parse(r.body);
 assert.equal(stats.traffic,null);assert.equal(stats.confirmedSubscribers.active,0);assert.equal(stats.accounts,null);assert.equal(stats.day,'2026-10-30');assert.equal(stats.dayTimeZone,'Asia/Jerusalem');assert.ok(stats.measuredAt);
});
test('visit uses daily pseudonymous identity; raw IP and URL query are never stored',async()=>{
 let args;const r=await call('visit',{body:{path:'/save.html'},query:async(sql,values)=>{args=values;return{rows:[]};}});
 assert.equal(r.statusCode,202);assert.equal(args[0],'2026-10-30');assert.match(args[1],/^[a-f0-9]{64}$/);assert.equal(JSON.stringify(args).includes('192.0.2.1'),false);
 assert.equal((await call('visit',{body:{path:'/save.html?email=private'}})).statusCode,400);
 assert.equal((await call('visit',{body:{path:'/'},headers:{'user-agent':'HeadlessChrome'}})).statusCode,202);
});
test('database, provider and configuration failures return unavailable, never fake success',async()=>{
 for(const options of [
  {config:{...env,RESEND_API_KEY:''}},
  {query:async()=>{throw Error('secret-database-string');}},
  {query:async sql=>({rows:sql.includes('rate_limits')?[{attempts:1}]:[{}]}),send:async()=>({ok:false})}
 ]){
  const r=await call('subscribe',{body:email,...options});assert.equal(r.statusCode,503);assert.doesNotMatch(r.body,/secret-database-string/);
 }
});
test('localized mail and stored-token language survive confirmation, expiry and unsubscribe',async()=>{
 const {messages}=require('../lib/language');
 for(const lang of ['en','ru','he']) {
  let mail;
  const sent=await call('subscribe',{body:{...email,lang},query:async sql=>({rows:sql.includes('rate_limits')?[{attempts:1}]:[{email:email.email}]}),send:async(url,options)=>{mail=JSON.parse(options.body);return{ok:true};}});
  assert.equal(sent.statusCode,202);assert.equal(mail.subject,messages[lang].subject);assert.match(mail.text,new RegExp('lang='+lang));
  for(const route of ['confirm','unsubscribe']) {
   const page=await call(route,{method:'GET',url:'/api/'+route+'?token='+'a'.repeat(64)+'&lang=en',query:async()=>({rows:[{language:lang}]})});
   assert.match(page.body,new RegExp('lang="'+lang+'"'));assert.match(page.body,new RegExp('dir="'+(lang==='he'?'rtl':'ltr')+'"'));assert.ok(page.body.includes(messages[lang][route]));
   const expired=await call(route,{body:{token:'a'.repeat(64)},query:async sql=>({rows:sql.startsWith('SELECT')?[{language:lang}]:[]})});
   assert.equal(expired.statusCode,400);assert.ok(expired.body.includes(messages[lang].invalid));
  }
 }
});
test('explicit provider rejection and ambiguous timeout are persisted differently without retrying',async()=>{
 for(const [send,state,delay] of [[async()=>({ok:false,status:422}),'failed',60],[async()=>({ok:false,status:503}),'unknown',900],[async()=>{throw Error('timeout');},'unknown',900]]) {
  const updates=[];
  const r=await call('subscribe',{body:email,query:async(sql,args)=>{
   if(sql.startsWith('UPDATE')) updates.push(args);
   return {rows:sql.includes('rate_limits')?[{attempts:1}]:[{email:email.email}]};
  },send});
  assert.equal(r.statusCode,503);assert.equal(JSON.parse(r.body).retryAfterSeconds,delay);assert.equal(r.headers['Retry-After'],String(delay));assert.equal(updates.length,2);assert.equal(updates[0][1],state);assert.equal(updates[1][1],state);
 }
});
