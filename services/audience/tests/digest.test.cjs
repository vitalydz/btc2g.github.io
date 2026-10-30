const test=require('node:test');
const assert=require('node:assert/strict');
const {validMarket,issueDay,createDigestHandler}=require('../lib/digest');
const {createHandler}=require('../lib/server');
const {randomBytes}=require('node:crypto');
const fs=require('node:fs');
const env={SITE_ORIGIN:'https://btc2g.com',API_ORIGIN:'https://api.example.com',RESEND_API_KEY:'fake',MAIL_FROM:'updates@example.com',CRON_SECRET:'test-secret',VISITOR_HASH_SECRET:'x'.repeat(40),DIGEST_ENABLED:'true'};
const stamp=new Date('2026-11-01T06:00:00Z');
const market={last_updated:'2026-10-30',ratio:20,btc_price_usd:80000,gold_price_usd:4000};
function response(){return{headers:{},setHeader(k,v){this.headers[k]=v;},end(body){this.body=body;}};}
async function cron(options={}){const res=response();await createDigestHandler({env,now:()=>stamp,...options})({method:'GET',headers:{authorization:'Bearer test-secret'}},res);return res;}
test('digest market validation rejects stale, future, impossible and non-finite data',()=>{
 assert.equal(validMarket(market,stamp),market);assert.equal(issueDay(new Date('2026-10-31T22:30:00Z')),'2026-11-01');
 for(const data of [{...market,last_updated:'2026-02-30'},{...market,last_updated:'2026-11-02'},{...market,last_updated:'2026-10-01'},{...market,ratio:Infinity},{...market,gold_price_usd:0}])assert.throws(()=>validMarket(data,stamp));
});
test('cron requires authorization, an enabled service, and Sunday in Israel',async()=>{
 const blocked=response();await createDigestHandler({env})({method:'GET',headers:{}},blocked);assert.equal(blocked.statusCode,401);
 for(const options of [{env:{...env,DIGEST_ENABLED:'false'}},{now:()=>new Date('2026-11-02T06:00:00Z')}])assert.equal((await cron(options)).statusCode,200);
});
test('provider failures never become accepted and claims are not retried automatically',async()=>{
 let sends=0;const writes=[];
 const db={query:async(sql,args)=>{
  if(sql.startsWith('SELECT s.email'))return{rows:[{email:'a@example.com',digest_unsubscribe_token:'a'.repeat(64)}]};
  if(sql.startsWith('SELECT count'))return{rows:[{n:0}]};
  if(sql.startsWith('UPDATE')){writes.push(args);return{rows:[]};}
  return{rows:[{email:'a@example.com'}]};
 }};
 const result=await cron({db,send:async url=>{if(url.includes('btc_signal'))return{ok:true,json:async()=>market};sends++;throw Error('ambiguous timeout');}});
 assert.equal(result.statusCode,200);assert.equal(sends,1);assert.equal(writes[0][2],'uncertain');assert.equal(JSON.parse(result.body).providerAccepted,0);
});
test('digest choice rejects non-booleans and an inactive service without sending',async()=>{
 for(const body of [{weeklyDigest:'yes'},{weeklyDigest:true}]){
  const res=response();await createHandler('digest-request',{env:{...env,DIGEST_ENABLED:'false'},db:{query:()=>{throw Error('must not query');}}})({method:'POST',url:'/api/digest-request',headers:{origin:env.SITE_ORIGIN,'x-forwarded-for':'192.0.2.1'},body:{email:'a@example.com',consent:true,consentVersion:'2026-10-29',...body}},res);
  assert.ok([400,503].includes(res.statusCode));
 }
});
test('real PostgreSQL digest lifecycle, ownership verification, duplicate cron and opt-out',{skip:!process.env.TEST_DATABASE_URL},async()=>{
 const {Pool}=require('pg');const admin=new Pool({connectionString:process.env.TEST_DATABASE_URL});const schema='digest_test_'+randomBytes(8).toString('hex');let pool;
 try{
  await admin.query('CREATE SCHEMA '+schema);pool=new Pool({connectionString:process.env.TEST_DATABASE_URL,options:'-c search_path='+schema});
  for(const file of fs.readdirSync(__dirname+'/../migrations').filter(f=>f.endsWith('.sql')).sort())await pool.query(fs.readFileSync(__dirname+'/../migrations/'+file,'utf8'));
  let mail,accepted=0;
  async function call(route,body={},method='POST',url='/api/'+route){const res=response();await createHandler(route,{db:pool,env,send:async(_url,options)=>{mail=JSON.parse(options.body);return{ok:true,status:200};}})({method,url,body,headers:{origin:env.SITE_ORIGIN,'x-forwarded-for':'192.0.2.1'}},res);return res;}
  const request={email:'a@example.com',consent:true,consentVersion:'2026-10-29',weeklyDigest:true,lang:'ru'};
  assert.equal((await call('digest-request',request)).statusCode,202);
  let row=(await pool.query('SELECT * FROM audience_subscriptions')).rows[0];assert.equal(row.weekly_digest,false);assert.equal(row.confirmed_at,null);
  const confirmation=new URL(mail.text.match(/https:\/\/[^\s]+\/api\/confirm\?[^\s]+/)[0]);const token=confirmation.searchParams.get('token');
  await call('confirm',{},'GET',confirmation.pathname+confirmation.search);assert.equal((await pool.query('SELECT weekly_digest FROM audience_subscriptions')).rows[0].weekly_digest,false);
  assert.equal((await call('confirm',{token})).statusCode,200);row=(await pool.query('SELECT * FROM audience_subscriptions')).rows[0];assert.equal(row.weekly_digest,true);assert.ok(row.confirmed_at);assert.ok(row.digest_consent_at);
  const transport=async(url,options)=>{if(url.includes('btc_signal'))return{ok:true,json:async()=>market};accepted++;const email=JSON.parse(options.body);assert.match(email.headers['List-Unsubscribe'],/digest-unsubscribe/);return{ok:true,json:async()=>({id:'mock-'+accepted})};};
  await Promise.all([cron({db:pool,send:transport}),cron({db:pool,send:transport})]);assert.equal(accepted,1);assert.equal((await pool.query('SELECT state FROM audience_digest_deliveries')).rows[0].state,'accepted');
  // Stage another enable request, then opt out: the older pending link must not re-enable the digest.
  await pool.query("UPDATE audience_subscriptions SET requested_at=now()-interval '16 minutes'");
  await call('digest-request',request);const olderToken=new URL(mail.text.match(/https:\/\/[^\s]+\/api\/confirm\?[^\s]+/)[0]).searchParams.get('token');
  const unsubscribe=row.digest_unsubscribe_token;
  await call('digest-unsubscribe',{},'GET','/api/digest-unsubscribe?token='+unsubscribe);assert.equal((await pool.query('SELECT weekly_digest FROM audience_subscriptions')).rows[0].weekly_digest,true);
  assert.equal((await call('digest-unsubscribe',{token:unsubscribe})).statusCode,200);assert.equal((await pool.query('SELECT weekly_digest FROM audience_subscriptions')).rows[0].weekly_digest,false);
  assert.equal((await call('confirm',{token:olderToken})).statusCode,400);
  await call('digest-unsubscribe',{token:unsubscribe}); // Idempotent opt-out.
  // Only a new explicit, verified request can turn it back on.
  await pool.query("UPDATE audience_subscriptions SET requested_at=now()-interval '16 minutes'");await call('digest-request',request);
  const fresh=new URL(mail.text.match(/https:\/\/[^\s]+\/api\/confirm\?[^\s]+/)[0]).searchParams.get('token');assert.equal((await call('confirm',{token:fresh})).statusCode,200);
  await pool.query("UPDATE audience_subscriptions SET requested_at=now()-interval '16 minutes'");await call('digest-request',{...request,weeklyDigest:false});
  const disable=new URL(mail.text.match(/https:\/\/[^\s]+\/api\/confirm\?[^\s]+/)[0]).searchParams.get('token');
  assert.equal((await pool.query('SELECT weekly_digest FROM audience_subscriptions')).rows[0].weekly_digest,true);
  assert.equal((await call('confirm',{token:disable})).statusCode,200);assert.equal((await pool.query('SELECT weekly_digest FROM audience_subscriptions')).rows[0].weekly_digest,false);
 }finally{if(pool)await pool.end();await admin.query('DROP SCHEMA IF EXISTS '+schema+' CASCADE');await admin.end();}
});
