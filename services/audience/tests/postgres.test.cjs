const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {randomBytes}=require('node:crypto');
const {createHandler}=require('../lib/server');
test('real PostgreSQL: confirmation, replay, unsubscribe, deduplication and aggregates',{skip:!process.env.TEST_DATABASE_URL},async()=>{
 const {Pool}=require('pg');
 const admin=new Pool({connectionString:process.env.TEST_DATABASE_URL});
 const schema='audience_test_'+randomBytes(8).toString('hex');
 let pool;
 try{
  await admin.query('CREATE SCHEMA '+schema);
  pool=new Pool({connectionString:process.env.TEST_DATABASE_URL,options:'-c search_path='+schema});
  for(const name of fs.readdirSync(__dirname+'/../migrations').filter(n=>n.endsWith('.sql')).sort()) await pool.query(fs.readFileSync(__dirname+'/../migrations/'+name,'utf8'));
  const env={SITE_ORIGIN:'https://btc2g.com',API_ORIGIN:'https://audience.example.com',VISITOR_HASH_SECRET:'x'.repeat(40),STATS_TOKEN:'test',RESEND_API_KEY:'fake',MAIL_FROM:'updates@example.com'};
  const tokens=['a','b','c','d','e','f','1','2'].map(c=>c.repeat(64));let sent=0, mailMode='accepted';
  const call=async(route,body={},method='POST')=>{
   const res={setHeader(){},end(value){this.body=value;}};
   await createHandler(route,{db:pool,env,random:()=>tokens.shift(),send:async()=>{sent++;if(mailMode==='unknown')throw Error('timeout');return{ok:mailMode==='accepted',status:mailMode==='accepted'?200:422};}})({method,body,url:'/api/'+route,headers:{origin:env.SITE_ORIGIN,'x-forwarded-for':'192.0.2.1','user-agent':'Browser',authorization:'Bearer test'}},res);
   return res;
  };
  assert.equal((await call('subscribe',{email:'person@example.com',consent:true,consentVersion:'2026-10-29'})).statusCode,202);assert.equal(sent,1);
  const request={email:'person@example.com',consent:true,consentVersion:'2026-10-29',lang:'he'};
  await pool.query("UPDATE audience_subscriptions SET requested_at=now()-interval '16 minutes'");
  mailMode='failed';
  assert.equal((await call('subscribe',request)).statusCode,503);
  assert.equal((await pool.query('SELECT delivery_state FROM audience_subscriptions')).rows[0].delivery_state,'failed');
  assert.equal((await call('subscribe',request)).statusCode,429);assert.equal(sent,2);
  await pool.query("UPDATE audience_subscriptions SET requested_at=now()-interval '61 seconds'");
  mailMode='unknown';
  assert.equal((await call('subscribe',request)).statusCode,503);assert.equal(sent,3);
  assert.equal((await pool.query('SELECT delivery_state FROM audience_subscriptions')).rows[0].delivery_state,'unknown');
  assert.equal((await pool.query('SELECT count(*)::int AS n FROM audience_tokens WHERE used_at IS NULL')).rows[0].n,3);
  // The original accepted link survives both a rejected replacement and an ambiguous timeout.
  assert.equal((await call('confirm',{token:'a'.repeat(64)})).statusCode,200);
  assert.equal((await call('confirm',{token:'a'.repeat(64)})).statusCode,400);
  for(let i=0;i<2;i++)assert.equal((await call('visit',{path:'/'})).statusCode,202);
  let stats=JSON.parse((await call('stats',{},'GET')).body);
  assert.equal(stats.traffic.pageviews,2);assert.equal(stats.traffic.today.estimatedVisitors,1);assert.equal(stats.confirmedSubscribers.active,1);
  await call('unsubscribe',{token:'b'.repeat(64)});
  stats=JSON.parse((await call('stats',{},'GET')).body);assert.equal(stats.confirmedSubscribers.active,0);
 } finally {if(pool)await pool.end();await admin.query('DROP SCHEMA IF EXISTS '+schema+' CASCADE');await admin.end();}
});
