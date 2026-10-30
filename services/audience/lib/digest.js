const {timingSafeEqual,createHash}=require('node:crypto');
const issueDay=now=>new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Jerusalem',year:'numeric',month:'2-digit',day:'2-digit'}).format(now);
function validMarket(data,now){
 const date=data?.last_updated;
 if(typeof date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(date))throw Error('Invalid market date');
 const ms=Date.parse(date+'T00:00:00Z');
 if(!Number.isFinite(ms)||new Date(ms).toISOString().slice(0,10)!==date||ms>now.getTime()||now.getTime()-ms>4*86400000)throw Error('Stale market data');
 for(const key of ['ratio','btc_price_usd','gold_price_usd'])if(typeof data[key]!=='number'||!Number.isFinite(data[key])||data[key]<=0)throw Error('Invalid market values');
 if(Math.abs(data.btc_price_usd/data.gold_price_usd-data.ratio)>Math.max(.02,data.ratio*.003))throw Error('Inconsistent market values');
 return data;
}
function createDigestHandler({db=require('./db'),env=process.env,send=fetch,now=()=>new Date()}={}){
 return async(req,res)=>{
  res.setHeader('Cache-Control','no-store');res.setHeader('Content-Type','application/json');
  const reply=(status,body)=>{res.statusCode=status;res.end(JSON.stringify(body));};
  if(req.method!=='GET')return reply(405,{error:'method_not_allowed'});
  const provided=Buffer.from(req.headers.authorization||''),expected=Buffer.from('Bearer '+(env.CRON_SECRET||''));
  if(!env.CRON_SECRET||provided.length!==expected.length||!timingSafeEqual(provided,expected))return reply(401,{error:'unauthorized'});
  if(env.DIGEST_ENABLED!=='true')return reply(200,{status:'disabled',attempted:0});
  const stamp=now(),day=issueDay(stamp);
  if(new Intl.DateTimeFormat('en-US',{timeZone:'Asia/Jerusalem',weekday:'short'}).format(stamp)!=='Sun')return reply(200,{status:'not_sunday',attempted:0});
  try{
   if(!env.RESEND_API_KEY||!env.MAIL_FROM||!env.API_ORIGIN)throw Error('Mail not configured');
   const site=new URL(env.SITE_ORIGIN||'https://btc2g.com');const api=new URL(env.API_ORIGIN);
   if(site.protocol!=='https:'||api.protocol!=='https:')throw Error('HTTPS required');
   const marketResponse=await send(new URL('/assets/btc_signal.json',site).href,{signal:AbortSignal.timeout(10000)});
   if(!marketResponse.ok)throw Error('Market unavailable');
   const market=validMarket(await marketResponse.json(),stamp);
   const eligible=`s.weekly_digest=true AND s.confirmed_at IS NOT NULL AND s.unsubscribed_at IS NULL AND s.digest_unsubscribe_token IS NOT NULL
    AND NOT EXISTS(SELECT 1 FROM audience_digest_deliveries d WHERE d.email=s.email AND d.issue_day=$1)`;
   const rows=await db.query(`SELECT s.email,s.digest_unsubscribe_token FROM audience_subscriptions s WHERE ${eligible} ORDER BY s.email LIMIT 1000`,[day]);
   let cursor=0,attempted=0,accepted=0;const deadline=Date.now()+240000;
   async function work(){
    while(cursor<rows.rows.length&&Date.now()<deadline){
     const row=rows.rows[cursor++];
     const claim=await db.query(`INSERT INTO audience_digest_deliveries(email,issue_day) VALUES($1,$2) ON CONFLICT DO NOTHING RETURNING email`,[row.email,day]);
     if(!claim.rows.length)continue;
     const current=await db.query(`SELECT email FROM audience_subscriptions WHERE email=$1 AND weekly_digest=true AND confirmed_at IS NOT NULL AND unsubscribed_at IS NULL`,[row.email]);
     if(!current.rows.length){await db.query("UPDATE audience_digest_deliveries SET state='cancelled' WHERE email=$1 AND issue_day=$2",[row.email,day]);continue;}
     attempted++;
     const unsubscribe=new URL('/api/digest-unsubscribe',api);unsubscribe.searchParams.set('token',row.digest_unsubscribe_token);
     let state='uncertain',id=null;
     try{
      const response=await send('https://api.resend.com/emails',{method:'POST',signal:AbortSignal.timeout(10000),headers:{Authorization:'Bearer '+env.RESEND_API_KEY,'Content-Type':'application/json','Idempotency-Key':'btc2g-digest-'+day+'-'+createHash('sha256').update(row.email).digest('hex')},body:JSON.stringify({from:env.MAIL_FROM,to:[row.email],subject:'BTC2G Sunday research digest — '+day,text:`BTC2G Sunday research digest\nPublished market date: ${market.last_updated}\nBTC / Gold ratio: ${market.ratio} oz t\nBitcoin: USD ${market.btc_price_usd}\nGold: USD ${market.gold_price_usd} / oz t\n\nPublished model and chart: ${new URL('/#forecast',site).href}\nResearch information, not investment advice or a guarantee.\n\nStop Sunday digests: ${unsubscribe.href}`,headers:{'List-Unsubscribe':'<'+unsubscribe.href+'>','List-Unsubscribe-Post':'List-Unsubscribe=One-Click'}})});
      if(!response.ok)throw Error('Provider rejected');
      const receipt=await response.json();if(typeof receipt.id!=='string'||!receipt.id)throw Error('Missing provider receipt');
      state='accepted';id=receipt.id;accepted++;
     }catch{}
     await db.query('UPDATE audience_digest_deliveries SET state=$3,provider_id=$4 WHERE email=$1 AND issue_day=$2',[row.email,day,state,id]);
    }
   }
   // Wait for every in-flight worker, even if another encounters a database failure.
   const jobs=await Promise.allSettled(Array.from({length:10},()=>work()));
   if(jobs.some(job=>job.status==='rejected'))throw Error('Digest storage failed');
   const pending=await db.query(`SELECT count(*)::int AS n FROM audience_subscriptions s WHERE ${eligible}`,[day]);
   return reply(pending.rows[0].n?503:200,{status:pending.rows[0].n?'more_work_required':'processed',issueDay:day,attempted,providerAccepted:accepted,remaining:pending.rows[0].n});
  }catch{return reply(503,{error:'digest_unavailable'});}
 };
}
module.exports={createDigestHandler,validMarket,issueDay};
