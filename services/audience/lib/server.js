const { createHash, createHmac, randomBytes, timingSafeEqual } = require('node:crypto');
const digest = value => createHash('sha256').update(value).digest('hex');
const dayAt = now => new Intl.DateTimeFormat('en-CA', {timeZone:'Asia/Jerusalem',year:'numeric',month:'2-digit',day:'2-digit'}).format(now);
const tokenPattern = /^[a-f0-9]{64}$/;
const genericReceipt = {status:'request_received', confirmed:false};
const {messages,language} = require('./language');

function createHandler(route, {db = require('./db'), env = process.env, send = globalThis.fetch, now = () => new Date(), random = () => randomBytes(32).toString('hex')} = {}) {
 return async (req, res) => {
  res.setHeader('Cache-Control','no-store');
  res.setHeader('Referrer-Policy','no-referrer');
  res.setHeader('X-Content-Type-Options','nosniff');
  const json = (status, body) => { res.statusCode=status; res.setHeader('Content-Type','application/json; charset=utf-8'); res.end(JSON.stringify(body)); };
  let lang = language(new URL(req.url,'https://local.invalid').searchParams.get('lang'));
  const html = (status, body) => {
   res.statusCode=status;
   res.setHeader('Content-Type','text/html; charset=utf-8');
   res.setHeader('Content-Security-Policy',"default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'");
   res.end('<!doctype html><html lang="'+lang+'" dir="'+(lang==='he'?'rtl':'ltr')+'"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BTC2G subscription</title><body>'+body+'</body></html>');
  };
  try {
   const site = new URL(env.SITE_ORIGIN || 'https://btc2g.com').origin;
   const origin = req.headers.origin;
   if (origin && origin !== site && origin !== env.API_ORIGIN) return json(403,{error:'origin_rejected'});
   if (origin) { res.setHeader('Access-Control-Allow-Origin',origin); res.setHeader('Vary','Origin'); }
   if (req.method==='OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods','GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers','Content-Type, Authorization');
    res.statusCode=204; return res.end();
   }
   const expected = route==='stats' ? ['GET'] : ['confirm','unsubscribe','digest-unsubscribe'].includes(route) ? ['GET','POST'] : ['POST'];
   if (!expected.includes(req.method)) {res.setHeader('Allow',expected.join(', '));return json(405,{error:'method_not_allowed'});}
   let body = {};
   if (req.method==='POST') {
    // Vercel parses JSON/form bodies. Refuse arrays, oversized bodies and malformed input.
    body = req.body;
    if (typeof body==='string') {
     if (body.length>2048) return json(413,{error:'body_too_large'});
     body = (req.headers['content-type'] || '').startsWith('application/x-www-form-urlencoded') ? Object.fromEntries(new URLSearchParams(body)) : JSON.parse(body);
    }
    if (!body || typeof body!=='object' || Array.isArray(body) || JSON.stringify(body).length>2048) return json(400,{error:'invalid_body'});
   }
   if (route==='stats') {
    const supplied = (req.headers.authorization || '').replace(/^Bearer /,'');
    const a=Buffer.from(supplied), b=Buffer.from(env.STATS_TOKEN || '');
    if (!b.length || a.length!==b.length || !timingSafeEqual(a,b)) return json(401,{error:'unauthorized'});
    const measuredAt=now().toISOString(), day=dayAt(now());
    const traffic=await db.query(`SELECT min(first_seen_at) AS since, coalesce(sum(views),0)::text AS views,
     count(*)::text AS visitor_days, count(*) FILTER(WHERE day=$1)::text AS today_visitors,
     coalesce(sum(views) FILTER(WHERE day=$1),0)::text AS today_views FROM audience_visitors`,[day]);
    const subscribers=await db.query(`SELECT (SELECT initialized_at FROM audience_meta WHERE id=1) AS since, count(*) FILTER(WHERE confirmed_at IS NOT NULL AND unsubscribed_at IS NULL)::text AS active,
     count(*) FILTER(WHERE confirmed_at IS NOT NULL AND (confirmed_at AT TIME ZONE 'Asia/Jerusalem')::date=$1::date)::text AS confirmed_today FROM audience_subscriptions`,[day]);
    const t=traffic.rows[0], s=subscribers.rows[0];
    return json(200,{source:'BTC2G PostgreSQL audience v1',measuredAt,day,dayTimeZone:'Asia/Jerusalem',
     traffic:t.since ? {since:t.since,through:measuredAt,pageviews:Number(t.views),visitorDays:Number(t.visitor_days),today:{estimatedVisitors:Number(t.today_visitors),pageviews:Number(t.today_views)},method:'Daily HMAC of platform IP and user-agent; estimate, not verified people or sessions'} : null,
     confirmedSubscribers:{active:Number(s.active),confirmedToday:Number(s.confirmed_today),period:{since:s.since,through:measuredAt},history:'Only confirmations collected by this service; no imported history'},accounts:null});
   }
   if (route==='digest-unsubscribe') {
    const token=req.method==='GET'?new URL(req.url,'https://local.invalid').searchParams.get('token'):body.token || new URL(req.url,'https://local.invalid').searchParams.get('token');
    if(typeof token!=='string' || !tokenPattern.test(token)) return html(400,'<p>'+messages[lang].invalid+'</p>');
    if(req.method==='GET') return html(200,`<h1>BTC2G</h1><p>${messages[lang].digestLeave}</p><form method="post"><input type="hidden" name="token" value="${token}"><button type="submit">${messages[lang].unsubscribe}</button></form>`);
    await db.query(`UPDATE audience_subscriptions SET weekly_digest=false,digest_preferences_at=now() WHERE digest_unsubscribe_token=$1`,[token]);
    return html(200,'<p>'+messages[lang].left+'</p>');
   }
   if (route==='confirm' || route==='unsubscribe') {
    const token = req.method==='GET' ? new URL(req.url,'https://local.invalid').searchParams.get('token') : body.token;
    if (typeof token!=='string' || !tokenPattern.test(token)) return html(400,'<p>'+messages[lang].invalid+'</p>');
    // Read language even for expired/used tokens; GET never mutates subscription state.
    const found=await db.query(`SELECT language,weekly_digest FROM audience_tokens WHERE ${route==='confirm'?'token_hash':'unsubscribe_hash'}=$1`,[digest(token)]);
    if(found.rows.length) lang=language(found.rows[0].language);
    const t=messages[lang], digestChoice=found.rows[0]?.weekly_digest;
    if (req.method==='GET') return html(200,`<h1>BTC2G</h1><p>${route==='confirm'?(typeof digestChoice==='boolean'?t.digestPrompt+(digestChoice?t.digestOn:t.digestOff):t.prompt):t.leave}</p><form method="post"><input type="hidden" name="token" value="${token}"><button type="submit">${route==='confirm'?t.confirm:t.unsubscribe}</button></form>`);
    const result = route==='confirm'
     ? await db.query(`WITH consumed AS (
       UPDATE audience_tokens SET used_at=now() WHERE token_hash=$1 AND expires_at>now() AND used_at IS NULL RETURNING email,weekly_digest,created_at,language
      ) UPDATE audience_subscriptions s SET confirmed_at=CASE WHEN c.weekly_digest=false THEN s.confirmed_at ELSE coalesce(s.confirmed_at,now()) END,
       weekly_digest=coalesce(c.weekly_digest,s.weekly_digest),
       digest_preferences_at=CASE WHEN c.weekly_digest IS NOT NULL THEN now() ELSE s.digest_preferences_at END,
       digest_consent_at=CASE WHEN c.weekly_digest=true THEN now() ELSE s.digest_consent_at END,
       digest_unsubscribe_token=coalesce(s.digest_unsubscribe_token,$2),
       language=c.language,unsubscribed_at=CASE WHEN c.weekly_digest=false THEN s.unsubscribed_at ELSE NULL END,token_hash=NULL,token_expires_at=NULL
       FROM consumed c WHERE s.email=c.email AND (s.unsubscribed_at IS NULL OR c.created_at>s.unsubscribed_at)
       AND (c.weekly_digest IS NULL OR s.digest_preferences_at IS NULL OR c.created_at>s.digest_preferences_at) RETURNING s.email`,[digest(token),random()])
     : await db.query(`WITH target AS (SELECT email FROM audience_tokens WHERE unsubscribe_hash=$1),
       invalidated AS (UPDATE audience_tokens SET used_at=coalesce(used_at,now()) WHERE email IN (SELECT email FROM target))
       UPDATE audience_subscriptions SET unsubscribed_at=coalesce(unsubscribed_at,now()),weekly_digest=false,digest_preferences_at=now(), token_hash=NULL,token_expires_at=NULL
       WHERE email IN (SELECT email FROM target) RETURNING email`,[digest(token)]);
    if (!result.rows.length) return html(400,'<p>'+t.invalid+'</p>');
    return html(200,'<p>'+(route==='confirm'?(typeof digestChoice==='boolean'?t.digestSaved:t.confirmed):t.left)+'</p>');
   }
   if (!env.VISITOR_HASH_SECRET || env.VISITOR_HASH_SECRET.length<32) return json(503,{error:'service_not_configured'});
   // Deploy only on Vercel, which supplies x-forwarded-for. Do not expose behind an untrusted proxy.
   const ip=String(req.headers['x-forwarded-for'] || '').split(',')[0].trim();
   if (!ip) return json(503,{error:'visitor_identity_unavailable'});
   const stamp=now(), day=dayAt(stamp);
   const hash=value=>createHmac('sha256',env.VISITOR_HASH_SECRET).update(value).digest('hex');
   if (route==='visit') {
    if (!origin) return json(403,{error:'origin_required'});
    if (!/^\/(?:[A-Za-z0-9_/-]*)(?:\.html)?$/.test(body.path || '') || body.path.length>160) return json(400,{error:'invalid_path'});
    if (/bot|crawler|spider|headless/i.test(req.headers['user-agent'] || '')) return json(202,{status:'ignored'});
    const visitor=hash(day+'\n'+ip+'\n'+String(req.headers['user-agent'] || '').slice(0,512));
    await db.query(`INSERT INTO audience_visitors(day,visitor_hash) VALUES($1,$2) ON CONFLICT(day,visitor_hash) DO UPDATE SET views=audience_visitors.views+1,last_seen_at=now()`,[day,visitor]);
    return json(202,{status:'recorded'});
   }
   if (route!=='subscribe' && route!=='digest-request') return json(404,{error:'not_found'});
   if (!origin) return json(403,{error:'origin_required'});
   if (!env.RESEND_API_KEY || !env.MAIL_FROM || !env.API_ORIGIN || new URL(env.API_ORIGIN).protocol!=='https:') return json(503,{error:'mail_not_configured'});
   const email=typeof body.email==='string'?body.email.trim().toLowerCase():'';
   if (email.length>254 || !/^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(email) || body.consent!==true || body.consentVersion!=='2026-10-29') return json(400,{error:'email_and_explicit_consent_required'});
   const isDigest=route==='digest-request';
   if(isDigest && typeof body.weeklyDigest!=='boolean') return json(400,{error:'explicit_digest_choice_required'});
   if(isDigest && body.weeklyDigest && env.DIGEST_ENABLED!=='true') return json(503,{error:'digest_not_enabled'});
   const bucket=hash('subscribe\n'+ip+'\n'+stamp.toISOString().slice(0,13));
   const rate=await db.query(`INSERT INTO audience_rate_limits(bucket,expires_at) VALUES($1,now()+interval '2 hours') ON CONFLICT(bucket) DO UPDATE SET attempts=audience_rate_limits.attempts+1 RETURNING attempts`,[bucket]);
   if (rate.rows[0].attempts>5) return json(429,{error:'rate_limited'});
   lang=language(body.lang);
   const t=messages[lang], token=random(), unsubscribe=random(), attempt=digest(token);
   const inserted=await db.query(`INSERT INTO audience_subscriptions(email,consent_version,token_hash,token_expires_at,unsubscribe_hash,language,delivery_state,last_attempt_hash)
    VALUES($1,$2,NULL,NULL,$4,$5,'pending',$3)
    ON CONFLICT(email) DO UPDATE SET requested_at=now(),consent_version=excluded.consent_version,language=excluded.language,
     delivery_state='pending',last_attempt_hash=excluded.last_attempt_hash
    WHERE (audience_subscriptions.confirmed_at IS NULL OR audience_subscriptions.unsubscribed_at IS NOT NULL OR $6)
     AND audience_subscriptions.requested_at < now() - CASE audience_subscriptions.delivery_state
      WHEN 'failed' THEN interval '1 minute' WHEN 'pending' THEN interval '2 minutes' ELSE interval '15 minutes' END
    RETURNING email`,[email,body.consentVersion,attempt,digest(unsubscribe),lang,isDigest]);
   // Do not turn an unsent/cooldown retry into a successful receipt.
   // Generic response for active subscriptions too, to avoid exposing membership.
   if (!inserted.rows.length) {res.setHeader('Retry-After','900');return json(429,{error:'request_not_sent',retryAfterSeconds:900,confirmed:false});}
   await db.query(`INSERT INTO audience_tokens(token_hash,unsubscribe_hash,email,language,expires_at,weekly_digest) VALUES($1,$2,$3,$4,now()+interval '24 hours',$5)`,[attempt,digest(unsubscribe),email,lang,isDigest?body.weeklyDigest:null]);
   const base=new URL(env.API_ORIGIN).origin;
   let state='unknown';
   try {
    const response=await send('https://api.resend.com/emails',{method:'POST',signal:AbortSignal.timeout(10000),headers:{Authorization:'Bearer '+env.RESEND_API_KEY,'Content-Type':'application/json','Idempotency-Key':'btc2g-'+attempt},body:JSON.stringify({from:env.MAIL_FROM,to:[email],subject:isDigest?t.digestSubject:t.subject,text:`${isDigest?t.digestIntro+(body.weeklyDigest?t.digestOn:t.digestOff):t.intro}\n${base}/api/confirm?token=${token}&lang=${lang}\n\n${t.ignore}\n${t.unsubscribe}: ${base}/api/unsubscribe?token=${unsubscribe}&lang=${lang}`})});
    state=response.ok?'accepted':(response.status>=400 && response.status<500 && response.status!==408?'failed':'unknown');
   } catch {} // Timeout is ambiguous: retain this token and every earlier valid link.
   await db.query(`UPDATE audience_tokens SET delivery_state=$2 WHERE token_hash=$1`,[attempt,state]);
   await db.query(`UPDATE audience_subscriptions SET delivery_state=$2 WHERE email=$1 AND last_attempt_hash=$3`,[email,state,attempt]);
   if(state!=='accepted') {
    const retry=state==='failed'?60:900;
    res.setHeader('Retry-After',String(retry));
    return json(503,{error:'mail_'+state,confirmed:false,retryAfterSeconds:retry,message:'Receipt could not be confirmed. Retry only by explicit user action after the stated delay.'});
   }
   return json(202,genericReceipt);
  } catch (error) {
   if (error instanceof SyntaxError) return json(400,{error:'invalid_body'});
   // Never log email, bearer tokens, connection strings or provider response bodies.
   return json(503,{error:'service_unavailable',message:'Receipt could not be confirmed. Do not retry automatically.'});
  }
 };
}
module.exports={createHandler,dayAt,digest};
