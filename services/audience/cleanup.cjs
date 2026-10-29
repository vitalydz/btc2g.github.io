const {Pool}=require('pg');
(async()=>{
 if(!process.env.DATABASE_URL)throw Error('Missing database');
 const pool=new Pool({connectionString:process.env.DATABASE_URL});
 try{
  await pool.query("DELETE FROM audience_rate_limits WHERE expires_at<now()");
  await pool.query("DELETE FROM audience_visitors WHERE day < (now() AT TIME ZONE 'Asia/Jerusalem')::date - 400");
  await pool.query("DELETE FROM audience_subscriptions WHERE confirmed_at IS NULL AND requested_at<now()-interval '30 days'");
  console.log('Audience retention cleanup completed');
 }finally{await pool.end();}
})().catch(()=>{console.error('Audience cleanup failed');process.exitCode=1;});
