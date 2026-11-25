// Staged adapter. Loading this file does not collect visits or submit anything.
// Instantiate only after the deployment and explicit consent UI have been reviewed.
(() => {
 window.createBTC2GAudienceClient = function(apiOrigin) {
  const base=new URL(apiOrigin);
  if(base.protocol!=='https:' || base.pathname!=='/' || base.search || base.hash || base.username || base.password) throw new Error('An HTTPS API origin is required');
  let visitRecorded=false;
  async function post(route,body) {
   const response=await fetch(new URL('/api/'+route,base),{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(15000)});
   if(!response.ok) {
    const error=new Error('Receipt could not be confirmed; do not retry automatically');
    error.retryAfterSeconds=Number(response.headers.get('Retry-After')) || null;
    throw error;
   }
   return response.json();
  }
  return {
   async subscribe(email,consent,lang='en') {
    if(consent!==true) throw new Error('Explicit newsletter consent is required');
    const result=await post('subscribe',{email,consent,consentVersion:'2026-10-29',lang:['en','ru','he'].includes(lang)?lang:'en'});
    if(result.status!=='request_received' || result.confirmed!==false) throw new Error('Unexpected response');
    return result; // UI: request received; check your email. Never display "subscribed" here.
   },
   async recordVisit() {
    if(visitRecorded || navigator.webdriver || location.protocol==='file:') return;
    visitRecorded=true;
    try {
     const result=await post('visit',{path:location.pathname});
     if(!['recorded','ignored'].includes(result.status)) throw new Error('Unexpected response');
    } catch(error) { visitRecorded=false; throw error; }
   }
  };
 };
})();
