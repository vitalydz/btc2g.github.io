(function () {
  'use strict';
  const $=id=>document.getElementById(id), texts=window.BTC2GContestText;
  const names={en:'English',ru:'Русский',he:'עברית',de:'Deutsch',fr:'Français',es:'Español',zh:'中文',it:'Italiano',pt:'Português',ar:'العربية',hi:'हिन्दी',ja:'日本語',ko:'한국어',tr:'Türkçe',pl:'Polski',nl:'Nederlands',uk:'Українська',id:'Bahasa Indonesia',vi:'Tiếng Việt'};
  let lang='en',state=null,connected=false,busy=false,offset=0;
  // Planned public deadlines are display-only; they never enable public writes.
  // Closing instants are exclusive. The UI shows the preceding final second.
  const publicSchedule = Object.freeze({
    opens: Date.parse('2026-11-01T00:00:00+02:00') / 1000,
    submits: Date.parse('2027-01-01T00:00:00+02:00') / 1000,
    closes: Date.parse('2027-01-02T00:00:00+02:00') / 1000,
    runoff_closes: Date.parse('2027-03-02T00:00:00+02:00') / 1000
  });
  const pending=new Map();
  lang=window.BTC2GLanguages.requested();
  if (!texts[lang]) lang='en';
  const t=key=>texts[lang][key];
  function node(tag,value,className) {const n=document.createElement(tag);if(value!==undefined)n.textContent=value;if(className)n.className=className;return n;}
  function message(value) {$('request-status').textContent=value;}
  function deadline(value, exclusive) {return value ? new Intl.DateTimeFormat(lang,{dateStyle:'medium',timeStyle:'medium',timeZone:'Asia/Jerusalem'}).format((value-(exclusive?1:0))*1000)+' · Asia/Jerusalem' : t('prep');}
  function canWrite(kind) {
    if (!connected||busy||!state?.me) return false;
    const r=state.round,now=(Date.now()+offset)/1000;
    return kind==='submit' ? r.phase==='submissions'&&now<r.submits : ['voting','runoff'].includes(r.phase)&&now<(r.phase==='voting'?r.closes:r.runoff_closes);
  }
  function showDetail(idea) {
    const area=$('detail-content');area.replaceChildren(node('h2',idea.title));
    for(const key of ['problem','solution','audience']) area.append(node('h3',t(key)),node('p',idea[key]));
    $('detail').showModal();
  }
  function render() {
    window.BTC2GLanguages.apply(lang);
    document.querySelectorAll('[data-t]').forEach(n=>{n.textContent=t(n.dataset.t);});
    $('language').value=lang;$('language').setAttribute('aria-label',t('language'));
    $('availability').textContent=t('prep');$('mode-note').textContent=t(state?'demo':'blocked');
    $('account').hidden=!state?.demo;
    $('logout').disabled=busy||!connected; $('retry').disabled=busy;
    $('login-form').hidden=!!state?.me;$('logout').hidden=!state?.me;
    $('idea-form').querySelectorAll('input,textarea,button').forEach(n=>{n.disabled=!canWrite('submit');});
    $('phases').replaceChildren();
    const phase=state?.round.phase==='runoff_pending'?'runoff':state?.round.phase||'draft';
    for(const key of ['draft','submissions','voting','runoff','completed']) {
      const li=node('li',t('p_'+key));if(key===phase)li.setAttribute('aria-current','step');$('phases').append(li);
    }
    $('deadlines').replaceChildren();
    for(const [key,field] of [['opens','opens'],['submissions','submits'],['voting','closes'],['runoff','runoff_closes']]) $('deadlines').append(node('dt',t(key)),node('dd',deadline(state?.round[field] ?? publicSchedule[field], field!=='opens')));
    $('empty').hidden=!!state?.ideas.length;$('cards').replaceChildren();
    for(const idea of state?.ideas||[]) {
      const card=node('article',undefined,'idea');card.append(node('h3',idea.title),node('p',idea.problem),node('span',`${t('votes')}: ${new Intl.NumberFormat(lang).format(idea.total)}`,'total'));
      const actions=node('div',undefined,'actions'),detail=node('button',t('details')),vote=node('button',state.me?.vote===idea.id?t('selected'):t('vote'),'primary');
      detail.type=vote.type='button';detail.addEventListener('click',()=>showDetail(idea));
      vote.disabled=!canWrite('vote')||!idea.eligible;vote.setAttribute('aria-pressed',String(state.me?.vote===idea.id));
      vote.addEventListener('click',()=>mutate('vote',{idea:idea.id},'confirmed'));
      actions.append(detail,vote);card.append(actions);$('cards').append(card);
    }
    $('own').replaceChildren();
    if(state?.own.length){$('own').append(node('h3',t('own')));for(const idea of state.own)$('own').append(node('p',`${idea.title} · ${t(idea.status)}${idea.reason?' · '+idea.reason:''}`));}
    $('review').hidden=state?.me?.role!=='admin';$('queue').replaceChildren();
    for(const idea of state?.queue||[]) {
      const card=node('article',undefined,'idea');card.append(node('h3',idea.title));
      for(const key of ['problem','solution','audience'])card.append(node('h4',t(key)),node('p',idea[key]));
      const label=node('label',t('reason')),reason=node('input');reason.maxLength=300;label.append(reason);card.append(label);
      for(const status of ['approved','rejected']) {
        const button=node('button',t(status==='approved'?'approve':'reject'));button.type='button';button.disabled=!connected||busy||!['submissions','draft'].includes(state.round.phase);
        button.addEventListener('click',()=>{if(status==='rejected'&&reason.value.trim().length<5){reason.focus();message(t('error'));return;}mutate('review',{idea:idea.id,status,reason:reason.value},'approved');});card.append(button);
      }
      $('queue').append(card);
    }
    if(state?.round.phase==='runoff_pending'){
      const form=node('form'),label=node('label',t('runoff')+' (UTC)'),input=node('input'),button=node('button',t('send'));
      input.type='datetime-local';input.required=true;label.append(input);form.append(label,button);
      form.addEventListener('submit',e=>{e.preventDefault();mutate('runoff',{deadline:Date.parse(input.value+'Z')/1000},'confirmed');});$('queue').append(node('p',t('prep')),form);
    }
    $('results').hidden=!state?.results.length;$('result-tables').replaceChildren();
    $('outcome').textContent=state?.round.winner ? `${t('winner')}: ${state.ideas.find(i=>i.id===state.round.winner)?.title||state.round.winner}` : state?.round.phase==='completed'?t('noWinner'):t('p_runoff');
    for(const ballot of [...new Set((state?.results||[]).map(r=>r.ballot))]) {
      const table=node('table');table.append(node('caption',ballot==='main'?t('p_voting'):t('p_runoff')+' '+ballot.split('-')[1]));
      const head=node('tr');head.append(node('th',t('ideaTitle')),node('th',t('votes')));table.append(head);
      for(const result of state.results.filter(r=>r.ballot===ballot)) {const row=node('tr');row.append(node('td',state.ideas.find(i=>i.id===result.idea)?.title||result.idea),node('td',String(result.total)));table.append(row);}
      $('result-tables').append(table);
    }
  }
  async function request(path,options={}) {
    const response=await fetch('/api/'+path,{...options,cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(10000)});
    if(!response.ok){let error;try{error=(await response.json()).error;}catch(_){error='error';}throw new Error(error);}
    return response.json();
  }
  async function refresh(clear=true) {
    try {const data=await request('contest');if(data.demo!==true||!Array.isArray(data.ideas))throw new Error('error');state=data;connected=true;offset=state.server_time*1000-Date.now();if(clear)message('');}
    catch(_){connected=false;if(state)message(t('error'));}
    render();
  }
  async function mutate(action,body,success) {
    if(busy||!connected)return;
    busy=true;render();
    const signature=JSON.stringify([action,body]);if(!pending.has(signature))pending.set(signature,crypto.randomUUID());
    try {
      await request(action,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf||'','Idempotency-Key':pending.get(signature)},body:JSON.stringify(body)});
      pending.delete(signature);if(action==='submit')$('idea-form').reset();await refresh(false);message(t(success));
    }catch(error){connected=false;message(t(error.message==='auth'?'auth':error.message==='rate'?'rate':'error'));}
    finally{busy=false;render();}
  }
  $('language').replaceChildren(...Object.entries(names).map(([code,name])=>{const o=node('option',name);o.value=code;return o;}));
  $('language').addEventListener('change',e=>{lang=e.target.value;window.BTC2GLanguages.apply(lang);message('');render();});
  $('login-form').addEventListener('submit',async e=>{e.preventDefault();if(busy)return;busy=true;try{await request('login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:$('code').value})});$('code').value='';await refresh();}catch(_){message(t('auth'));}finally{busy=false;render();}});
  $('logout').addEventListener('click',()=>mutate('logout',{},'prep'));
  $('retry').addEventListener('click',()=>refresh());
  $('idea-form').addEventListener('submit',e=>{e.preventDefault();if(!canWrite('submit')||!$('consent').checked)return;mutate('submit',{title:$('idea-title').value,problem:$('problem').value,solution:$('solution').value,audience:$('audience').value},'submitted');});
  $('detail-close').addEventListener('click',()=>$('detail').close());
  $('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText('https://btc2g.com/contest.html#ideas');message(t('copied'));}catch(_){message(t('error'));}});
  setInterval(()=>{if(state&&!busy)refresh(false);},30000);
  render();refresh();
})();
