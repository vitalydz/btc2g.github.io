(() => {
  const languages=['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
  const pages=['index.html','save.html','scenario.html','premium.html'];
  const parts=location.pathname.split('/').filter(Boolean);
  const language=languages.includes(parts[0])?parts[0]:null;
  const page=(language?parts[1]:parts[0])||'index.html';
  if(!pages.includes(page))return;
  const target=(lang,file)=>`${lang==='en'?'/':`/${lang}/`}${file==='index.html'?'':file}`;
  const preservedSearch=(file,search=location.search)=>{
    const allowed={
      'index.html':['chart','utm_source'],
      'save.html':['utm_source'],
      'scenario.html':['btc','gold','target','utm_source'],
      'premium.html':['utm_source'],
    }[file]||[];
    const source=new URLSearchParams(search);const kept=new URLSearchParams();
    for(const name of allowed){const value=source.get(name);if(value!==null)kept.set(name,value);}
    const query=kept.toString();return query?`?${query}`:'';
  };
  const requested=new URLSearchParams(location.search).get('lang');
  if(requested && languages.includes(requested) && requested!==(language||'en')) {
    location.replace(target(requested,page)+preservedSearch(page)+location.hash);return;
  }
  try{localStorage.setItem('lang',language||'en');}catch{}
  document.addEventListener('change',event=>{
    if(!['language-select','save-language','scenario-language','premium-language'].includes(event.target.id))return;
    const lang=event.target.value;if(!languages.includes(lang))return;
    event.stopImmediatePropagation();location.assign(target(lang,page)+preservedSearch(page)+location.hash);
  },true);
  document.addEventListener('click',event=>{
    const a=event.target.closest?.('a[href]');if(!a || event.ctrlKey || event.metaKey || a.target==='_blank')return;
    const url=new URL(a.href,location.href);
    if(url.origin!==location.origin)return;
    const pathParts=url.pathname.split('/').filter(Boolean);
    const targetLanguage=languages.includes(pathParts[0])?pathParts[0]:null;
    const file=(targetLanguage?pathParts[1]:pathParts[0])||'index.html';
    if(!pages.includes(file))return;
    if(!language)return;
    if(targetLanguage && targetLanguage!==language)return;
    const search=url.search||(file===page?location.search:'');
    event.preventDefault();location.assign(target(language,file)+preservedSearch(file,search)+url.hash);
  });
})();
