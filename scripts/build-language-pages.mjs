import {parseHTML} from 'linkedom';
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const languages = ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
export const pages = ['index.html','save.html','scenario.html','premium.html'];
const origin = 'https://btc2g.com';
export const pageURL = (page, lang) => lang === 'en' ? `${origin}/${page === 'index.html' ? '' : page}` : `${origin}/${lang}/${page === 'index.html' ? '' : page}`;

export async function render(page, lang) {
  if (!pages.includes(page) || !languages.includes(lang)) throw new Error('Unsupported language page');
  const {document, Event} = parseHTML(fs.readFileSync(path.join(root,page),'utf8'));
  // linkedom's SELECT exposes a getter; browser localization code sets its value.
  for (const select of document.querySelectorAll('select')) {
    Object.defineProperty(select,'value',{configurable:true,get(){return this.querySelector('option[selected]')?.value || this.querySelector('option')?.value || '';},set(value){for(const option of this.querySelectorAll('option')) option.toggleAttribute('selected',option.value===String(value));}});
  }
  const store = new Map([['lang',lang]]);
  const location = new URL(`${origin}/${page}?lang=${lang}`);
  const marketTime=Date.parse(JSON.parse(fs.readFileSync(path.join(root,'assets/btc_gold_forecast_meta.json'),'utf8')).last_updated_utc);
  class BuildDate extends Date {constructor(...args){super(...(args.length?args:[marketTime]));}static now(){return marketTime;}}
  const sandbox = {document, Event, location, URL, URLSearchParams, Intl, Date:BuildDate, AbortController, AbortSignal,
    navigator:{language:lang}, localStorage:{getItem:key=>store.get(key)||null,setItem:(key,value)=>store.set(key,String(value)),removeItem:key=>store.delete(key)},
    setTimeout:()=>0,clearTimeout(){},setInterval:()=>0,clearInterval(){},addEventListener(){},
    matchMedia:()=>({matches:false,addEventListener(){}}), requestAnimationFrame:()=>0,
    history:{replaceState(){}}, console:{log(){},warn(){},error(){}},
    // Build-time localization must never contact providers or send forms/analytics.
    fetch:()=>new Promise(()=>{}), getComputedStyle:()=>({getPropertyValue:()=>''})};
  sandbox.window=sandbox;
  sandbox.globalThis=sandbox;
  const context=vm.createContext(sandbox);
  for(const script of [...document.querySelectorAll('script')]) {
    if(script.type && script.type !== 'text/javascript') continue;
    const src=script.getAttribute('src');
    if(src && (!src.startsWith('assets/js/') || /traffic|loading-promo/.test(src))) continue;
    const code=src ? fs.readFileSync(path.join(root,src.split('?')[0]),'utf8') : script.textContent;
    vm.runInContext(code,context,{filename:src||page,timeout:3000});
  }
  document.dispatchEvent(new Event('DOMContentLoaded'));
  await Promise.resolve();
  document.documentElement.lang=lang;
  document.documentElement.dir=['he','ar'].includes(lang)?'rtl':'ltr';
  document.documentElement.dataset.staticLanguage=lang;
  const title=document.querySelector('h1')?.textContent.trim();
  if(!title) throw new Error(`Missing heading: ${page}/${lang}`);
  document.title=`${title} | BTC2G`;
  const description=`${title} — BTC2G.`;
  for(const meta of document.querySelectorAll('meta[name="description"],meta[property="og:description"],meta[name="twitter:description"]')) meta.content=description;
  for(const meta of document.querySelectorAll('meta[property="og:title"],meta[name="twitter:title"]')) meta.content=document.title;
  for(const meta of document.querySelectorAll('meta[property="og:url"]')) meta.content=pageURL(page,lang);
  const canonical=document.querySelector('link[rel="canonical"]');
  if(!canonical) throw new Error(`Missing canonical: ${page}`);
  canonical.href=pageURL(page,lang);
  document.querySelectorAll('link[hreflang]').forEach(node=>node.remove());
  for(const code of [...languages,'x-default']) {
    const link=document.createElement('link');link.setAttribute('rel','alternate');link.setAttribute('hreflang',code);link.setAttribute('href',pageURL(page,code==='x-default'?'en':code));document.head.append(link);
  }
  // Resolve original assets against the root, never against a language folder.
  for(const element of document.querySelectorAll('[href],[src]')) for(const attribute of ['href','src']) {
    const value=element.getAttribute(attribute);
    if(!value || /^(?:[a-z]+:|\/|#)/i.test(value)) continue;
    element.setAttribute(attribute,`/${value}`);
  }
  for(const a of document.querySelectorAll('a[href]')) {
    if(a.getAttribute('href').startsWith('#')) {a.href=new URL(pageURL(page,lang)).pathname+a.getAttribute('href');continue;}
    const url=new URL(a.href,origin);
    const name=url.pathname.slice(1)||'index.html';
    if(url.origin===origin && pages.includes(name)) {
      const target=new URL(pageURL(name,lang));target.hash=url.hash;a.href=target.pathname+target.hash;
    }
  }
  const base=document.createElement('base');base.setAttribute('href',origin+'/');document.head.insertBefore(base,document.head.firstChild);
  const bootstrap=document.createElement('script');
  bootstrap.src='/assets/js/static-language.js';
  document.querySelectorAll('script[src="/assets/js/static-language.js"]').forEach(node=>node.remove());
  document.head.insertBefore(bootstrap,document.head.firstChild);
  return '<!DOCTYPE html>\n'+document.documentElement.outerHTML.replace(/^[ \t]+(?=\r?$)/gm,'')+'\n';
}

export async function build() {
  let count=0;
  for(const lang of languages.filter(code=>code!=='en')) for(const page of pages) {
    const directory=path.join(root,lang);fs.mkdirSync(directory,{recursive:true});
    fs.writeFileSync(path.join(directory,page),await render(page,lang));count++;
  }
  // English pages stay the source of truth; advertise reciprocal variants there.
  for(const page of pages) {
    let html=fs.readFileSync(path.join(root,page),'utf8');
    html=html.replace(/\s*<link[^>]+hreflang=["'][^"']+["'][^>]*>/g,'');
    const links=[...languages,'x-default'].map(lang=>`  <link rel="alternate" hreflang="${lang}" href="${pageURL(page,lang==='x-default'?'en':lang)}" />`).join('\n');
    html=html.replace('</head>',links+'\n</head>');
    if(!html.includes('assets/js/static-language.js'))html=html.replace('</head>','  <script src="/assets/js/static-language.js"></script>\n</head>');
    fs.writeFileSync(path.join(root,page),html);
  }
  let sitemap=fs.readFileSync(path.join(root,'sitemap.xml'),'utf8');
  sitemap=sitemap.replace(/\s*<url><loc>https:\/\/btc2g\.com\/(?:ru|he|de|fr|es|zh|it|pt|ar|hi|ja|ko|tr|pl|nl|uk|id|vi)\/[^<]*<\/loc><\/url>/g,'');
  const urls=languages.filter(lang=>lang!=='en').flatMap(lang=>pages.map(page=>`  <url><loc>${pageURL(page,lang)}</loc></url>`));
  fs.writeFileSync(path.join(root,'sitemap.xml'),sitemap.replace('</urlset>',urls.join('\n')+'\n</urlset>'));
  console.log(`Built ${count} static language pages from the existing UI translations.`);
}
if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) await build();
