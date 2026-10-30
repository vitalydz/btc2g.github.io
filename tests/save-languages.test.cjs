const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const read=p=>fs.readFileSync(p,'utf8');
const expected=['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
function page(lang,blocked=false){
 const nodes=new Map(); const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,textContent:'',dataset:{},listeners:{},addEventListener(k,f){this.listeners[k]=f},setAttribute(k,v){this[k]=v},getAttribute(k){return this[k]}});return nodes.get(id)};
 for(const id of ['save-amount','save-moonpay','save-topper','save-confirm','save-currency','save-result','save-language','save-compare','save-clear','moonpay-rate','topper-rate'])node(id);
 node('save-currency').value='USD';node('save-amount').value='100';
 const fields=[...read('save.html').matchAll(/data-text="([^"]+)"/g)].map(x=>({dataset:{text:x[1]},textContent:''}));
 const links=[{href:'index.html',getAttribute(){return this.href},setAttribute(k,v){this[k]=v}},{href:'index.html#calculator',getAttribute(){return this.href},setAttribute(k,v){this[k]=v}}];
 let saved='en';const document={documentElement:{},getElementById:node,querySelector:()=>({content:''}),querySelectorAll(s){if(s==='[data-text]')return fields;if(s.startsWith('a['))return links;return []}};
 const c={window:{},document,location:{search:'?lang='+lang,href:'https://btc2g.com/save.html?lang='+lang},localStorage:{getItem(){if(blocked)throw Error();return saved},setItem(k,v){if(blocked)throw Error();saved=v}},navigator:{language:'en'},URL,URLSearchParams,Intl};vm.createContext(c);
 for(const p of ['languages.js','ui-text.js','quote-compare.js','save-text.js','save.js'])vm.runInContext(read('assets/js/'+p),c);
 return {c,node,fields,links,saved:()=>saved};
}
test('SAVE selector and dictionaries contain the same 19 languages and complete templates',()=>{
 const c={window:{}};vm.runInNewContext(read('assets/js/save-text.js'),c);assert.deepEqual(Object.keys(c.window.BTC2GSaveText).sort(),[...expected].sort());
 const opts=[...read('save.html').matchAll(/option value="([a-z]{2})"/g)].map(x=>x[1]);assert.deepEqual(opts.sort(),[...expected].sort());
 const keys=Object.keys(c.window.BTC2GSaveText.en).sort();for(const l of expected){const t=c.window.BTC2GSaveText[l];assert.deepEqual(Object.keys(t).sort(),keys);for(const k of keys)assert.ok(t[k]?.trim(),l+':'+k);for(const p of ['{provider}','{sats}','{percent}'])assert.ok(t.more.includes(p));}
});
for(const lang of expected)test('SAVE renders and calculates in '+lang,()=>{
 const {c,node,fields,links,saved}=page(lang);assert.equal(c.document.documentElement.lang,lang);assert.equal(c.document.documentElement.dir,['he','ar'].includes(lang)?'rtl':'ltr');assert.equal(saved(),lang);
 for(const f of fields)assert.equal(f.textContent,c.window.BTC2GSaveText[lang][f.dataset.text]);
 node('save-moonpay').value='0.001';node('save-topper').value='0.0011';node('save-confirm').checked=true;node('save-compare').listeners.click();assert.ok(node('save-result').textContent.includes('Topper'));assert.ok(!node('save-result').textContent.includes('{'));
 for(const a of links)assert.ok(a.href.includes('lang='+lang));
 node('save-language').listeners.change({target:{value:lang==='en'?'de':'en'}});assert.ok(node('save-result').textContent.includes('Topper'));
});
test('regional language tags and denied storage are supported',()=>{assert.equal(page('ar-SA',true).c.document.documentElement.lang,'ar');assert.equal(page('xx',true).c.document.documentElement.lang,'en')});
test('homepage no longer restricts SAVE to three languages',()=>{const s=read('index.html');assert.ok(!s.includes('EN / RU / HE'));assert.ok(s.includes("'save.html?lang=' + encodeURIComponent(lang)"));});
function premiumPage(lang){
 const nodes=new Map();
 const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',children:[],setAttribute(k,v){this[k]=v},replaceChildren(...v){this.children=v},addEventListener(k,v){this[k]=v}});return nodes.get(id)};
 const document={documentElement:{},title:'',getElementById:node,querySelector:node,querySelectorAll(){return []},createElement(){return node('created-'+nodes.size)},createTextNode(text){return {textContent:text}}};
 let saved=null;const location={search:'?lang='+lang,href:'https://btc2g.com/premium.html?lang='+lang,assign(value){this.assigned=value}};
 const c={window:{},document,URL,URLSearchParams,location,localStorage:{getItem:()=>saved,setItem(k,v){saved=v}},navigator:{language:'en'}};
 vm.createContext(c);vm.runInContext(read('assets/js/languages.js'),c);vm.runInContext(read('assets/js/ui-text.js'),c);vm.runInContext(read('assets/js/subscription-request.js'),c);vm.runInContext(read('assets/js/premium-language.js'),c);return {c,nodes};
}
test('Premium selector and dictionaries contain the same 19 complete languages',()=>{
 const {c}=premiumPage('en');const html=read('premium.html');const options=[...html.matchAll(/<option value="([a-z]{2})"/g)].map(match=>match[1]);
 assert.deepEqual(options.sort(),[...expected].sort());assert.deepEqual(Object.keys(c.window.BTC2GPremiumTranslations).sort(),[...expected].sort());
 const keys=Object.keys(c.window.BTC2GPremiumTranslations.en).sort();for(const lang of expected){const t=c.window.BTC2GPremiumTranslations[lang];assert.deepEqual(Object.keys(t).sort(),keys);for(const key of keys)assert.ok(t[key]?.trim(),lang+':'+key);}
});
for(const lang of expected)test('Premium page renders in '+lang,()=>{
 const {c,nodes}=premiumPage(lang);const t=c.window.BTC2GPremiumTranslations[lang];
 assert.equal(c.document.documentElement.lang,lang);assert.equal(c.document.documentElement.dir,['he','ar'].includes(lang)?'rtl':'ltr');
 assert.equal(nodes.get('early-access').lang,lang);assert.equal(nodes.get('early-access').dir,['he','ar'].includes(lang)?'rtl':'ltr');
 assert.equal(nodes.get('#early-access-email')['aria-label'],t.email);assert.equal(nodes.get('.hero h1').textContent,t.heroTitle);
 assert.equal(c.window.BTC2GPremiumText,t);assert.ok(t.received);assert.ok(t.error);
});
