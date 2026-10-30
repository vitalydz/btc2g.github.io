const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {calculateScenario}=require('../assets/js/scenario-tools');
const source=fs.readFileSync(require.resolve('../assets/js/scenario-tools'),'utf8');
async function setup({blocked=false,failed=false,search=''}={}){
 const nodes=new Map();let init,copied='',requests=0;
 const el=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',disabled:true,listeners:{},attributes:{},setAttribute(k,v){this.attributes[k]=v;},addEventListener(n,fn){this.listeners[n]=fn;}});return nodes.get(id);};
 const context={URL,URLSearchParams,Date,Intl,AbortSignal,document:{documentElement:{},getElementById:el,addEventListener(n,fn){init=fn;}},location:{href:'https://btc2g.com/scenario.html'+search,search},localStorage:{getItem(){if(blocked)throw Error('SecurityError');return'en';},setItem(){}},navigator:{language:'en',clipboard:{async writeText(value){copied=value;}}},fetch:async()=>{requests++;if(failed)throw Error('network');return{ok:true,json:async()=>({btc_price_usd:80000,gold_price_usd:4000,ratio:20,last_updated:'2026-10-29'})};}};
 context.window=context;vm.runInNewContext(source,context);init();await new Promise(resolve=>setImmediate(resolve));
 return{el,context,copied:()=>copied,requests:()=>requests};
}
test('invalid target clears previous values and blocks copying; correction recovers',async()=>{
 const h=await setup();assert.equal(h.el('required-btc').textContent,'$120,000.00');
 h.el('target-ratio').value='';h.el('target-ratio').listeners.input();
 for(const id of ['required-btc','future-btc','future-gold','future-ratio','ratio-difference'])assert.equal(h.el(id).textContent,'—');
 assert.equal(h.el('target-ratio').attributes['aria-invalid'],'true');
 assert.equal(h.el('copy-scenario').disabled,true);assert.equal(h.el('share-scenario').disabled,true);
 await h.el('copy-scenario').listeners.click();assert.equal(h.copied(),'');
 h.el('target-ratio').value='25';h.el('target-ratio').listeners.input();
 assert.equal(h.el('required-btc').textContent,'$100,000.00');assert.equal(h.el('copy-scenario').disabled,false);
 await h.el('copy-scenario').listeners.click();assert.match(h.copied(),/100,000/);
});
test('blocked storage does not prevent initialization or market loading',async()=>{
 const h=await setup({blocked:true});assert.equal(h.requests(),1);assert.equal(h.el('target-ratio').disabled,false);assert.equal(h.context.document.documentElement.lang,'en');
});
test('changing language preserves failed/loaded states instead of inventing a new load',async()=>{
 for(const failed of [true,false]){
  const h=await setup({failed});h.el('scenario-language').listeners.change({target:{value:'ru'}});
  assert.match(h.el('data-status').textContent,failed?/недоступны/:/загружены/);assert.equal(h.requests(),1);
 }
});
test('invalid input, out-of-range target and overflowing derived results are rejected',()=>{
 const market={btc:80000,gold:4000,ratio:20};
 for(const value of ['', ' ', null, false, 0,-1,10001,1e308,Infinity])assert.equal(calculateScenario(market,0,0,value),null,String(value));
 assert.equal(calculateScenario({btc:1e308,gold:1e308,ratio:1},0,0,10000),null);
 assert.equal(calculateScenario(market,301,0,30),null);assert.equal(calculateScenario(market,0,101,30),null);
});
test('inherited language names and regional tags safely fall back or normalize',async()=>{
 assert.equal((await setup({search:'?lang=constructor'})).context.document.documentElement.lang,'en');
 assert.equal((await setup({search:'?lang=he-IL'})).context.document.documentElement.dir,'rtl');
});
