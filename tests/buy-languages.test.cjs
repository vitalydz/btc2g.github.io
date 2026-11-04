const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'buy.html'), 'utf8');
const moduleScript = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const expected = ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
const freeMarkers = {en:'free',ru:'бесплатный',he:'בחינם',de:'Kostenloses',fr:'gratuit',es:'gratis',zh:'免费',it:'gratuito',pt:'grátis',ar:'مجاني',hi:'मुफ़्त',ja:'無料',ko:'무료',tr:'Ücretsiz',pl:'bezpłatne',nl:'Gratis',uk:'безкоштовний',id:'gratis',vi:'miễn phí'};

function render(lang) {
  const nodes = new Map();
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, { textContent:'', value:'', href:'', title:'', setAttribute(name,value){ this[name]=value; }, addEventListener(){} });
    return nodes.get(id);
  };
  const context = {
    window: {}, document: { documentElement:{}, getElementById:element },
    localStorage: { getItem(){ return null; }, setItem(){} },
    navigator: { language:'en' }, location: { search:'?lang=' + lang }, URLSearchParams,
  };
  vm.createContext(context);
  vm.runInContext(moduleScript, context);
  return { context, element };
}

test('Buy Bitcoin offers and renders all 19 site languages', () => {
  const options = [...html.matchAll(/<option value="([a-z]{2})"/g)].map(match => match[1]);
  assert.deepEqual(options, expected);
  for (const lang of expected) {
    const { context, element } = render(lang);
    assert.equal(context.document.documentElement.lang, lang);
    assert.equal(context.document.documentElement.dir, ['he','ar'].includes(lang) ? 'rtl' : 'ltr');
    assert.ok(element('title').textContent.trim(), lang);
    assert.ok(element('provider-notice').textContent.trim(), lang);
    assert.equal(element('save-link').href, 'save.html?lang=' + lang);
    assert.equal(element('account-link').href, 'https://btc2g-accounts.btc2g.workers.dev/?lang=' + lang);
    assert.ok(element('account-link').textContent.trim(), lang);
    assert.ok(element('account-link').textContent.includes(freeMarkers[lang]), `${lang} free account label`);
    assert.equal(element('back-home').href, 'index.html?lang=' + lang);
  }
});
