import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import {parseHTML} from 'linkedom';
import {root,languages,openGraphLocales,pages,pageURL,render} from './build-language-pages.mjs';

test('published language HTML has real translated headings, reciprocal alternates and correct asset roots',()=>{
  for(const page of pages) {
    const english=parseHTML(fs.readFileSync(path.join(root,page),'utf8')).document.querySelector('h1').textContent.trim();
    for(const lang of languages.filter(code=>code!=='en')) {
      const {document:d}=parseHTML(fs.readFileSync(path.join(root,lang,page),'utf8'));
      assert.equal(d.documentElement.lang,lang);
      assert.equal(d.documentElement.dir,['he','ar'].includes(lang)?'rtl':'ltr');
      assert.notEqual(d.querySelector('h1').textContent.trim(),english,`${lang}/${page} must contain a translated heading without running JS`);
      assert.equal(d.querySelector('link[rel="canonical"]').href,pageURL(page,lang));
      const openGraphLocale=d.querySelector('meta[property="og:locale"]');
      if(openGraphLocale) assert.equal(openGraphLocale.content,openGraphLocales[languages.indexOf(lang)],`${lang}/${page} must declare its own Open Graph locale`);
      assert.equal(d.querySelector('base').getAttribute('href'),'https://btc2g.com/');
      const alternates=[...d.querySelectorAll('link[hreflang]')];
      assert.equal(alternates.length,20);
      for(const code of languages) assert.equal(alternates.find(a=>a.getAttribute('hreflang')===code).href,pageURL(page,code));
      for(const input of d.querySelectorAll('input[type="email"],input[type="password"]')) assert.ok(!input.getAttribute('value'));
      for(const script of d.querySelectorAll('script[src]'))assert.ok(script.src.startsWith('/')||script.src.startsWith('https://'));
      assert.equal(d.querySelectorAll('script[src="/assets/js/static-language.js"]').length,1);
      if(page==='index.html') {
        assert.equal(d.querySelector('#refresh-chart-btn').hasAttribute('disabled'),false,`${lang}: initial market refresh must not be locked`);
        assert.equal(d.querySelector('#refresh-chart-btn').hasAttribute('aria-busy'),false);
        const expected=d.querySelector('#site-purpose').textContent.trim();
        assert.equal(d.querySelector('meta[name="description"]').content,expected);
        assert.equal(d.querySelector('meta[property="og:description"]').content,expected);
        assert.equal(d.querySelector('meta[name="twitter:description"]').content,expected);
        assert.equal(JSON.parse(d.querySelector('script[type="application/ld+json"]').textContent).description,expected);
      }
      if(page==='save.html') {
        assert.equal(d.querySelector('meta[name="description"]').content,d.querySelector('.save-shell > .intro').textContent.trim());
      }
      if(page==='scenario.html') {
        const expected=`${d.querySelector('#scenario-intro').textContent.trim()} ${d.querySelector('#disclaimer-title').textContent.trim()}`;
        assert.equal(d.querySelector('meta[name="description"]').content,expected);
        assert.equal(d.querySelector('meta[property="og:description"]').content,expected);
        assert.equal(d.querySelector('meta[name="twitter:description"]').content,expected);
        const structured=JSON.parse(d.querySelector('script[type="application/ld+json"]').textContent);
        assert.equal(structured.description,expected);
        assert.equal(structured.inLanguage,lang);
      }
    }
  }
});

test('language generation rejects unsupported routes and preserves model/privacy disclosures',async()=>{
  await assert.rejects(render('admin.html','ru'));
  await assert.rejects(render('index.html','xx'));
  const html=await render('scenario.html','ru');
  assert.match(html,/не прогноз/);
  assert.match(html,/не сохраняются/);
  assert.equal(html,await render('scenario.html','ru'),'Repeated builds must be deterministic');
});

test('language switch uses a crawlable path and internal navigation preserves language',()=>{
  const listeners=new Map();const calls=[];
  const context=vm.createContext({URL,URLSearchParams,location:{pathname:'/he/',search:'?chart=purchasingPower&utm_source=telegram&secret=drop',hash:'#forecast',origin:'https://btc2g.com',href:'https://btc2g.com/he/?chart=purchasingPower&utm_source=telegram&secret=drop#forecast',assign:url=>calls.push(url),replace:url=>calls.push(url)},localStorage:{setItem(){}},document:{addEventListener:(type,fn)=>listeners.set(type,fn)}});
  vm.runInContext(fs.readFileSync(path.join(root,'assets/js/static-language.js'),'utf8'),context);
  let stopped=false;
  listeners.get('change')({target:{id:'language-select',value:'ar'},stopImmediatePropagation(){stopped=true;}});
  assert.equal(stopped,true);assert.equal(calls.pop(),'/ar/?chart=purchasingPower&utm_source=telegram#forecast');
  listeners.get('click')({target:{closest:()=>({href:'https://btc2g.com/scenario.html?lang=he&btc=10&gold=-5&target=20&utm_source=telegram&secret=drop',target:''})},preventDefault(){}});
  assert.equal(calls.pop(),'/he/scenario.html?btc=10&gold=-5&target=20&utm_source=telegram');
  listeners.get('click')({target:{closest:()=>({href:'https://btc2g.com/he/#calculator',target:''})},preventDefault(){}});
  assert.equal(calls.pop(),'/he/?chart=purchasingPower&utm_source=telegram#calculator');
});

test('localized homepage build releases its mocked market refresh lock',async()=>{
  const {document}=parseHTML(await render('index.html','ru'));
  assert.equal(document.querySelector('#refresh-chart-btn').hasAttribute('disabled'),false);
  assert.equal(document.querySelector('#refresh-chart-btn').hasAttribute('aria-busy'),false);
});
