const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const homepage = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

test('homepage social preview names BTC2G and describes its image', () => {
  assert.match(homepage, /property="og:site_name" content="BTC2G"/);
  assert.match(homepage, /property="og:image:type" content="image\/png"/);
  assert.match(homepage, /property="og:image:width" content="2800"/);
  assert.match(homepage, /property="og:image:height" content="1400"/);
  assert.match(homepage, /property="og:image:alt" content="[^"]*Bitcoin-to-gold ratio[^"]*"/);
  assert.match(homepage, /name="twitter:image:alt" content="[^"]*Bitcoin-to-gold ratio[^"]*"/);
});

test('homepage social preview declares all supported locale variants', () => {
  assert.match(homepage, /property="og:locale" content="en_US"/);
  const alternateLocales = [...homepage.matchAll(/property="og:locale:alternate" content="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(alternateLocales, ['ru_RU','he_IL','de_DE','fr_FR','es_ES','zh_CN','it_IT','pt_PT','ar_SA','hi_IN','ja_JP','ko_KR','tr_TR','pl_PL','nl_NL','uk_UA','id_ID','vi_VN']);
});

test('homepage publishes valid multilingual WebSite structured data', () => {
  const match = homepage.match(/<script type="application\/ld\+json">\s*([\s\S]*?)\s*<\/script>/);
  assert.ok(match);
  const data = JSON.parse(match[1]);
  assert.equal(data['@context'], 'https://schema.org');
  assert.equal(data['@type'], 'WebSite');
  assert.equal(data.name, 'BTC2G');
  assert.equal(data.url, 'https://btc2g.com/');
  assert.deepEqual(data.featureList, [
    'Bitcoin-to-gold ratio history',
    'BTC and gold purchasing-power index',
    'Reference calculator',
    'What-if scenario simulator'
  ]);
  assert.equal(data.inLanguage.length, 19);
});

test('homepage allows large search-result image previews', () => {
  assert.match(homepage, /name="robots" content="index,follow,max-image-preview:large"/);
});

test('TimeT research hub publishes a complete preview using its existing illustration', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'research.html'), 'utf8');
  assert.match(html, /name="robots" content="index,follow,max-image-preview:large"/);
  assert.match(html, /property="og:title" content="TimeT Independent Research · BTC2G"/);
  assert.match(html, /property="og:url" content="https:\/\/btc2g\.com\/research\.html"/);
  assert.match(html, /property="og:image" content="https:\/\/btc2g\.com\/assets\/images\/timet-document-2-figure-no-logo\.png"/);
  assert.match(html, /property="og:image:width" content="1055"/);
  assert.match(html, /property="og:image:height" content="1491"/);
  assert.match(html, /property="og:image:alt" content="TimeT illustration of discrete temporal transitions"/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/);
  assert.match(html, /name="twitter:image:alt" content="TimeT illustration of discrete temporal transitions"/);
});

test('Buy page preview identifies external providers without implying payment processing', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'buy.html'), 'utf8');
  assert.match(html, /<title>Buy Bitcoin Provider Links \| BTC2G<\/title>/);
  assert.match(html, /name="description" content="[^"]*official MoonPay and Topper[^"]*BTC2G does not process payments/);
  assert.match(html, /property="og:title" content="Buy Bitcoin Provider Links \| BTC2G"/);
  assert.match(html, /property="og:url" content="https:\/\/btc2g\.com\/buy\.html"/);
  assert.match(html, /property="og:image" content="https:\/\/btc2g\.com\/icons\/icon-512\.png"/);
  assert.match(html, /property="og:image:width" content="512"/);
  assert.match(html, /property="og:image:height" content="496"/);
  assert.match(html, /name="twitter:card" content="summary"/);
  assert.match(html, /name="twitter:description" content="[^"]*BTC2G does not process payments/);
  assert.doesNotMatch(html, /guaranteed return|guaranteed profit|best rate/i);
});

test('scenario simulator publishes localized large-image social previews', () => {
  const root = path.join(__dirname, '..');
  for (const lang of ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi']) {
    const file = lang === 'en' ? path.join(root, 'scenario.html') : path.join(root, lang, 'scenario.html');
    const html = fs.readFileSync(file, 'utf8');
    const title = html.match(/<title>([^<]+)<\/title>/)?.[1];
    const description = html.match(/<meta name="description" content="([^"]+)">/)?.[1];
    const canonical = html.match(/<link rel="canonical" href="([^"]+)">/)?.[1];
    assert.ok(title && description && canonical, lang);
    assert.match(html, /name="robots" content="index,follow,max-image-preview:large"/);
    assert.ok(html.includes(`<meta property="og:title" content="${title}" />`), lang);
    assert.ok(html.includes(`<meta property="og:description" content="${description}" />`), lang);
    assert.ok(html.includes(`<meta property="og:url" content="${canonical}" />`), lang);
    assert.match(html, /property="og:image" content="https:\/\/btc2g\.com\/assets\/btc_gold_forecast\.png"/);
    assert.match(html, /name="twitter:card" content="summary_large_image"/);
    assert.ok(html.includes(`<meta name="twitter:title" content="${title}" />`), lang);
    assert.ok(html.includes(`<meta name="twitter:description" content="${description}" />`), lang);
  }
});

test('scenario simulator publishes localized WebApplication structured data', () => {
  const root = path.join(__dirname, '..');
  for (const lang of ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi']) {
    const file = lang === 'en' ? path.join(root, 'scenario.html') : path.join(root, lang, 'scenario.html');
    const html = fs.readFileSync(file, 'utf8');
    const title = html.match(/<title>([^<]+)<\/title>/)?.[1];
    const description = html.match(/<meta name="description" content="([^"]+)">/)?.[1];
    const canonical = html.match(/<link rel="canonical" href="([^"]+)">/)?.[1];
    const scripts = [...html.matchAll(/<script type="application\/ld\+json">\s*([\s\S]*?)\s*<\/script>/g)];
    assert.equal(scripts.length, 1, lang);
    const data = JSON.parse(scripts[0][1]);
    assert.equal(data['@context'], 'https://schema.org', lang);
    assert.equal(data['@type'], 'WebApplication', lang);
    assert.equal(data.name, title, lang);
    assert.equal(data.url, canonical, lang);
    assert.equal(data.description, description, lang);
    assert.equal(data.applicationCategory, 'FinanceApplication', lang);
    assert.equal(data.operatingSystem, 'Any', lang);
    assert.equal(data.inLanguage, lang, lang);
    assert.equal('offers' in data, false, lang);
  }
});

test('research articles publish complete large-image social previews', () => {
  const root = path.join(__dirname, '..');
  for (const file of ['bitcoin-vs-gold.html', 'bitcoin-gold-ratio.html', 'bitcoin-vs-gold-forecast-2026.html']) {
    const html = fs.readFileSync(path.join(root, file), 'utf8');
    assert.match(html, /<meta name="robots" content="index,follow,max-image-preview:large"\s*\/>/);
    assert.match(html, /<meta property="og:type" content="article"\s*\/>/);
    assert.match(html, /<meta property="og:site_name" content="BTC2G"\s*\/>/);
    assert.match(html, /<meta property="og:locale" content="en_US"\s*\/>/);
    assert.match(html, /<meta property="og:title" content="[^"]+"\s*\/>/);
    assert.match(html, /<meta property="og:description" content="[^"]+"\s*\/>/);
    assert.match(html, /<meta property="og:url" content="https:\/\/btc2g\.com\/[^"]+"\s*\/>/);
    assert.match(html, /<meta property="og:image" content="https:\/\/btc2g\.com\/assets\/btc_gold_forecast\.png"\s*\/>/);
    assert.match(html, /<meta property="og:image:type" content="image\/png"\s*\/>/);
    assert.match(html, /<meta property="og:image:width" content="2800"\s*\/>/);
    assert.match(html, /<meta property="og:image:height" content="1400"\s*\/>/);
    assert.match(html, /<meta property="og:image:alt" content="BTC2G Bitcoin-to-gold research chart"\s*\/>/);
    assert.match(html, /<meta name="twitter:card" content="summary_large_image"\s*\/>/);
    assert.match(html, /<meta name="twitter:title" content="[^"]+"\s*\/>/);
    assert.match(html, /<meta name="twitter:description" content="[^"]+"\s*\/>/);
    assert.match(html, /<meta name="twitter:image" content="https:\/\/btc2g\.com\/assets\/btc_gold_forecast\.png"\s*\/>/);
    assert.match(html, /<meta name="twitter:image:alt" content="BTC2G Bitcoin-to-gold research chart"\s*\/>/);
  }
});

test('research articles keep canonical, Open Graph and structured-data URLs aligned', () => {
  const root = path.join(__dirname, '..');
  for (const file of ['bitcoin-vs-gold.html', 'bitcoin-gold-ratio.html', 'bitcoin-vs-gold-forecast-2026.html']) {
    const html = fs.readFileSync(path.join(root, file), 'utf8');
    const head = html.match(/<head>([\s\S]*?)<\/head>/)?.[1];
    const canonicals = [...html.matchAll(/<link rel="canonical" href="([^"]+)"\s*\/>/g)];
    const canonical = canonicals[0]?.[1];
    const openGraphUrl = html.match(/<meta property="og:url" content="([^"]+)"\s*\/>/)?.[1];
    const script = html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)?.[1];
    assert.equal(canonicals.length, 1, file);
    assert.ok(head.includes(canonicals[0][0]), file);
    assert.ok(canonical, file);
    assert.equal(openGraphUrl, canonical, file);
    assert.equal(JSON.parse(script).mainEntityOfPage, canonical, file);
  }
});

test('Premium publishes complete large-image previews in all 19 languages', () => {
  const root = path.join(__dirname, '..');
  const languages = ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
  for (const language of languages) {
    const file = language === 'en' ? 'premium.html' : path.join(language, 'premium.html');
    const html = fs.readFileSync(path.join(root, file), 'utf8');
    assert.match(html, /<meta property="og:image:type" content="image\/png"\s*\/?>/);
    assert.match(html, /<meta property="og:image:width" content="2800"\s*\/?>/);
    assert.match(html, /<meta property="og:image:height" content="1400"\s*\/?>/);
    assert.match(html, /<meta property="og:image:alt" content="BTC2G Bitcoin-to-gold research dashboard"\s*\/?>/);
    assert.match(html, /<meta name="twitter:image:alt" content="BTC2G Bitcoin-to-gold research dashboard"\s*\/?>/);
  }
});

test('research articles publish valid Article structured data without invented dates', () => {
  const root = path.join(__dirname, '..');
  for (const file of ['bitcoin-vs-gold.html', 'bitcoin-gold-ratio.html', 'bitcoin-vs-gold-forecast-2026.html']) {
    const html = fs.readFileSync(path.join(root, file), 'utf8');
    const scripts = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
    assert.equal(scripts.length, 1, file);
    const article = JSON.parse(scripts[0][1]);
    assert.equal(article['@context'], 'https://schema.org');
    assert.equal(article['@type'], 'Article');
    assert.match(article.headline, /Bitcoin|BTC/);
    assert.match(article.mainEntityOfPage, /^https:\/\/btc2g\.com\//);
    assert.equal(article.image, 'https://btc2g.com/assets/btc_gold_forecast.png');
    assert.equal(article.author.name, 'BTC2G');
    assert.equal(article.publisher.logo.url, 'https://btc2g.com/icons/icon-512.png');
    assert.equal('datePublished' in article, false);
    assert.equal('dateModified' in article, false);
  }
});
