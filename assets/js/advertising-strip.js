/* Reserved for future paid links. Current links below are our own sites, not paid ads. */
(() => {
  'use strict';
  const script = document.currentScript;
  const ownSite = script?.dataset.site || '';
  const assetBase = new URL(ownSite === 'btc2g.com' ? '../brands/' : './advertising-brands/', script.src).href;
  const labels = {"en": ["Advertising space", "Our sites"], "ru": ["Место для рекламы", "Наши сайты"], "he": ["מקום לפרסום", "האתרים שלנו"], "ar": ["مساحة إعلانية", "مواقعنا"], "de": ["Werbeplatz", "Unsere Websites"], "fr": ["Espace publicitaire", "Nos sites"], "es": ["Espacio publicitario", "Nuestros sitios"], "zh": ["广告位", "我们的网站"], "it": ["Spazio pubblicitario", "I nostri siti"], "pt": ["Espaço publicitário", "Nossos sites"], "hi": ["विज्ञापन स्थान", "हमारी वेबसाइटें"], "ja": ["広告スペース", "関連サイト"], "ko": ["광고 공간", "관련 사이트"], "tr": ["Reklam alanı", "Sitelerimiz"], "pl": ["Miejsce na reklamę", "Nasze strony"], "nl": ["Advertentieruimte", "Onze websites"], "uk": ["Місце для реклами", "Наші сайти"], "id": ["Ruang iklan", "Situs kami"], "vi": ["Không gian quảng cáo", "Các trang của chúng tôi"]};
  const sites = [['BTC2G','btc2g.com','btc2g','btc2g.png'],['VitdArt','vitdart.com','vitdart','vitdart.svg'],['bit2g','bit2g.com','bit2g','bit2g.svg'],['findbBest','findbbest.com','findbbest','findbbest.svg'],['AIKickAI','aikickai.com','aikickai','aikickai.webp'],['BMyChance','bmychance.com','bmychance','bmychance.png']];
  function mount() {
    if (document.querySelector('[data-advertising-strip]')) return;
    const strip = document.createElement('aside');
    strip.className = 'site-advertising-strip';
    strip.dataset.advertisingStrip = '';
    const inner = document.createElement('div'); inner.className = 'site-advertising-inner';
    const slot = document.createElement('span'); slot.className = 'site-advertising-slot'; slot.dataset.advertisingSlot = 'empty';
    const nav = document.createElement('nav'); nav.className = 'site-advertising-links';
    for (const [name, host, brand, logo] of sites) {
      if (host === ownSite) continue;
      const link = document.createElement('a'); link.href = 'https://' + host + '/';
      link.className = 'site-brand site-brand-' + brand; link.setAttribute('aria-label', name); link.dir = 'ltr'; link.translate = false;
      const image = document.createElement('img'); image.className = 'site-brand-logo'; image.src = assetBase + logo; image.alt = ''; image.width = 32; image.height = 32; image.loading = 'lazy'; image.decoding = 'async';
      const wordmark = document.createElement('span'); wordmark.className = 'site-brand-name'; wordmark.setAttribute('aria-hidden','true');
      if (brand === 'findbbest') {
        for (const [part, className] of [['find','site-brand-metal'],['b','site-brand-blue'],['Best','site-brand-best']]) {
          const span = document.createElement('span'); span.className = className;
          if (part === 'find') {
            for (const letter of part) { const glyph = document.createElement('i'); glyph.className = 'site-brand-metal-letter site-brand-metal-' + letter; glyph.textContent = letter; span.append(glyph); }
          } else { span.textContent = part; }
          wordmark.append(span);
        }
      } else { wordmark.textContent = brand === 'bmychance' ? 'mychance' : name; }
      link.append(image, wordmark); nav.append(link);
    }
    inner.append(slot, nav); strip.append(inner); document.body.append(strip);
    function localize() {
      const lang = (document.documentElement.lang || 'en').toLowerCase().split('-')[0];
      const [advertising, projects] = labels[lang] || labels.en;
      strip.lang = lang; strip.dir = ['he','ar'].includes(lang) ? 'rtl' : 'ltr';
      strip.setAttribute('aria-label', advertising); slot.textContent = advertising;
      nav.setAttribute('aria-label', projects);
    }
    localize(); new MutationObserver(localize).observe(document.documentElement, {attributes:true,attributeFilter:['lang']});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once:true}); else mount();
})();
