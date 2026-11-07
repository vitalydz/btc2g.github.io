/* Reserved for future paid links. Current links below are our own sites, not paid ads. */
(() => {
  'use strict';
  const ownSite = document.currentScript?.dataset.site || '';
  const labels = {"en": ["Advertising space", "Our sites"], "ru": ["Место для рекламы", "Наши сайты"], "he": ["מקום לפרסום", "האתרים שלנו"], "ar": ["مساحة إعلانية", "مواقعنا"], "de": ["Werbeplatz", "Unsere Websites"], "fr": ["Espace publicitaire", "Nos sites"], "es": ["Espacio publicitario", "Nuestros sitios"], "zh": ["广告位", "我们的网站"], "it": ["Spazio pubblicitario", "I nostri siti"], "pt": ["Espaço publicitário", "Nossos sites"], "hi": ["विज्ञापन स्थान", "हमारी वेबसाइटें"], "ja": ["広告スペース", "関連サイト"], "ko": ["광고 공간", "관련 사이트"], "tr": ["Reklam alanı", "Sitelerimiz"], "pl": ["Miejsce na reklamę", "Nasze strony"], "nl": ["Advertentieruimte", "Onze websites"], "uk": ["Місце для реклами", "Наші сайти"], "id": ["Ruang iklan", "Situs kami"], "vi": ["Không gian quảng cáo", "Các trang của chúng tôi"]};
  const sites = [['BTC2G','btc2g.com'],['VitdArt','vitdart.com'],['Bit2G','bit2g.com'],['findbBest','findbbest.com'],['AIKickAI','aikickai.com'],['BMyChance','bmychance.com']];
  function mount() {
    if (document.querySelector('[data-advertising-strip]')) return;
    const strip = document.createElement('aside');
    strip.className = 'site-advertising-strip';
    strip.dataset.advertisingStrip = '';
    const inner = document.createElement('div'); inner.className = 'site-advertising-inner';
    const slot = document.createElement('span'); slot.className = 'site-advertising-slot'; slot.dataset.advertisingSlot = 'empty';
    const nav = document.createElement('nav'); nav.className = 'site-advertising-links';
    const caption = document.createElement('span'); caption.className = 'site-advertising-caption'; nav.append(caption);
    for (const [name, host] of sites) {
      if (host === ownSite) continue;
      const link = document.createElement('a'); link.href = 'https://' + host + '/'; link.textContent = name;
      link.hreflang = 'x-default'; nav.append(link);
    }
    inner.append(slot, nav); strip.append(inner); document.body.append(strip);
    function localize() {
      const lang = (document.documentElement.lang || 'en').toLowerCase().split('-')[0];
      const [advertising, projects] = labels[lang] || labels.en;
      strip.lang = lang; strip.dir = ['he','ar'].includes(lang) ? 'rtl' : 'ltr';
      strip.setAttribute('aria-label', advertising); slot.textContent = advertising;
      caption.textContent = projects + ':'; nav.setAttribute('aria-label', projects);
    }
    localize(); new MutationObserver(localize).observe(document.documentElement, {attributes:true,attributeFilter:['lang']});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once:true}); else mount();
})();
