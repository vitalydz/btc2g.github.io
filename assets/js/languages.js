/* Shared language selection. No network requests or translation-service dependency. */
(function (root) {
  'use strict';
  const names = Object.freeze({en:'English',ru:'Русский',he:'עברית',de:'Deutsch',fr:'Français',es:'Español',zh:'中文',it:'Italiano',pt:'Português',ar:'العربية',hi:'हिन्दी',ja:'日本語',ko:'한국어',tr:'Türkçe',pl:'Polski',nl:'Nederlands',uk:'Українська',id:'Bahasa Indonesia',vi:'Tiếng Việt'});
  const menus = {
    en:['Open menu','Close menu'],ru:['Открыть меню','Закрыть меню'],he:['פתיחת תפריט','סגירת תפריט'],
    de:['Menü öffnen','Menü schließen'],fr:['Ouvrir le menu','Fermer le menu'],es:['Abrir menú','Cerrar menú'],
    zh:['打开菜单','关闭菜单'],it:['Apri menu','Chiudi menu'],pt:['Abrir menu','Fechar menu'],ar:['فتح القائمة','إغلاق القائمة'],
    hi:['मेन्यू खोलें','मेन्यू बंद करें'],ja:['メニューを開く','メニューを閉じる'],ko:['메뉴 열기','메뉴 닫기'],tr:['Menüyü aç','Menüyü kapat'],
    pl:['Otwórz menu','Zamknij menu'],nl:['Menu openen','Menu sluiten'],uk:['Відкрити меню','Закрити меню'],id:['Buka menu','Tutup menu'],vi:['Mở menu','Đóng menu']
  };
  function menu(value, open) { return menus[normalize(value) || 'en'][Number(Boolean(open))]; }
  function normalize(value) {
    if (typeof value !== 'string') return null;
    const code = value.trim().toLowerCase().split(/[-_]/)[0];
    return Object.hasOwn(names, code) ? code : null;
  }
  function requested() {
    let saved;
    try { saved = localStorage.getItem('lang'); } catch (_) {}
    return normalize(new URLSearchParams(location.search).get('lang')) || normalize(saved) || normalize(navigator.language) || 'en';
  }
  function links(lang) {
    document.querySelectorAll('a[href]').forEach(link => {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('#')) return;
      let url;
      try { url = new URL(href, location.href); } catch (_) { return; }
      const account = url.origin === 'https://btc2g-accounts.btc2g.workers.dev' && ['/', '/admin/'].includes(url.pathname);
      const localPage = url.origin === location.origin && /\/(?:index|save|buy|contest|premium|premium-success|scenario|research|support|about|privacy|terms|bitcoin-vs-gold|bitcoin-gold-ratio|bitcoin-vs-gold-forecast-2026)\.html$/.test(url.pathname);
      if (!account && !localPage) return;
      url.searchParams.set('lang', lang);
      link.setAttribute('href', account ? url.href : url.pathname + url.search + url.hash);
    });
  }
  function remember(value) {
    const lang = normalize(value) || 'en';
    try { localStorage.setItem('lang', lang); } catch (_) {}
    try { const url = new URL(location.href); url.searchParams.set('lang', lang); history.replaceState(null, '', url); } catch (_) {}
    return lang;
  }
  function apply(value) {
    const lang = remember(value);
    document.documentElement.lang = lang;
    document.documentElement.dir = ['he', 'ar'].includes(lang) ? 'rtl' : 'ltr';
    links(lang);
    return lang;
  }
  function options(select) {
    select.replaceChildren(...Object.entries(names).map(([code, name]) => {
      const option = document.createElement('option'); option.value = code; option.textContent = name; return option;
    }));
  }
  function decimal(value) {
    return String(value).replace(/[٠-٩۰-۹०-९０-９]/g, digit => {
      const point = digit.charCodeAt(0);
      for (const zero of [0x0660, 0x06f0, 0x0966, 0xff10]) {
        if (point >= zero && point <= zero + 9) return String(point - zero);
      }
      return digit;
    }).replace(/٫/g, '.');
  }
  root.BTC2GLanguages = Object.freeze({ names, normalize, requested, remember, apply, links, options, decimal, menu });
})(window);
