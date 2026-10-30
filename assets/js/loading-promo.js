(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.BTC2GLoadingPromo = api;
  if (root && root.document) api.mount(root, root.document);
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const SESSION_KEY = 'btc2g:loading-promo:seen';
  const VIP_HIDE_KEY = 'btc2g:loading-promo:vip-hidden';
  const VIDEO_PATH = 'product-hunt/btc2g-demo-42s-f.mp4';
  const POSTER_PATH = 'product-hunt/btc2g-logo-512.png';

  const copy = {
    en: ['BTC2G presentation', 'Turn on sound', 'Continue to BTC2G', "VIP: Don't show this again", 'Close presentation'],
    ru: ['Презентация BTC2G', 'Включить звук', 'Перейти к BTC2G', 'VIP: больше не показывать', 'Закрыть презентацию'],
    he: ['מצגת BTC2G', 'הפעלת צליל', 'המשך ל-BTC2G', 'VIP: לא להציג שוב', 'סגירת המצגת'],
    de: ['BTC2G-Präsentation', 'Ton einschalten', 'Weiter zu BTC2G', 'VIP: Nicht erneut anzeigen', 'Präsentation schließen'],
    fr: ['Présentation BTC2G', 'Activer le son', 'Continuer vers BTC2G', 'VIP : ne plus afficher', 'Fermer la présentation'],
    es: ['Presentación de BTC2G', 'Activar sonido', 'Continuar a BTC2G', 'VIP: no volver a mostrar', 'Cerrar presentación'],
    zh: ['BTC2G 演示', '开启声音', '继续进入 BTC2G', 'VIP：不再显示', '关闭演示'],
    it: ['Presentazione BTC2G', 'Attiva audio', 'Continua su BTC2G', 'VIP: non mostrare più', 'Chiudi presentazione'],
    pt: ['Apresentação BTC2G', 'Ativar som', 'Continuar para BTC2G', 'VIP: não mostrar novamente', 'Fechar apresentação'],
    ar: ['عرض BTC2G', 'تشغيل الصوت', 'المتابعة إلى BTC2G', 'VIP: عدم العرض مرة أخرى', 'إغلاق العرض'],
    hi: ['BTC2G प्रस्तुति', 'ध्वनि चालू करें', 'BTC2G पर जाएँ', 'VIP: फिर न दिखाएँ', 'प्रस्तुति बंद करें'],
    ja: ['BTC2G 紹介動画', '音声をオン', 'BTC2Gへ進む', 'VIP：今後表示しない', '紹介動画を閉じる'],
    ko: ['BTC2G 소개', '소리 켜기', 'BTC2G로 계속', 'VIP: 다시 표시하지 않기', '소개 닫기'],
    tr: ['BTC2G sunumu', 'Sesi aç', "BTC2G'ye devam et", 'VIP: bir daha gösterme', 'Sunumu kapat'],
    pl: ['Prezentacja BTC2G', 'Włącz dźwięk', 'Przejdź do BTC2G', 'VIP: nie pokazuj ponownie', 'Zamknij prezentację'],
    nl: ['BTC2G-presentatie', 'Geluid aan', 'Doorgaan naar BTC2G', 'VIP: niet opnieuw tonen', 'Presentatie sluiten'],
    uk: ['Презентація BTC2G', 'Увімкнути звук', 'Перейти до BTC2G', 'VIP: більше не показувати', 'Закрити презентацію'],
    id: ['Presentasi BTC2G', 'Aktifkan suara', 'Lanjut ke BTC2G', 'VIP: jangan tampilkan lagi', 'Tutup presentasi'],
    vi: ['Giới thiệu BTC2G', 'Bật âm thanh', 'Tiếp tục đến BTC2G', 'VIP: không hiển thị lại', 'Đóng giới thiệu'],
  };

  function normalizeTier(value) {
    return String(value || '').trim().toLowerCase();
  }

  function isVipTier(value) {
    return ['vip', 'premium'].includes(normalizeTier(value));
  }

  function shouldShow({ sessionSeen, vipHidden, isVip }) {
    return !sessionSeen && !(isVip && vipHidden);
  }

  function readStorage(storage, key) {
    try {
      return storage?.getItem(key) || null;
    } catch (_) {
      return null;
    }
  }

  function writeStorage(storage, key, value) {
    try {
      storage?.setItem(key, value);
    } catch (_) {
      // Storage can be disabled; the promo remains dismissible for this page view.
    }
  }

  function selectedLanguage(win) {
    const requested = new URLSearchParams(win.location.search).get('lang');
    const saved = readStorage(win.localStorage, 'lang');
    const language = requested || saved || 'en';
    return copy[language] ? language : 'en';
  }

  function accessTier(win, doc) {
    return win.BTC2G_ACCESS?.tier || doc.documentElement.dataset.btc2gTier || '';
  }

  function mount(win, doc) {
    const start = () => {
      if (doc.getElementById('btc2g-loading-promo')) return;
      const isVip = isVipTier(accessTier(win, doc));
      const sessionSeen = readStorage(win.sessionStorage, SESSION_KEY) === '1';
      const vipHidden = readStorage(win.localStorage, VIP_HIDE_KEY) === '1';
      if (!shouldShow({ sessionSeen, vipHidden, isVip })) return;

      const text = copy[selectedLanguage(win)];
      const overlay = doc.createElement('section');
      overlay.id = 'btc2g-loading-promo';
      overlay.className = 'loading-promo';
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');
      overlay.setAttribute('aria-labelledby', 'btc2g-loading-promo-title');
      overlay.innerHTML = `
        <div class="loading-promo-panel">
          <button class="loading-promo-close" type="button" aria-label="${text[4]}">×</button>
          <h2 id="btc2g-loading-promo-title">${text[0]}</h2>
          <video class="loading-promo-video" muted autoplay playsinline preload="auto" poster="${POSTER_PATH}">
            <source src="${VIDEO_PATH}" type="video/mp4">
          </video>
          <div class="loading-promo-actions">
            <button class="loading-promo-sound" type="button">${text[1]}</button>
            <button class="loading-promo-continue" type="button">${text[2]}</button>
          </div>
          <label class="loading-promo-vip"${isVip ? '' : ' hidden'}>
            <input type="checkbox"> <span>${text[3]}</span>
          </label>
        </div>`;
      doc.body.appendChild(overlay);
      doc.body.classList.add('loading-promo-open');

      const video = overlay.querySelector('video');
      const closeButton = overlay.querySelector('.loading-promo-close');
      const continueButton = overlay.querySelector('.loading-promo-continue');
      const soundButton = overlay.querySelector('.loading-promo-sound');
      const hideCheckbox = overlay.querySelector('.loading-promo-vip input');
      let closingTimer = null;

      const close = () => {
        if (!overlay.isConnected) return;
        writeStorage(win.sessionStorage, SESSION_KEY, '1');
        if (isVip && hideCheckbox.checked) writeStorage(win.localStorage, VIP_HIDE_KEY, '1');
        win.clearTimeout(closingTimer);
        video.pause();
        overlay.remove();
        doc.body.classList.remove('loading-promo-open');
      };

      closeButton.addEventListener('click', close);
      continueButton.addEventListener('click', close);
      soundButton.addEventListener('click', () => {
        video.muted = false;
        video.volume = 1;
        video.play().catch(() => {});
        soundButton.hidden = true;
      });
      video.addEventListener('ended', () => {
        closingTimer = win.setTimeout(close, 1400);
      });
      doc.addEventListener('keydown', event => {
        if (event.key === 'Escape' && overlay.isConnected) close();
      }, {once: true});
      video.play().catch(() => {});
      closeButton.focus({preventScroll: true});
    };

    if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', start, {once: true});
    else start();
  }

  return {copy, normalizeTier, isVipTier, shouldShow, readStorage, writeStorage, mount};
});
