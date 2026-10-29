(function () {
  'use strict';
  const texts = window.BTC2GSaveText;

  const byId = id => document.getElementById(id);
  let lang = 'en';
  let compared = false;
  const fields = ['save-amount', 'save-moonpay', 'save-topper'];
  const t = () => texts[lang];
  function resetResult() {
    compared = false;
    byId('save-confirm').checked = false;
    byId('save-result').textContent = t().empty;
    for (const provider of ['moonpay', 'topper']) byId(provider + '-rate').textContent = t().required;
  }
  function clearQuotes() {
    byId('save-moonpay').value = '';
    byId('save-topper').value = '';
    resetResult();
  }
  function renderComparison() {
    const result = window.BTC2GQuotes.compare(...fields.map(id => byId(id).value));
    if (result.status !== 'ready') {
      byId('save-result').textContent = t().invalid;
      return;
    }
    if (!byId('save-confirm').checked) {
      byId('save-result').textContent = t().confirmNeeded;
      return;
    }
    const currency = byId('save-currency').value;
    const money = new Intl.NumberFormat(lang, { style: 'currency', currency, maximumFractionDigits: 2 });
    const number = new Intl.NumberFormat(lang, { maximumFractionDigits: 2 });
    byId('moonpay-rate').textContent = t().rate + ': ' + money.format(result.firstRate) + ' / BTC';
    byId('topper-rate').textContent = t().rate + ': ' + money.format(result.secondRate) + ' / BTC';
    const summary = result.winner ? t().more.replace('{provider}', result.winner).replace('{sats}', number.format(result.differenceSats)).replace('{percent}', number.format(result.moreBTCPercent)) : t().equal;
    byId('save-result').textContent = summary + '\n' + t().reminder;
    compared = true;
  }
  function setLanguage(value) {
    value = String(value || '').toLowerCase().split(/[-_]/)[0];
    lang = Object.hasOwn(texts, value) ? value : 'en';
    document.documentElement.lang = lang;
    document.documentElement.dir = ['he', 'ar'].includes(lang) ? 'rtl' : 'ltr';
    byId('save-language').value = lang;
    byId('save-language').setAttribute('aria-label', t().languageLabel);
    document.querySelectorAll('.presets').forEach(e => e.setAttribute('aria-label', t().amount));
    try { localStorage.setItem('lang', lang); } catch {}
    document.title = t().title + ' | BTC2G SAVE';
    document.querySelector('meta[name=description]').content = t().intro;
    document.querySelectorAll('a[href^="index.html"]').forEach(a => {
      const u = new URL(a.getAttribute('href'), location.href); u.searchParams.set('lang', lang); a.setAttribute('href', 'index.html' + u.search + u.hash);
    });
    document.querySelectorAll('[data-text]').forEach(element => { element.textContent = t()[element.dataset.text]; });
    if (compared) renderComparison();
    else {
      byId('save-result').textContent = t().empty;
      for (const provider of ['moonpay', 'topper']) byId(provider + '-rate').textContent = t().required;
    }
  }
  byId('save-language').addEventListener('change', event => setLanguage(event.target.value));
  byId('save-amount').addEventListener('input', clearQuotes);
  byId('save-currency').addEventListener('change', clearQuotes);
  for (const id of ['save-moonpay', 'save-topper']) byId(id).addEventListener('input', resetResult);
  byId('save-confirm').addEventListener('change', () => {
    if (!byId('save-confirm').checked) {
      compared = false;
      byId('save-result').textContent = t().empty;
      for (const provider of ['moonpay', 'topper']) byId(provider + '-rate').textContent = t().required;
    }
  });
  document.querySelectorAll('[data-amount]').forEach(button => button.addEventListener('click', () => {
    byId('save-amount').value = button.dataset.amount;
    clearQuotes();
  }));
  byId('save-compare').addEventListener('click', renderComparison);
  byId('save-clear').addEventListener('click', clearQuotes);
  let saved;
  try { saved = localStorage.getItem('lang'); } catch {}
  setLanguage(new URLSearchParams(location.search).get('lang') || saved || navigator.language.slice(0, 2));
})();
