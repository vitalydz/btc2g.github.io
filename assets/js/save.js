(function () {
  'use strict';
  const texts = {
    en: {
      eyebrow: 'More clarity before you buy', title: 'Compare what reaches your wallet', intro: 'Enter the final BTC from two quotes for the same total payment. No fees or exchange rates are assumed.',
      budget: '1. Set the total you pay', amount: 'Total payment, including fees', currency: 'Currency', resetNote: 'Changing the amount or currency clears both quotes. Request new quotes for each scenario.',
      quotes: '2. Enter comparable quotes', instructions: 'Use the same country, payment method and Bitcoin network. Enter the final BTC delivered after deductions; do not subtract network fees again. Quotes can change before payment.',
      open: 'Get a quote ↗', received: 'Final BTC received', confirm: 'Both quotes are current and use the same total payment, currency, payment method and network.', compare: 'Compare quotes', clear: 'Clear quotes',
      method: 'How the comparison works', formula: 'Effective cost per BTC = total payment ÷ final BTC received. The difference is shown in satoshis (1 BTC = 100,000,000 sats). More BTC for the same payment means a lower effective cost in these entered quotes.',
      limits: 'This does not isolate provider fees or spread, verify your quotes, or predict future savings. Bank charges outside a quote are not included. Compare the final provider checkout before purchasing.',
      topperFees: 'Topper fee information', moonpayHelp: 'MoonPay quote guidance', privacy: 'Calculations stay in this page. Quotes are not saved or sent to BTC2G. Provider links open external sites; BTC2G does not process purchases.', home: 'Explore the BTC ↔ gold calculator',
      required: 'Quote required', empty: 'Enter both quotes to compare.', invalid: 'Enter a positive total with up to 2 decimal places and both BTC quotes with up to 8 decimal places. Use a decimal point or comma, without thousands separators.', confirmNeeded: 'Confirm that the two quotes use comparable conditions.',
      rate: 'Effective cost', equal: 'Both entered quotes deliver the same BTC.', more: '{provider} delivers {sats} more sats ({percent}% more BTC) for the same total payment in your entered quotes.', reminder: 'Manual comparison only. Recheck the final quotes before payment.'
    },
    ru: {
      eyebrow: 'Больше ясности перед покупкой', title: 'Сравните, сколько придёт в кошелёк', intro: 'Введите итоговое количество BTC из двух котировок для одинаковой полной суммы оплаты. Комиссии и курсы не подставляются автоматически.',
      budget: '1. Укажите полную сумму оплаты', amount: 'Полная оплата с комиссиями', currency: 'Валюта', resetNote: 'Изменение суммы или валюты очищает обе котировки. Для каждого сценария получите новые предложения.',
      quotes: '2. Введите сопоставимые котировки', instructions: 'Используйте одну страну, способ оплаты и сеть Bitcoin. Введите итоговые BTC после удержаний; не вычитайте сетевую комиссию повторно. Котировки могут измениться до оплаты.',
      open: 'Получить котировку ↗', received: 'Итоговое количество BTC', confirm: 'Обе котировки актуальны и рассчитаны для одинаковой полной оплаты, валюты, способа оплаты и сети.', compare: 'Сравнить котировки', clear: 'Очистить котировки',
      method: 'Как работает сравнение', formula: 'Итоговая стоимость BTC = полная оплата ÷ полученные BTC. Разница показана в сатоши (1 BTC = 100 000 000 сатоши). Больше BTC за одну сумму означает меньшую стоимость BTC в введённых котировках.',
      limits: 'Расчёт не выделяет комиссию или спред, не проверяет котировки и не прогнозирует будущую экономию. Банковские списания вне котировки не учтены. Перед покупкой сравните итоговые условия провайдеров.',
      topperFees: 'Комиссии Topper', moonpayHelp: 'Инструкция MoonPay', privacy: 'Расчёт выполняется на этой странице. Котировки не сохраняются и не отправляются BTC2G. Ссылки ведут на внешние сайты; BTC2G не обрабатывает покупки.', home: 'Открыть калькулятор BTC ↔ золото',
      required: 'Нужна котировка', empty: 'Введите обе котировки для сравнения.', invalid: 'Введите положительную сумму: до 2 знаков после запятой для оплаты и до 8 для BTC. Допустимы точка или запятая без разделителей тысяч.', confirmNeeded: 'Подтвердите сопоставимость условий двух котировок.',
      rate: 'Итоговая стоимость', equal: 'Обе введённые котировки дают одинаковое количество BTC.', more: '{provider} даёт на {sats} сатоши больше ({percent}% больше BTC) за одну полную сумму в введённых котировках.', reminder: 'Сравнение введённых данных. Проверьте итоговые котировки перед оплатой.'
    },
    he: {
      eyebrow: 'יותר בהירות לפני הקנייה', title: 'השוו כמה יגיע לארנק שלכם', intro: 'הזינו את כמות ה־BTC הסופית משתי הצעות עבור אותו תשלום כולל. אין הנחות לגבי עמלות או שערי חליפין.',
      budget: '1. הגדירו את התשלום הכולל', amount: 'תשלום כולל עמלות', currency: 'מטבע', resetNote: 'שינוי הסכום או המטבע מנקה את שתי ההצעות. בקשו הצעות חדשות לכל תרחיש.',
      quotes: '2. הזינו הצעות בתנאים זהים', instructions: 'השתמשו באותה מדינה, שיטת תשלום ורשת Bitcoin. הזינו את כמות ה־BTC הסופית לאחר ניכויים; אל תנכו שוב את עמלת הרשת. ההצעות עשויות להשתנות לפני התשלום.',
      open: 'קבלת הצעה ↗', received: 'כמות BTC סופית לקבלה', confirm: 'שתי ההצעות עדכניות ומתייחסות לאותו תשלום כולל, מטבע, אמצעי תשלום ורשת.', compare: 'השוואת הצעות', clear: 'ניקוי ההצעות',
      method: 'איך ההשוואה עובדת', formula: 'עלות אפקטיבית ל־BTC = התשלום הכולל ÷ כמות ה־BTC הסופית. ההפרש מוצג בסאטושי (1 BTC = 100,000,000 סאטושי). יותר BTC עבור אותו תשלום פירושו עלות נמוכה יותר בהצעות שהוזנו.',
      limits: 'החישוב אינו מפריד עמלות או מרווח, אינו מאמת את ההצעות ואינו חוזה חיסכון עתידי. חיובי בנק מחוץ להצעה אינם כלולים. בדקו את התנאים הסופיים לפני הקנייה.',
      topperFees: 'מידע על עמלות Topper', moonpayHelp: 'מדריך MoonPay', privacy: 'החישוב מתבצע בעמוד זה. ההצעות אינן נשמרות או נשלחות ל־BTC2G. הקישורים מובילים לאתרים חיצוניים; BTC2G אינו מעבד רכישות.', home: 'למחשבון BTC ↔ זהב',
      required: 'נדרשת הצעה', empty: 'הזינו שתי הצעות להשוואה.', invalid: 'הזינו תשלום חיובי עם עד 2 ספרות אחרי הנקודה וכמויות BTC עם עד 8 ספרות. השתמשו בנקודה או בפסיק ללא מפרידי אלפים.', confirmNeeded: 'אשרו ששתי ההצעות משתמשות בתנאים זהים.',
      rate: 'עלות אפקטיבית', equal: 'שתי ההצעות מספקות אותה כמות BTC.', more: '{provider} מספקת {sats} סאטושי נוספים ({percent}% יותר BTC) עבור אותו תשלום כולל בהצעות שהוזנו.', reminder: 'השוואה ידנית בלבד. בדקו שוב את ההצעות הסופיות לפני התשלום.'
    }
  };
  Object.assign(texts, window.BTC2GSaveLocales || {});
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
    lang = Object.hasOwn(texts, value) ? value : 'en';
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === 'he' ? 'rtl' : 'ltr';
    byId('save-language').value = lang;
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
  setLanguage(new URLSearchParams(location.search).get('lang') || navigator.language.slice(0, 2));
})();
