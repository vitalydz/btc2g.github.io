(() => {
  const text = {
    en: {button:'Join early access',invalid:'Enter a valid email address.',sending:'Joining…',received:'Request received. Thank you for your interest in BTC2G.',error:'Signup is unavailable right now. Please try again later.',email:'Email address',title:'Help shape a more useful research layer.',help:'Receive BTC2G product updates. Privacy Policy'},
    ru: {button:'Оставить заявку',invalid:'Введите корректный email.',sending:'Отправка…',received:'Заявка получена. Спасибо за интерес к BTC2G.',error:'Не удалось отправить заявку. Попробуйте позже.',email:'Электронная почта',title:'Помогите сделать исследования полезнее.',help:'Получать новости продукта BTC2G. Политика конфиденциальности'},
    he: {button:'שליחת בקשה',invalid:'הזינו כתובת אימייל תקינה.',sending:'שולח…',received:'הבקשה התקבלה. תודה על ההתעניינות ב-BTC2G.',error:'שליחת הבקשה נכשלה. נסו שוב מאוחר יותר.',email:'כתובת אימייל',title:'עזרו לנו לשפר את כלי המחקר.',help:'קבלת עדכוני מוצר BTC2G. מדיניות פרטיות'}
  };
  let saved;
  try { saved = localStorage.getItem('lang'); } catch {}
  const requested = new URLSearchParams(location.search).get('lang') || saved || navigator.language.slice(0,2);
  const lang = Object.hasOwn(text, requested) ? requested : 'en';
  const t = text[lang];
  window.BTC2GPremiumText = t;
  const section = document.getElementById('early-access');
  section.lang = lang;
  section.dir = lang === 'he' ? 'rtl' : 'ltr';
  document.getElementById('early-access-button').textContent = t.button;
  document.getElementById('early-access-email').setAttribute('aria-label', t.email);
  document.getElementById('early-access-title').textContent = t.title;
  const help = document.getElementById('form-help');
  const link = document.createElement('a'); link.href = 'privacy.html'; link.textContent = t.help;
  help.replaceChildren(link);
})();
