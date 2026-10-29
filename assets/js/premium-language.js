(() => {
  const text = {
  "en": {
    "button": "Join early access",
    "invalid": "Enter a valid email address.",
    "sending": "Joining…",
    "received": "Request received. Thank you for your interest in BTC2G.",
    "error": "Signup is unavailable right now. Please try again later.",
    "email": "Email address",
    "title": "Help shape a more useful research layer.",
    "help": "Receive BTC2G product updates. Privacy Policy"
  },
  "ru": {
    "button": "Оставить заявку",
    "invalid": "Введите корректный email.",
    "sending": "Отправка…",
    "received": "Заявка получена. Спасибо за интерес к BTC2G.",
    "error": "Не удалось отправить заявку. Попробуйте позже.",
    "email": "Электронная почта",
    "title": "Помогите сделать исследования полезнее.",
    "help": "Получать новости продукта BTC2G. Политика конфиденциальности"
  },
  "he": {
    "button": "שליחת בקשה",
    "invalid": "הזינו כתובת אימייל תקינה.",
    "sending": "שולח…",
    "received": "הבקשה התקבלה. תודה על ההתעניינות ב-BTC2G.",
    "error": "שליחת הבקשה נכשלה. נסו שוב מאוחר יותר.",
    "email": "כתובת אימייל",
    "title": "עזרו לנו לשפר את כלי המחקר.",
    "help": "קבלת עדכוני מוצר BTC2G. מדיניות פרטיות"
  },
  "de": {
    "button": "Abonnieren",
    "invalid": "Geben Sie eine gültige E-Mail-Adresse ein.",
    "sending": "Wird abonniert...",
    "received": "Anfrage erhalten. Vielen Dank für Ihr Interesse an BTC2G.",
    "error": "Abonnement fehlgeschlagen. Bitte versuchen Sie es später erneut.",
    "title": "BTC2G Premium",
    "help": "Datenschutz"
  },
  "fr": {
    "button": "S’abonner",
    "invalid": "Saisissez une adresse e-mail valide.",
    "sending": "Inscription...",
    "received": "Demande reçue. Merci de votre intérêt pour BTC2G.",
    "error": "L’inscription a échoué. Veuillez réessayer plus tard.",
    "title": "BTC2G Premium",
    "help": "Confidentialité"
  },
  "es": {
    "button": "Suscribirse",
    "invalid": "Ingresa un correo electrónico válido.",
    "sending": "Suscribiendo...",
    "received": "Solicitud recibida. Gracias por tu interés en BTC2G.",
    "error": "La suscripción falló. Inténtalo de nuevo más tarde.",
    "title": "BTC2G Premium",
    "help": "Política de privacidad"
  },
  "zh": {
    "button": "订阅",
    "invalid": "请输入有效的电子邮箱。",
    "sending": "正在订阅...",
    "received": "申请已收到。感谢您关注 BTC2G。",
    "error": "订阅失败。请稍后再试。",
    "title": "BTC2G Premium",
    "help": "隐私政策"
  },
  "it": {
    "button": "Iscriviti",
    "invalid": "Inserisci un indirizzo email valido.",
    "sending": "Iscrizione...",
    "received": "Richiesta ricevuta. Grazie per il tuo interesse in BTC2G.",
    "error": "Iscrizione non riuscita. Riprova più tardi.",
    "title": "BTC2G Premium",
    "help": "Privacy"
  },
  "pt": {
    "button": "Subscrever",
    "invalid": "Insira um email válido.",
    "sending": "A subscrever...",
    "received": "Pedido recebido. Obrigado pelo interesse no BTC2G.",
    "error": "A subscrição falhou. Tente novamente mais tarde.",
    "title": "BTC2G Premium",
    "help": "Privacidade"
  },
  "ar": {
    "button": "اشتراك",
    "invalid": "أدخل بريدا إلكترونيا صالحا.",
    "sending": "جار الاشتراك...",
    "received": "تم استلام الطلب. شكرًا لاهتمامك بـ BTC2G.",
    "error": "فشل الاشتراك. حاول مرة أخرى لاحقا.",
    "title": "BTC2G Premium",
    "help": "الخصوصية"
  },
  "hi": {
    "button": "सदस्यता लें",
    "invalid": "मान्य ईमेल दर्ज करें।",
    "sending": "सदस्यता ली जा रही है...",
    "received": "अनुरोध प्राप्त हुआ। BTC2G में रुचि के लिए धन्यवाद।",
    "error": "सदस्यता विफल रही। बाद में फिर कोशिश करें।",
    "title": "BTC2G Premium",
    "help": "गोपनीयता"
  },
  "ja": {
    "button": "登録",
    "invalid": "有効なメールアドレスを入力してください。",
    "sending": "登録中...",
    "received": "リクエストを受け付けました。BTC2Gへのご関心ありがとうございます。",
    "error": "登録に失敗しました。後でもう一度お試しください。",
    "title": "BTC2G Premium",
    "help": "プライバシー"
  },
  "ko": {
    "button": "구독",
    "invalid": "유효한 이메일 주소를 입력하세요.",
    "sending": "구독 중...",
    "received": "요청이 접수되었습니다. BTC2G에 관심을 가져 주셔서 감사합니다.",
    "error": "구독에 실패했습니다. 나중에 다시 시도하세요.",
    "title": "BTC2G Premium",
    "help": "개인정보"
  },
  "tr": {
    "button": "Abone ol",
    "invalid": "Geçerli bir e-posta adresi girin.",
    "sending": "Abone olunuyor...",
    "received": "Talep alındı. BTC2G ilginiz için teşekkürler.",
    "error": "Abonelik başarısız oldu. Daha sonra tekrar deneyin.",
    "title": "BTC2G Premium",
    "help": "Gizlilik"
  },
  "pl": {
    "button": "Subskrybuj",
    "invalid": "Wpisz prawidłowy adres e-mail.",
    "sending": "Subskrybowanie...",
    "received": "Zgłoszenie otrzymane. Dziękujemy za zainteresowanie BTC2G.",
    "error": "Subskrypcja nie powiodła się. Spróbuj później.",
    "title": "BTC2G Premium",
    "help": "Prywatność"
  },
  "nl": {
    "button": "Abonneren",
    "invalid": "Voer een geldig e-mailadres in.",
    "sending": "Abonneren...",
    "received": "Aanvraag ontvangen. Bedankt voor je interesse in BTC2G.",
    "error": "Abonneren mislukt. Probeer het later opnieuw.",
    "title": "BTC2G Premium",
    "help": "Privacy"
  },
  "uk": {
    "button": "Підписатися",
    "invalid": "Введіть дійсну електронну адресу.",
    "sending": "Підписка...",
    "received": "Заявку отримано. Дякуємо за інтерес до BTC2G.",
    "error": "Підписка не вдалася. Спробуйте пізніше.",
    "title": "BTC2G Premium",
    "help": "Конфіденційність"
  },
  "id": {
    "button": "Berlangganan",
    "invalid": "Masukkan alamat email yang valid.",
    "sending": "Berlangganan...",
    "received": "Permintaan diterima. Terima kasih atas minat Anda pada BTC2G.",
    "error": "Langganan gagal. Coba lagi nanti.",
    "title": "BTC2G Premium",
    "help": "Privasi"
  },
  "vi": {
    "button": "Đăng ký",
    "invalid": "Nhập địa chỉ email hợp lệ.",
    "sending": "Đang đăng ký...",
    "received": "Đã nhận yêu cầu. Cảm ơn bạn đã quan tâm đến BTC2G.",
    "error": "Đăng ký thất bại. Vui lòng thử lại sau.",
    "title": "BTC2G Premium",
    "help": "Quyền riêng tư"
  }
};

  let saved;
  try { saved = localStorage.getItem('lang'); } catch {}
  const requested = new URLSearchParams(location.search).get('lang') || saved || navigator.language.slice(0,2);
  const lang = Object.hasOwn(text, requested) ? requested : 'en';
  const t = text[lang];
  window.BTC2GPremiumText = t;
  const section = document.getElementById('early-access');
  section.lang = lang;
  section.dir = ['he', 'ar'].includes(lang) ? 'rtl' : 'ltr';
  document.getElementById('early-access-button').textContent = t.button;
  document.getElementById('early-access-email').setAttribute('aria-label', window.BTC2GSubscription.text(lang).email);
  document.getElementById('early-access-title').textContent = t.title;
  const help = document.getElementById('form-help');
  const link = document.createElement('a'); link.href = 'privacy.html'; link.textContent = t.help;
  help.replaceChildren(link);
})();
