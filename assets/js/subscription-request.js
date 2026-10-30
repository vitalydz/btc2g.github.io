(() => {
  const messages = {
    en: ['Email address', 'Could not confirm receipt of your request. It may have arrived. No automatic retry was sent.'],
    ru: ['Электронная почта', 'Не удалось подтвердить получение заявки. Она могла поступить. Автоматическая повторная отправка не выполнялась.'],
    he: ['כתובת אימייל', 'לא ניתן לאשר שהבקשה התקבלה. ייתכן שהיא התקבלה. לא בוצע ניסיון שליחה נוסף באופן אוטומטי.'],
    de: ['E-Mail-Adresse', 'Der Eingang konnte nicht bestätigt werden. Die Anfrage könnte angekommen sein. Kein automatischer erneuter Versand.'],
    fr: ['Adresse e-mail', 'Réception non confirmée. La demande a peut-être été reçue. Aucun nouvel envoi automatique.'],
    es: ['Correo electrónico', 'No se pudo confirmar la recepción. Puede que la solicitud haya llegado. No se reenvió automáticamente.'],
    zh: ['电子邮箱', '无法确认是否收到申请。申请可能已送达。没有自动重新发送。'],
    it: ['Indirizzo email', 'Ricezione non confermata. La richiesta potrebbe essere arrivata. Nessun nuovo invio automatico.'],
    pt: ['Endereço de email', 'Não foi possível confirmar a receção. O pedido pode ter chegado. Não houve reenvio automático.'],
    ar: ['البريد الإلكتروني', 'تعذر تأكيد استلام الطلب. ربما وصل. لم تتم إعادة الإرسال تلقائيًا.'],
    hi: ['ईमेल पता', 'अनुरोध मिलने की पुष्टि नहीं हो सकी। वह पहुँच गया हो सकता है। अपने आप दोबारा नहीं भेजा गया।'],
    ja: ['メールアドレス', '受信を確認できませんでした。届いている可能性があります。自動再送はしていません。'],
    ko: ['이메일 주소', '요청 수신을 확인할 수 없습니다. 도착했을 수도 있습니다. 자동으로 다시 보내지 않았습니다.'],
    tr: ['E-posta adresi', 'İsteğin alındığı doğrulanamadı. Ulaşmış olabilir. Otomatik olarak yeniden gönderilmedi.'],
    pl: ['Adres e-mail', 'Nie potwierdzono odbioru. Zgłoszenie mogło dotrzeć. Nie wysłano go ponownie automatycznie.'],
    nl: ['E-mailadres', 'Ontvangst niet bevestigd. De aanvraag kan zijn aangekomen. Niet automatisch opnieuw verzonden.'],
    uk: ['Електронна пошта', 'Не вдалося підтвердити отримання заявки. Вона могла надійти. Автоматичного повторного надсилання не було.'],
    id: ['Alamat email', 'Penerimaan belum dapat dikonfirmasi. Permintaan mungkin sudah diterima. Tidak dikirim ulang secara otomatis.'],
    vi: ['Địa chỉ email', 'Không thể xác nhận đã nhận yêu cầu. Yêu cầu có thể đã đến. Không tự động gửi lại.']
  };

  const receiptNotes = {"en": "Receiving a request does not confirm an active email subscription.", "ru": "Получение заявки не подтверждает активную подписку по электронной почте.", "he": "קבלת הבקשה אינה מאשרת הרשמה פעילה לדואר אלקטרוני.", "de": "Der Eingang der Anfrage bestätigt kein aktives E-Mail-Abonnement.", "fr": "La réception de la demande ne confirme pas un abonnement e-mail actif.", "es": "Recibir la solicitud no confirma una suscripción de correo activa.", "zh": "收到申请并不代表电子邮件订阅已生效。", "it": "La ricezione della richiesta non conferma un abbonamento email attivo.", "pt": "A receção do pedido não confirma uma subscrição de email ativa.", "ar": "استلام الطلب لا يؤكد وجود اشتراك نشط بالبريد الإلكتروني.", "hi": "अनुरोध प्राप्त होना सक्रिय ईमेल सदस्यता की पुष्टि नहीं है।", "ja": "リクエストの受信は、メール購読が有効になったことを意味しません。", "ko": "요청 수신은 이메일 구독 활성화를 의미하지 않습니다.", "tr": "Talebin alınması etkin bir e-posta aboneliğini doğrulamaz.", "pl": "Otrzymanie zgłoszenia nie potwierdza aktywnej subskrypcji e-mail.", "nl": "Ontvangst van de aanvraag bevestigt geen actief e-mailabonnement.", "uk": "Отримання заявки не підтверджує активної підписки електронною поштою.", "id": "Penerimaan permintaan tidak mengonfirmasi langganan email aktif.", "vi": "Việc nhận yêu cầu không xác nhận đăng ký email đang hoạt động."};

  async function send(url, options) {
    const controller = new AbortController();
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => {
        const error = new Error('Subscription receipt could not be confirmed in time');
        error.name = 'TimeoutError';
        reject(error);
        controller.abort();
      }, 15000);
    });
    try {
      return await Promise.race([fetch(url, { ...options, signal: controller.signal }), timeout]);
    } finally {
      clearTimeout(timer);
    }
  }

  window.BTC2GSubscription = {
    send,
    text(language) {
      const [email, uncertain] = Object.hasOwn(messages, language) ? messages[language] : messages.en;
      return { email, uncertain, receiptNote: Object.hasOwn(receiptNotes, language) ? receiptNotes[language] : receiptNotes.en };
    }
  };
})();
