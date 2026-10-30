(function () {
  'use strict';
  const rows = {
    en: ['Payment confirmation pending','Opening this page does not confirm payment or activate access. Access requires verified server-side payment confirmation.','BTC2G signals are model-based and are not financial advice.','Back to BTC2G'],
    ru: ['Ожидается подтверждение оплаты','Открытие страницы не подтверждает оплату и не активирует доступ. Доступ требует проверки оплаты на сервере.','Сигналы BTC2G основаны на модели и не являются финансовой рекомендацией.','Вернуться к BTC2G'],
    he: ['ממתינים לאישור תשלום','פתיחת העמוד אינה מאשרת תשלום ואינה מפעילה גישה. הגישה דורשת אימות תשלום בצד השרת.','איתותי BTC2G מבוססים על מודל ואינם ייעוץ פיננסי.','חזרה אל BTC2G'],
    de: ['Zahlungsbestätigung ausstehend','Diese Seite bestätigt keine Zahlung und aktiviert keinen Zugang. Der Zugang erfordert eine serverseitig geprüfte Zahlungsbestätigung.','BTC2G-Signale sind modellbasiert und keine Finanzberatung.','Zurück zu BTC2G'],
    fr: ['Confirmation du paiement en attente','Ouvrir cette page ne confirme pas le paiement et n’active pas l’accès. L’accès nécessite une confirmation vérifiée côté serveur.','Les signaux BTC2G sont basés sur un modèle et ne constituent pas un conseil financier.','Retour à BTC2G'],
    es: ['Confirmación de pago pendiente','Abrir esta página no confirma el pago ni activa el acceso. El acceso requiere confirmar el pago en el servidor.','Las señales BTC2G se basan en un modelo y no son asesoramiento financiero.','Volver a BTC2G'],
    zh: ['等待付款确认','打开此页面不代表付款成功，也不会激活访问权限。访问权限需要服务器验证付款确认。','BTC2G信号基于模型，不构成财务建议。','返回BTC2G'],
    it: ['Conferma del pagamento in attesa','Aprire questa pagina non conferma il pagamento né attiva l’accesso. L’accesso richiede una conferma verificata dal server.','I segnali BTC2G si basano su un modello e non sono consulenza finanziaria.','Torna a BTC2G'],
    pt: ['Confirmação de pagamento pendente','Abrir esta página não confirma o pagamento nem ativa o acesso. O acesso requer confirmação verificada pelo servidor.','Os sinais BTC2G baseiam-se num modelo e não são aconselhamento financeiro.','Voltar à BTC2G'],
    ar: ['بانتظار تأكيد الدفع','فتح هذه الصفحة لا يؤكد الدفع ولا يفعّل الوصول. يتطلب الوصول تأكيد دفع متحققًا منه على الخادم.','إشارات BTC2G قائمة على نموذج وليست نصيحة مالية.','العودة إلى BTC2G'],
    hi: ['भुगतान की पुष्टि लंबित है','यह पेज खोलने से भुगतान की पुष्टि या पहुँच चालू नहीं होती। पहुँच के लिए सर्वर पर सत्यापित भुगतान पुष्टि आवश्यक है।','BTC2G संकेत मॉडल पर आधारित हैं और वित्तीय सलाह नहीं हैं।','BTC2G पर वापस जाएँ'],
    ja: ['支払い確認待ち','このページを開いても支払いの確認やアクセスの有効化にはなりません。サーバー側で検証した支払い確認が必要です。','BTC2Gのシグナルはモデルに基づくもので、金融助言ではありません。','BTC2Gに戻る'],
    ko: ['결제 확인 대기 중','이 페이지를 열어도 결제가 확인되거나 접근 권한이 활성화되지 않습니다. 서버에서 검증된 결제 확인이 필요합니다.','BTC2G 신호는 모델 기반이며 금융 조언이 아닙니다.','BTC2G로 돌아가기'],
    tr: ['Ödeme onayı bekleniyor','Bu sayfayı açmak ödemeyi onaylamaz veya erişimi etkinleştirmez. Erişim, sunucuda doğrulanmış ödeme onayı gerektirir.','BTC2G sinyalleri modele dayalıdır ve finansal tavsiye değildir.','BTC2G’ye dön'],
    pl: ['Oczekiwanie na potwierdzenie płatności','Otwarcie strony nie potwierdza płatności ani nie aktywuje dostępu. Dostęp wymaga zweryfikowanego potwierdzenia płatności na serwerze.','Sygnały BTC2G są oparte na modelu i nie stanowią porady finansowej.','Powrót do BTC2G'],
    nl: ['Wachten op betalingsbevestiging','Deze pagina openen bevestigt geen betaling en activeert geen toegang. Toegang vereist een op de server geverifieerde betalingsbevestiging.','BTC2G-signalen zijn gebaseerd op een model en zijn geen financieel advies.','Terug naar BTC2G'],
    uk: ['Очікується підтвердження оплати','Відкриття сторінки не підтверджує оплату й не активує доступ. Потрібне перевірене підтвердження оплати на сервері.','Сигнали BTC2G засновані на моделі та не є фінансовою порадою.','Повернутися до BTC2G'],
    id: ['Menunggu konfirmasi pembayaran','Membuka halaman ini tidak mengonfirmasi pembayaran atau mengaktifkan akses. Akses memerlukan konfirmasi pembayaran yang diverifikasi server.','Sinyal BTC2G berbasis model dan bukan nasihat keuangan.','Kembali ke BTC2G'],
    vi: ['Đang chờ xác nhận thanh toán','Mở trang này không xác nhận thanh toán hay kích hoạt quyền truy cập. Cần xác nhận thanh toán đã được máy chủ xác minh.','Tín hiệu BTC2G dựa trên mô hình, không phải tư vấn tài chính.','Quay lại BTC2G']
  };
  const select = document.getElementById('status-language');
  window.BTC2GLanguages.options(select);
  function render(value) {
    const lang = window.BTC2GLanguages.apply(value);
    const [title, pending, disclaimer, home] = rows[lang];
    select.value = lang;
    select.setAttribute('aria-label', window.BTC2GLanguages.names[lang]);
    const data = {title, pending, disclaimer, home};
    document.querySelectorAll('[data-status]').forEach(el => { el.textContent = data[el.dataset.status]; });
    document.title = 'BTC2G · ' + title;
  }
  select.addEventListener('change', event => render(event.target.value));
  render(window.BTC2GLanguages.requested());
})();
