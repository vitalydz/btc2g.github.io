import { request } from './metrics-view.js';
import { accountLanguage } from './account-language.js';

const copy = {
  en:['ACCOUNT EMAIL','Verify your email','Confirm the address for the BTC2G account that is currently signed in.','Confirm email','Return to account','Email verification does not subscribe you to marketing or activate Premium.','Invalid verification link.','Email verified. You can return to your account.'],
  ru:['EMAIL АККАУНТА','Подтвердите email','Подтвердите адрес для аккаунта BTC2G, в который вы сейчас вошли.','Подтвердить email','Вернуться в аккаунт','Подтверждение email не подписывает вас на рассылку и не активирует Premium.','Недействительная ссылка подтверждения.','Email подтверждён. Можно вернуться в аккаунт.'],
  he:['אימייל החשבון','אימות כתובת האימייל','אשרו את הכתובת עבור חשבון BTC2G שאליו אתם מחוברים כעת.','אימות אימייל','חזרה לחשבון','אימות האימייל אינו מצרף אתכם לדיוור ואינו מפעיל Premium.','קישור האימות אינו תקין.','האימייל אומת. אפשר לחזור לחשבון.'],
  de:['KONTO-E-MAIL','E-Mail bestätigen','Bestätigen Sie die Adresse für das aktuell angemeldete BTC2G-Konto.','E-Mail bestätigen','Zurück zum Konto','Die E-Mail-Bestätigung meldet Sie nicht für Marketing an und aktiviert Premium nicht.','Ungültiger Bestätigungslink.','E-Mail bestätigt. Sie können zu Ihrem Konto zurückkehren.'],
  fr:['E-MAIL DU COMPTE','Vérifiez votre e-mail','Confirmez l’adresse du compte BTC2G actuellement connecté.','Confirmer l’e-mail','Retour au compte','La vérification ne vous inscrit pas au marketing et n’active pas Premium.','Lien de vérification non valide.','E-mail vérifié. Vous pouvez revenir à votre compte.'],
  es:['CORREO DE LA CUENTA','Verifica tu correo','Confirma la dirección de la cuenta BTC2G en la que has iniciado sesión.','Confirmar correo','Volver a la cuenta','La verificación no te suscribe a marketing ni activa Premium.','Enlace de verificación no válido.','Correo verificado. Puedes volver a tu cuenta.'],
  zh:['账户邮箱','验证您的邮箱','确认当前已登录 BTC2G 账户的邮箱地址。','确认邮箱','返回账户','邮箱验证不会订阅营销邮件，也不会激活 Premium。','验证链接无效。','邮箱已验证。您可以返回账户。'],
  it:['EMAIL DELL’ACCOUNT','Verifica la tua email','Conferma l’indirizzo dell’account BTC2G attualmente connesso.','Conferma email','Torna all’account','La verifica non ti iscrive al marketing e non attiva Premium.','Link di verifica non valido.','Email verificata. Puoi tornare al tuo account.'],
  pt:['EMAIL DA CONTA','Verifique seu email','Confirme o endereço da conta BTC2G conectada no momento.','Confirmar email','Voltar à conta','A verificação não inscreve você em marketing nem ativa o Premium.','Link de verificação inválido.','Email verificado. Você pode voltar à sua conta.'],
  ar:['بريد الحساب','أكّد بريدك الإلكتروني','أكّد العنوان لحساب BTC2G المسجّل دخوله حالياً.','تأكيد البريد','العودة إلى الحساب','لا يؤدي تأكيد البريد إلى الاشتراك في التسويق أو تفعيل Premium.','رابط التأكيد غير صالح.','تم تأكيد البريد. يمكنك العودة إلى حسابك.'],
  hi:['खाता ईमेल','अपना ईमेल सत्यापित करें','अभी साइन इन किए गए BTC2G खाते के पते की पुष्टि करें।','ईमेल की पुष्टि करें','खाते पर लौटें','ईमेल सत्यापन आपको मार्केटिंग में सदस्य नहीं बनाता और Premium सक्रिय नहीं करता।','सत्यापन लिंक अमान्य है।','ईमेल सत्यापित हो गया। आप अपने खाते पर लौट सकते हैं।'],
  ja:['アカウントのメール','メールを確認','現在ログイン中の BTC2G アカウントのアドレスを確認してください。','メールを確認','アカウントに戻る','メール確認によってマーケティング配信への登録や Premium の有効化は行われません。','確認リンクが無効です。','メールを確認しました。アカウントに戻れます。'],
  ko:['계정 이메일','이메일 확인','현재 로그인한 BTC2G 계정의 이메일 주소를 확인하세요.','이메일 확인','계정으로 돌아가기','이메일 확인은 마케팅 구독이나 Premium 활성화로 이어지지 않습니다.','확인 링크가 유효하지 않습니다.','이메일이 확인되었습니다. 계정으로 돌아갈 수 있습니다.'],
  tr:['HESAP E-POSTASI','E-postanızı doğrulayın','Şu anda giriş yapılmış BTC2G hesabının adresini doğrulayın.','E-postayı doğrula','Hesaba dön','E-posta doğrulaması sizi pazarlamaya kaydetmez veya Premium’u etkinleştirmez.','Geçersiz doğrulama bağlantısı.','E-posta doğrulandı. Hesabınıza dönebilirsiniz.'],
  pl:['E-MAIL KONTA','Zweryfikuj e-mail','Potwierdź adres aktualnie zalogowanego konta BTC2G.','Potwierdź e-mail','Wróć do konta','Weryfikacja nie zapisuje do marketingu ani nie aktywuje Premium.','Nieprawidłowy link weryfikacyjny.','E-mail zweryfikowany. Możesz wrócić do konta.'],
  nl:['ACCOUNT-E-MAIL','Verifieer uw e-mail','Bevestig het adres van het BTC2G-account dat momenteel is aangemeld.','E-mail bevestigen','Terug naar account','E-mailverificatie schrijft u niet in voor marketing en activeert Premium niet.','Ongeldige verificatielink.','E-mail geverifieerd. U kunt terug naar uw account.'],
  uk:['EMAIL ОБЛІКОВОГО ЗАПИСУ','Підтвердьте email','Підтвердьте адресу для облікового запису BTC2G, у який ви зараз увійшли.','Підтвердити email','Повернутися до облікового запису','Підтвердження email не підписує вас на маркетинг і не активує Premium.','Недійсне посилання підтвердження.','Email підтверджено. Можна повернутися до облікового запису.'],
  id:['EMAIL AKUN','Verifikasi email Anda','Konfirmasikan alamat untuk akun BTC2G yang sedang masuk.','Konfirmasi email','Kembali ke akun','Verifikasi email tidak mendaftarkan Anda ke pemasaran atau mengaktifkan Premium.','Tautan verifikasi tidak valid.','Email terverifikasi. Anda dapat kembali ke akun.'],
  vi:['EMAIL TÀI KHOẢN','Xác minh email','Xác nhận địa chỉ cho tài khoản BTC2G đang đăng nhập.','Xác nhận email','Quay lại tài khoản','Xác minh email không đăng ký tiếp thị hoặc kích hoạt Premium.','Liên kết xác minh không hợp lệ.','Email đã được xác minh. Bạn có thể quay lại tài khoản.']
};

const text = copy[accountLanguage] || copy.en;
document.documentElement.lang = accountLanguage;
document.documentElement.dir = ['he','ar'].includes(accountLanguage) ? 'rtl' : 'ltr';
document.title = `${text[1]} · BTC2G`;
[['verify-eyebrow',0],['verify-title',1],['verify-intro',2],['confirm-email',3],['verify-return',4],['verify-disclaimer',5]]
  .forEach(([id,index]) => { document.getElementById(id).textContent = text[index]; });
document.querySelector('#verify-account-link').setAttribute('aria-label', text[4]);

const button = document.querySelector('#confirm-email');
const message = document.querySelector('#verify-message');
const token = new URLSearchParams(location.search).get('token');

if (!token) {
  button.disabled = true;
  message.textContent = text[6];
}

button.addEventListener('click', async () => {
  button.disabled = true;
  message.textContent = '';
  message.classList.remove('success');
  try {
    await request('/api/email-verification/confirm', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ token })
    });
    message.textContent = text[7];
    message.classList.add('success');
  } catch (error) {
    message.textContent = error.message;
    button.disabled = false;
  }
});
