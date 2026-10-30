const languages = ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
const rows = {
  en:['Your BTC2G account','Create an account with a verified email for future research features. No payment details are collected.','new accounts today','total active accounts','confirmed email subscribers','estimated visitors today','recorded page views today','Refresh statistics','Create account','Sign in','Username','Email address','Password','Show','Hide','Use at least 12 characters.','Enter the password for this account.'],
  ru:['Ваш аккаунт BTC2G','Создайте аккаунт с подтверждённым email для будущих исследовательских функций. Платёжные данные не собираются.','новых аккаунтов сегодня','всего активных аккаунтов','подтверждённых email-подписчиков','примерно посетителей сегодня','просмотров страниц сегодня','Обновить статистику','Создать аккаунт','Войти','Имя пользователя','Адрес электронной почты','Пароль','Показать','Скрыть','Используйте не менее 12 символов.','Введите пароль этого аккаунта.'],
  he:['חשבון BTC2G שלך','צרו חשבון עם אימייל מאומת לתכונות מחקר עתידיות. לא נאספים פרטי תשלום.','חשבונות חדשים היום','חשבונות פעילים','מנויי אימייל מאומתים','מבקרים משוערים היום','צפיות היום','רענון נתונים','יצירת חשבון','כניסה','שם משתמש','כתובת אימייל','סיסמה','הצגה','הסתרה','לפחות 12 תווים.','הזינו את סיסמת החשבון.'],
  de:['Ihr BTC2G-Konto','Erstellen Sie ein Konto mit bestätigter E-Mail für künftige Forschungsfunktionen. Keine Zahlungsdaten.','neue Konten heute','aktive Konten gesamt','bestätigte E-Mail-Abonnenten','geschätzte Besucher heute','Seitenaufrufe heute','Statistik aktualisieren','Konto erstellen','Anmelden','Benutzername','E-Mail-Adresse','Passwort','Anzeigen','Ausblenden','Mindestens 12 Zeichen verwenden.','Kontopasswort eingeben.'],
  fr:['Votre compte BTC2G','Créez un compte avec une adresse e-mail vérifiée pour les futures fonctions de recherche. Aucune donnée de paiement.','nouveaux comptes aujourd’hui','comptes actifs','abonnés e-mail confirmés','visiteurs estimés aujourd’hui','pages vues aujourd’hui','Actualiser les statistiques','Créer un compte','Se connecter','Nom d’utilisateur','Adresse e-mail','Mot de passe','Afficher','Masquer','Utilisez au moins 12 caractères.','Saisissez le mot de passe du compte.'],
  es:['Tu cuenta BTC2G','Crea una cuenta con correo verificado para futuras funciones de investigación. No se recopilan datos de pago.','cuentas nuevas hoy','cuentas activas','suscriptores confirmados','visitantes estimados hoy','páginas vistas hoy','Actualizar estadísticas','Crear cuenta','Iniciar sesión','Usuario','Correo electrónico','Contraseña','Mostrar','Ocultar','Usa al menos 12 caracteres.','Ingresa la contraseña de la cuenta.'],
  zh:['您的 BTC2G 账户','使用已验证邮箱创建账户，以使用未来的研究功能。不会收集付款信息。','今日新账户','活跃账户总数','已确认邮件订阅者','今日估算访客','今日页面浏览量','刷新统计','创建账户','登录','用户名','电子邮箱','密码','显示','隐藏','至少使用 12 个字符。','输入此账户的密码。'],
  it:['Il tuo account BTC2G','Crea un account con email verificata per le future funzioni di ricerca. Nessun dato di pagamento.','nuovi account oggi','account attivi','iscritti email confermati','visitatori stimati oggi','visualizzazioni oggi','Aggiorna statistiche','Crea account','Accedi','Nome utente','Indirizzo email','Password','Mostra','Nascondi','Usa almeno 12 caratteri.','Inserisci la password dell’account.'],
  pt:['Sua conta BTC2G','Crie uma conta com email verificado para futuras funções de pesquisa. Nenhum dado de pagamento é coletado.','novas contas hoje','contas ativas','assinantes confirmados','visitantes estimados hoje','visualizações hoje','Atualizar estatísticas','Criar conta','Entrar','Nome de usuário','Email','Senha','Mostrar','Ocultar','Use pelo menos 12 caracteres.','Digite a senha da conta.'],
  ar:['حساب BTC2G الخاص بك','أنشئ حساباً ببريد إلكتروني موثّق لميزات البحث المستقبلية. لا تُجمع بيانات الدفع.','حسابات جديدة اليوم','إجمالي الحسابات النشطة','مشتركو البريد المؤكدون','الزوار المقدرون اليوم','مشاهدات الصفحات اليوم','تحديث الإحصاءات','إنشاء حساب','تسجيل الدخول','اسم المستخدم','البريد الإلكتروني','كلمة المرور','إظهار','إخفاء','استخدم 12 حرفاً على الأقل.','أدخل كلمة مرور الحساب.'],
  hi:['आपका BTC2G खाता','भविष्य की शोध सुविधाओं के लिए सत्यापित ईमेल से खाता बनाएँ। भुगतान विवरण एकत्र नहीं किए जाते।','आज के नए खाते','कुल सक्रिय खाते','पुष्ट ईमेल सदस्य','आज के अनुमानित दर्शक','आज के पेज व्यू','आँकड़े रीफ़्रेश करें','खाता बनाएँ','साइन इन','उपयोगकर्ता नाम','ईमेल पता','पासवर्ड','दिखाएँ','छिपाएँ','कम से कम 12 अक्षर उपयोग करें।','इस खाते का पासवर्ड दर्ज करें।'],
  ja:['BTC2G アカウント','確認済みメールでアカウントを作成し、今後の研究機能を利用できます。支払い情報は収集しません。','本日の新規アカウント','有効なアカウント','確認済みメール購読者','本日の推定訪問者','本日のページビュー','統計を更新','アカウント作成','ログイン','ユーザー名','メールアドレス','パスワード','表示','非表示','12文字以上を使用してください。','アカウントのパスワードを入力してください。'],
  ko:['BTC2G 계정','확인된 이메일로 계정을 만들고 향후 연구 기능을 이용하세요. 결제 정보는 수집하지 않습니다.','오늘 새 계정','전체 활성 계정','확인된 이메일 구독자','오늘 예상 방문자','오늘 페이지 조회','통계 새로고침','계정 만들기','로그인','사용자 이름','이메일 주소','비밀번호','표시','숨기기','12자 이상 사용하세요.','계정 비밀번호를 입력하세요.'],
  tr:['BTC2G hesabınız','Gelecekteki araştırma özellikleri için doğrulanmış e-postayla hesap oluşturun. Ödeme bilgisi toplanmaz.','bugünkü yeni hesaplar','toplam aktif hesap','doğrulanmış e-posta aboneleri','bugünkü tahmini ziyaretçi','bugünkü sayfa görüntüleme','İstatistikleri yenile','Hesap oluştur','Giriş yap','Kullanıcı adı','E-posta adresi','Parola','Göster','Gizle','En az 12 karakter kullanın.','Hesap parolasını girin.'],
  pl:['Twoje konto BTC2G','Utwórz konto ze zweryfikowanym e-mailem dla przyszłych funkcji badawczych. Nie zbieramy danych płatniczych.','nowe konta dzisiaj','aktywne konta','potwierdzeni subskrybenci','szacowani odwiedzający dzisiaj','wyświetlenia dzisiaj','Odśwież statystyki','Utwórz konto','Zaloguj się','Nazwa użytkownika','Adres e-mail','Hasło','Pokaż','Ukryj','Użyj co najmniej 12 znaków.','Wpisz hasło konta.'],
  nl:['Uw BTC2G-account','Maak een account met geverifieerde e-mail voor toekomstige onderzoeksfuncties. Geen betaalgegevens.','nieuwe accounts vandaag','actieve accounts','bevestigde e-mailabonnees','geschatte bezoekers vandaag','paginaweergaven vandaag','Statistieken vernieuwen','Account maken','Aanmelden','Gebruikersnaam','E-mailadres','Wachtwoord','Tonen','Verbergen','Gebruik minimaal 12 tekens.','Voer het accountwachtwoord in.'],
  uk:['Ваш обліковий запис BTC2G','Створіть обліковий запис із підтвердженим email для майбутніх дослідницьких функцій. Платіжні дані не збираються.','нових облікових записів сьогодні','активних облікових записів','підтверджених email-підписників','приблизно відвідувачів сьогодні','переглядів сьогодні','Оновити статистику','Створити обліковий запис','Увійти','Ім’я користувача','Електронна пошта','Пароль','Показати','Сховати','Використовуйте щонайменше 12 символів.','Введіть пароль облікового запису.'],
  id:['Akun BTC2G Anda','Buat akun dengan email terverifikasi untuk fitur riset mendatang. Data pembayaran tidak dikumpulkan.','akun baru hari ini','total akun aktif','pelanggan email terkonfirmasi','perkiraan pengunjung hari ini','tayangan halaman hari ini','Segarkan statistik','Buat akun','Masuk','Nama pengguna','Alamat email','Kata sandi','Tampilkan','Sembunyikan','Gunakan minimal 12 karakter.','Masukkan kata sandi akun.'],
  vi:['Tài khoản BTC2G của bạn','Tạo tài khoản bằng email đã xác minh cho các tính năng nghiên cứu tương lai. Không thu thập thông tin thanh toán.','tài khoản mới hôm nay','tổng tài khoản hoạt động','người đăng ký email đã xác nhận','khách ước tính hôm nay','lượt xem trang hôm nay','Làm mới thống kê','Tạo tài khoản','Đăng nhập','Tên người dùng','Địa chỉ email','Mật khẩu','Hiện','Ẩn','Dùng ít nhất 12 ký tự.','Nhập mật khẩu của tài khoản.']
};

const keys = ['title','intro','today','total','subscribers','visitors','views','refresh','create','signIn','username','email','password','show','hide','passwordHint','loginPasswordHint'];
const dictionaries = Object.fromEntries(Object.entries(rows).map(([lang, values]) => [lang, Object.fromEntries(keys.map((key, index) => [key, values[index]]))]));
const extras = {
  en: { dataUnavailable:'Data unavailable', statsUnavailable:'Data unavailable. The statistics service could not be reached. Retry below.' },
  ru: { dataUnavailable:'Данные недоступны', statsUnavailable:'Данные недоступны. Не удалось связаться со службой статистики. Повторите попытку ниже.' }
};
const requested = new URLSearchParams(location.search).get('lang');
let saved = '';
try { saved = localStorage.getItem('lang') || ''; } catch {}
const browser = (navigator.language || 'en').slice(0, 2).toLowerCase();
export let accountLanguage = languages.includes(requested) ? requested : (languages.includes(saved) ? saved : (languages.includes(browser) ? browser : 'en'));

export function accountText(key) { return dictionaries[accountLanguage]?.[key] || extras[accountLanguage]?.[key] || dictionaries.en[key] || extras.en[key] || key; }

export function applyAccountLanguage() {
  document.documentElement.lang = accountLanguage;
  document.documentElement.dir = ['he','ar'].includes(accountLanguage) ? 'rtl' : 'ltr';
  try { localStorage.setItem('lang', accountLanguage); } catch {}
  const selector = document.querySelector('#account-language');
  selector.value = accountLanguage;
  const targets = { title:'account-title', intro:'account-intro', today:'stats-title', total:'total-label', subscribers:'subscribers-label', visitors:'visitors-label', views:'views-label', refresh:'refresh-stats', create:'register-tab', signIn:'login-tab', username:'username-label', email:'registration-email-label', password:'password-label', show:'toggle-password' };
  Object.entries(targets).forEach(([key,id]) => { const node = document.getElementById(id); if (node) node.textContent = accountText(key); });
  document.querySelector('#account-home-link').href = `https://btc2g.com/?lang=${encodeURIComponent(accountLanguage)}`;
  if (accountLanguage === 'ru') {
    const map = new Map([
      ['EARLY ACCESS','РАННИЙ ДОСТУП'],['Create account','Создать аккаунт'],['Choose a username and secure password.','Выберите имя пользователя и надёжный пароль.'],['Check your inbox','Проверьте входящие письма'],['Open the verification link within 30 minutes.','Откройте ссылку подтверждения в течение 30 минут.'],['Email verified','Email подтверждён'],['Your registration is complete.','Регистрация завершена.'],['3–32 characters. Start with a letter; use letters, numbers, or underscores.','3–32 символа. Начните с буквы; используйте буквы, цифры или подчёркивания.'],['Required. Registration is complete only after you verify this address.','Обязательно. Регистрация завершится только после подтверждения адреса.'],['I accept the','Я принимаю'],['Terms','Условия'],['and','и'],['Privacy Policy','Политику конфиденциальности'],['.','.'],['BTC2G provides informational research, not financial advice. Account totals include email-verified registrations, not verified investors or customers.','BTC2G предоставляет исследовательскую информацию, а не финансовые рекомендации. В статистику входят регистрации с подтверждённым email, а не проверенные инвесторы или клиенты.']
    ]);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) { const value = walker.currentNode.nodeValue.trim(); if (map.has(value)) walker.currentNode.nodeValue = walker.currentNode.nodeValue.replace(value, map.get(value)); }
  }
  document.querySelector('#password-hint').textContent = accountText('passwordHint');
  document.querySelector('#submit-button').textContent = accountText('create');
  selector.addEventListener('change', () => {
    const url = new URL(location.href);
    url.searchParams.set('lang', selector.value);
    location.assign(url.toString());
  }, { once:true });
}

export { dictionaries, languages };
