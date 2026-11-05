/* Shared home-screen shortcut UI. No service worker, cache or automatic install. */
(() => {
  'use strict';
  const script = document.currentScript;
  const name = script?.dataset.siteName || document.title.split('—')[0].trim();
  const lang = (document.documentElement.lang || 'en').split('-')[0];
  const messages = {
    en:['Add to home screen','Open the browser menu and choose “Install app” or “Add to Home screen”.','On iPhone or iPad, open this page in Safari. Tap Share, then “Add to Home Screen”.','Close'],
    ru:['На главный экран','Откройте меню браузера и выберите «Установить приложение» или «Добавить на главный экран».','На iPhone или iPad откройте страницу в Safari. Нажмите «Поделиться», затем «На экран Домой».','Закрыть'],
    he:['הוספה למסך הבית','פתחו את תפריט הדפדפן ובחרו התקנת אפליקציה או הוספה למסך הבית.','ב-iPhone או iPad פתחו את הדף ב-Safari. לחצו על שיתוף ואז על הוספה למסך הבית.','סגירה'],
    de:['Zum Startbildschirm','Im Browsermenü „App installieren“ oder „Zum Startbildschirm hinzufügen“ wählen.','Auf iPhone oder iPad diese Seite in Safari öffnen. Auf Teilen tippen, dann „Zum Home-Bildschirm“.','Schließen'],
    fr:['Ajouter à l’écran d’accueil','Ouvrez le menu du navigateur et choisissez « Installer l’application » ou « Ajouter à l’écran d’accueil ».','Sur iPhone ou iPad, ouvrez cette page dans Safari. Touchez Partager, puis « Sur l’écran d’accueil ».','Fermer'],
    es:['Añadir a inicio','Abre el menú del navegador y elige «Instalar aplicación» o «Añadir a pantalla de inicio».','En iPhone o iPad, abre esta página en Safari. Pulsa Compartir y luego «Añadir a pantalla de inicio».','Cerrar'],
    zh:['添加到主屏幕','打开浏览器菜单，选择“安装应用”或“添加到主屏幕”。','在 iPhone 或 iPad 上用 Safari 打开此页面。点击分享，然后选择“添加到主屏幕”。','关闭'],
    it:['Aggiungi alla schermata Home','Apri il menu del browser e scegli «Installa app» o «Aggiungi alla schermata Home».','Su iPhone o iPad apri questa pagina in Safari. Tocca Condividi, poi «Aggiungi alla schermata Home».','Chiudi'],
    pt:['Adicionar à tela inicial','Abra o menu do navegador e escolha «Instalar aplicativo» ou «Adicionar à tela inicial».','No iPhone ou iPad, abra esta página no Safari. Toque em Compartilhar e depois em «Adicionar à Tela de Início».','Fechar'],
    ar:['إضافة إلى الشاشة الرئيسية','افتح قائمة المتصفح واختر تثبيت التطبيق أو إضافة إلى الشاشة الرئيسية.','على iPhone أو iPad افتح الصفحة في Safari، ثم اضغط مشاركة وإضافة إلى الشاشة الرئيسية.','إغلاق'],
    hi:['होम स्क्रीन पर जोड़ें','ब्राउज़र मेनू खोलें और ऐप इंस्टॉल करें या होम स्क्रीन पर जोड़ें चुनें।','iPhone या iPad पर यह पेज Safari में खोलें। शेयर पर टैप करें, फिर होम स्क्रीन पर जोड़ें चुनें।','बंद करें'],
    ja:['ホーム画面に追加','ブラウザーのメニューで「アプリをインストール」または「ホーム画面に追加」を選んでください。','iPhone・iPadではSafariでこのページを開き、共有から「ホーム画面に追加」を選んでください。','閉じる'],
    ko:['홈 화면에 추가','브라우저 메뉴에서 앱 설치 또는 홈 화면에 추가를 선택하세요.','iPhone 또는 iPad에서 Safari로 이 페이지를 열고 공유를 누른 다음 홈 화면에 추가를 선택하세요.','닫기'],
    tr:['Ana ekrana ekle','Tarayıcı menüsünden Uygulamayı yükle veya Ana ekrana ekle seçeneğini seçin.','iPhone veya iPad’de sayfayı Safari’de açın. Paylaş’a, ardından Ana Ekrana Ekle’ye dokunun.','Kapat'],
    pl:['Dodaj do ekranu głównego','Otwórz menu przeglądarki i wybierz Zainstaluj aplikację lub Dodaj do ekranu głównego.','Na iPhonie lub iPadzie otwórz stronę w Safari. Stuknij Udostępnij, a następnie Do ekranu początkowego.','Zamknij'],
    nl:['Op beginscherm zetten','Open het browsermenu en kies App installeren of Toevoegen aan startscherm.','Open deze pagina op iPhone of iPad in Safari. Tik op Deel en daarna op Zet op beginscherm.','Sluiten'],
    uk:['На головний екран','Відкрийте меню браузера й виберіть «Установити застосунок» або «Додати на головний екран».','На iPhone або iPad відкрийте сторінку в Safari. Натисніть «Поширити», потім «На початковий екран».','Закрити'],
    id:['Tambahkan ke layar utama','Buka menu browser dan pilih Instal aplikasi atau Tambahkan ke layar utama.','Di iPhone atau iPad, buka halaman ini di Safari. Ketuk Bagikan, lalu Tambah ke Layar Utama.','Tutup'],
    vi:['Thêm vào màn hình chính','Mở menu trình duyệt và chọn Cài đặt ứng dụng hoặc Thêm vào màn hình chính.','Trên iPhone hoặc iPad, mở trang này trong Safari. Nhấn Chia sẻ rồi Thêm vào màn hình chính.','Đóng']
  };
  const text = messages[lang] || messages.en;
  const standalone = () => navigator.standalone === true || matchMedia('(display-mode: standalone)').matches;
  let pending = null, button, dialog, mobileEntry, focusTarget;
  addEventListener('beforeinstallprompt', event => {
    if (standalone()) return;
    event.preventDefault(); pending = event;
  });
  addEventListener('appinstalled', () => {
    pending = null;
    if (button) button.hidden = true;
    if (mobileEntry) mobileEntry.hidden = true;
    if (dialog?.open) dialog.close();
  });
  function init() {
    if (standalone() || document.querySelector('[data-site-install]')) return;
    button = document.createElement('button');
    button.type = 'button'; button.className = 'site-install-button';
    button.dataset.siteInstall = ''; button.textContent = text[0];
    (document.querySelector('footer') || document.body).append(button);
    mobileEntry = document.createElement('div');
    mobileEntry.className = 'site-install-mobile-entry';
    const mobileButton = document.createElement('button');
    mobileButton.type = 'button'; mobileButton.className = 'site-install-button';
    mobileButton.dataset.siteInstallMobile = ''; mobileButton.textContent = text[0];
    mobileEntry.append(mobileButton);
    const header = document.querySelector('header');
    if (header) header.after(mobileEntry);
    else document.body.prepend(mobileEntry);
    mobileButton.addEventListener('click', () => button.click());
    button.addEventListener('click', async () => {
      focusTarget = document.activeElement || button;
      if (pending) {
        const request = pending; pending = null;
        try { await request.prompt(); await request.userChoice; return; }
        catch { /* If the browser rejects a stale prompt, show manual steps. */ }
      }
      if (!dialog) {
        dialog = document.createElement('dialog');
        dialog.className = 'site-install-dialog';
        dialog.setAttribute('aria-labelledby', 'site-install-title');
        const heading = document.createElement('h2');
        heading.id = 'site-install-title'; heading.textContent = `${text[0]} · ${name}`;
        const instructions = document.createElement('p');
        const ios = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
        instructions.textContent = text[ios ? 2 : 1];
        const close = document.createElement('button');
        close.type = 'button'; close.textContent = text[3];
        close.addEventListener('click', () => { dialog.close(); (focusTarget || button).focus(); });
        dialog.append(heading, instructions, close); document.body.append(dialog);
      }
      dialog.showModal();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once:true});
  else init();
})();
