(function () {
  'use strict';
  const rows = {
    it: ['Acquista Bitcoin','Apri il sito ↗','I link aprono i siti ufficiali in una nuova scheda. BTC2G non elabora pagamenti.','Torna a BTC2G','La disponibilità dipende dal paese.','Confronta i preventivi BTC →'],
    pt: ['Comprar Bitcoin','Abrir site ↗','Os links abrem os sites oficiais numa nova aba. A BTC2G não processa pagamentos.','Voltar à BTC2G','A disponibilidade depende do país.','Comparar cotações BTC →'],
    ar: ['شراء بيتكوين','فتح الموقع ↗','تفتح الروابط مواقع المزودين الرسمية في علامة تبويب جديدة. لا تعالج BTC2G المدفوعات.','العودة إلى BTC2G','يعتمد توفر المزود على البلد.','مقارنة عروض BTC ←'],
    hi: ['बिटकॉइन खरीदें','वेबसाइट खोलें ↗','लिंक प्रदाताओं की आधिकारिक वेबसाइट नए टैब में खोलते हैं। BTC2G भुगतान संसाधित नहीं करता।','BTC2G पर वापस जाएँ','प्रदाता की उपलब्धता देश पर निर्भर है।','BTC कोटेशन की तुलना करें →'],
    ja: ['ビットコインを購入','公式サイトを開く ↗','リンクはサービスの公式サイトを新しいタブで開きます。BTC2Gは決済を処理しません。','BTC2Gに戻る','サービスの利用可否は国によって異なります。','BTC見積もりを比較 →'],
    ko: ['비트코인 구매','웹사이트 열기 ↗','링크는 서비스의 공식 사이트를 새 탭에서 엽니다. BTC2G는 결제를 처리하지 않습니다.','BTC2G로 돌아가기','서비스 이용 가능 여부는 국가에 따라 다릅니다.','BTC 견적 비교 →'],
    tr: ['Bitcoin satın al','Siteyi aç ↗','Bağlantılar sağlayıcıların resmi sitelerini yeni sekmede açar. BTC2G ödeme işlemez.','BTC2G’ye dön','Sağlayıcının kullanılabilirliği ülkeye bağlıdır.','BTC tekliflerini karşılaştır →'],
    pl: ['Kup Bitcoin','Otwórz stronę ↗','Linki otwierają oficjalne strony dostawców w nowej karcie. BTC2G nie obsługuje płatności.','Powrót do BTC2G','Dostępność dostawcy zależy od kraju.','Porównaj oferty BTC →'],
    nl: ['Bitcoin kopen','Website openen ↗','Links openen de officiële websites in een nieuw tabblad. BTC2G verwerkt geen betalingen.','Terug naar BTC2G','Beschikbaarheid hangt af van het land.','BTC-offertes vergelijken →'],
    uk: ['Купити біткоїн','Відкрити сайт ↗','Посилання відкривають офіційні сайти провайдерів у новій вкладці. BTC2G не обробляє платежі.','Повернутися до BTC2G','Доступність провайдера залежить від країни.','Порівняти котирування BTC →'],
    id: ['Beli Bitcoin','Buka situs ↗','Tautan membuka situs resmi penyedia di tab baru. BTC2G tidak memproses pembayaran.','Kembali ke BTC2G','Ketersediaan penyedia bergantung pada negara.','Bandingkan penawaran BTC →'],
    vi: ['Mua Bitcoin','Mở trang web ↗','Liên kết mở trang chính thức của nhà cung cấp trong thẻ mới. BTC2G không xử lý thanh toán.','Quay lại BTC2G','Dịch vụ khả dụng tùy quốc gia.','So sánh báo giá BTC →']
  };
  const texts = {}, availability = {}, saveLabels = {};
  for (const [lang, [title, soon, message, home, footer, save]] of Object.entries(rows)) {
    texts[lang] = {title, topperTitle: 'Topper', languageLabel: window.BTC2GLanguages.names[lang]};
    availability[lang] = {soon, message, home, footer};
    saveLabels[lang] = save;
  }
  window.BTC2GBuyLocales = {texts, availability, saveLabels};
})();
