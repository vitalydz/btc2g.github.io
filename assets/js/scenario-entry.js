(function (root) {
  const values = {
    en:['INTERACTIVE TOOL','Test a BTC / Gold scenario','Adjust hypothetical Bitcoin and gold moves, compare the resulting ratio, or reverse-calculate a target BTC price.','Open simulator'],
    ru:['ИНТЕРАКТИВНЫЙ ИНСТРУМЕНТ','Проверьте сценарий BTC / золото','Настройте гипотетические движения Биткоина и золота, сравните итоговый ratio или рассчитайте цену BTC для целевого ratio.','Открыть симулятор'],
    he:['כלי אינטראקטיבי','בדיקת תרחיש BTC / זהב','התאימו שינויי מחיר היפותטיים, השוו את היחס או חשבו מחיר BTC ליחס יעד.','פתיחת הסימולטור'],
    de:['INTERAKTIVES TOOL','BTC-/Gold-Szenario testen','Passen Sie hypothetische Bewegungen an, vergleichen Sie das Verhältnis oder berechnen Sie einen BTC-Preis für ein Zielverhältnis.','Simulator öffnen'],
    fr:['OUTIL INTERACTIF','Tester un scénario BTC / or','Ajustez des variations hypothétiques, comparez le ratio ou calculez le prix BTC correspondant à un ratio cible.','Ouvrir le simulateur'],
    es:['HERRAMIENTA INTERACTIVA','Prueba un escenario BTC / oro','Ajusta movimientos hipotéticos, compara el ratio o calcula el precio BTC para un ratio objetivo.','Abrir simulador'],
    zh:['交互式工具','测试 BTC / 黄金情景','调整比特币和黄金的假设变化，比较所得比率，或反向计算目标比率对应的 BTC 价格。','打开模拟器'],
    it:['STRUMENTO INTERATTIVO','Prova uno scenario BTC / oro','Regola movimenti ipotetici, confronta il rapporto o calcola il prezzo BTC per un rapporto obiettivo.','Apri il simulatore'],
    pt:['FERRAMENTA INTERATIVA','Teste um cenário BTC / ouro','Ajuste movimentos hipotéticos, compare o rácio ou calcule o preço BTC para um rácio alvo.','Abrir simulador'],
    ar:['أداة تفاعلية','اختبر سيناريو BTC / الذهب','اضبط تحركات افتراضية وقارن النسبة أو احسب سعر BTC لنسبة مستهدفة.','فتح المحاكي'],
    hi:['इंटरैक्टिव टूल','BTC / गोल्ड परिदृश्य जाँचें','काल्पनिक बदलाव समायोजित करें, अनुपात की तुलना करें या लक्ष्य अनुपात के लिए BTC मूल्य निकालें।','सिम्युलेटर खोलें'],
    ja:['インタラクティブツール','BTC / ゴールドのシナリオを試す','仮定の変動を調整し、比率を比較するか、目標比率に必要なBTC価格を逆算します。','シミュレーターを開く'],
    ko:['대화형 도구','BTC / 금 시나리오 테스트','가상 변동을 조정하고 비율을 비교하거나 목표 비율의 BTC 가격을 역산하세요.','시뮬레이터 열기'],
    tr:['ETKİLEŞİMLİ ARAÇ','BTC / Altın senaryosu deneyin','Varsayımsal hareketleri ayarlayın, oranı karşılaştırın veya hedef oran için BTC fiyatını hesaplayın.','Simülatörü aç'],
    pl:['NARZĘDZIE INTERAKTYWNE','Przetestuj scenariusz BTC / złoto','Dostosuj hipotetyczne ruchy, porównaj wskaźnik lub oblicz cenę BTC dla wskaźnika docelowego.','Otwórz symulator'],
    nl:['INTERACTIEVE TOOL','Test een BTC / goud-scenario','Pas hypothetische bewegingen aan, vergelijk de ratio of bereken een BTC-prijs voor een doelratio.','Simulator openen'],
    uk:['ІНТЕРАКТИВНИЙ ІНСТРУМЕНТ','Перевірте сценарій BTC / золото','Налаштуйте гіпотетичні рухи, порівняйте ratio або розрахуйте ціну BTC для цільового ratio.','Відкрити симулятор'],
    id:['ALAT INTERAKTIF','Uji skenario BTC / emas','Sesuaikan pergerakan hipotetis, bandingkan rasio, atau hitung harga BTC untuk rasio target.','Buka simulator'],
    vi:['CÔNG CỤ TƯƠNG TÁC','Thử kịch bản BTC / vàng','Điều chỉnh biến động giả định, so sánh tỷ lệ hoặc tính giá BTC cho tỷ lệ mục tiêu.','Mở trình mô phỏng']
  };
  function setLanguage(lang) {
    const t=values[lang]||values.en;
    [['scenario-entry-tag',0],['scenario-entry-title',1],['scenario-entry-note',2],['scenario-entry-link',3]].forEach(([id,index])=>{const el=root.document?.getElementById(id);if(el)el.textContent=t[index];});
    const link=root.document?.getElementById('scenario-entry-link');if(link)link.href=`scenario.html?lang=${encodeURIComponent(values[lang]?lang:'en')}`;
  }
  root.BTC2GScenarioEntry={values,setLanguage};
})(typeof window!=='undefined'?window:globalThis);
