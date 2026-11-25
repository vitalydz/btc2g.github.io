(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BTC2GRegimes = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const REGIMES = ['btc', 'neutral', 'gold'];
  const labels = {
    en: ['BTC / Gold regimes','Current','BTC dominates','Neutral','Gold dominates','Avg daily ratio change','Typical run','market observations','30-observation moving average; BTC or Gold dominates outside a ±5% band. Statistics use published history only.','Extended regime history in Premium →'],
    ru: ['Режимы BTC / Gold','Сейчас','Доминирует BTC','Нейтрально','Доминирует золото','Среднее дневное изменение ratio','Типичная длительность','рыночных наблюдений','Скользящая средняя за 30 наблюдений; доминирование BTC или золота фиксируется за пределами коридора ±5%. Только опубликованная история.','Расширенная история режимов в Premium →'],
    he: ['משטרי BTC / Gold','כעת','BTC דומיננטי','ניטרלי','זהב דומיננטי','שינוי יומי ממוצע ביחס','משך טיפוסי','תצפיות שוק','ממוצע נע של 30 תצפיות; דומיננטיות BTC או זהב מחוץ לטווח ±5%. נתונים היסטוריים שפורסמו בלבד.','היסטוריית משטרים מורחבת ב-Premium ←'],
    de: ['BTC-/Gold-Regime','Aktuell','BTC dominiert','Neutral','Gold dominiert','Ø tägliche Ratio-Änderung','Typische Dauer','Marktbeobachtungen','Gleitender Durchschnitt über 30 Beobachtungen; BTC- oder Gold-Dominanz außerhalb eines ±5%-Bands. Nur veröffentlichte Historie.','Erweiterte Regime-Historie in Premium →'],
    fr: ['Régimes BTC / or','Actuel','BTC domine','Neutre','L’or domine','Variation quotidienne moyenne du ratio','Durée typique','observations de marché','Moyenne mobile sur 30 observations ; domination BTC ou or hors d’une bande de ±5 %. Historique publié uniquement.','Historique étendu des régimes dans Premium →'],
    es: ['Regímenes BTC / oro','Actual','BTC domina','Neutral','El oro domina','Cambio diario medio del ratio','Duración típica','observaciones de mercado','Media móvil de 30 observaciones; BTC u oro domina fuera de una banda de ±5 %. Solo historial publicado.','Historial ampliado de regímenes en Premium →'],
    zh: ['BTC / 黄金状态','当前','BTC 占优','中性','黄金占优','比率日均变化','典型持续时间','个市场观测','30 个观测值移动平均线；超出 ±5% 区间时判定 BTC 或黄金占优。仅使用已发布历史。','Premium 扩展状态历史 →'],
    it: ['Regimi BTC / oro','Attuale','BTC domina','Neutrale','L’oro domina','Variazione media giornaliera del ratio','Durata tipica','osservazioni di mercato','Media mobile su 30 osservazioni; BTC o oro domina fuori da una fascia del ±5%. Solo storico pubblicato.','Storico esteso dei regimi in Premium →'],
    pt: ['Regimes BTC / ouro','Atual','BTC domina','Neutro','Ouro domina','Variação diária média do ratio','Duração típica','observações de mercado','Média móvel de 30 observações; BTC ou ouro domina fora de uma faixa de ±5%. Apenas histórico publicado.','Histórico estendido de regimes no Premium →'],
    ar: ['أنظمة BTC / الذهب','الحالي','هيمنة BTC','محايد','هيمنة الذهب','متوسط التغير اليومي للنسبة','المدة المعتادة','مشاهدات سوقية','متوسط متحرك لـ30 مشاهدة؛ هيمنة BTC أو الذهب خارج نطاق ±5%. يعتمد على السجل المنشور فقط.','سجل موسع للأنظمة في Premium ←'],
    hi: ['BTC / Gold व्यवस्थाएँ','वर्तमान','BTC प्रभावी','तटस्थ','Gold प्रभावी','औसत दैनिक ratio बदलाव','सामान्य अवधि','बाज़ार अवलोकन','30-अवलोकन चल औसत; ±5% बैंड के बाहर BTC या Gold प्रभावी। केवल प्रकाशित इतिहास।','Premium में विस्तृत व्यवस्था इतिहास →'],
    ja: ['BTC / Gold レジーム','現在','BTC 優位','中立','Gold 優位','比率の平均日次変化','通常の期間','市場観測','30観測の移動平均。±5%帯の外側でBTCまたはGold優位。公開履歴のみを使用。','Premium の拡張レジーム履歴 →'],
    ko: ['BTC / Gold 국면','현재','BTC 우위','중립','Gold 우위','평균 일간 비율 변화','일반 지속 기간','시장 관측치','30개 관측 이동평균; ±5% 범위 밖에서 BTC 또는 Gold 우위. 공개된 기록만 사용.','Premium 확장 국면 기록 →'],
    tr: ['BTC / Altın rejimleri','Güncel','BTC baskın','Nötr','Altın baskın','Ortalama günlük oran değişimi','Tipik süre','piyasa gözlemi','30 gözlemlik hareketli ortalama; ±%5 bandı dışında BTC veya altın baskın. Yalnızca yayımlanmış geçmiş.','Premium’da genişletilmiş rejim geçmişi →'],
    pl: ['Reżimy BTC / złoto','Aktualny','BTC dominuje','Neutralny','Złoto dominuje','Średnia dzienna zmiana ratio','Typowy czas','obserwacji rynkowych','Średnia krocząca z 30 obserwacji; dominacja BTC lub złota poza pasmem ±5%. Tylko opublikowana historia.','Rozszerzona historia reżimów w Premium →'],
    nl: ['BTC-/goudregimes','Huidig','BTC domineert','Neutraal','Goud domineert','Gem. dagelijkse ratioverandering','Typische duur','marktwaarnemingen','Voortschrijdend gemiddelde van 30 waarnemingen; BTC- of gouddominantie buiten een band van ±5%. Alleen gepubliceerde historie.','Uitgebreide regimehistorie in Premium →'],
    uk: ['Режими BTC / Gold','Поточний','Домінує BTC','Нейтрально','Домінує золото','Середня денна зміна ratio','Типова тривалість','ринкових спостережень','Ковзне середнє за 30 спостережень; домінування BTC або золота поза коридором ±5%. Лише опублікована історія.','Розширена історія режимів у Premium →'],
    id: ['Rezim BTC / emas','Saat ini','BTC dominan','Netral','Emas dominan','Rata-rata perubahan ratio harian','Durasi tipikal','observasi pasar','Rata-rata bergerak 30 observasi; BTC atau emas dominan di luar rentang ±5%. Hanya riwayat terpublikasi.','Riwayat rezim lengkap di Premium →'],
    vi: ['Chế độ BTC / vàng','Hiện tại','BTC chiếm ưu thế','Trung lập','Vàng chiếm ưu thế','Thay đổi ratio trung bình ngày','Thời lượng điển hình','quan sát thị trường','Trung bình động 30 quan sát; BTC hoặc vàng chiếm ưu thế ngoài biên ±5%. Chỉ dùng lịch sử đã công bố.','Lịch sử chế độ mở rộng trong Premium →']
  };
  let lastAnalysis = null;
  let language = 'en';

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function analyze(series, options = {}) {
    const windowSize = Number.isInteger(options.windowSize) ? options.windowSize : 30;
    const threshold = Number.isFinite(options.threshold) ? options.threshold : 0.05;
    if (!Array.isArray(series) || series.length < windowSize || windowSize < 2 || threshold <= 0 || threshold >= 1) return null;
    if (!series.every((row, index) => /^\d{4}-\d{2}-\d{2}$/.test(row?.date) && Number.isFinite(row.ratio) && row.ratio > 0 && (!index || row.date > series[index - 1].date))) return null;
    const assignments = Array(series.length).fill(null);
    for (let index = windowSize - 1; index < series.length; index += 1) {
      let total = 0;
      for (let offset = index - windowSize + 1; offset <= index; offset += 1) total += series[offset].ratio;
      const average = total / windowSize;
      assignments[index] = series[index].ratio > average * (1 + threshold) ? 'btc'
        : series[index].ratio < average * (1 - threshold) ? 'gold' : 'neutral';
    }
    const runs = [];
    for (let index = windowSize - 1; index < assignments.length;) {
      const regime = assignments[index];
      let end = index;
      while (end + 1 < assignments.length && assignments[end + 1] === regime) end += 1;
      runs.push({ regime, observations: end - index + 1, start: series[index].date, end: series[end].date });
      index = end + 1;
    }
    const stats = Object.fromEntries(REGIMES.map(regime => {
      const changes = [];
      assignments.forEach((assigned, index) => {
        if (assigned === regime && index > 0) changes.push((series[index].ratio / series[index - 1].ratio - 1) * 100);
      });
      const durations = runs.filter(run => run.regime === regime).map(run => run.observations);
      return [regime, {
        observations: assignments.filter(value => value === regime).length,
        averageDailyChangePct: changes.length ? changes.reduce((sum, value) => sum + value, 0) / changes.length : null,
        typicalRunObservations: median(durations)
      }];
    }));
    return { assignments, runs, stats, current: assignments.at(-1), windowSize, threshold };
  }

  function formatChange(value) {
    if (!Number.isFinite(value)) return '—';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  }

  function render(analysis, lang = language) {
    if (!analysis || typeof document === 'undefined') return;
    lastAnalysis = analysis;
    language = labels[lang] ? lang : 'en';
    const text = labels[language];
    const names = { btc: text[2], neutral: text[3], gold: text[4] };
    const detector = document.getElementById('regime-detector');
    if (!detector) return;
    document.getElementById('regime-title').textContent = text[0];
    const current = document.getElementById('regime-current');
    current.textContent = `${text[1]}: ${names[analysis.current]}`;
    current.dataset.regime = analysis.current;
    const summary = document.getElementById('regime-summary');
    summary.replaceChildren(...REGIMES.map(regime => {
      const card = document.createElement('article');
      card.className = 'regime-card'; card.dataset.regime = regime; card.setAttribute('role', 'listitem');
      const heading = document.createElement('strong');
      const swatch = document.createElement('i'); swatch.className = 'regime-swatch'; swatch.setAttribute('aria-hidden', 'true');
      heading.append(swatch, document.createTextNode(names[regime]));
      const change = document.createElement('span'); change.textContent = `${text[5]}: ${formatChange(analysis.stats[regime].averageDailyChangePct)}`;
      const duration = document.createElement('span'); duration.textContent = `${text[6]}: ${analysis.stats[regime].typicalRunObservations ?? '—'} ${text[7]}`;
      card.append(heading, change, duration);
      return card;
    }));
    document.getElementById('regime-method').textContent = text[8];
    const premiumLink = document.getElementById('regime-premium-link');
    premiumLink.textContent = text[9];
    premiumLink.href = `premium.html?lang=${encodeURIComponent(language)}#early-access`;
    detector.hidden = false;
  }

  function setLanguage(lang) {
    language = labels[lang] ? lang : 'en';
    if (lastAnalysis) render(lastAnalysis, language);
  }

  return { analyze, render, setLanguage, labels, regimes: [...REGIMES] };
});
