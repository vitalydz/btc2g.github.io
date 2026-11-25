(function (root, factory) {
  'use strict';
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BTC2GMultiAsset = api;
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';
  const DATA_PATH = 'assets/multi_asset_strength.json';
  const PAIRS = ['btc_gold', 'btc_silver', 'btc_spx', 'gold_spx'];
  const BTC_PAIRS = ['btc_gold', 'btc_silver', 'btc_spx'];
  const text = {
    en: ['Bitcoin across major reference assets','Compare published relative-price ratios on the same market date.','Strongest Bitcoin 30-day relative move','Current ratio','7 days','30 days','Source','Relative price ratios, not exchange rates or investment recommendations.'],
    ru: ['Биткоин относительно ключевых активов','Сравнение опубликованных относительных цен на одну и ту же рыночную дату.','Сильнейшее относительное движение Биткоина за 30 дней','Текущее соотношение','7 дней','30 дней','Источник','Относительные цены, а не обменные курсы или инвестиционные рекомендации.'],
    he: ['ביטקוין מול נכסי ייחוס מרכזיים','השוואת יחסי מחיר שפורסמו לאותו תאריך שוק.','המהלך היחסי החזק ביותר של ביטקוין ב-30 יום','יחס נוכחי','7 ימים','30 ימים','מקור','יחסי מחירים יחסיים, לא שערי חליפין או המלצות השקעה.'],
    de: ['Bitcoin gegenüber wichtigen Referenzwerten','Veröffentlichte relative Preisverhältnisse zum selben Marktdatum.','Stärkste relative 30-Tage-Bewegung von Bitcoin','Aktuelles Verhältnis','7 Tage','30 Tage','Quelle','Relative Preisverhältnisse, keine Wechselkurse oder Anlageempfehlungen.'],
    fr: ['Bitcoin face aux principaux actifs de référence','Comparaison des ratios de prix publiés à la même date de marché.','Plus fort mouvement relatif du Bitcoin sur 30 jours','Ratio actuel','7 jours','30 jours','Source','Ratios de prix relatifs, pas des taux de change ni des recommandations.'],
    es: ['Bitcoin frente a activos de referencia','Ratios de precios publicados en la misma fecha de mercado.','Movimiento relativo más fuerte de Bitcoin en 30 días','Ratio actual','7 días','30 días','Fuente','Ratios de precios relativos, no tipos de cambio ni recomendaciones.'],
    zh: ['比特币相对主要参考资产','比较同一市场日期发布的相对价格比率。','比特币 30 天最强相对走势','当前比率','7 天','30 天','来源','相对价格比率，并非汇率或投资建议。'],
    it: ['Bitcoin rispetto ai principali riferimenti','Rapporti di prezzo pubblicati alla stessa data di mercato.','Movimento relativo Bitcoin più forte a 30 giorni','Rapporto attuale','7 giorni','30 giorni','Fonte','Rapporti di prezzo relativi, non tassi di cambio o raccomandazioni.'],
    pt: ['Bitcoin face aos principais ativos','Rácios de preços publicados na mesma data de mercado.','Movimento relativo mais forte do Bitcoin em 30 dias','Rácio atual','7 dias','30 dias','Fonte','Rácios de preços relativos, não taxas de câmbio ou recomendações.'],
    ar: ['بيتكوين مقابل الأصول المرجعية الرئيسية','مقارنة نسب الأسعار المنشورة في تاريخ السوق نفسه.','أقوى حركة نسبية لبيتكوين خلال 30 يومًا','النسبة الحالية','7 أيام','30 يومًا','المصدر','نسب أسعار نسبية وليست أسعار صرف أو توصيات استثمارية.'],
    hi: ['प्रमुख संदर्भ परिसंपत्तियों के मुकाबले Bitcoin','एक ही बाज़ार तारीख के प्रकाशित सापेक्ष मूल्य अनुपात।','Bitcoin की सबसे मजबूत 30-दिन की सापेक्ष चाल','वर्तमान अनुपात','7 दिन','30 दिन','स्रोत','सापेक्ष मूल्य अनुपात, विनिमय दर या निवेश सलाह नहीं।'],
    ja: ['主要な参照資産に対するBitcoin','同じ市場日の公表相対価格比率を比較します。','Bitcoinの最も強い30日相対変動','現在の比率','7日','30日','出典','相対価格比率であり、為替レートや投資推奨ではありません。'],
    ko: ['주요 기준 자산 대비 Bitcoin','동일한 시장 날짜의 공개 상대 가격 비율을 비교합니다.','Bitcoin의 가장 강한 30일 상대 움직임','현재 비율','7일','30일','출처','상대 가격 비율이며 환율이나 투자 권유가 아닙니다.'],
    tr: ['Başlıca referans varlıklara karşı Bitcoin','Aynı piyasa tarihindeki yayımlanmış göreli fiyat oranları.','Bitcoin’in en güçlü 30 günlük göreli hareketi','Güncel oran','7 gün','30 gün','Kaynak','Göreli fiyat oranlarıdır; döviz kuru veya yatırım tavsiyesi değildir.'],
    pl: ['Bitcoin względem głównych aktywów','Opublikowane względne relacje cen z tej samej daty rynkowej.','Najsilniejszy 30-dniowy ruch względny Bitcoin','Bieżąca relacja','7 dni','30 dni','Źródło','Względne relacje cen, nie kursy wymiany ani rekomendacje.'],
    nl: ['Bitcoin tegenover belangrijke referentieactiva','Gepubliceerde relatieve prijsverhoudingen op dezelfde marktdatum.','Sterkste relatieve Bitcoin-beweging over 30 dagen','Huidige verhouding','7 dagen','30 dagen','Bron','Relatieve prijsverhoudingen, geen wisselkoersen of beleggingsadvies.'],
    uk: ['Біткоїн відносно ключових активів','Опубліковані відносні ціни на одну ринкову дату.','Найсильніший відносний рух Біткоїна за 30 днів','Поточне співвідношення','7 днів','30 днів','Джерело','Відносні ціни, а не обмінні курси чи інвестиційні рекомендації.'],
    id: ['Bitcoin terhadap aset acuan utama','Rasio harga relatif terpublikasi pada tanggal pasar yang sama.','Pergerakan relatif Bitcoin terkuat selama 30 hari','Rasio saat ini','7 hari','30 hari','Sumber','Rasio harga relatif, bukan nilai tukar atau rekomendasi investasi.'],
    vi: ['Bitcoin so với các tài sản tham chiếu','So sánh tỷ lệ giá tương đối đã công bố cùng ngày thị trường.','Biến động tương đối mạnh nhất của Bitcoin trong 30 ngày','Tỷ lệ hiện tại','7 ngày','30 ngày','Nguồn','Tỷ lệ giá tương đối, không phải tỷ giá hay khuyến nghị đầu tư.']
  };
  let currentData = null;
  let language = 'en';

  function validData(data) {
    if (data?.source !== 'yfinance' || data?.tickers?.Bitcoin !== 'BTC-USD' || data?.tickers?.Gold !== 'GC=F' || data?.tickers?.Silver !== 'SI=F' || data?.tickers?.SPX !== '^GSPC') return false;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(data.updated_for_market_date || '') || !Array.isArray(data.series) || data.series.length < 31) return false;
    if (!data.current?.pairs || !BTC_PAIRS.includes(data.current.strongest_btc_30d)) return false;
    if (!PAIRS.every(key => {
      const pair = data.current.pairs[key];
      return Number.isFinite(pair?.value) && pair.value > 0 && ['change_7d_pct', 'change_30d_pct'].every(field => pair[field] === null || Number.isFinite(pair[field]));
    })) return false;
    return data.series.every((row, index) => /^\d{4}-\d{2}-\d{2}$/.test(row.date) && (!index || row.date > data.series[index - 1].date) && PAIRS.every(key => Number.isFinite(row[key]) && row[key] > 0)) && data.series.at(-1).date === data.updated_for_market_date;
  }

  function formatValue(value) {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: value >= 100 ? 2 : value >= 1 ? 3 : 5 }).format(value);
  }

  function formatChange(value) {
    if (!Number.isFinite(value)) return '—';
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  }

  function changeClass(value) {
    return value > 0 ? 'strength-positive' : value < 0 ? 'strength-negative' : 'strength-neutral';
  }

  function render(data = currentData, lang = language) {
    if (!validData(data) || typeof document === 'undefined') return false;
    currentData = data; language = text[lang] ? lang : 'en';
    const words = text[language];
    const section = document.getElementById('relative-strength');
    document.getElementById('relative-strength-title').textContent = words[0];
    document.getElementById('relative-strength-intro').textContent = words[1];
    const leaderKey = data.current.strongest_btc_30d;
    const leader = data.current.pairs[leaderKey];
    document.getElementById('relative-strength-leader').textContent = `${words[2]}: ${data.pairs[leaderKey].label} (${formatChange(leader.change_30d_pct)})`;
    const cards = document.getElementById('relative-strength-cards');
    cards.replaceChildren(...PAIRS.map(key => {
      const pair = data.current.pairs[key];
      const card = document.createElement('article'); card.className = `strength-card${key === leaderKey ? ' is-leader' : ''}`;
      const title = document.createElement('h3'); title.textContent = data.pairs[key].label;
      const value = document.createElement('strong'); value.className = 'strength-value'; value.textContent = formatValue(pair.value); value.setAttribute('aria-label', `${words[3]} ${formatValue(pair.value)}`);
      const changes = document.createElement('div'); changes.className = 'strength-changes';
      [[words[4], pair.change_7d_pct], [words[5], pair.change_30d_pct]].forEach(([label, amount]) => {
        const block = document.createElement('div'); const caption = document.createElement('span'); const result = document.createElement('strong');
        caption.textContent = label; result.textContent = formatChange(amount); result.className = changeClass(amount); block.append(caption, result); changes.append(block);
      });
      card.append(title, value, changes); return card;
    }));
    document.getElementById('relative-strength-source').textContent = `${words[6]}: yfinance · BTC-USD / GC=F / SI=F / ^GSPC · ${data.updated_for_market_date} (UTC)`;
    document.getElementById('relative-strength-disclaimer').textContent = words[7];
    section.hidden = false;
    return true;
  }

  function setLanguage(lang) { language = text[lang] ? lang : 'en'; if (currentData) render(currentData, language); }

  async function load(version = Date.now()) {
    try {
      const response = await fetch(`${DATA_PATH}?v=${encodeURIComponent(version)}`, { cache: 'no-store' });
      if (!response.ok) throw new Error('Relative-strength data unavailable.');
      const data = await response.json();
      if (!render(data, document.documentElement.lang || language)) throw new Error('Relative-strength data invalid.');
    } catch {
      const section = document.getElementById('relative-strength');
      if (section) section.hidden = true;
    }
  }

  if (root.document) load();
  return { validData, render, setLanguage, load, translations: text, pairs: [...PAIRS] };
});
