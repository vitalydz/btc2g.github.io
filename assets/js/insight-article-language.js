(function (root) {
  'use strict';

  const supported = new Set(['en', 'ru', 'he', 'de', 'fr', 'es', 'zh', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'tr', 'pl', 'nl', 'uk', 'id', 'vi']);
  const requested = new URLSearchParams(root.location.search).get('lang');
  let saved = '';
  try { saved = root.localStorage.getItem('lang') || ''; } catch (_) { /* Storage is optional. */ }
  const browser = (root.navigator.language || 'en').slice(0, 2).toLowerCase();
  const lang = supported.has(requested) ? requested : (supported.has(saved) ? saved : (supported.has(browser) ? browser : 'en'));

  root.document.documentElement.lang = lang;
  root.document.documentElement.dir = (lang === 'he' || lang === 'ar') ? 'rtl' : 'ltr';
  try { root.localStorage.setItem('lang', lang); } catch (_) { /* Storage is optional. */ }

  root.document.querySelectorAll('a[href]').forEach((link) => {
    const raw = link.getAttribute('href');
    if (!raw || /^(?:https?:|mailto:|tel:|#)/i.test(raw)) return;
    const url = new URL(raw, root.location.href);
    url.searchParams.set('lang', lang);
    link.setAttribute('href', url.pathname.replace(/^\//, '') + url.search + url.hash);
  });

  if (lang !== 'ru') return;

  const russian = {
    'Bitcoin vs Gold': 'Биткоин и золото',
    'Bitcoin and gold are often compared because both are scarce assets that people use to think about long-term value storage. Gold has centuries of market history, broad cultural recognition, and deep liquidity. Bitcoin is newer, digital, globally transferable, and defined by a fixed supply schedule.': 'Биткоин и золото часто сравнивают, поскольку оба актива ограничены в предложении и рассматриваются как способы долгосрочного сохранения ценности. У золота — многовековая история, широкое признание и глубокая ликвидность. Биткоин появился позже: он цифровой, доступен по всему миру и имеет заранее заданный график выпуска.',
    'Why investors compare them': 'Почему их сравнивают',
    'Gold is a physical asset with industrial, jewelry, and monetary demand. Bitcoin is a digital asset secured by a decentralized network. The comparison is not about claiming one asset is always better. It is about understanding how scarcity, liquidity, adoption, and volatility can shape long-term market behavior.': 'Золото — физический актив с промышленным, ювелирным и денежным спросом. Биткоин — цифровой актив, защищённый децентрализованной сетью. Смысл сравнения не в том, чтобы объявить один актив всегда лучшим, а в том, чтобы понять влияние дефицита, ликвидности, распространения и волатильности на долгосрочное поведение рынка.',
    'Scarcity and supply': 'Ограниченность и предложение',
    'Gold supply grows gradually through mining. Bitcoin has a transparent supply cap and issuance schedule. This difference is one reason Bitcoin is sometimes described as digital scarcity, while gold is viewed as physical scarcity.': 'Предложение золота постепенно растёт благодаря добыче. У Биткоина есть прозрачный предел предложения и график выпуска. Поэтому Биткоин иногда называют цифровым дефицитным активом, а золото — физическим.',
    'Volatility and risk': 'Волатильность и риск',
    'Bitcoin has historically been more volatile than gold. Gold may move more slowly, while Bitcoin can reprice quickly when market sentiment changes. A BTC vs Gold model should be treated as a comparison tool, not a guarantee of future price movement.': 'Исторически Биткоин был волатильнее золота. Золото может двигаться медленнее, тогда как Биткоин способен быстро переоцениваться при изменении настроений рынка. Модель BTC/Gold следует воспринимать как инструмент сравнения, а не как гарантию будущего движения цены.',
    'How BTC2G helps': 'Как помогает BTC2G',
    'BTC2G visualizes Bitcoin relative to gold so users can compare broad long-term trends. The chart and signal are model-based and designed to support research, not replace personal judgment or professional advice.': 'BTC2G показывает Биткоин относительно золота, чтобы пользователи могли сравнивать долгосрочные тенденции. График и сигнал основаны на модели и предназначены для исследований, а не для замены личной оценки или профессиональной консультации.',
    'Bitcoin Gold Ratio': 'Соотношение Биткоина и золота',
    'The Bitcoin Gold Ratio is a simple way to compare Bitcoin against gold. Instead of looking only at the dollar price of BTC, the ratio asks how Bitcoin is performing relative to a traditional store-of-value asset.': 'Соотношение Биткоина и золота — простой способ сравнить эти активы. Вместо оценки только долларовой цены BTC оно показывает, как Биткоин ведёт себя относительно традиционного средства сохранения ценности.',
    'BTC/Gold ratio = BTC price (USD per BTC) / gold price (USD per troy ounce)': 'Соотношение BTC/Gold = цена BTC (USD за BTC) / цена золота (USD за тройскую унцию)',
    'The result is troy ounces per BTC. BTC2G uses GC=F gold futures as a reference, not a physical-gold exchange quote.': 'Результат выражается в тройских унциях на один BTC. BTC2G использует фьючерс GC=F как ориентир, а не биржевую котировку физического золота.',
    'A simple example': 'Простой пример',
    'With hypothetical prices of $100,000 per BTC and $4,000 per troy ounce, the ratio is 25 troy ounces per BTC. At those same prices, 0.1 BTC represents 2.5 troy ounces, and 1 troy ounce represents 0.04 BTC. These are example prices, not current quotes.': 'При условных ценах $100 000 за BTC и $4 000 за тройскую унцию соотношение составит 25 тройских унций на BTC. При тех же ценах 0,1 BTC соответствует 2,5 тройской унции, а 1 тройская унция — 0,04 BTC. Это пример, а не текущие котировки.',
    'Try the BTC ↔ gold reference calculator': 'Откройте справочный калькулятор BTC ↔ золото',
    ' with the published market date. Estimates exclude fees, spreads and taxes.': ' с опубликованной рыночной датой. Расчёты не учитывают комиссии, спреды и налоги.',
    'Data and units': 'Данные и единицы измерения',
    'BTC2G compares BTC-USD and GC=F daily price observations on dates available for both series. Check the market date shown by the calculator; the chart generation time is a separate timestamp.': 'BTC2G сопоставляет дневные значения BTC-USD и GC=F за даты, доступные в обоих рядах. Проверяйте рыночную дату в калькуляторе: время построения графика указывается отдельно.',
    'A standard GC futures contract covers 100 troy ounces, but its price is quoted per ounce. The ratio uses that per-ounce price, not the value of a whole contract. See the ': 'Стандартный фьючерсный контракт GC охватывает 100 тройских унций, но цена указывается за унцию. В соотношении используется цена одной унции, а не стоимость всего контракта. См. ',
    'CME Group contract specifications (PDF)': 'спецификацию контракта CME Group (PDF)',
    'What the ratio can show': 'Что показывает соотношение',
    'When the ratio rises, Bitcoin is gaining value relative to gold. When the ratio falls, Bitcoin is weakening relative to gold. This can help users study broad market phases, but it should not be treated as a short-term trading signal by itself.': 'Когда соотношение растёт, Биткоин укрепляется относительно золота; когда снижается — ослабевает. Это помогает изучать крупные рыночные фазы, но само по себе не является краткосрочным торговым сигналом.',
    'Why relative valuation matters': 'Зачем нужна относительная оценка',
    'Dollar prices can be affected by inflation, liquidity, interest rates, and market cycles. A ratio can reduce some of that noise by comparing two scarce assets directly. BTC2G uses this idea to make long-term BTC vs Gold trends easier to see.': 'На долларовые цены влияют инфляция, ликвидность, процентные ставки и рыночные циклы. Прямое сравнение двух дефицитных активов помогает уменьшить часть этого шума. BTC2G использует такой подход, чтобы сделать долгосрочные тенденции BTC/Gold нагляднее.',
    'Limits of the ratio': 'Ограничения соотношения',
    'No single ratio explains the full market. Bitcoin and gold have different risk profiles, investor bases, trading behavior, and macro sensitivity. The ratio is best used as one research input among many.': 'Одно соотношение не объясняет весь рынок. У Биткоина и золота разные профили риска, группы инвесторов, торговое поведение и чувствительность к макроэкономике. Это лишь один из исследовательских показателей.',
    'How BTC2G uses it': 'Как BTC2G использует соотношение',
    'BTC2G turns the BTC vs Gold comparison into a visual chart and model-based signal. The goal is to help users quickly understand whether Bitcoin appears strong, weak, or neutral compared with gold according to the model.': 'BTC2G превращает сравнение BTC и золота в наглядный график и модельный сигнал. Цель — помочь быстро понять, выглядит ли Биткоин сильным, слабым или нейтральным относительно золота согласно модели.',
    'Bitcoin vs Gold Forecast 2026': 'Прогноз Биткоина относительно золота на 2026 год',
    'A Bitcoin vs Gold forecast for 2026 should be read as a model-based scenario, not a promise. BTC2G focuses on relative valuation: how Bitcoin appears compared with gold over time.': 'Прогноз Биткоина относительно золота на 2026 год следует рассматривать как модельный сценарий, а не обещание. BTC2G сосредоточен на относительной оценке Биткоина и золота во времени.',
    'The BTC2G model is designed to help users compare long-term trends. It does not guarantee a price, return, or market outcome.': 'Модель BTC2G помогает сравнивать долгосрочные тенденции. Она не гарантирует цену, доходность или результат на рынке.',
    'What may influence the 2026 comparison': 'Что может повлиять на сравнение в 2026 году',
    'Several factors can affect Bitcoin and gold differently, including monetary policy, market liquidity, institutional demand, regulation, mining economics, and investor appetite for risk. Gold may benefit from defensive demand, while Bitcoin may respond more strongly to liquidity and adoption cycles.': 'На Биткоин и золото по-разному влияют денежно-кредитная политика, ликвидность, институциональный спрос, регулирование, экономика добычи и готовность инвесторов к риску. Золото может выигрывать от защитного спроса, а Биткоин — сильнее реагировать на ликвидность и циклы распространения.',
    'Why use a BTC vs Gold model': 'Зачем использовать модель BTC/Gold',
    'Looking at Bitcoin only in dollars can miss relative performance against other scarce assets. A BTC vs Gold model helps frame whether Bitcoin appears strong or weak compared with a long-established store-of-value benchmark.': 'Оценка Биткоина только в долларах не показывает его динамику относительно других дефицитных активов. Модель BTC/Gold помогает понять, выглядит ли Биткоин сильным или слабым по сравнению с давно известным средством сохранения ценности.',
    'Reading the BTC2G chart': 'Как читать график BTC2G',
    'The BTC2G chart combines historical comparison with a forward-looking projection. Users can review the latest signal, confidence value, and chart update date on the homepage. These tools are intended for research and education.': 'График BTC2G сочетает историческое сравнение с перспективным сценарием. На главной странице доступны последний сигнал, оценка модели и дата обновления графика. Эти инструменты предназначены для исследований и обучения.',
    'Risk-aware interpretation': 'Интерпретация с учётом риска',
    'Bitcoin can move sharply in both directions, and gold can also change with macro conditions. A forecast should be combined with risk management, independent research, and awareness that market conditions can change quickly.': 'Биткоин способен резко двигаться в обе стороны, а золото меняется вместе с макроэкономическими условиями. Прогноз следует сочетать с управлением рисками, самостоятельным исследованием и пониманием того, что рынок может быстро измениться.',
    'Back to BTC2G': 'Вернуться к BTC2G',
    'Buy Bitcoin': 'Купить Биткоин',
    'This content is for informational purposes only and is not financial advice.': 'Материал предназначен только для информации и не является финансовой рекомендацией.',
    'Get BTC2G model updates': 'Получать обновления модели BTC2G',
    'A short email when the published model direction changes or the ratio moves significantly. No daily spam.': 'Короткое письмо при изменении направления модели или значительном движении соотношения. Без ежедневного спама.',
    'Subscribe free': 'Подписаться бесплатно',
    'Email address': 'Адрес электронной почты'
  };

  const walker = root.document.createTreeWalker(root.document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const leading = node.nodeValue.match(/^\s*/)[0];
    const trailing = node.nodeValue.match(/\s*$/)[0];
    const value = node.nodeValue.trim();
    let translated = russian[value] || value;
    translated = translated
      .replace('with the published market date. Estimates exclude fees, spreads and taxes.', 'с опубликованной рыночной датой. Расчёты не учитывают комиссии, спреды и налоги.')
      .replace('A standard GC futures contract covers 100 troy ounces, but its price is quoted per ounce. The ratio uses that per-ounce price, not the value of a whole contract. See the', 'Стандартный фьючерсный контракт GC охватывает 100 тройских унций, но цена указывается за унцию. В соотношении используется цена одной унции, а не стоимость всего контракта. См.');
    if (translated !== value) node.nodeValue = leading + translated + trailing;
  });
  root.document.querySelectorAll('[aria-label]').forEach((node) => {
    const value = node.getAttribute('aria-label');
    if (russian[value]) node.setAttribute('aria-label', russian[value]);
  });
  root.document.title = root.document.title
    .replace('Bitcoin vs Gold: Store of Value Comparison', 'Биткоин и золото: сравнение способов сохранения ценности')
    .replace('Bitcoin Gold Ratio Explained', 'Соотношение Биткоина и золота')
    .replace('Bitcoin vs Gold Forecast 2026', 'Прогноз Биткоина относительно золота на 2026 год');
})(window);
