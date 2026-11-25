(function (root) {
  'use strict';
  // Parse quoted amounts as integer minor units; reject ambiguous separators.
  function units(value, places) {
    const text = String(value ?? '').trim();
    if (!new RegExp('^\\d{1,12}(?:[.,]\\d{1,' + places + '})?$').test(text)) return null;
    const parts = text.replace(',', '.').split('.');
    const result = Number(parts[0]) * 10 ** places + Number((parts[1] || '').padEnd(places, '0'));
    return Number.isSafeInteger(result) && result > 0 ? result : null;
  }
  function compare(amount, firstBTC, secondBTC) {
    const cents = units(amount, 2);
    const first = units(firstBTC, 8);
    const second = units(secondBTC, 8);
    if (cents === null || first === null || second === null) return { status: 'incomplete' };
    const difference = Math.abs(first - second);
    return {
      status: 'ready',
      firstRate: cents * 1000000 / first,
      secondRate: cents * 1000000 / second,
      differenceSats: difference,
      moreBTCPercent: difference / Math.min(first, second) * 100,
      winner: first === second ? null : first > second ? 'MoonPay' : 'Topper'
    };
  }
  const api = Object.freeze({ units, compare });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BTC2GQuotes = api;
})(typeof window === 'undefined' ? globalThis : window);
