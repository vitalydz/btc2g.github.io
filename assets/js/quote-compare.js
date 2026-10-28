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
  function comparePurchases(firstQuote, secondQuote, combinedQuote) {
    const quotes = [firstQuote, secondQuote, combinedQuote];
    if (quotes.some(q => !q || !['USD', 'EUR', 'GBP', 'ILS'].includes(q.currency))) return { status: 'incomplete' };
    if (quotes.some(q => q.currency !== combinedQuote.currency)) return { status: 'currency-mismatch' };
    const amounts = quotes.map(q => units(q.amount, 2));
    const sats = quotes.map(q => units(q.btc, 8));
    if ([...amounts, ...sats].some(value => value === null)) return { status: 'incomplete' };
    const separateAmount = amounts[0] + amounts[1];
    const separateSats = sats[0] + sats[1];
    if (!Number.isSafeInteger(separateAmount) || !Number.isSafeInteger(separateSats)) return { status: 'incomplete' };
    if (separateAmount !== amounts[2]) return { status: 'amount-mismatch' };
    return {
      status: 'ready', currency: combinedQuote.currency,
      separateSats, combinedSats: sats[2],
      differenceSats: sats[2] - separateSats,
      separateRate: separateAmount * 1000000 / separateSats,
      combinedRate: amounts[2] * 1000000 / sats[2]
    };
  }
  const api = Object.freeze({ units, compare, comparePurchases });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BTC2GQuotes = api;
})(typeof window === 'undefined' ? globalThis : window);
