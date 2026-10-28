# BTC2G SAVE — manual quote comparison

Release: 2026.10.28-save.1. Prepared on 28 October 2026 from main 4743fe7.

The agreed SAVE/EARN/PLAY plan is present in remote commit 949dd186 (docs/BTC2G_SAVE_EARN_PLAY_PLAN.md). This release implements a small, usable part of SAVE: compare final delivered BTC for the same total fiat payment. It does not claim the complete purchase-comparison roadmap is finished.

The new page uses a responsive dark layout, white text and yellow actions, English/Russian/Hebrew, RTL and accessible form labels. The initial payment is 50; 50/100/200 presets require fresh quotes. Currency or amount changes clear quoted BTC and results. Editing a quote clears the previous comparison and its comparability confirmation. Calculations do not use cookies, storage, analytics or a server. Existing provider destinations are preserved. Homepage and buy-page links expose the tool; unsupported SAVE locales fall back explicitly to English.

Quotes are entered manually. No live provider API, fee schedule, spread, partnership, commission or guaranteed saving is fabricated. The comparison shows effective fiat cost per BTC and the difference in integer satoshis. Users confirm equivalent conditions. Final delivered BTC already includes deductions, so no network fee is subtracted again. No purchase is initiated by calculation.

Sources reviewed:
- https://support.topperpay.com/hc/en-us/articles/8926549324444-Topper-costs-and-fees — fee and network charge distinctions; no static schedule copied into calculation.
- https://support.moonpay.com/en/articles/380500-how-do-i-buy-cryptocurrency-with-moonpay — review the final rate and payment method before confirming.

Validation: three new arithmetic/invalid-input tests plus seven existing Node checks passed. Chrome checks covered equal-budget calculation, currency/amount reset, confirmation, input validation, English/Russian/Hebrew, RTL, 360/390/768/1240 CSS-pixel widths and mobile navigation from the homepage. No horizontal overflow observed. This is viewport emulation on Win11, not a physical Android or remote Win10 test.

Also completed the previously prepared ratio article: explicit oz t per BTC units, hypothetical example, calculator link and CME contract-unit source. Its mobile layout was inspected after the local preview was restored. Removed unconfigured analytics from the two changed legacy pages (buy and ratio article). Forecast PNG/JSON, Premium, contest dates and backend state are outside this change.

Remaining: real quote API access and permitted terms; comparable multi-purchase scenarios; full SAVE localization; verified partner rewards; authenticated private statistics; production contest identity/storage. No public contest or reward ledger enabled.

## Homepage entry — 2026.10.28-save.2

After Daily Builder review, added one compact SAVE card between the chart and reference calculator. It states that comparison is manual and uses equal total payment; it promises no saving before quotes are entered. All 19 homepage languages have title, explanation and action text. A visible EN/RU/HE tag identifies the languages currently available in SAVE. The link preserves those three languages and otherwise opens English. The card uses a restrained teal border and yellow/white text, stacks on mobile, and preserves the chart and calculator.

Validation: four existing Node suites passed, all 19 language selections rendered the expected title and target in Chrome, 390px Russian and 360px Hebrew RTL layouts inspected, no horizontal overflow. Release/cache identifiers updated together. Prior release ff329a6 deployed successfully; live calculation and secondary-copy preservation were verified.
