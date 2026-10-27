# BTC2G audience readiness — draft for owner review

Base `origin/main`: `9babf125f42ac02c12efae9fc103fb5b9bfb16f8`, checked 27 October 2026.
Prepared in branch: `codex/audience-readiness`. The owner authorized the website
release on 27 October 2026 after review. External posts and profile edits remain
drafts and have not been applied.
The earlier Topper review is preserved separately in local branch
`codex/topper-revenue-review`, commit `6ddd8d7`; it is not included in this branch.

## First-visit audit and smallest changes

The old heading emphasized a forecast but did not immediately explain the ratio's
units or its use. The subtitle used today's date even when the latest common
market close was earlier. The displayed confidence percentage is a heuristic
based on distance from a rolling mean, not calibrated probability. The PNG plots
separate USD prices, while the cards show the BTC/gold ratio; dashed polynomial
curves can be mistaken for dependable future prices.

The draft adds a short purpose statement, daily/not-live subtitle, jump to the
calculator, distinct market/chart-generation date labels, the label “Model score”,
and a plain chart-method explanation. No model or generated data is changed.
The chart's embedded 2030 title and historical copyright are existing PNG content;
the new copy does not present its projection horizon as a guaranteed outcome.

No existing converter was found. The new client-side calculator reuses the
already-loaded `assets/btc_signal.json` ratio and `assets/btc_gold_forecast_meta.json`
source metadata. No new market API, payment backend or analytics service is used.

- BTC → gold: multiply by the ratio. Gold → BTC: divide by the ratio.
- Gold is in **troy ounces**, using Yahoo Finance `GC=F` gold futures as a proxy,
  not physical bullion prices. Bitcoin is `BTC-USD`; both inputs are USD prices.
  [CME's GC specifications](https://www.cmegroup.com/trading/metals/files/fact-card-gold-futures-options.pdf)
  quote the gold price per troy ounce, even though one contract contains 100 ounces.
- Approximate output uses the published three-decimal ratio and excludes trading
  fees, spreads and taxes. It is a reference comparison, not an executable quote.
- Missing/bad ratio or source metadata, invalid/future market dates, or market
  dates more than three UTC calendar days old disable conversion and clear the
  previous result. The age rule matches the existing site freshness threshold.
- This daily-close policy is not a claim of intraday freshness. Weekends and
  holidays can delay the last common close. Old context can remain in existing
  model cards with their warning, while the calculator refuses a numerical result.
- Zero, positive decimals, decimal commas and both conversion directions are
  supported. Ambiguous grouping, negative values, non-numbers and overflow are
  rejected. Tiny nonzero outputs display a less-than threshold instead of zero.
- Calculator inputs remain in the page; this code sends or stores none of them.
  Existing page services, including the placeholder analytics script, are unchanged.

New copy is supplied for all 19 existing languages. A short creator section uses
the three exact owner-supplied links. No audience totals or subscriber claims are
displayed. Public Stihi.ru and Proza.ru pages were readable. The supplied LinkedIn
URL opened “QA in Israel” in the existing signed-in browser; its About dialog
showed a private, listed group with post approval. No group/profile was modified.

## English LinkedIn group post — not posted

Use only after the owner publishes and verifies this version. The draft is a
QA discussion, not a purchase offer. Destination: [QA in Israel](https://www.linkedin.com/groups/1408997/).

> What would you test first in a small financial-data website?
>
> I’m developing BTC2G, a personal project comparing Bitcoin with gold. The
> engineering challenge is making units, dates and uncertainty clear—not just
> drawing a chart.
>
> The latest version adds a BTC ↔ gold reference calculator. Tests cover both
> directions, invalid inputs, missing data, UTC freshness boundaries and recovery
> after failed refreshes. I’m also checking multilingual and mobile layouts.
> The model score is not a probability of profit, and projections are scenarios.
>
> I’d welcome QA feedback on confusing wording, edge cases and accessibility:
> https://btc2g.com/?utm_source=linkedin&utm_medium=community&utm_campaign=btc2g_intro
>
> Which failure case would you add to the test suite?

[LinkedIn's current group-post guidance](https://www.linkedin.com/help/linkedin/answer/a539929/)
requires group-appropriate original content and allows admin review. Recheck the
group rules at posting time. Do not publish an advertisement, mass-tag members or
send unsolicited messages. The owner's group membership is not a BTC2G subscription.

## Russian literary-profile descriptions — not applied

These are neutral biographical drafts, not calls to buy cryptocurrency. Keep the
existing literary biography; any addition is for owner/moderator review only.

Stihi.ru:

> Кроме стихов, занимаюсь инженерными проектами. Один из них — BTC2G:
> справочное сравнение Bitcoin и золота, график и калькулятор. https://btc2g.com/

Proza.ru:

> Пишу прозу и занимаюсь инженерными проектами. BTC2G — мой небольшой проект
> для наглядного сравнения Bitcoin и золота. https://btc2g.com/

Important limitation: [Stihi.ru rules](https://o.stihi.ru/pravila) and
[Proza.ru rules](https://o.proza.ru/pravila), sections 3 and 4.2, extend to profile
resumes and prohibit advertising publications/messages. A neutral personal link
is not guaranteed to be accepted. Do not place these as poems, stories, reviews,
comments or mass messages; seek moderator clarification if an unrelated project
link is considered advertising. No contact or external edit has been made.

## Privacy-conscious source measurement proposal

Repository audit found the placeholder `G-XXXXXXXXXX` and `gtag` event calls, but
no verified real GA property or usable reporting. This is **not working analytics**.
Group activity statistics and literary reader counters do not measure BTC2G visits.

Use one non-personal tagged URL per source after owner approval:

- LinkedIn: `https://btc2g.com/?utm_source=linkedin&utm_medium=community&utm_campaign=btc2g_intro`
- Stihi.ru: `https://btc2g.com/?utm_source=stihi&utm_medium=author_profile&utm_campaign=btc2g_intro`
- Proza.ru: `https://btc2g.com/?utm_source=proza&utm_medium=author_profile&utm_campaign=btc2g_intro`

No names, email addresses, member IDs or reader IDs belong in these parameters.
Tags distinguish links; **they do not create a counter or reporting dashboard**.
The initial browser check only verifies that a tagged landing URL loads and its
parameters survive navigation to the landing page. No visit attribution was
verified in a collector, and no analytics account or paid plan was created.

For actual measurement, the owner can later approve a cookieless aggregate
analytics setup with an allowlist of these three source values, no fingerprinting,
no calculator-input collection and no persistent visitor IDs. Verify a test visit
from each source in its dashboard, update the privacy notice, then begin counting.
Until then record post/profile publication dates manually, and do not report
clicks, conversion rates or commission attribution from URL tags alone.

## Review and verification

- Automated: calculator arithmetic, freshness, malformed/future/stale data, missing
  source metadata, rejected input and overflow, stale-result removal and recovery;
  translation coverage for 19 languages; existing ratio/freshness/refresh tests.
- Browser: actual published ratio 19.811 produced 39.622 oz t for 2 BTC and 1 BTC
  for 19.811 oz t; negative input produced an error without a numeric result.
- Browser language selection passed for all 19 languages. At 360 and 390 CSS
  pixels, English/Russian/Hebrew/Arabic layouts had no horizontal overflow;
  Hebrew/Arabic kept RTL direction. These are desktop viewport checks.
- Three tagged local landing URLs loaded and preserved their source values.
  This checks URL handling, not analytics collection or campaign attribution.
- Four Node test suites and seven Python tests passed; whitespace checks passed.
- Real Android, assistive-device testing and native-speaker review of all 19
  languages remain outside the verified checks.
- Generated PNG/JSON and provider URLs must remain byte-for-byte unchanged.

Next owner action: review the diff and drafts, choose any biography wording, and
publish only after checking the exact version. No Topper application is needed
to review or use the reference calculator.

Daily Builder reviewed the described behavior, not the local test execution.
Its two checks were confirmed against `scripts/update_btc_gold_chart.py`: the
ratio divides Bitcoin USD price by the per-ounce gold price, and `last_updated`
comes from the last common market row. A regression test confirms that a newly
generated metadata timestamp cannot make an old market date fresh.

## Follow-up code review and real-browser failure checks

Daily Builder subsequently reviewed the actual `dataStatus` and `convert`
functions and found no reproducible defect in those functions; it did not
independently execute the browser adapter. Ratio cards now explicitly use `oz t`
as well as the calculator, and the ratio-format regression was updated and passed.

An isolated local HTTP test server served synthetic JSON responses in memory;
the repository forecast files were not changed. Desktop Chrome checks passed for
stale data, invalid market date, an unrecognized gold ticker, missing signal HTTP
503 and missing metadata HTTP 503. Conversion controls remained disabled and no
numeric result was shown. A same-page manual refresh from valid data to HTTP 503
removed the previous numeric result; restoring valid data and refreshing again
re-enabled conversion. These were local failure simulations, not production outages.
