# Growth coordination and verified fixes — 2026-11-05

## Scope and ownership

Read-only coordination preceded changes. BTC2G account work in chat `01a04984-9c06-7ce1-9eeb-9fe8ea9208a8` was completed/idle at commit `6b81953`; relevant source paths were clean and origin/main matched that SHA. Changes were made in separate worktree `C:/Users/vitly/PycharmProjects/MPA_04/btc2g-growth`, branch `codex/growth-attribution`. Existing dirty main changes, staged PDFs and other worktrees were not changed. VitdArt code and artist outreach were not changed. Priorities were added to the existing organizational `Сайты/Общие настройки/README.md`.

## Confirmed BTC2G defects and corrections

- Production GET of `assets/js/traffic.js` confirmed Telegram-only UTM forwarding. Core/token verification supported only Product Hunt, Scenario and Telegram. Added fixed labels Reddit, LinkedIn and directory plus explicit Product Hunt UTM; HTTPS referral host matching for Reddit/LinkedIn. Arbitrary UTM values and full referrers are not stored. Directory uses `utm_source=directory`, without arbitrary directory names.
- Localized Scenario paths previously returned null. All 19 supported prefixes now retain classification; unknown languages and lookalike paths remain rejected.
- Browser-bound signed same-day tokens remain required. Tokens now align with the first known daily visitor source. Referrers/UTM remain user-controlled observations, not independent proof of referral. Network/browser changes and expired tokens lose attribution; old records cannot be accurately backfilled.
- Added aggregate channel table to protected admin statistics using existing columns. Visitor estimates, accounts created today, email confirmations today and all pending accounts are distinct. Existing summary/daily statistics remain present. Counts are not a cohort funnel; no new form-start or failed-registration telemetry, personal identifiers or migration.
- Production GET of `assets/js/article-signup.js` confirmed Formspree on three research articles, while homepage uses configured Audience API. Articles now use the same subscription helper, unchecked consent, privacy link and honest pending-confirmation receipt. Existing account link retained; subscription does not create an account. Premium early-access Formspree and digest sending were not changed. No production subscription form submitted.
- No defect established in basic BTC2G metadata; it was not rewritten.

## VitdArt live evidence

HTTP and www each redirect 308 to `https://vitdart.com/`, then 200. Root has title, H1, description, canonical and substantive HTML without JavaScript. Robots allows public crawling, excludes private paths and links sitemap. Index and three child maps returned 200; URL counts: pages 95, works 760, artists 19 (language variants included, not unique works/person counts).

Sample work `https://vitdart.com/works/08900886-7772-5560-a192-f3df6f92d57d` and artist `https://vitdart.com/artist/64854977-6e1c-4de4-b042-4b8809eac3db`, including English variants, returned 200 with title, description, H1, self canonical and server-rendered authored content. No noindex directive found on these public samples. Sampling is not a complete site audit; no confirmed defect requiring a VitdArt code change found.

Authenticated Google Search Console read on 05.11.2026, property `https://vitdart.com/`: selected range 3 months, web search; displayed chart coverage 31.10–03.11.2026, 3 impressions, 1 click, CTR 33.3%, average position 29.7; UI updated 2.5 hours ago. This small sample does not establish registrations or campaign causality. Indexing report says data is processing: indexed page count unknown, not zero. Sitemap success: 494 discovered URLs, processed 03.11. Discovered is not indexed; live map has changed since processing. No indexing request or new sitemap submission sent. Ambiguous brand-search results do not establish absence of indexing.

## Verification before release

186 Node 24 tests passed (176 existing + 10 added): real in-memory SQLite registration/confirmation stages for new sources with mocked mail, forged token rejection, daily first-source consistency, all 19 Scenario prefixes, executed browser-script tests for token forwarding and article consent/success/failure without retries. 27 Python tests passed; IndexNow output in this suite is mocked, not a live submission. Source diff checks passed. Production checks are read-only; no accounts, subscriptions, visit API events or mail created. Publication outcome recorded separately after deployment. No social posts, paid advertising, outreach or access expansion.
