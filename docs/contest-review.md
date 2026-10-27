# BTC2G community ideas contest — LOCAL DEMONSTRATION / owner review

Base: `60208380675c6c5164f506c76d33041986c66172` (fetched origin/main). Branch `codex/contest-review`, isolated worktree `C:\Users\vitly\PycharmProjects\MPA_04\tmp\btc2g-contest-review`. The unpublished `codex/services-launch` work remains untouched in its original checkout. Owner subsequently authorized version update and push on 27 October 2026. Release 2026.10.27-contest-preview.3 publishes the static preparation page only; public submissions/voting remain disabled. No payment, provider contact or external profile change is authorized. The loopback demonstration backend is not deployed.

## Implemented

A navy/charcoal responsive contest page, restrained blue/teal accents, structured cards, submission form, detail dialog, private moderation queue, explicit phases and Asia/Jerusalem deadlines, accurate server-side totals, vote confirmation/change, runoff and completed results at `contest.html#results`. Keyboard focus and reduced-motion support are included. All UI keys have 19 language translations; user-submitted text stays in its original language. The homepage change is one localized contest entry link, not a redesign of unrelated sections. Provider URLs, calculator, Premium and forecast data are unchanged. No author biography/outbound audience links.

GitHub Pages has no server-side writes. The existing Telegram/VIP code is a payment/content service, not an established public web identity service for voting. Therefore this is deliberately **loopback-only**, with private, randomly issued local demo-account codes. No Telegram login, public registration or production-authentication claim. Never expose this development HTTP server publicly. The page without its API remains **Contest in preparation**, with public entry and voting disabled. The server cannot be launched with a database inside the public checkout.

SQLite holds ideas, sessions, votes, idempotency records, rate limits, moderation audit and frozen results. `BEGIN IMMEDIATE` and database uniqueness `(account, ballot)` serialize concurrent votes. A fresh vote replaces that account's previous choice; a delayed retry with the original idempotency key does not restore the older choice. Identical submissions from the same account return the existing entry. Different authors' similar ideas require human review, not automatic merging. Three submissions per demo account maximum. Basic persistent limits: 6 login attempts/IP/minute, 5 submission requests/account/minute, 30 other authenticated requests/account/minute. These are basic controls, not anti-Sybil protection.

Sessions use random opaque tokens, server-side account binding, one-hour expiry, HttpOnly/SameSite=Strict cookies, exact loopback Origin/Host checks and CSRF tokens. Usernames/client user IDs do not establish identity. Private codes/cookies/submission bodies are not logged. Public data excludes account IDs and pending entries; the author sees their own status/rejection reason; the owner sees the queue. Submitted content renders via textContent, never HTML. No paid promotion, visibility controls, Stars or payment routes exist.

The first ballot opens after submissions close. Moderation must finish before voting; approving/removing entries during voting is blocked. Highest positive valid vote count wins. Zero votes means no winner. A positive tie starts a separate runoff among tied ideas, with one active vote/account in that ballot. A repeated tie pauses for a new owner-announced deadline; it never picks randomly. Closed totals remain in the results table across restarts. Self-voting is allowed and disclosed. Accounts are not proof of unique humans.

## Local preview / running

- Submission/moderation demo: http://127.0.0.1:8771/contest.html
- Voting demo with explicitly labelled DEMO entries: http://127.0.0.1:8772/contest.html#ideas
- Completed round uses the same persistent page at `#results`; automatic closure is tested with controlled server times, not an invented live winner.

The two previews use separate disposable SQLite databases outside Git. The voting fixture starts with zero votes, not fabricated popularity. Any votes cast during QA are real database writes by local test accounts, not community participation. Authentication codes are printed once by the local launcher; keep them private. Restart without `--init-demo` to retain accounts, sessions and votes.

```
python contest/server.py --db ..\btc2g-contest-demo.sqlite --init-demo --port 8771
python contest/server.py --db ..\btc2g-contest-voting-demo.sqlite --init-demo --voting-demo --port 8772
# Subsequent restart: omit --init-demo and --voting-demo.
```

Do not reuse fixture accounts, codes or databases for a public contest. Local URLs work on this PC, not an Android phone's localhost.

## Rules — draft, not legally approved or open

1. **Organizer/eligibility:** owner must supply legal organizer name, contact, residence/age eligibility and excluded regions before publication. Proposed minimum 18; organizer/operators and their household should be excluded from prize eligibility. These are proposed terms, not verified legal requirements. Local demo accounts have no prize eligibility.
2. **Schedule:** the owner supplied planned dates on 27 October 2026. Asia/Jerusalem: opening 1 November 2026 00:00; submissions through 31 December 2026 23:59:59; voting through 1 January 2027 23:59:59; runoff through 1 March 2027 23:59:59. The corresponding exclusive closing instants are 31 December 2026 22:00 UTC, 1 January 2027 22:00 UTC and 1 March 2027 22:00 UTC. Opening is 31 October 2026 22:00 UTC. These are display-only dates; they do not enable public submissions or voting. Local demo rounds retain their own clearly labelled test schedule. Real writes remain blocked pending production infrastructure, rules and prize approval. No retroactive change to an active ballot is made by this UI update.
3. **Submission:** title, concrete problem, proposed solution and intended users; useful to BTC2G, original or authorized, no personal data, infringement, spam, investment promises or abuse. Maximum three entries/account. A private review queue precedes public display. Rejected authors receive a reason. Similar proposals require transparent review before voting, not hidden preferential visibility.
4. **Voting:** free, one active vote/account/ballot, changeable until closing; self-voting allowed. No purchases, deposits, Stars, paid advantages, referral obligations or paid idea promotion. Highest valid positive total wins. Ties lead only to another vote among tied entries. No votes means no winner. Repeated ties require newly announced runoff dates.
5. **Abuse:** reject spam/duplicates before publication with a recorded reason. Rate limits are not a one-human guarantee. Suspected coordinated voting needs a documented owner investigation, evidence, notice/appeal and a published reasoned resolution. This demo does NOT silently delete or recalculate allegedly invalid votes. Production launch remains blocked until a freeze/investigation/appeal and auditable invalidation process is approved and implemented. Do not manually rewrite the votes database to pick a winner.
6. **Prize:** proposed 12 months VIP, no cash alternative or payment requirement. Included services, delivery channel, update frequency, support, exact start/expiry, claim window and remedy for unavailable delivery MUST be reviewed before launch. Proposed activation: within 7 days after private acceptance, 12 calendar months from activation, no renewal/card requirement. These proposed terms are NOT yet an offer. Do not substitute 12 monthly purchases or charge Stars.
7. **Privacy:** collect only provider-stable account reference, submission text, votes, timestamps and necessary moderation/security records; no wallet, payment card or public account ID. Proposed retention: sessions 1 hour; demo rate-limit events up to 1 hour; contest/audit records 90 days after result/appeals, then anonymize/delete identity links, with separately agreed handling of published entries. Owner must approve real retention, deletion requests/contact, backup deletion and lawful handling before production. Current demo data is disposable and must not contain actual personal information. Add a tested operator deletion/backup procedure before public use; it is not implemented as a public endpoint here.
8. **Rights:** entrant keeps ownership. Proposed limited, nonexclusive permission to display submitted text for the contest and discuss/credit the idea; implementation may require separately agreed rights for contributed code/artwork. General ideas may inform BTC2G development without a promise to implement or employ the author. No transfer of all intellectual property is requested.

## VIP audit and launch blocker

The existing bot defines **30 days** of access with a **7-day forecast horizon**, USD4.95 / 200 Stars as separate existing paid options. Its README reports synthetic-payment tests and sales disabled by default. It can validate supplied weekly content, check an entitlement, and deliver `/forecast`; it does not establish 12 months of reliable future service. The five-year public polynomial chart is not a weekly VIP product. No live forecast delivery, year-long fulfillment, support SLA or prize grant workflow has been verified. No payments or VIP grants were activated in this task. Consequently the public state remains preparation.

Before launch the owner must define the included current services and verify end-to-end free prize activation, expiry, weekly content maintenance and support without payments. Either implement/review a free twelve-calendar-month prize entitlement separately, or revise the prize with separate approval. Do not quietly reuse a payment grant as a contest prize.

## Production requirements / costs

Reuse an existing persistent host only after verifying ownership, available resources and deployment permission. Minimum: HTTPS application runtime, persistent SQLite volume (single writer host) or transactional database for multiple instances, restricted backups and recovery tests, secret storage, scheduled monitoring, privacy/security/abuse procedures and a real moderator. GitHub Pages alone is insufficient. Free ephemeral serverless disks are not suitable storage.

A future Telegram Mini App is one possible identity route, not implemented here. Validate signed initData server-side per [official Telegram guidance](https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app), check auth_date age and replay/session handling, and trust only the verified user. Keep bot token server-side. Never accept initDataUnsafe, username or submitted ID as proof. An existing verified identity provider is also possible after owner choice; avoid a new registration project.

Cost estimates, not vendor quotes: local demo $0 incremental. Existing capable host may have no additional hosting fee, subject to its contract/quota. A small managed runtime/persistent volume/backups might need roughly $5–20/month; availability, taxes, provider limits and moderation/support time are unverified. No service was purchased. VIP fulfillment has ongoing owner labor/content/support cost even if the prize is free. No guaranteed audience growth or income forecast.

## Promotion drafts — not posted

Only use after the owner approves launch/prize/rules and the service is publicly reachable. Before that, invite feedback on a preparation preview, not entries for a prize.

**English LinkedIn QA group draft**

> What would make BTC2G more useful to you? I’m preparing a free community ideas contest around its Bitcoin-versus-gold tools. I’d value QA feedback on the submission flow, date clarity, accessibility and voting safeguards. Please inspect the site and suggest one specific problem, your proposed solution and who would benefit.
>
> The contest is currently in preparation: public submissions and voting are not open, and the proposed VIP prize still needs final terms. No purchase or invitation of friends is required. This is not an investment or income promise.
>
> https://btc2g.com/contest.html?utm_source=linkedin&utm_medium=group&utm_campaign=ideas_preparation#ideas

**Russian draft for explicitly permitted placements**

> Какие инструменты сделали бы BTC2G полезнее? Готовлю бесплатный конкурс идей: конкретная проблема, решение и кому оно поможет. Пока конкурс не открыт, правила и предполагаемый VIP-приз согласуются. Буду благодарен за замечания по понятности сайта и работе на телефоне. Покупки, платежи и приглашение друзей не нужны; доходность не обещается.
>
> https://btc2g.com/contest.html?utm_source=other&utm_medium=permitted_community&utm_campaign=ideas_preparation#ideas

These future public links are not evidence of publication or analytics. This task adds no collector. UTM labels do not count visits. [LinkedIn](https://www.linkedin.com/help/linkedin/answer/a539929/) requires group-specific content and compliance with group rules/admin review. [Stihi](https://o.stihi.ru/pravila) and [Proza](https://o.proza.ru/pravila) prohibit advertising; do not place the draft in works, reviews or profile resumes without explicit platform permission. No outgoing links to the owner's audiences were added to BTC2G.

A navy/teal product share-preview asset is included; rendering in actual social crawlers after owner-approved deployment is not verified.

## Tests and known limits

Executed: 10 new Python tests, plus 7 existing Python tests (17 total); existing calculator/ratio/freshness/refresh Node suites plus 3 contest UI checks (7 top-level Node checks). Server tests cover unauthorized identity/writes/moderation, CSRF/Origin/cookie boundaries, private paths, uniqueness and concurrent voting, vote changes, delayed retries, pending privacy, deadline boundaries, tied runoff and repeated tie, persistent counts/results after store restart, duplicate submission and persistent rate limits. Frontend renders submitted text through safe DOM APIs. Browser QA used an HTML payload and confirmed literal text with no injected image.

Chrome: demo-code login, submission confirmation, private queue, approval, detail dialog/Escape; all 19 locales at 360px without empty text or horizontal page overflow. Also verified 390px, 768px tablet and 1240px desktop without horizontal overflow; browser vote replacement changed totals from 1/0 to 0/1, keeping exactly one active vote. Stopping the demo server disabled voting and displayed the last-confirmed-data warning; restarting the same database and refreshing restored the same saved vote and totals. Static hosting without the API stayed in preparation with writes disabled. Desktop/RTL screenshots and a 1200×630 JPEG social preview were captured. Native dialog Escape was checked; an exhaustive keyboard accessibility audit was not performed. No real Android, Win10 remote, actual Telegram identity, public shared deployment, production attack test, legal approval, real prize activation or social publication was performed. This is a bounded local prototype, not a production-ready competition platform.

## Remaining launch steps (in order)

1. The owner approved publishing this reviewed preparation page. Opening a real contest and deploying a production backend still require separate approval.
2. Define legal organizer, eligibility, dates and full rules/privacy terms; obtain appropriate review without claiming automatic legal approval.
3. Define and demonstrate the 12-month VIP prize delivery/support/expiry. Until then keep preparation and all public writes closed.
4. Select and approve real identity plus persistent HTTPS hosting; implement provider verification, backup/deletion and audit/abuse/appeal controls. Remove all demo codes, fixture data and loopback server assumptions. Do not expose this server publicly.
5. Repeat concurrency/security/deadline/recovery tests on the actual deployment, test Android and restore a backup. Approve production configuration/rules and a public opening as a separate decision.
6. Only then use owner-approved permitted promotion drafts. Traffic, contest votes, Topper orders and commissions are separate metrics.
