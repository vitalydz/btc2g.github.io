# Sunday digest: production service and controlled rollout

The homepage choice is wired to POST `/api/digest-request` at the verified Vercel production origin in `assets/audience-config.json`. Loading the form alone sends no email. Formspree is not used to save digest choices. Production sending stays disabled until more than three confirmed subscribers have explicitly selected `weekly_digest=true` and the owner gives final approval immediately before enabling it.

## User flow

1. The existing main subscription request still uses its existing receipt flow. The restored digest form starts unchecked, explains that the digest is in English and requires email confirmation, and never silently opts in an account or request.
2. Saving `{weeklyDigest:true|false}` starts a rate-limited ownership-verification email request. It does **not** change the active preference. Existing active subscribers may request a verified change without being deactivated while waiting.
3. The mail contains an expiring, single-use confirmation link in EN/RU/HE, followed by a confirmation button. Merely opening the URL does not mutate state; security scanners cannot opt people in with GET. The page displays the exact intended enabled/disabled choice.
4. Only POST confirmation atomically persists the choice and consent date. An opt-out/confirmed newer choice invalidates older pending intents. A failed replacement mail preserves every prior still-valid link, subject to the newer-intent guard. Disabled preferences are not counted as confirmed subscribers for a never-confirmed email.
5. Every issued digest has both a visible unsubscribe link and RFC one-click headers. GET shows a button; POST disables digest delivery. This is independent of mail-provider availability. A global subscription opt-out also clears the digest choice. Only a newly requested and verified intent can re-enable it.

No public API can change a preference by just knowing the address. Changes require mailbox access; unsubscribe capabilities can only disable. Confirmation hashes remain private. `digest_unsubscribe_token` is an opaque random disable-only capability stored in the private database so it can be included in each weekly issue. Never log or publish it.

## Scheduled sending

`vercel.json` calls `/api/digest-cron` Sundays at 06:00 UTC (08:00 Israel winter / 09:00 summer). It requires `Authorization: Bearer CRON_SECRET`, `DIGEST_ENABLED=true`, a working PostgreSQL database, verified MAIL_FROM and RESEND_API_KEY. Keep disabled in every preview/test deployment.

The handler checks Sunday in Asia/Jerusalem. It reads both the published BTC2G signal artifact and the published BTC/Gold history, then validates dates, arithmetic consistency, ordering and freshness (maximum four days). The email reports the published market date, ratio/prices, model action and score. It calculates the measured seven-day ratio change from the last published observation at or before the calendar cutoff and reports the observed 30-day low/high as key-level context. If the two artifacts do not agree, the issue is rejected instead of sending a guessed or stale summary.

The basic Sunday digest is free for confirmed, opted-in subscribers. Extended explanatory model notes remain a planned Premium feature; the email must describe them as planned and must not imply that Premium is live.

Only confirmed, opted-in, non-unsubscribed addresses are selected. Atomic `(email,issue_day)` claims and stable provider idempotency keys suppress duplicates across concurrent runs. Consent is checked again before sending. An already in-flight message may arrive after opt-out.

A successful provider response must include an email ID; state is `accepted`, **not delivered**. Errors, malformed receipts and timeouts become `uncertain` and are not automatically resent. Interrupted claims remain `sending`; investigate rather than resetting these rows and risking duplicates.

Each invocation selects at most 1000 candidates, uses up to 10 concurrent sends, and stops starting work after 240 seconds. Remaining unsent work returns HTTP 503 with `remaining`, never a false all-done message. For a larger/slow audience, use a queue or explicitly invoke the same authenticated job again on Sunday. Observe the response and delivery rows after the first issue; this initial bounded implementation is not an unbounded bulk mail platform. Verify Fluid Compute / 300-second function support for the chosen Vercel project and plan before enabling.

## Deployment and verification

1. Import repository with Vercel **Root Directory `services/audience`**, Node 22, Framework Other. Do not deploy repository-root Stripe/Topper endpoints as this service.
2. Create PostgreSQL; set DATABASE_URL (verified TLS), SITE_ORIGIN=https://btc2g.com, API_ORIGIN=the actual production origin, MAIL_FROM, RESEND_API_KEY, VISITOR_HASH_SECRET, STATS_TOKEN and CRON_SECRET as server secrets. A browser login alone does not grant this session deployment access.
3. Back up data and run `npm run migrate`; migrations 001–003 are additive and idempotent. Start with DIGEST_ENABLED=false.
4. Verify staging and automated tests. Use only a separately authorized mailbox for a real mail test; this implementation was tested using mocked mail, not live recipients.
5. Confirm production sender/domain, scheduler and privacy notice. Set DIGEST_ENABLED=true and update only the public `apiOrigin` in `assets/audience-config.json` to the verified deployment origin. No database/mail/cron secrets belong in frontend code.
6. Monitor first Sunday and the accepted/uncertain/sending states. To suspend delivery set DIGEST_ENABLED=false; keep opt-out accessible. Roll back origin config only if needed; do not delete preferences or consent records.

## Tests

- Root `npm test` covers restored form labels in 19 languages, explicit unchecked choice, API wiring, pending-confirmation message, invalid replies, failed configuration, timeout recovery and concurrent click suppression.
- `services/audience`: `npm test` covers authentication, strict choices, Sunday/market validation, provider failures and the general confirmation flow.
- With `TEST_DATABASE_URL` pointing only to an isolated disposable database, the real PostgreSQL test checks all migrations, confirmation-only mutation, concurrent cron deduplication, durable preferences, unsubscribe, old-intent rejection and explicit re-enable/disable. All mail is mocked.

Passing these tests is not a claim that every possible bug is absent or that production is running. Deployment, real DNS/sender verification and permission to send a test mail remain separate checks.
