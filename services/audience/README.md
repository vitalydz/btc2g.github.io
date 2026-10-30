# BTC2G audience service — staged Vercel/PostgreSQL replacement

This directory is a standalone Vercel project. No Cloudflare SDK, Worker, D1 or account authentication dependency. The homepage digest form uses the dedicated API when assets/audience-config.json specifies its verified deployment origin. With apiOrigin:null no mail or preference write is attempted. Existing Cloudflare accounts, SAVE/EARN/PLAY, calculator, forecast and provider links are not switched. It does not migrate accounts or historical data. Contest remains closed.

## Setup

1. Create a separate Vercel project from this repository with **Root Directory `services/audience`**, Framework Preset **Other**, no build command, no output directory override. Do not use the repository root: its unrelated payment endpoints are outside this service.
2. Provision PostgreSQL and set server environment variables listed in `.env.example`. Use the provider's verified TLS connection string; do not disable certificate verification. Restrict the runtime database role to the audience tables. Use a separate migration role for DDL.
3. Configure a verified sending domain and sender with Resend. The mail adapter is isolated in `lib/server.js` and can be replaced; no provider has been contacted or configured by this patch.
4. From this directory run `npm ci`, then `npm run migrate` with the migration connection. The migration runner applies numbered SQL files in order, including additive migrations 002 and 003 for existing installations. It is idempotent and only changes `audience_*` tables. No existing D1 data is deleted or imported.
5. Deploy the service, setting API_ORIGIN to its fixed HTTPS production origin. Preview deployments must use a separate database and mail sandbox. Do not enable production traffic in previews.
6. Verify read-only `GET /api/stats` with server-side `Authorization: Bearer STATS_TOKEN`. Missing database, schema or runtime configuration returns 503, never a fabricated zero. Empty installed traffic storage returns `traffic: null`. A successful database query may report 0 active confirmed subscribers in this **new database**, with its initialization period; this is not a historical total for BTC2G.

Do not send test emails or submit production forms without separate permission. Verification before that permission uses the mock tests and a disposable PostgreSQL database only.

## API and semantics

- POST `/api/visit`, JSON `{ "path": "/save.html" }`, Origin `https://btc2g.com`: one stored pageview per accepted request; deduplicates a daily visitor estimate using HMAC(day, platform IP, truncated user-agent). Raw IP, raw user-agent, email, path and query parameters are not stored in traffic rows. Recognizable bots are excluded, but unknown bots, NAT, VPN and browser changes remain limitations. Counts are **estimates**, not verified humans; sessions are not measured. `visitorDays` sums daily estimates and is not period-wide unique users. No geographic totals or campaign attribution yet.
- POST `/api/subscribe`, JSON `{ "email": "...", "consent": true, "consentVersion": "2026-10-29" }`: explicit unchecked-by-default newsletter consent must be given by the user. Rate limit 5 requests per platform IP per UTC hour, plus a persistent per-email cooldown: 1 minute after explicit mail rejection, 2 minutes after an interrupted pending attempt, 15 minutes after acceptance or an ambiguous timeout. Active subscriptions and cooldown duplicates receive the same generic 429 with no new send, not a misleading 202. No membership details are disclosed. Only an accepted new mail request gets 202, which still never means confirmed or delivered. Rejections/timeouts return 503 with Retry-After. Retry must be a new explicit user action. Pending, accepted, failed and unknown outcomes are stored separately; stale attempts cannot overwrite a newer attempt state. Every prior valid link is retained until expiry, confirmation or unsubscribe; replacement is not destructive.
- GET `/api/confirm?token=...`: confirmation page only; mail scanners cannot confirm with GET. POST the form to consume the random single-use token (24-hour expiry). Only then is `confirmed_at` stored. Token plaintext exists in the email only; database stores SHA-256 hashes. EN/RU/HE are supported end to end, with Hebrew RTL. The client passes allowlisted `lang`, saved on each request/token; stored token language wins over URL overrides. Invalid/expired links use the saved language or allowlisted URL fallback.
- GET `/api/unsubscribe?token=...`: shows an unsubscribe button; POST removes active status and invalidates any pending confirmation. The unsubscribe link is included in the confirmation email. Any future newsletter sender **must include this same per-recipient unsubscribe capability**; this patch does not implement bulk newsletters. Repeated unsubscribe is idempotent. A new explicit request after unsubscribe can start a fresh confirmation flow after cooldown.
- GET `/api/stats`: private server-side bearer authentication, no browser access token. Returns source, measuredAt, Israel day/timezone, traffic observation range, active confirmed subscribers and confirmations dated today. `accounts: null` because registrations are not subscribers. No subscriber addresses or tokens are exposed. No prior confirmed baseline is available; callers must retain dated snapshots to report changes. Account totals and old Formspree requests must never be imported as confirmed subscribers.

Vercel supplies the trusted `x-forwarded-for` used here. Do not deploy this handler directly behind a proxy that accepts caller-supplied IP headers. Origin checks reduce unwanted browser requests, but do not authenticate human visitors. Enable host-level abuse protection before promoting. Use direct Vercel hosting with no unconfigured intermediate proxy.

## Frontend cutover after deployment

`client/audience-client.js` is an inactive adapter example, not loaded by the public site. After API validation:

1. Copy the adapter into site assets; configure only the public API origin. Never put database, email or stats secrets in JavaScript.
2. Add an unchecked, localized newsletter consent checkbox and privacy link to both forms. Pass its checked state to `client.subscribe(email, checked, currentLang)`; do not hardcode true. Preserve RU/HE/RTL labels, validation, existing button recovery and ambiguous timeout messages. Show “Request received. If eligible, check your email to confirm”; never “confirmed” from this call.
3. Replace the old traffic collector with one `recordVisit()` call per document. Remove the old Worker collector at cutover so pageviews are not counted in two datasets. Do not merge the old same-origin PR #5 alongside this separate-origin implementation.
4. Keep current Formspree and Worker frontend wiring until the new service is actually deployed and separately approved for live use. Roll back frontend wiring if the new endpoint fails. Neither localStorage nor test submissions are an audience source.
5. Update the site's privacy notice with the actual database/mail processors, HMAC visitor method, email retention, confirmation and unsubscribe process before enabling the adapter.

## Tests and maintenance

`npm test` runs handler tests with injected storage/mail, without any network requests. `TEST_DATABASE_URL=... npm test` additionally creates an isolated random schema in a **disposable PostgreSQL test database**, applies the real SQL and exercises subscribe → confirm → replay rejection → visit deduplication → stats → unsubscribe. Never point TEST_DATABASE_URL at production. Without it, the real PostgreSQL test is explicitly skipped; mock tests do not validate PostgreSQL execution, Vercel routing or deliverability.

Schedule `node cleanup.cjs` daily through your deployment scheduler. It removes expired rate limits, daily visitor hashes older than 400 days, and unconfirmed requests older than 30 days. Active and unsubscribed consent records are retained until an explicit deletion policy/request; agree this policy before production. Traffic totals are over the retained observation period, not lifetime site totals. Database backup and restore remain an operator responsibility.

Official references checked 2026-10-29:
- https://vercel.com/docs/functions/runtimes/node-js
- https://vercel.com/docs/headers/request-headers
- https://node-postgres.com/features/queries
- https://resend.com/docs/api-reference/emails/send-email

## Sunday digest

See [DIGEST.md](DIGEST.md) for the restored homepage flow, verification, storage, scheduling, measured weekly-change and key-level calculations, opt-out, capacity and rollout checks. The homepage is wired to assets/js/digest-preferences.js; it never sends weekly choices to Formspree. Do not confuse source integration with a deployed service.
