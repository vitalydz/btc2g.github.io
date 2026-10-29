# BTC2G Accounts Worker

This Cloudflare Worker hosts a same-origin account page and API backed by D1. Public totals count only active, email-verified `user` rows; pending accounts and the bootstrap administrator are deliberately excluded from growth metrics.

## Security model

- PBKDF2-SHA-256 passwords with a unique random salt and 210,000 iterations.
- Random, hashed, 30-day `HttpOnly; Secure; SameSite=Lax` sessions.
- Same-origin checks on browser mutations.
- Hourly registration limits keyed by an HMAC-like salted hash of the Cloudflare client IP. Raw IP addresses are never stored.
- Failed sign-ins are rate-limited across hashed login/network pairs, hashed usernames and hashed networks. Five pair failures trigger a 15-minute lock; broader limits slow distributed password spraying without storing raw IP addresses or attempted usernames.
- Administrators can enroll WebAuthn passkeys from an authenticated dashboard after confirming their password. Once the first passkey is enrolled, password verification alone never creates an administrator session; Windows Hello, a phone, or a FIDO2 security key must also verify the five-minute, one-time challenge. Enrollment marks the current session as passkey-verified and revokes the account's other sessions. Additional passkeys require a session that was itself passkey-verified.
- Reserved public names, including `Codex_admin`.
- One-time administrator bootstrap protected by a Cloudflare secret and disabled after the first admin exists.
- Mandatory account-email verification for new public registrations through Resend. Only token hashes are stored; links expire after 30 minutes. Pending accounts can sign in to request another message but are not counted as completed registrations. A verified account email is not marketing consent and does not activate Premium.
- Explicit Terms and Privacy Policy acceptance recorded with a version and timestamp.
- Daily traffic uses a salted hash of IP + user agent; raw IP addresses and user-agent strings are not stored. Product Hunt attribution uses a signed same-day token tied to that daily estimate and stores only the normalized source label.

The protected `/admin/` dashboard shows visitor estimates, page views, registrations, registration/visitor ratios, Product Hunt-attributed visits and accounts, daily history, top paths, and recent usernames. Only a live session for a D1 user with `role = 'admin'` can read `/api/admin/stats`.

### Reading statistics correctly

- Use `https://btc2g-accounts.btc2g.workers.dev/api/stats` for the public summary. GitHub Pages does not route `https://btc2g.com/api/stats` to this Worker.
- The public account page and protected dashboard show traffic estimates, recorded page views, completed email-verified registrations, pending accounts and confirmed email subscriptions separately.
- Responses include `source`, `measuredAt`, `day` and `timeZone` (`Asia/Jerusalem`). Traffic before the earliest retained recorded visit is unknown, not zero. `firstRecordedVisitAt` is the earliest retained evidence, not a claim that full-day collection began then.
- IP/browser hashes estimate browsers, not unique people. Ad blockers, shared networks and automated traffic affect counts. Registration/visitor ratios are aggregate ratios, not attributed conversions.
- `confirmedSubscribers.status` is `unavailable` and counts are `null` until a confirmed email subscription source is connected. Account creation and Formspree acceptance are not subscription confirmation. The old `dailyGoal`/`remainingToday` fields refer only to accounts and are retained for API compatibility; they do not measure the owner's confirmed-subscriber goal.
- Use **Refresh statistics** to recheck. A failed request clears visible values to **Data unavailable**, rather than displaying stale numbers as current. No traffic event or test email is needed to read the counters.
- If Cloudflare returns an access error (for example `403 / 1010`), the counters are unavailable from that client. Do not infer zero or bypass the restriction; deployment/access configuration must be checked through the authorized Cloudflare account.

### Validation and release of metrics changes

Run `npm test` with Node 24 (the SQL integration tests use built-in `node:sqlite`). Tests run the Worker against an in-memory SQLite database, including all migrations. Fixtures never write to production or send emails.

Apply all migrations through `0004_email_verification.sql`. Deploy the Worker and its assets together with `npm run auth:deploy`, then read `/api/stats` and the protected dashboard to verify `measuredAt`, the reporting day and the separated metrics. Confirm that a forced *local* request failure clears the display; do not submit production test forms or page-view events. Live deployment requires an authorized Cloudflare session and is separate from a GitHub source commit.

### Free email provider setup

BTC2G uses the Resend HTTP API. Its free plan currently supports 100 transactional emails per day. Before deploying email verification:

1. Create the owner's Resend account and verify `btc2g.com` (or change `EMAIL_FROM` to an address on another verified domain).
2. Create a sending-only API key and store it with `npx wrangler secret put RESEND_API_KEY --config cloudflare/auth/wrangler.jsonc`.
3. Apply migration `0004_email_verification.sql` remotely, deploy, and verify using the owner's real account. Never commit the key or verification token.

If `RESEND_API_KEY` is absent or delivery fails, the pending account remains accessible so its owner can retry, but it is not counted as a completed registration and the UI does not claim that mail was sent.

This is an account count, not an identity-verification system. A verified-human metric requires an email or identity provider and is intentionally not claimed.

## Local verification

```powershell
npm test
npx wrangler d1 migrations apply btc2g-users --local --config cloudflare/auth/wrangler.jsonc
npx wrangler dev --local --config cloudflare/auth/wrangler.jsonc
```

Set `IP_HASH_SECRET` in `cloudflare/auth/.dev.vars` before testing public registration locally. Do not commit that file.

## Production deployment

1. Set `IP_HASH_SECRET` and `BOOTSTRAP_TOKEN` with `wrangler secret put`.
2. Apply D1 migrations with `wrangler d1 migrations apply btc2g-users --remote`.
3. Deploy with `npm run auth:deploy`.
4. Create `Codex_admin` once through `POST /api/bootstrap-admin`, then remove the bootstrap secret.
5. Add a Turnstile widget before broad promotion if abuse becomes material. The stored totals must never be presented as verified people without verification.
