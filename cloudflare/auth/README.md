# BTC2G Accounts Worker

This Cloudflare Worker hosts a same-origin account page and API backed by D1. Public totals count only active `user` rows; the bootstrap administrator is deliberately excluded from growth metrics.

## Security model

- PBKDF2-SHA-256 passwords with a unique random salt and 210,000 iterations.
- Random, hashed, 30-day `HttpOnly; Secure; SameSite=Lax` sessions.
- Same-origin checks on browser mutations.
- Hourly registration limits keyed by an HMAC-like salted hash of the Cloudflare client IP. Raw IP addresses are never stored.
- Reserved public names, including `Codex_admin`.
- One-time administrator bootstrap protected by a Cloudflare secret and disabled after the first admin exists.
- Explicit Terms and Privacy Policy acceptance recorded with a version and timestamp.
- Daily traffic uses a salted hash of IP + user agent; raw IP addresses and user-agent strings are not stored. Product Hunt attribution uses a signed same-day token tied to that daily estimate and stores only the normalized source label.

The protected `/admin/` dashboard shows visitor estimates, page views, registrations, registration/visitor ratios, Product Hunt-attributed visits and accounts, daily history, top paths, and recent usernames. Only a live session for a D1 user with `role = 'admin'` can read `/api/admin/stats`.

### Reading statistics correctly

- Use `https://btc2g-accounts.btc2g.workers.dev/api/stats` for the public summary. GitHub Pages does not route `https://btc2g.com/api/stats` to this Worker.
- The public account page and protected dashboard show traffic estimates, recorded page views, account registrations and confirmed email subscriptions separately.
- Responses include `source`, `measuredAt`, `day` and `timeZone` (`Asia/Jerusalem`). Traffic before the earliest retained recorded visit is unknown, not zero. `firstRecordedVisitAt` is the earliest retained evidence, not a claim that full-day collection began then.
- IP/browser hashes estimate browsers, not unique people. Ad blockers, shared networks and automated traffic affect counts. Registration/visitor ratios are aggregate ratios, not attributed conversions.
- `confirmedSubscribers.status` is `unavailable` and counts are `null` until a confirmed email subscription source is connected. Account creation and Formspree acceptance are not subscription confirmation. The old `dailyGoal`/`remainingToday` fields refer only to accounts and are retained for API compatibility; they do not measure the owner's confirmed-subscriber goal.
- Use **Refresh statistics** to recheck. A failed request clears visible values to **Data unavailable**, rather than displaying stale numbers as current. No traffic event or test email is needed to read the counters.
- If Cloudflare returns an access error (for example `403 / 1010`), the counters are unavailable from that client. Do not infer zero or bypass the restriction; deployment/access configuration must be checked through the authorized Cloudflare account.

### Validation and release of metrics changes

Run `npm test` with Node 24 (the SQL integration tests use built-in `node:sqlite`). Tests run the Worker against an in-memory SQLite database, including both migrations. Fixtures never write to production or send emails.

Apply `0001_initial.sql`, `0002_traffic_metrics.sql`, and `0003_acquisition_attribution.sql`. Deploy the Worker and its assets together with `npm run auth:deploy`, then read `/api/stats` and the protected dashboard to verify `measuredAt`, the reporting day and the separated metrics. Confirm that a forced *local* request failure clears the display; do not submit production test forms or page-view events. Live deployment requires an authorized Cloudflare session and is separate from a GitHub source commit.

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
