# BTC2G Accounts Worker

This Cloudflare Worker hosts a same-origin account page and API backed by D1. Public totals count only active, email-verified `user` rows; pending accounts and the bootstrap administrator are deliberately excluded from growth metrics.

The release version is stored in the repository-root `VERSION` file and mirrored by `APP_VERSION` in `wrangler.jsonc`. The protected administrator dashboard reads it from `/api/admin/stats`, so the GUI reports the deployed Worker version rather than a browser-only label.

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
- Email-verified ordinary users can privately submit up to three ideas. D1 stores the text, status and private moderation note. Only the author can read their own submissions; only an administrator can read the review queue. No public list or voting endpoint exists.

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

Apply all migrations through `0007_private_ideas.sql`. Deploy the Worker and its assets together with `npm run auth:deploy`, then read `/api/stats` and the protected dashboard to verify `measuredAt`, the reporting day and the separated metrics. Confirm that a forced *local* request failure clears the display; do not submit production test forms, ideas, or page-view events. Live deployment requires an authorized Cloudflare session and is separate from a GitHub source commit.

### Free email provider setup

BTC2G uses the Resend HTTP API. Its free plan currently supports 100 transactional emails per day. Before deploying email verification:

1. Create the owner's Resend account and verify `btc2g.com` (or change `EMAIL_FROM` to an address on another verified domain).
2. Create a sending-only API key and store it with `npx wrangler secret put RESEND_API_KEY --config cloudflare/auth/wrangler.jsonc`.
3. Apply migration `0004_email_verification.sql` remotely, deploy, and verify using the owner's real account. Never commit the key or verification token.

Verification mail sets `support@btc2g.com` as its reply address. Incoming support mail is expected to use the domain's configured mail-forwarding service; Resend remains the transactional sender, not the support inbox.

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


## Administrator MFA review (2026-10-30)

### Local evidence and limits

Run `node --test cloudflare/auth/tests/metrics.test.mjs` on Node 24 after installing
repository dependencies. The suite contains 21 tests, including six added MFA and
upgrade cases. The local software authenticator generates an ephemeral P-256 key,
COSE public key, CBOR `none` attestation and real ES256 authentication signatures.
The production SimpleWebAuthn verifier is never mocked. `none` attestation validates
the registration format, RP, origin and UV; the subsequent signed assertion proves
possession of the registered private key. This is not hardware attestation.

Coverage includes primary/backup enrollment and login, challenge reuse after both
success and failure, replaced and expired challenges (including exact expiry),
wrong signed client origin, wrong RP hash, missing UV, bad signatures, revoked
sessions, surviving unverified session rows, password enrollment lockout and
transaction rollback on an injected session-update failure. Existing 15 integration
tests remain included. Test identities and keys exist only in memory.

Fixes: enforce existing login lockouts before enrollment password checks; commit
credential insertion, current-session verification, other-session revocation and
pending-challenge invalidation in one D1 batch; reject unverified admin sessions
whenever an enrolled key exists, even if an old row survived or was created by an
in-flight password login. Before the fixes, the new lockout, surviving-session and
rollback regression tests fail.

Not established by these tests: real Windows Hello/phone/security-key interaction,
Cloudflare runtime behavior, concurrent distributed request ordering, production
schema state or deployment. Full repository `npm test` was not run in this scoped
checkout. A local SQLite adapter exercises real SQL and transactions, not remote D1.

### Register and verify an independent backup

1. Keep the first authenticator available. Use the configured Worker HTTPS origin;
   the current RP is `btc2g-accounts.btc2g.workers.dev`, not `btc2g.com`.
2. Sign out, then sign in using the password **and** the existing passkey.
3. In that verified session, confirm the password and enroll a second authenticator.
   Name it distinctly. Prefer a separate FIDO2 key stored safely; another credential
   synced through the same phone/cloud account may share its failure modes.
4. Sign out and complete a fresh password-plus-passkey login using the backup alone.
   Check the admin dashboard, then separately check the primary key. Merely seeing
   a second row does not prove the backup works. Keep at least two tested keys.
5. The display API exposes a shortened credential ID. Never use that suffix to
   identify a key for a database deletion; use the full stored ID and exact user ID.

No real enrollment was performed for this review. During future staging validation,
use a separate test D1 database and its own origin/RP, never production credentials.

### Lost device: controlled recovery, not a public bypass

There is deliberately no password-only reset endpoint or email fallback. The
bootstrap endpoint cannot recover an existing administrator. Do not change a role,
re-enable bootstrap or delete the last key on the publicly reachable application.

If a tested backup survives: authenticate with it and enroll/test a replacement.
An authorized D1 operator can then revoke the lost full credential ID for the exact
administrator. Treat this as a separate approved maintenance change: restrict all
Worker ingress first (including workers.dev and any alternate routes), drain
in-flight requests, verify the exact row and another tested credential remain,
then delete only that credential and all of that administrator's sessions and
pending WebAuthn challenges in one transaction. Reopen ingress only after a fresh
password-plus-replacement/backup login succeeds. Revocation also requires deletion
of sessions: a stolen device may already possess a verified session.

If every key is lost: recovery requires independently authenticated control of the
Cloudflare account/D1 and explicit owner authorization. Use a separately protected
maintenance window; confirm external requests cannot reach any API route and drain
requests before changing authentication data. Back up the database, record exact
administrator ID, and atomically revoke that user's sessions, challenges and
passkeys. Reset the password through a separately reviewed operator process if
compromise is suspected; this service has no password-change endpoint. Under the
same independently enforced access restriction, the owner signs in and enrolls a
new primary and independent backup, then tests both through fresh MFA logins.
Verify no unverified admin sessions remain before restoring public access. If
exclusive maintenance access cannot be guaranteed, leave the account locked and
prepare a separately reviewed offline recovery implementation. Never leave a
zero-key administrator publicly accessible. Do not restore an old whole-database
backup just to recover a key: it can resurrect revoked credentials and sessions.

These are operator runbooks, not newly implemented self-service recovery or key
removal routes. No production SQL or account changes were executed.

### D1 compatibility and migration order

The configured database is `btc2g-users`, binding `DB`; use the explicit config
path in every Wrangler command. This patch requires no new migration and does not
edit historical migrations. A populated pre-MFA schema (0001–0005) upgrades through
0006 preserving user/password/session data; old sessions receive NULL
`passkey_verified_at`. Fresh local integration databases apply 0001–0007. Local
foreign-key checks pass. This verifies schema compatibility, not the contents of
production D1.

Before an approved release:

1. Read `wrangler d1 migrations list btc2g-users --remote --config cloudflare/auth/wrangler.jsonc`
   and inspect `PRAGMA table_info(sessions)`, passkey/challenge tables and migration
   history in the intended account/database. Read-only inspection is still separate
   from the local evidence above; it has not been performed here.
2. Preserve a D1 export/recovery bookmark and validate the release against a separate
   staging database. Apply pending migrations in order using
   `wrangler d1 migrations apply btc2g-users --local --config cloudflare/auth/wrangler.jsonc`
   for local validation, then the corresponding `--remote` command only with release
   authorization. Required order is 0001 through 0007, including 0006 before this
   Worker can serve requests.
3. Use Wrangler's migration ledger. The `ALTER TABLE ... ADD COLUMN` in 0006 is not
   idempotent: manually rerunning its SQL fails with a duplicate column. If schema
   and ledger disagree, stop and reconcile under a separate reviewed plan; do not
   drop tables or blindly replay/mark migrations.
4. Deploy Worker/assets only after successful remote migrations and explicit approval.
   Verify secrets and exact APP_ORIGIN/RP ID; changing RP to the main website would
   invalidate access to the existing credentials on this RP. Keep bootstrap disabled.
5. Perform owner-authorized hardware smoke checks and backup enrollment separately.
   Read deployed version and schema evidence before declaring production readiness.

Production migrations, Worker deployment and live hardware ceremonies remain
unconfirmed. Source commits alone provide no deployment evidence.

References: [SimpleWebAuthn server](https://simplewebauthn.dev/docs/packages/server),
[passkey verification](https://simplewebauthn.dev/docs/advanced/passkeys/),
[Cloudflare D1 migration commands](https://developers.cloudflare.com/d1/wrangler-commands/).
