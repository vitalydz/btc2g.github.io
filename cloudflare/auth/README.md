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
