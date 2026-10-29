# First-party visitor collection

This change is **not live until the Worker routes and the client are deployed**.
The browser sends only a page pathname to `https://btc2g.com/api/visit` and
reads aggregate statistics at `https://btc2g.com/api/stats`. There are no Google
Analytics, external counter calls, cookies or persistent browser IDs in this
collector. The storage remains the existing BTC2G Cloudflare D1 database;
this is independence from external analytics websites, not from hosting.

The existing collector called a separate `workers.dev` origin. That is a
cross-origin dependency; it is not proof of the cause of any live outage.
No live root cause or production count was confirmed from this environment.

## Deploy in this order

1. Use the authorized Cloudflare account to check that the `btc2g.com` zone
   is active and that its DNS record is proxied. Preserve the GitHub Pages
   origin, TLS configuration, existing routes and existing D1 database.
   Stop if those prerequisites are absent; do not change nameservers blindly.
2. Review existing routes for overlap. Apply both existing D1 migrations if
   not yet applied and verify the existing `IP_HASH_SECRET` is configured.
   Do not recreate/reset the database or rotate the secret for this release.
3. From the repository root, deploy the existing accounts Worker and assets:
   `npx wrangler deploy --config cloudflare/auth/wrangler.jsonc`.
   The two added route patterns cover only visitor collection and the public
   summary. Account login/admin routes remain on the existing account host;
   the site HTML, payment/provider APIs and forecasts are not moved.
4. Read `https://btc2g.com/api/stats`: require HTTP 200, JSON, `source`,
   `measuredAt`, Jerusalem reporting day, `traffic` and `confirmedSubscribers`.
   HTML from Pages or a 403/404/5xx means the routing/deployment is not ready.
   Do not publish the new client while this fails.
5. Publish `assets/js/traffic.js` through the normal GitHub Pages process.
   In an ordinary browser verify a real page load has one same-origin POST,
   HTTP 202 and `{ "ok": true }`, then read the summary again. Label any
   owner/test traffic when reviewing growth; do not manufacture registrations.
6. Check a real mobile browser and the source site's privacy notice. Confirm
   its description still matches the existing IP/browser daily hashing and
   retention policy. This patch adds no identity data or storage schema.

Cloudflare route prerequisites and matching rules:
https://developers.cloudflare.com/workers/configuration/routing/routes/

## Failure and counting semantics

One attempt per document. No retries: an ambiguous network failure may have
committed, so retrying would risk double counting. Failed/HTML/malformed
responses emit `btc2g:traffic` with `detail.status = 'unavailable'`; only an
acknowledged 202 JSON response emits `recorded`. Nothing claims that a failed
request has incremented the counter. Preview/localhost/automation visits are
not sent. Canonical production host is `btc2g.com`; a separate `www` deployment
requires deliberate route/origin review, not a third-party fallback.

The daily unique value remains an IP + browser hash estimate, not people.
Refreshing a page adds a view, not another daily unique browser estimate.
Missing historical data is unknown, and accounts are not email subscriptions.
Confirmed subscriptions remain unavailable until a real confirmation source
exists. An HTTP success from Formspree does not supply that evidence.

## Verification

Run `node --test cloudflare/auth/tests/*.test.mjs` with Node 24. Integration
fixtures use only local SQLite and real Worker SQL. The client tests cover the
same-origin request, no credential/referrer leakage, response rejection,
duplicate script execution and skipped non-production/automation surfaces.

No production counter endpoint, routing, visual layout or DNS change was
verified here. A GitHub source commit is not a Cloudflare deployment.

## Rollback

Restore the previous client asset first, then remove only the two added route
patterns if necessary. Preserve D1 and the existing accounts host. Keep missing
traffic marked unknown; never backfill estimated visits as observed traffic.
