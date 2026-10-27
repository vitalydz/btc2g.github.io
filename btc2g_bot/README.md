# BTC2G VIP — website + Telegram Stars

## Product

- Website: **USD $4.95** for **30 days** of VIP access.
- Telegram: **200 Stars** for the same 30 days.
- Forecast horizon: the next **7 days**, delivered by `/forecast` in a private chat.
- One-time purchases; no automatic renewal. Additional paid periods extend existing access.
- Forecasts are not a guarantee of profit.

## Current state

Implementation is local and tested with synthetic payments. Real payments, polling and public deployment have not been started. `VIP_SALES_ENABLED=false` by default. The existing GitHub Pages site and its forecast PNG/JSON are unchanged.

The old manual crypto-proof approval flow and unrestricted channel invitations are replaced by time-limited delivery in the bot. Screenshots, TXIDs and the website return URL cannot grant VIP access. Existing channel membership is not changed or revoked by this code.

This folder is the deployment unit: the bot and `web_app.py` share one persistent SQLite database. It is not a Vercel serverless function and must not use an ephemeral database. Existing Stripe TypeScript placeholders in `btc2g.github.io/api/stripe` are not connected to this service.

## Setup

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env`; never commit it.
3. Set `BOT_TOKEN`, `VIP_SUPPORT_CONTACT`, `VIP_PUBLIC_URL` (an HTTPS origin serving this app), `STRIPE_SECRET_KEY`, and `STRIPE_WEBHOOK_SECRET`.
4. Keep `VIP_STARS_PRICE=200`. The USD amount is enforced server-side as 495 cents.
5. Both processes must point to the same `VIP_DATABASE` on persistent storage. Back it up.
6. Provide a real reviewed weekly forecast at `VIP_FORECAST_FILE`. Missing, future, expired or empty forecasts block new purchases.
7. Test Telegram and Stripe in their test environments. Only then set `VIP_SALES_ENABLED=true`.

Run in two terminals from this directory:

```powershell
python bot.py
python -m uvicorn web_app:app --host 127.0.0.1 --port 8766 --no-access-log
```

For public use, put the web process behind an HTTPS reverse proxy on an existing host. Configure Stripe webhook URL `https://YOUR-HOST/stripe/webhook`, subscribing to `checkout.session.completed`, `checkout.session.async_payment_succeeded`, and `charge.refunded`. Point the main site's VIP link to `https://YOUR-HOST/vip` after review and deployment. Do not point it at localhost.

## User flows

Telegram: `/vip` → accept displayed terms → pay 200 Stars → Telegram `successful_payment` → VIP → `/forecast`.

Website: obtain a private account-binding code via `/connect` → enter it on `/vip` → accept terms → Stripe Checkout for $4.95 → signed webhook activates the same account → `/status` and `/forecast` in Telegram. The code expires in 24 hours and cannot be used after payment. Retrying checkout with the same code uses the same Stripe idempotency key. Do not share codes.

Other commands: `/terms`, `/support`, `/paysupport`. No external checkout link is promoted inside the Telegram purchase flow; Telegram purchases use Stars.

## Weekly forecast file

Do not use a fabricated sample as a paid forecast. Supply reviewed content with this structure:

```json
{
  "issued_on": "YYYY-MM-DD",
  "valid_until": "YYYY-MM-DD exactly seven days after issued_on",
  "horizon_days": 7,
  "text": "Reviewed weekly BTC/Gold forecast with its method, data date, scenarios and limitations"
}
```

Dates are UTC; validity ends at midnight on `valid_until`. Replace this file atomically when publishing a new forecast. This implementation delivers supplied forecasts; it does not invent or generate the market forecast. The public five-year chart is not substituted for a weekly product.

## Payment correctness and refunds

Orders bind the provider, amount, currency and Telegram user. Access is granted only by verified payment events. SQLite transactions and unique charge IDs prevent duplicate access extensions, including concurrent callbacks and restarts. Paid events are still handled when new sales are disabled.

Full-refund notifications from Stripe and Telegram revoke one paid 30-day period once. Refunds arriving before payment confirmation are recorded to prevent a later access grant. Partial refunds and disputes require manual reconciliation. Refund initiation remains with the operator using the provider's tools; `/paysupport` exposes the configured contact.

Do not delete the database to resolve payment problems. Backups must include the ledger. Application logs omit payment codes, tokens, message bodies and card information; configure the reverse proxy likewise.

## Verification

```powershell
python -m unittest -v
python -m py_compile bot.py web_app.py vip_store.py vip_service.py vip_settings.py
```

Tests exercise both prices, shared access, duplicate/concurrent notifications, restart persistence, invalid signatures, wrong amount/user/currency, consent, disabled sales, forecast validity, full refunds, code expiration, and direct access checks. Real provider checkout and public deployment still require integration testing with the configured accounts.
