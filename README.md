# BTC2G

BTC2G is a static GitHub Pages website with a Vercel Serverless Function for creating Topper bootstrap tokens.

## Topper serverless token endpoint

Topper bootstrap tokens must be signed on a backend because `TOPPER_PRIVATE_JWK` is a private signing key. Never put `TOPPER_PRIVATE_JWK`, private keys, or server secrets in `index.html`, `buy.html`, or any other frontend file.

The endpoint is:

```text
POST /api/topper/token
```

It returns:

```json
{
  "bt": "...jwt...",
  "appUrl": "https://app.topperpay.com/"
}
```

## Deploy to Vercel Hobby

1. Import this GitHub repository into Vercel.
2. Keep the default framework settings. Vercel will detect the `api/topper/token.ts` Serverless Function.
3. Add these environment variables in Vercel Project Settings > Environment Variables:

```env
TOPPER_ENV=sandbox
TOPPER_WIDGET_ID=<your Topper widget id>
TOPPER_KEY_ID=<your Topper key id>
TOPPER_PRIVATE_JWK=<your private EC P-256 JWK JSON>
```

4. Deploy the project.
5. Copy the Vercel deployment URL and replace `YOUR-VERCEL-PROJECT` in `buy.html`:

```js
const TOPPER_API_URL = "https://YOUR-VERCEL-PROJECT.vercel.app/api/topper/token";
```

6. Keep the frontend on GitHub Pages. The API endpoint runs on Vercel and is called from the static site.

## Sandbox and production

Use sandbox while testing:

```env
TOPPER_ENV=sandbox
```

Switch to production only when Topper production credentials are ready:

```env
TOPPER_ENV=production
```

`TOPPER_ENV=sandbox` returns `https://app.sandbox.topperpay.com/`.
`TOPPER_ENV=production` returns `https://app.topperpay.com/`.

## Wallet/address logic

The current wallet address comes from user input in `buy.html`:

```js
const address = document.getElementById('topper-address').value.trim();
```

To use a fixed configured wallet instead, replace that line with your own safe frontend logic. Do not put private keys or server secrets there.

## Local commands

Run dependencies install from the site repository:

```powershell
cd C:\Users\vitly\PycharmProjects\BTC2G\btc2g.github.io
npm install
```

This repository currently has no `test` or `build` scripts in `package.json`, so `npm test` and `npm run build` are not available unless scripts are added later.

## Monetization flow

The homepage keeps the BTC vs Gold forecast chart first, then shows a short educational explanation and two buy buttons:

- MoonPay: `index.html` links to `buy.html#moonpay`, where the existing MoonPay widget button remains visible.
- Topper: `index.html` links to `buy.html#topper`, where the Topper address and amount form calls the serverless token endpoint.

Topper API URL configuration lives in `buy.html`:

```js
const TOPPER_API_URL = "https://YOUR-VERCEL-PROJECT.vercel.app/api/topper/token";
```

Future affiliate or revenue share tracking can be added to these button links or provider parameters, but private keys and signing secrets must stay out of frontend files. `TOPPER_PRIVATE_JWK` belongs only in Vercel environment variables.

The homepage disclaimer is intentionally conservative:

```text
This forecast is for informational purposes only and is not financial advice.
```

## Email Subscription Setup

The homepage email capture form uses Formspree because it does not require a custom backend or frontend secrets.

The placeholder endpoint lives in `index.html`:

```js
const SUBSCRIPTION_ENDPOINT = 'https://formspree.io/f/YOUR_ID';
```

Setup steps:

1. Create a free Formspree form at `https://formspree.io/`.
2. Copy the form endpoint URL.
3. Replace `https://formspree.io/f/YOUR_ID` in `index.html` with your real endpoint.
4. Open the homepage locally or on GitHub Pages.
5. Submit a test email and confirm it appears in Formspree.

Local test command:

```powershell
cd C:\Users\vitly\PycharmProjects\BTC2G\btc2g.github.io
python -m http.server 8765
```

Then open:

```text
http://127.0.0.1:8765/index.html
```

Do not commit private API keys. Formspree form URLs are meant for frontend use, but do not add any private provider secret to `index.html`.

## Localized dates

Dates are stored internally as ISO `YYYY-MM-DD` and localized only in the UI. The homepage uses `Intl.DateTimeFormat` to render the forecast/model date according to the selected site language.

## BTC Signal System

The homepage displays a simple model-based BTC signal loaded from:

```text
assets/btc_signal.json
```

Supported values:

- `BUY`: the model sees BTC as relatively undervalued versus gold.
- `HOLD`: the model sees BTC as near neutral versus gold.
- `SELL`: the model sees BTC as relatively overvalued versus gold.

The signal is based on the BTC/Gold ratio model and is not financial advice. It reflects relative valuation, not guaranteed profit or short-term price predictions.

The MVP signal logic is implemented in:

```text
scripts/update_btc_gold_chart.py
```

Look for:

```python
def build_signal(df: pd.DataFrame) -> dict:
```

Current MVP rule:

- Compute the latest BTC/Gold ratio.
- Compare it with the last 365 days of the BTC/Gold ratio.
- Below the rolling range threshold becomes `BUY`.
- Near the rolling average becomes `HOLD`.
- Above the rolling range threshold becomes `SELL`.

The daily GitHub Actions workflow runs the chart script and commits:

```text
assets/btc_gold_forecast.png
assets/btc_gold_forecast_meta.json
assets/btc_signal.json
```

Future accuracy improvements can replace `build_signal(...)` with a more robust model while keeping the JSON shape stable for the frontend.

## Premium Signals (MVP)

The homepage includes a UI-only Premium BTC Signals layer. It does not add payments, authentication, or backend user management yet.

Current MVP behavior:

- Free users still see the current model signal, confidence, and basic explanation.
- Premium-only areas are marked with a lock and careful wording.
- `const isPremiumUser = false;` in `index.html` is the temporary frontend gating flag.
- The "Unlock Premium" button opens a lightweight modal.
- The modal captures early-access emails through the same Formspree subscription helper used by the main email form.

Premium early access is configured through the same endpoint:

```js
const SUBSCRIPTION_ENDPOINT = 'https://formspree.io/f/YOUR_ID';
```

Future paid subscription work can connect Stripe or a similar provider from the Premium modal button flow. Keep payment secrets, webhook signing secrets, and user entitlement checks on a backend. Do not put private keys or payment secrets in frontend files.

No sensitive user data is stored in this repository. Email collection depends on the external form provider configured in `SUBSCRIPTION_ENDPOINT`.

## Daily BTC vs Gold chart update

The public chart is saved at:

```text
assets/btc_gold_forecast.png
```

`index.html` loads that stable file with a daily UTC cache-busting query string:

```js
assets/btc_gold_forecast.png?v=YYYY-MM-DD
```

The chart is generated by:

```text
scripts/update_btc_gold_chart.py
```

The script preserves the existing data source logic and uses `yfinance` with these public Yahoo Finance tickers:

- Gold futures: `GC=F`
- Bitcoin: `BTC-USD`

No paid API keys are required. If data download or chart generation fails, the script does not delete the last working chart. If a previous chart exists, it logs the error and exits without replacing it. If no chart exists yet, it fails clearly.

The generated metadata file is:

```text
assets/btc_gold_forecast_meta.json
```

### Run the chart update locally

```powershell
cd C:\Users\vitly\PycharmProjects\BTC2G\btc2g.github.io
python -m pip install -r requirements.txt
python scripts/update_btc_gold_chart.py
```

After running it, confirm `assets/btc_gold_forecast.png` exists and opens.

### Automatic GitHub Actions update

The workflow is:

```text
.github/workflows/update-chart.yml
```

It runs once per day at `03:15 UTC` and can also be started manually from GitHub Actions with `workflow_dispatch`.

The workflow installs Python and Node dependencies, runs the chart script, and commits only when `assets/btc_gold_forecast.png` or `assets/btc_gold_forecast_meta.json` changes. The commit message is:

```text
Update BTC vs Gold forecast chart
```

### Manual browser refresh

The "Update chart manually" button on the homepage reloads the image by changing the query string to:

```js
assets/btc_gold_forecast.png?v=Date.now()
```

This refreshes the user's browser cache without changing the server file.

### If the chart is broken

Check these items:

- `assets/btc_gold_forecast.png` exists in the repository and is published by GitHub Pages.
- `.github/workflows/update-chart.yml` completed successfully.
- `scripts/update_btc_gold_chart.py` can download data from Yahoo Finance through `yfinance`.
- `index.html` points to `assets/btc_gold_forecast.png`, not an old date-stamped filename.
- The browser is not blocked from loading PNG assets from the same GitHub Pages site.

## Changed files after implementation

- `api/topper/token.ts`
- `.env.example`
- `README.md`
- `package.json`
- `package-lock.json`
- `index.html`
- `buy.html`
- `.github/workflows/update-chart.yml`
- `assets/btc_gold_forecast.png`
- `assets/btc_gold_forecast_meta.json`
- `requirements.txt`
- `scripts/update_btc_gold_chart.py`

## Risks and TODOs

- Replace `YOUR-VERCEL-PROJECT` before using the live GitHub Pages site.
- Confirm Topper's expected bootstrap token payload with real sandbox credentials.
- Update the CORS allowlist in `api/topper/token.ts` if the public frontend domain changes.
- Keep `TOPPER_PRIVATE_JWK` only in Vercel environment variables and never commit it.
