# BTC2G — SAVE · EARN · PLAY delivery plan
Updated: 2026-10-28
Repository: vitalydz/btc2g.github.io
Branch: codex/btc2g-design-save-earn-play-20261028

## Purpose
Make BTC2G a useful, trustworthy and attractive multilingual platform with:
1. SAVE — explain and reduce the *total delivered BTC cost* of purchases and transfers.
2. EARN — disclose verified reward opportunities and actual cashback earned.
3. PLAY — free, moderated creative-AI competitions, with honest voting and published rules.

## Release sequence
### R1 — Quality, design and trust
- Review current responsive design using desktop and mobile viewport sizes. Improve navigation, readable graph, contrast, footer and consistency of the buying and premium pages.
- Replace statements about unfinished *features* with localized "Coming soon" (or equivalent), without specific release dates. Never suppress truthful price/data staleness or outage warnings.
- Add a **real** private visit statistics view in the footer only for authenticated admin. Server-side access control and persistence required; do not expose counts or keys in client assets. Gracefully omit feature if backend unavailable; never display fabricated traffic.
- Ensure the existing contest rules and dates stay accurate and distinguish demo from live.
- Add regression tests for changed behaviors.

### R2 — SAVE first functional release
- Default purchase example USD 50, user-selectable amounts and currencies.
- Display fee breakdown in fiat and percentage: quoted BTC received, effective rate, provider fee, possible spread, transfer/network fees, final amount in wallet, and quote timestamp.
- Compare Topper and MoonPay only when *both* have verifiable equivalent live quotes or display "Quote required". Check whether rate/fee APIs and use terms permit live comparison.
- Show $50 + $50 separately vs $100 combined as an *illustration only* using real comparable quotes, not assumed fixed fees. Waiting can change BTC market price.
- Never guess provider fees or claim Topper/MoonPay affiliate agreements. Use explicit affiliate disclosures once agreements exist.
- Avoid auto-purchasing, storing private wallet keys or promises of savings.

### R3 — EARN (non-custodial preview)
- "BTC2G Profit Passport" dashboard with separate balances: reported savings (estimated vs confirmed), pending cashback, confirmed sats, paid sats, BTC2G referral income.
- Show sample data marked DEMO in preview only. Real rewards require approved partner contract, secure backend ledger, anti-fraud, payout policy and verification.
- Avoid misleading descriptions such as guaranteed profit, APY, passive returns, or confirmed affiliate links without a contract.
- Before payouts, review local tax/regulatory, privacy and money-transmission obligations in relevant jurisdictions.
- Handle reversals/refunds, duplicate events and idempotency.

### R4 — PLAY
- Free AI art / character creation contest: participant-owned submissions, clear theme, no fake votes, transparent judging rubric, privacy/copyright and moderation controls.
- Distinguish AI-assisted judging from user voting. Protect against duplicate votes, spam, manipulation and improper submissions.
- Prize "12 months Premium" is **proposed**, not guaranteed until service fulfillment, eligibility, prize rules and regulatory review are complete.
- Reuse existing contest infrastructure rather than replacing it; do not expose private identities.

## Implementation constraints
- Work in small commits with clear names, tests and release identifiers; do not overwrite existing main changes.
- Inspect actual code before modifying. Backward compatibility with existing chart, translations, subscribe flow, and BUY navigation.
- Never commit secrets, private keys, passwords or payment credentials.
- Site front-end is GitHub Pages; Vercel serverless APIs deploy separately. Confirm which backend is actually configured before relying on it.
- Run Python unit tests and existing Node .cjs tests. Add tests for auth boundaries, quote arithmetic, language fallback and mobile.
- Create pull requests for reviewed merges; verify GitHub Pages deployment and real website behavior before claiming success.
- Keep a changelog with commit SHA, tests, and blockers.

## Immediate Codex Cloud task
Start with R1; inspect the latest main, implement a small safe release, run tests, push branch and open a pull request. Do not stop after planning. Report exactly which parts are implemented vs awaiting external configuration.
