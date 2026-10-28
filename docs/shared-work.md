# Shared BTC2G development

Owner instruction, 28 October 2026: use one shared branch and continue together.

Working branch: `codex/btc2g-shared`. Initial base: published `main` at `94d61a357076dfab71466d4c80cd1652d1a3c4f2` (2026.10.28-save.2).

The shared branch includes the agreed SAVE/EARN/PLAY plan. Older services-launch, Topper review, cloud snapshots and local Premium edits are preserved separately; they are not blanket-merged or deleted.

## Coordination

- Cloud Codex chat “Подключиться к vitalydz/btc2g.github” owns the pending R1 homepage/buy/platform styling changes. Reapply its reviewed diff onto this branch, preserving the published SAVE calculator, homepage SAVE card, ratio article and analytics removal. Do not replace current files with an old source snapshot.
- Local Codex owns SAVE localization (`save.html`, `assets/js/save.js`, SAVE-specific tests) and integration, browser checks, main publication and secondary-copy synchronization. Coordinate any overlapping files before editing them.
- Daily Builder provides review; it does not independently publish.
- Fetch the shared branch before work and again before push. Push normal fast-forward commits only. If another writer has advanced it, preserve local work and reconcile deliberately; never force-push. This is a coordination agreement, not an automated lock.
- Only local Codex integrates tested shared commits into `main`, verifies Pages and the live site, and synchronizes the secondary checkout while preserving its user diff. Do not publish unfinished shared work.
- Keep each handoff explicit: base SHA, changed files, actual tests, blockers, next file ownership. A sent message is not proof that the other session has acknowledged it.

## Preserved boundaries

Forecast/signal PNG and JSON, calculator, provider URLs, Premium user changes and intentional PNG deletions must be preserved. Public contest, payments, analytics, partner commissions, reward payouts and external outreach remain outside this integration. No credentials in the repository. Development/testing stays in MPA_04 on the local PC; do not claim remote Win10 tests without execution.

Current owner deadline: 22:00 Asia/Jerusalem on 28 October 2026. Stop project work then; no automatic continuation tomorrow.
