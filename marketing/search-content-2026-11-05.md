# Search-content follow-up — 2026-11-05

Main BTC2G was safely fast-forwarded from 6b81953 to e1f0b0b after af47a57 and the scheduled data refresh. Before/after checks confirmed all 85 original dirty/staged paths retained the same status and file hashes; staged binary patch was unchanged. Backups and hashes: `C:/Users/vitly/PycharmProjects/MPA_04/reports/btc2g-sync-2026-11-05/`. No stash, reset or overwrite of others' work. Further edits use the isolated `btc2g-growth` checkout.

## Evidence and inventory

Authenticated Search Console, web search, selected 3 months, chart coverage 31.10–03.11.2026: BTC2G 5 impressions/1 click. Visible query `биткоин голд прогноз`: 1 impression/0 clicks. Root page: 5/1; privacy and terms each 1/0. Hidden queries cannot be reconstructed and page totals need not sum to property totals. One ambiguous query does not establish large demand or registration conversion. BTG name checked against the project's source `https://btgofficial.org/`.

85 sitemap HTML URLs: dashboard/scenario/save/premium each in 19 variants, three research articles, buy, contest, research, support, terms and privacy. Existing research, ratio explanations and tools are the useful content base; no duplicate or mass-generated landing pages added. Existing metadata was not rewritten without a defect.

Production D1 stats at 05.11.2026 10:59:08 UTC (12:59:08 Israel): 0 active ordinary email-confirmed accounts total and today. Separately 1 daily visitor estimate/2 views; PostgreSQL audience reports 0 confirmed subscriptions. None of these categories is substituted for another. VitdArt D1 at 10:59:04 UTC: 0 confirmed ordinary accounts, 2 visitor estimates/13 views, newsletter source unavailable/null. Snapshot and complete sitemap URL inventory: `C:/Users/vitly/PycharmProjects/MPA_04/reports/search-content-2026-11-05/inventory-and-stats.json`.

## Change and limits

Existing forecast article now clarifies BTC versus gold versus the separate Bitcoin Gold (BTG) name, gives a reproducible 10%/5% relative-price example (about 4.76%), and links the existing scenario calculator and dashboard market-date/source context. Russian client copy added without claiming new server-translated article URLs. This improves clarity; search benefit remains a hypothesis. No price target, financial return or registration promise. Cache version updated for the modified language script.

Earlier audit page counts for VitdArt were corrected: 494 page URLs = 95 static + 380 artwork + 19 artist variants; 380 image references are counted separately. Sitemap discovered count is not indexed count.

186 Node tests passed after content edits. Previous 27 Python tests passed in af47a57; they were not repeated for this text-only update. No production forms, traffic events, emails, indexing requests or social posts sent by these checks.

VitdArt help was prepared separately at `C:/Users/vitly/PycharmProjects/MPA_04/vitdart-search-content`, commit f0f9bd5, but NOT published: live 0.20.0 is newer than canonical main 0.17.3. Its existing branch includes outreach features; this task did not merge or redeploy them. Production-source reconciliation is required before that draft can be released without rollback. 147 Node + 13 Python tests passed for that draft.


## Later release outcome in this same task

The VitdArt baseline mismatch was resolved for this patch without merging outreach work into main: the isolated change was rebased onto clean production-source commit 690284b (0.20.0), preserving existing tests and code, then released as 0.20.1 from b07998f. Worker version 54e15bf7-1440-4f96-9c84-b6b4da7579f9. 185 Node + 16 Python tests passed on that actual baseline. This supersedes the draft-only status above; canonical main remains 0.17.3 and is unsafe as a production release base until reconciled. Dedicated branch codex/search-content stores the release.

BTC2G content commit b7d1e38 is published: forecast-scope section and cache version 20261105-2 verified in production. Read-only checks and account snapshot: `C:/Users/vitly/PycharmProjects/MPA_04/reports/search-content-2026-11-05/release.json`. Neither release establishes new registrations or improved search performance. No external social messages or posts sent.
