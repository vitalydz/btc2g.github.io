# Search discovery improvements — 2026-10-28

Base: 6661cc2274bb6c2e29f8c4036b27c7b33949712f. Release: 2026.10.28-discovery.1.

The homepage and purchase page lacked canonical URLs; the published checkout had no sitemap or robots.txt. Add self-referencing canonical URLs and a small sitemap of existing public pages, advertised through robots.txt. Do not invent last-modified dates or publish unpublished services. The contest URL describes preparation only. Premium remains untouched because its local content is separately modified by the owner.

Sources consulted:
- https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap
- https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls

These signals help discovery and URL consolidation; they do not guarantee indexing or audience growth. No analytics or external submission was enabled. XML, canonical consistency, local targets, HTTP status and published content are checked for this release. UI and layout are unchanged; no new Android verification is claimed.

Next: evaluate homepage usefulness and sharing, without repeating this metadata change. Search Console ownership is not verified; no search performance measurements are available.

## Educational pages — discovery.2

Audited the existing Bitcoin vs Gold and Bitcoin Gold Ratio articles, both already linked from the homepage. Added their self-referencing canonical URLs and sitemap entries; content, links, layout and languages are unchanged. These existing explanatory pages are a more useful search entry point than adding duplicate material. Reviewed the interpretation/source pattern at https://www.longtermtrends.com/bitcoin-vs-gold/; no text or code copied.

Validation: all five sitemap URLs resolve to existing local pages with matching unique canonical URLs and return HTTP 200. XML and diff checks passed. No new browser/mobile checks are needed for head-only metadata; no Android check claimed. Next content opportunity: the ratio article formula should explain USD units, troy ounces and the futures proxy, consistent with the existing homepage calculator; implement only with source verification and responsive review.

## 2026.10.28-cleanup.1

Removed the homepage Google Analytics loader and configuration using the placeholder G-XXXXXXXXXX. Existing event calls already check typeof window.gtag before calling it. No analytics enabled, no layout changes. Calculator, ratio/inline JavaScript, freshness and refresh tests passed (4 suites). Article content remains a separate unpublished diff awaiting permitted visual preview. Daily Builder reviewed the supplied formula/example and confirmed the arithmetic; it did not inspect the local file or publish.
