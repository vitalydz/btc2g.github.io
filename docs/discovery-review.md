# Search discovery improvements — 2026-10-28

Base: 6661cc2274bb6c2e29f8c4036b27c7b33949712f. Release: 2026.10.28-discovery.1.

The homepage and purchase page lacked canonical URLs; the published checkout had no sitemap or robots.txt. Add self-referencing canonical URLs and a small sitemap of existing public pages, advertised through robots.txt. Do not invent last-modified dates or publish unpublished services. The contest URL describes preparation only. Premium remains untouched because its local content is separately modified by the owner.

Sources consulted:
- https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap
- https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls

These signals help discovery and URL consolidation; they do not guarantee indexing or audience growth. No analytics or external submission was enabled. XML, canonical consistency, local targets, HTTP status and published content are checked for this release. UI and layout are unchanged; no new Android verification is claimed.

Next: evaluate homepage usefulness and sharing, without repeating this metadata change. Search Console ownership is not verified; no search performance measurements are available.
