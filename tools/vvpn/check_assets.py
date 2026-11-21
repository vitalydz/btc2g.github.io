#!/usr/bin/env python3
"""Test discovered CSS/JS URLs from Russian Globalping probes.

The input is a prior VVPN HTTP report. Only publicly linked resources are checked.
This is still not a complete browser screenshot or interactive functional test.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from vvpn import globalping, write_report


def probe_assets(source: Path, city: str = "", limit: int = 6):
    report = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for page in report.get("results", []):
        if page.get("state") != "ok" or (city and page.get("city", "").casefold() != city.casefold()):
            continue
        assets = sorted(page.get("asset_candidates", []), key=lambda a: 0 if a.get("type") == "stylesheet" else 1)
        seen = set()
        selected = []
        for asset in assets:
            url = asset.get("url", "")
            if urlparse(url).scheme not in {"http", "https"} or url in seen:
                continue
            seen.add(url)
            selected.append(asset)
            if len(selected) >= limit:
                break
        if not selected:
            rows.append({"provider": "VVPN", "city": page.get("city"), "url": page.get("url"),
                         "state": "no_discovered_assets"})
        for asset in selected:
            for row in globalping(asset["url"], [page["city"]]):
                row["page_url"] = page["url"]
                row["asset_type"] = asset.get("type", "resource")
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Check page CSS/JS from remote Russian probes")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--city", default="")
    parser.add_argument("--max-per-site", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("vvpn-assets-report"))
    args = parser.parse_args()
    if not (1 <= args.max_per_site <= 30):
        parser.error("--max-per-site must be between 1 and 30")
    rows = probe_assets(args.source, args.city, args.max_per_site)
    write_report(rows, args.output)
    for r in rows:
        print(r.get("city"), r.get("asset_type", "page"), r.get("state"), r.get("code"), r.get("content_type", ""), r.get("url"))
    print("Assets checked:", len(rows))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
