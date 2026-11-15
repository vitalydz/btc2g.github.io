#!/usr/bin/env python3
"""VVPN: Moscow image/script HTTP probes and independent browser smoke tests.

Only Globalping probes are in Moscow. Chromium runs on GitHub Actions, outside
Russia, and NEVER serves as proof of browser behavior from a Moscow IP.
No logins, forms, purchases, or user-provided credentials are used.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse, parse_qsl
from urllib.request import Request, urlopen

from vvpn import SITES, globalping, site_url, write_report

CSS_URL = re.compile(r"url\(\s*['\"]?([^'\"\)]+)", re.I)
DANGEROUS_QUERY_KEYS = {"token", "secret", "auth", "key", "signature", "sig", "access_token", "password", "jwt", "credential"}


def safe_asset(base, value):
    if not value or str(value).startswith(("data:", "blob:", "#", "javascript:")):
        return None
    url = urljoin(base, str(value).strip())
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        return None
    if any(k.lower() in DANGEROUS_QUERY_KEYS for k, _ in parse_qsl(parsed.query)):
        return None
    return url


def srcset_list(value):
    return [part.strip().split()[0] for part in (value or "").split(",") if part.strip()]


class MediaParser(HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.images, self.scripts = [], []
        self.in_style, self.styles = False, []

    def add(self, kind, candidate):
        target = self.images if kind == "image" else self.scripts
        url = safe_asset(self.base, candidate)
        if url and url not in target:
            target.append(url)

    def handle_starttag(self, tag, values):
        a = dict(values)
        if tag == "style":
            self.in_style = True
        if tag == "script":
            self.add("script", a.get("src"))
        if tag == "img":
            for field in ("src", "data-src", "data-lazy-src"):
                self.add("image", a.get(field))
        if tag in ("img", "source"):
            for field in ("srcset", "data-srcset"):
                for val in srcset_list(a.get(field)):
                    self.add("image", val)
        if tag == "link":
            if "icon" in (a.get("rel") or "").lower() or (a.get("as") or "").lower() == "image":
                self.add("image", a.get("href"))
        if tag == "meta" and (a.get("property") or a.get("name") or "").lower() in ("og:image", "twitter:image"):
            self.add("image", a.get("content"))
        if a.get("style"):
            self.styles.append(a["style"])

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.styles.append(data)

    def finish(self):
        for stylesheet in self.styles:
            for url in CSS_URL.findall(stylesheet):
                self.add("image", url)


def discover_html(url):
    request = Request(url, headers={"User-Agent": "VVPN/1.2 public media diagnostic", "Accept": "text/html"})
    with urlopen(request, timeout=20) as response:
        typ = response.headers.get("Content-Type", "")
        if "text/html" not in typ:
            raise RuntimeError("Page is not HTML: " + typ)
        return response.read(2_000_000).decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def verify_mime(row, kind):
    row["asset_kind"] = kind
    if row.get("state") == "ok" and row.get("code") == 200:
        typ = str(row.get("content_type", "")).lower()
        if kind == "image" and typ and not typ.startswith("image/"):
            row["state"] = "invalid_image_mime"
        if kind == "script" and typ and "javascript" not in typ:
            row["state"] = "invalid_script_mime"
    return row


def remote_moscow(output, images=7, scripts=6, browser_assets=None):
    output.mkdir(parents=True, exist_ok=True)
    by_site = {r["site"]: r for r in (browser_assets or [])}
    rows, findings = [], []
    for site in SITES:
        url = site_url(site)
        info = {"site": site, "city": "Moscow",
                "discovery_location": "GitHub hosted runner (NOT Moscow)"}
        try:
            html = discover_html(url)
            parser = MediaParser(url)
            parser.feed(html)
            parser.finish()
            dynamic = by_site.get(site, {})
            for src in dynamic.get("images", []):
                parser.add("image", src)
            for src in dynamic.get("scripts", []):
                parser.add("script", src)
            info["discovered_images"] = len(parser.images)
            info["discovered_scripts"] = len(parser.scripts)
            info["image_candidates"] = parser.images[:images]
            info["script_candidates"] = parser.scripts[:scripts]
            if not parser.images:
                info["image_warning"] = "No static image URL found; browser-generated images may exist"
            for kind, urls, cap in (("script", parser.scripts, scripts), ("image", parser.images, images)):
                for asset in urls[:cap]:
                    for result in globalping(asset, ["Moscow"], wait=30):
                        result["page_url"] = url
                        rows.append(verify_mime(result, kind))
                        print("MOSCOW", site, kind, result.get("state"),
                              result.get("code"), result.get("content_type", ""), asset, flush=True)
        except Exception as exc:
            info["discovery_error"] = str(exc)[:220]
            rows.append({"provider": "VVPN-discovery", "city": "Moscow", "url": url,
                         "state": "discovery_error", "detail": info["discovery_error"]})
        findings.append(info)
        print("DISCOVERY", json.dumps(info, ensure_ascii=False), flush=True)
    write_report(rows, output)
    (output / "discovery.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = Counter(r["state"] for r in rows)
    print("MOSCOW SUMMARY", dict(counts), "probes", len(rows), flush=True)
    return 0 if rows and all(r.get("state") == "ok" for r in rows) else 1


async def browser_smoke(output, timeout=25):
    """Render sites in Chromium on the GitHub runner (not a Moscow network)."""
    from playwright.async_api import async_playwright
    output.mkdir(parents=True, exist_ok=True)
    rows, discovered = [], []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            for site in SITES:
                url = site_url(site)
                context = await browser.new_context(viewport={"width": 1365, "height": 900}, service_workers="block")
                page = await context.new_page()
                found = {"site": site, "images": [], "scripts": []}
                failed, js_errors, console_errors, resources = [], [], [], []

                def on_request(request):
                    if request.resource_type in ("script", "image", "stylesheet"):
                        resources.append({"type": request.resource_type, "url": request.url, "status": None})
                    if request.resource_type in ("script", "image") and safe_asset(url, request.url):
                        bucket = "scripts" if request.resource_type == "script" else "images"
                        if request.url not in found[bucket]:
                            found[bucket].append(request.url)

                def on_response(response):
                    for item in reversed(resources):
                        if item["url"] == response.url and item["status"] is None:
                            item["status"] = response.status
                            break

                page.on("request", on_request)
                page.on("response", on_response)
                page.on("requestfailed", lambda request: failed.append({
                    "type": request.resource_type, "url": request.url, "error": str(request.failure)[:140]}))
                page.on("pageerror", lambda error: js_errors.append(str(error)[:200]))
                page.on("console", lambda message: console_errors.append(message.text[:200]) if message.type == "error" else None)
                report = {"site": site, "browser_exit": "GitHub hosted runner (NOT Moscow)", "state": "browser_error"}
                try:
                    main_response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                    await page.wait_for_timeout(3000)
                    image_nodes = await page.evaluate("""() => [...document.images].slice(0,150).map(img => ({
                        src: img.currentSrc || img.src, complete: img.complete,
                        width: img.naturalWidth, height: img.naturalHeight
                    }))""")
                    broken = [img for img in image_nodes if img["complete"] and img["width"] == 0]
                    for img in image_nodes:
                        candidate = safe_asset(url, img.get("src"))
                        if candidate and candidate not in found["images"]:
                            found["images"].append(candidate)
                    report.update({
                        "http": main_response.status if main_response else None,
                        "title": await page.title(),
                        "css": await page.evaluate("""() => ({
                            stylesheet_count: document.styleSheets.length,
                            body_display: getComputedStyle(document.body).display,
                            body_background: getComputedStyle(document.body).backgroundColor
                        })"""),
                        "image_count": len(image_nodes),
                        "broken_images": broken[:20],
                        "broken_image_count": len(broken),
                    })
                    screenshot = output / (site + ".png")
                    await page.screenshot(path=str(screenshot), full_page=True, animations="disabled")
                    report["screenshot"] = str(screenshot)
                    report["state"] = "loaded" if report["http"] and 200 <= report["http"] < 300 else "http_error"
                    if broken or js_errors:
                        report["state"] = "loaded_with_errors"
                except Exception as exc:
                    report["error"] = str(exc)[:240]
                report["failed_requests"] = failed[:40]
                report["javascript_exceptions"] = js_errors[:20]
                report["console_errors"] = console_errors[:20]
                report["resource_responses"] = resources[:100]
                rows.append(report)
                discovered.append(found)
                print("BROWSER US", site, report["state"], "HTTP", report.get("http"),
                      "images", report.get("image_count"), "broken", report.get("broken_image_count"),
                      "JS exceptions", len(js_errors), "failed requests", len(failed), flush=True)
                await context.close()
        finally:
            await browser.close()
    (output / "report.json").write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "discovered_assets.json").write_text(json.dumps(discovered, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["state"] == "loaded" for r in rows) else 1


def main():
    cli = argparse.ArgumentParser(description="VVPN Moscow image/script reachability and US browser execution smoke")
    cli.add_argument("--mode", choices=("moscow", "browser"), required=True)
    cli.add_argument("--images", type=int, default=7)
    cli.add_argument("--scripts", type=int, default=6)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--browser-assets", type=Path)
    args = cli.parse_args()
    if not 0 <= args.images <= 15 or not 0 <= args.scripts <= 15:
        cli.error("Caps must be 0-15")
    if args.mode == "browser":
        return asyncio.run(browser_smoke(args.output))
    assets = None
    if args.browser_assets and args.browser_assets.is_file():
        assets = json.loads(args.browser_assets.read_text(encoding="utf-8"))
    return remote_moscow(args.output, args.images, args.scripts, assets)


if __name__ == "__main__":
    raise SystemExit(main())
