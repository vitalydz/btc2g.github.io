#!/usr/bin/env python3
"""VVPN: remote Russian HTTP probes and optional real browser proxy checks.

Standard-library HTTP checks use Check-Host and Globalping. The browser mode
requires Playwright and an independently provided Russian proxy/VPN endpoint.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

SITES = ["bmychance.com", "btc2g.com", "bit2g.com", "vitdart.com", "findbbest.com", "aikickai.com"]
CITIES = ["Moscow", "Saint Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan", "Chelyabinsk", "Krasnoyarsk", "Vladivostok"]
CITY_ALIASES = {"st petersburg": "Saint Petersburg", "saint petersburg": "Saint Petersburg", "sankt peterburg": "Saint Petersburg", "ekaterinburg": "Yekaterinburg"}
CHECK_HOST = "https://check-host.net"
GLOBALPING = "https://api.globalping.io/v1/measurements"


def city_key(city: str) -> str:
    key = re.sub(r"[^a-z]+", " ", city.casefold()).strip()
    return CITY_ALIASES.get(key, city).casefold()


def site_url(site: str) -> str:
    value = site if "://" in site else "https://" + site
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an HTTP(S) website URL or hostname")
    return value


def request_json(url: str, *, params: dict | None = None, payload: dict | None = None, timeout: int = 15) -> dict:
    if params:
        url += "?" + urlencode(params, doseq=True)
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers={"Accept": "application/json", "User-Agent": "VVPN/1.0 (site diagnostics)", **({"Content-Type": "application/json"} if data else {})})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read(400).decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Provider unreachable: {exc}") from exc


def categorize_http(code: int | None, finished: bool = True) -> str:
    if not finished:
        return "pending"
    if code is None:
        return "network_error"
    if 200 <= code < 300:
        return "ok"
    if 300 <= code < 400:
        return "redirect"
    return "http_error"


def parse_checkhost_result(raw: object) -> dict:
    if raw is None:
        return {"state": "pending", "code": None}
    record = raw[0] if isinstance(raw, list) and raw else None
    if not isinstance(record, list) or len(record) < 4:
        return {"state": "network_error", "code": None, "detail": str(raw)[:160]}
    code_str = record[3]
    code = int(str(code_str)) if code_str is not None and str(code_str).isdigit() else None
    return {"state": categorize_http(code), "code": code, "seconds": record[1], "detail": record[2]}


def checkhost(url: str, cities: list[str], wait: int = 18) -> list[dict]:
    """Only uses nodes whose API-reported location is Russia, never guessed IPs."""
    nodes = request_json(f"{CHECK_HOST}/nodes/hosts").get("nodes", {})
    available = {}
    for node, props in nodes.items():
        loc = props.get("location") or []
        if len(loc) >= 3 and str(loc[0]).lower() == "ru":
            available.setdefault(city_key(loc[2]), []).append(node)
    out = []
    for city in cities:
        node_ids = available.get(city_key(city), [])
        if not node_ids:
            out.append({"provider": "Check-Host", "city": city, "url": url, "state": "no_probe"})
            continue
        for node in node_ids[:2]:
            item = {"provider": "Check-Host", "city": city, "node": node, "url": url}
            try:
                started = request_json(f"{CHECK_HOST}/check-http", params={"host": url, "node": node})
                request_id = started["request_id"]
                item["report_url"] = started.get("permanent_link")
                deadline = time.monotonic() + wait
                while True:
                    response = request_json(f"{CHECK_HOST}/check-result/{request_id}")
                    outcome = parse_checkhost_result(response.get(node))
                    if outcome["state"] != "pending" or time.monotonic() >= deadline:
                        item.update(outcome)
                        break
                    time.sleep(2)
            except (RuntimeError, KeyError, ValueError) as exc:
                item.update({"state": "provider_error", "detail": str(exc)[:240]})
            out.append(item)
    return out


class PageAssetsParser(HTMLParser):
    """Extract CSS and JS URLs from the remote HTTP response body (up to 10 KiB)."""

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.assets: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        kind = None
        target = None
        if tag == "script" and values.get("src"):
            kind, target = "script", values["src"]
        if tag == "link" and values.get("href"):
            rel = (values.get("rel") or "").casefold()
            if "stylesheet" in rel:
                kind, target = "stylesheet", values["href"]
            elif "preload" in rel or "modulepreload" in rel:
                kind, target = values.get("as") or "preload", values["href"]
        if not target:
            return
        url = urljoin(self.base, target)
        if urlparse(url).scheme not in ("https", "http"):
            return
        if not any(r["url"] == url for r in self.assets):
            self.assets.append({"type": kind or "resource", "url": url})


def globalping_payload(url: str, city: str) -> dict:
    """Globalping takes a hostname target, NOT a full URL."""
    parsed = urlparse(site_url(url))
    if not parsed.hostname:
        raise ValueError("Missing hostname")
    opts = {
        "protocol": "HTTPS" if parsed.scheme == "https" else "HTTP",
        "request": {"method": "GET", "path": parsed.path or "/"},
    }
    if parsed.query:
        opts["request"]["query"] = parsed.query
    if parsed.port:
        opts["port"] = parsed.port
    return {"type": "http", "target": parsed.hostname,
            "locations": [{"country": "RU", "city": city}],
            "limit": 1, "measurementOptions": opts}


def parse_globalping_result(record: dict, requested_url: str = "") -> dict:
    result = record.get("result", {})
    if result.get("status") in ("offline", "failed"):
        return {"state": "network_error", "detail": result.get("rawOutput", "probe offline")[:240]}
    status = result.get("statusCode", result.get("statusCodeValue"))
    if status is None:
        status = (result.get("response") or {}).get("statusCode")
    try:
        code = int(status) if status is not None else None
    except (ValueError, TypeError):
        code = None
    out = {"state": categorize_http(code), "code": code,
           "detail": str(result.get("rawOutput", ""))[:240]}
    if isinstance(result.get("rawBody"), str) and requested_url:
        parser = PageAssetsParser(requested_url)
        parser.feed(result["rawBody"])
        out["asset_candidates"] = parser.assets[:30]
        out["html_preview_truncated"] = bool(result.get("truncated"))
    headers = result.get("headers") or {}
    content_type = str(next((v for k, v in headers.items() if k.lower() == "content-type"), ""))
    if content_type:
        out["content_type"] = content_type
    path = urlparse(requested_url).path.lower()
    if code == 200 and content_type:
        if path.endswith(".css") and not content_type.lower().startswith("text/css"):
            out["state"] = "invalid_mime"
        if path.endswith(".js") and "javascript" not in content_type.lower():
            out["state"] = "invalid_mime"
    if result.get("resolvedAddress"):
        out["resolved_ip"] = result["resolvedAddress"]
    return out


def globalping(url: str, cities: list[str], wait: int = 25) -> list[dict]:
    out = []
    for city in cities:
        item = {"provider": "Globalping", "city": city, "url": url}
        try:
            start = request_json(GLOBALPING, payload=globalping_payload(url, city))
            measurement_id = start["id"]
            item["measurement_id"] = measurement_id
            deadline = time.monotonic() + wait
            while True:
                result = request_json(f"{GLOBALPING}/{measurement_id}")
                if result.get("status") != "in-progress" or time.monotonic() >= deadline:
                    break
                time.sleep(2)
            records = result.get("results") or []
            valid = [r for r in records if str(r.get("probe", {}).get("country", "")).upper() == "RU" and city_key(r.get("probe", {}).get("city", "")) == city_key(city)]
            if not valid:
                item.update({"state": "no_probe" if not records else "location_mismatch"})
            else:
                item["probe"] = valid[0]["probe"].get("network", "")
                item.update(parse_globalping_result(valid[0], url))
            if result.get("status") == "in-progress":
                item["state"] = "pending"
        except RuntimeError as exc:
            msg = str(exc)
            item.update({"state": "no_probe" if ("no_probes_found" in msg or "not enough probes" in msg.lower()) else "provider_error", "detail": msg[:240]})
        except (KeyError, ValueError) as exc:
            item.update({"state": "provider_error", "detail": str(exc)[:240]})
        out.append(item)
    return out


async def browser_check(urls: list[str], proxy_server: str, output: Path, city: str, timeout: int) -> list[dict]:
    """Real CSS/JS screenshot test; traffic exits through user-provided proxy."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Install Playwright: pip install playwright && python -m playwright install chromium") from exc
    if not proxy_server or not proxy_server.startswith(("socks5://", "http://", "https://")):
        raise ValueError("--proxy-server must be a SOCKS5 or HTTP(S) proxy URL")
    proxy = {"server": proxy_server}
    if os.environ.get("VVPN_PROXY_USER"):
        proxy["username"] = os.environ["VVPN_PROXY_USER"]
        proxy["password"] = os.environ.get("VVPN_PROXY_PASSWORD", "")
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, proxy=proxy)
        try:
            context = await browser.new_context(ignore_https_errors=False, viewport={"width": 1365, "height": 850})
            for ix, url in enumerate(urls):
                page = await context.new_page()
                errors = []
                resources = []
                page.on("requestfailed", lambda req: errors.append({"url": req.url, "error": req.failure, "type": req.resource_type}))
                page.on("response", lambda resp: resources.append({"url": resp.url, "status": resp.status, "type": resp.request.resource_type}) if resp.request.resource_type in ("stylesheet", "script", "font") else None)
                item = {"provider": "Playwright proxy", "city": city, "url": url, "state": "network_error"}
                try:
                    resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                    await page.wait_for_timeout(3500)
                    item["code"] = resp.status if resp else None
                    item["final_url"] = page.url
                    item["title"] = await page.title()
                    item["stylesheets"] = await page.evaluate("document.styleSheets.length")
                    item["resources"] = resources
                    item["failures"] = errors
                    item["state"] = categorize_http(item["code"])
                    if any(r["status"] >= 400 for r in resources if r["type"] == "stylesheet") or any(e["type"] == "stylesheet" for e in errors):
                        item["state"] = "css_failure"
                    screenshot = output / f"browser_{ix}_{urlparse(url).hostname}.png"
                    await page.screenshot(path=str(screenshot), full_page=True)
                    item["screenshot"] = str(screenshot)
                except Exception as exc:
                    item["detail"] = str(exc)[:280]
                finally:
                    await page.close()
                results.append(item)
        finally:
            await browser.close()
    return results


def write_report(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"tool": "VVPN", "created_utc": datetime.now(timezone.utc).isoformat(), "results": rows}
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# VVPN Russia site test", "", "HTTP 200/3xx proves one endpoint responded; it does NOT prove browser CSS/JS work.", "", "| Provider | City | Website | State | HTTP |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "—")).replace("|", "\\|") for k in ("provider", "city", "url", "state", "code")) + " |")
    lines += ["", "`no_probe` means the provider has no usable node in the selected city, not a block.", "`provider_error` means a provider/API issue, not necessarily site downtime."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VVPN: check sites from Russian HTTP probes and optional browser proxy")
    parser.add_argument("sites", nargs="*", help="Site hostnames/URLs; default: bmychance.com")
    parser.add_argument("--all", action="store_true", help="Check all six project sites")
    parser.add_argument("--cities", default=",".join(CITIES), help="Comma-separated English city names")
    parser.add_argument("--backend", choices=("checkhost", "globalping", "both", "browser"), default="checkhost")
    parser.add_argument("--proxy-server", default=os.environ.get("VVPN_PROXY_SERVER"), help="Browser proxy URL; no proxy credentials in argument")
    parser.add_argument("--proxy-city", default="Unverified proxy city", help="Label only; cannot verify proxy geolocation")
    parser.add_argument("--output", default="vvpn-report", help="Report folder")
    parser.add_argument("--timeout", type=int, default=25)
    args = parser.parse_args(argv)
    if args.all and args.sites:
        parser.error("Choose --all or explicit sites")
    sites = SITES if args.all else (args.sites or ["bmychance.com"])
    try:
        urls = [site_url(s) for s in sites]
    except ValueError as exc:
        parser.error(str(exc))
    cities = [s.strip() for s in args.cities.split(",") if s.strip()]
    if not cities:
        parser.error("At least one city required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.backend == "browser":
        if not args.proxy_server:
            parser.error("Browser mode requires --proxy-server (Russian proxy/VPN endpoint)")
        try:
            rows = asyncio.run(browser_check(urls, args.proxy_server, output, args.proxy_city, args.timeout))
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        for url in urls:
            for provider, fn in (("checkhost", checkhost), ("globalping", globalping)):
                if args.backend not in ("both", provider):
                    continue
                try:
                    rows.extend(fn(url, cities, args.timeout))
                except RuntimeError as exc:
                    rows.extend({"provider": provider, "city": city, "url": url, "state": "provider_error", "detail": str(exc)[:240]} for city in cities)
    write_report(rows, output)
    for r in rows:
        print(f"{r['provider']:17} {r['city']:19} {r['url']:32} {r['state']:17} {r.get('code', '—')}")
    print(f"\nReports: {output / 'report.md'} and {output / 'report.json'}")
    return 0 if any(r["state"] in ("ok", "redirect") for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())