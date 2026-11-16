#!/usr/bin/env python3
"""VVPN: independent Moscow HTTP and DNS probe diversity check.

Uses public Globalping community probes, NOT a full browser. Requests up to
two Moscow probes for six public websites, records each actual network and
every result without inventing a second provider when only one responds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from vvpn import (
    GLOBALPING, SITES, city_key, globalping_payload,
    parse_globalping_result, request_json, site_url,
)
from moscow_media import MediaParser, discover_html, verify_mime

CITY = "Moscow"


def measurement_payload(site: str, kind: str, probes: int = 2) -> dict:
    if kind not in ("http", "dns") or not (1 <= probes <= 5):
        raise ValueError("Expected kind http/dns and 1-5 probes")
    if kind == "http":
        payload = globalping_payload(site_url(site), CITY)
        payload.pop("limit", None)  # Per-location and global limits are mutually exclusive.
    else:
        payload = {"type": "dns", "target": site, "measurementOptions": {"query": {"type": "A"}},
                   "locations": [{"country": "RU", "city": CITY}]}
    payload["locations"][0]["limit"] = probes
    return payload


def classify_dns(result: dict) -> str:
    if result.get("status") in ("offline", "failed"):
        return "network_error"
    if result.get("status") != "finished":
        return "incomplete"
    if result.get("statusCode") != 0:
        return "dns_error"
    answers = result.get("answers") or []
    if not answers:
        return "dns_no_answers"
    return "ok" if any(isinstance(a, dict) and a.get("type") == "A" and a.get("value")
                       for a in answers) else "dns_no_a_record"


def analyze_measurement(site: str, kind: str, response: dict, expected: int = 2,
                        asset_kind: str | None = None) -> dict:
    rows = []
    for raw in response.get("results", []):
        probe = raw.get("probe") or {}
        if str(probe.get("country", "")).upper() != "RU" or city_key(str(probe.get("city", ""))) != city_key(CITY):
            continue
        result = raw.get("result") or {}
        if kind == "http":
            parsed = parse_globalping_result(raw, site_url(site))
            if asset_kind:
                parsed = verify_mime(parsed, asset_kind)
            state, code = parsed["state"], parsed.get("code")
        else:
            state, code = classify_dns(result), result.get("statusCode")
        rows.append({
            "city": probe.get("city"), "network": probe.get("network", ""),
            "asn": probe.get("asn", ""), "probe_tags": probe.get("tags", []),
            "state": state, "code": code,
            "resolved_ip": result.get("resolvedAddress"),
            "content_type": parsed.get("content_type") if kind == "http" else None,
            "resolver": result.get("resolver") if kind == "dns" else None,
            "answer_count": len(result.get("answers") or []) if kind == "dns" else None,
            "error_excerpt": str(result.get("rawOutput", ""))[:220] if state != "ok" else "",
        })
    networks = sorted({str(r["asn"] or r["network"]).strip() for r in rows if r["asn"] or r["network"]})
    observed_ok = bool(rows) and all(r["state"] == "ok" for r in rows)
    coverage_ok = (observed_ok and len(rows) >= expected and len(networks) >= 2
                   and response.get("status") == "finished")
    return {
        "site": site, "kind": kind, "asset_kind": asset_kind, "city": CITY,
        "requested_probes": expected, "returned_probes": len(rows),
        "independent_network_count": len(networks),
        "networks": networks,
        "all_observed_ok": observed_ok,
        "two_networks_confirmed": len(networks) >= 2,
        "full_coverage_ok": coverage_ok,
        "measurement_id": response.get("id"),
        "status": response.get("status"),
        "probes": rows,
    }


def query(site: str, kind: str, wait: int = 30,
          asset_kind: str | None = None) -> dict:
    try:
        started = request_json(GLOBALPING, payload=measurement_payload(site, kind))
        result_url = GLOBALPING + "/" + started["id"]
        deadline = time.monotonic() + wait
        while True:
            response = request_json(result_url)
            if response.get("status") != "in-progress" or time.monotonic() >= deadline:
                break
            time.sleep(1)
        report = analyze_measurement(site, kind, response, asset_kind=asset_kind)
        if response.get("status") == "in-progress":
            report["status"] = "pending"
            report["all_observed_ok"] = False
            report["full_coverage_ok"] = False
        return report
    except (RuntimeError, KeyError, ValueError, TypeError) as exc:
        return {"site": site, "kind": kind, "asset_kind": asset_kind,
                "city": CITY, "status": "provider_error",
                "requested_probes": 2, "returned_probes": 0,
                "all_observed_ok": False, "two_networks_confirmed": False,
                "full_coverage_ok": False,
                "error": str(exc)[:240], "probes": []}



def critical_assets(site: str) -> list[tuple[str, str]]:
    """Select one public HTML-linked CSS, JS and image URL for independent probes.

    HTML discovery uses the GitHub runner; only the selected asset requests
    originate from Moscow. This sample is not a full browser or all-resource test.
    """
    base = site_url(site)
    parser = MediaParser(base)
    parser.feed(discover_html(base))
    parser.finish()
    selected = []
    for kind, candidates in (
        ("stylesheet", parser.stylesheets),
        ("script", parser.scripts),
        ("image", parser.images),
    ):
        if not candidates:
            raise ValueError(f"No {kind} URL discovered for {site}")
        if kind == "image":
            url = next((u for u in candidates
                        if not any(t in u.lower() for t in
                                   ("favicon", "/icon", "social.svg", "/logo"))),
                       candidates[0])
        else:
            url = candidates[0]
        selected.append((kind, url))
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check Moscow HTTP/DNS using up to two Globalping probes per site")
    parser.add_argument("--output", type=Path, default=Path("vvpn-results/moscow-diversity"))
    parser.add_argument("--wait", type=int, default=30)
    parser.add_argument("--critical-assets", action="store_true",
                        help="Probe first HTML-linked CSS, JS and image from two Moscow networks")
    args = parser.parse_args(argv)
    if not 5 <= args.wait <= 90:
        parser.error("--wait must be 5-90 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in SITES:
        for kind in ("http", "dns"):
            data = query(site, kind, args.wait)
            rows.append(data)
            observed = [(r["network"], r["state"], r["code"]) for r in data.get("probes", [])]
            print(site, kind, data["status"], "probes", data["returned_probes"],
                  "two distinct ASNs/networks", data.get("two_networks_confirmed"),
                  "full coverage", data.get("full_coverage_ok"),
                  "observed", observed, "error", data.get("error", ""), flush=True)
        if args.critical_assets:
            try:
                selected = critical_assets(site)
            except Exception as exc:
                rows.append({"site": site, "kind": "asset_discovery", "city": CITY,
                             "asset_kind": None, "status": "discovery_error",
                             "requested_probes": 0, "returned_probes": 0,
                             "independent_network_count": 0, "all_observed_ok": False,
                             "full_coverage_ok": False, "two_networks_confirmed": False,
                             "probes": [], "error": str(exc)[:200]})
                print(site, "critical asset discovery failed", str(exc)[:200], flush=True)
                continue
            for asset_kind, asset_url in selected:
                data = query(asset_url, "http", args.wait, asset_kind=asset_kind)
                data["parent_site"] = site
                rows.append(data)
                print(site, asset_kind, "Moscow independent networks",
                      data["returned_probes"], "full coverage", data.get("full_coverage_ok"),
                      "codes", [p["code"] for p in data.get("probes", [])],
                      "error", data.get("error", ""), flush=True)
    report = {"tool": "VVPN", "measurement": "Moscow public-node comparison",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "meaning": "Community HTTP/DNS/selected asset probes, NOT browser rendering",
              "results": rows}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Moscow multi-probe HTTP / DNS", "",
             "A missing second probe is **not** proof of website failure; it is incomplete coverage.",
             "HTTP 200 and DNS NOERROR **do not** prove Chromium rendering from Moscow.", "",
             "| Site/asset | Kind | Probes | Distinct networks | Fully confirmed | Codes |",
             "|---|---|---:|---:|---|---|"]
    for r in rows:
        codes = ", ".join(str(n.get("code")) for n in r.get("probes", [])) or "no probes"
        label = str(r['site']).replace("|", "%7C")
        if r.get("asset_kind"):
            label = f"{r.get('parent_site', '')} — {r['asset_kind']}: {label}"
        lines.append(f"| {label} | {r['kind']} | {r['returned_probes']} | "
                     f"{r.get('independent_network_count', 0)} | {r.get('full_coverage_ok', False)} | {codes} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # An API outage is not a site outage, but it must not become green silently.
    return 0 if rows and all(x.get("full_coverage_ok", False) for x in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
