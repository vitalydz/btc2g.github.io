#!/usr/bin/env python3
"""VVPN: Moscow *eyeball-network* HTTPS probes, closer to real home/office ISPs.

A Globalping eyeball probe is an HTTP client, not Chromium. It does NOT
prove pixel-perfect rendering or that all Moscow households can reach sites.
No-probe is a coverage gap, not a website outage. No secrets or forms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from vvpn import GLOBALPING, SITES, city_key, parse_globalping_result, request_json
from moscow_diversity import measurement_payload

TARGET_CITY = "Moscow"
TAG = "eyeball-network"


def eyeball_payload(site: str) -> dict:
    payload = measurement_payload(site, "http", probes=1)
    payload["locations"][0]["tags"] = [TAG]
    return payload


def classify(site: str, response: dict) -> dict:
    probes = []
    foreign_or_unverified = 0
    for raw in response.get("results", []):
        meta = raw.get("probe") or {}
        tags = meta.get("tags") or []
        if (str(meta.get("country", "")).upper() != "RU"
                or city_key(str(meta.get("city", ""))) != city_key(TARGET_CITY)
                or TAG not in tags):
            foreign_or_unverified += 1
            continue
        parsed = parse_globalping_result(raw, "https://" + site)
        probes.append({
            "network": meta.get("network"), "asn": meta.get("asn"),
            "city": meta.get("city"), "country": meta.get("country"),
            "probe_tags": tags, "state": parsed["state"], "http_code": parsed.get("code"),
            "content_type": parsed.get("content_type"),
        })
    if not probes:
        state = "unverified_location_or_type" if foreign_or_unverified else "no_eyeball_probe"
    elif all(p["state"] == "ok" and p["http_code"] == 200 for p in probes):
        state = "http_200"
    else:
        state = "observed_http_failure"
    if response.get("status") == "in-progress":
        state = "pending"
    return {
        "site": site, "city": TARGET_CITY, "requested_probe_tag": TAG,
        "measurement_id": response.get("id"), "provider_status": response.get("status"),
        "state": state, "returned_confirmed_eyeball_probes": len(probes),
        "discarded_unverified_probes": foreign_or_unverified,
        "probes": probes,
    }


def query(site: str, wait: int = 30) -> dict:
    try:
        start = request_json(GLOBALPING, payload=eyeball_payload(site))
        end = time.monotonic() + wait
        while True:
            response = request_json(GLOBALPING + "/" + start["id"])
            if response.get("status") != "in-progress" or time.monotonic() >= end:
                break
            time.sleep(1)
        return classify(site, response)
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        return {"site": site, "city": TARGET_CITY, "requested_probe_tag": TAG,
                "state": "provider_error", "error": str(exc)[:220], "probes": [],
                "returned_confirmed_eyeball_probes": 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description="Check the six websites via Moscow consumer-network probes")
    cli.add_argument("--output", type=Path, default=Path("vvpn-results/moscow-eyeball"))
    cli.add_argument("--wait", type=int, default=30)
    args = cli.parse_args(argv)
    if not 5 <= args.wait <= 90:
        cli.error("--wait must be 5-90 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in SITES:
        row = query(site, args.wait)
        rows.append(row)
        print(site, row["state"], "MOSCOW EYEBALL", [(p["network"], p["http_code"]) for p in row.get("probes", [])],
              "error", row.get("error", ""), flush=True)

    report = {
        "tool": "VVPN", "created_utc": datetime.now(timezone.utc).isoformat(),
        "meaning": "Moscow Globalping HTTP checks targeted to eyeball-network (end-user ISPs), NOT full browser",
        "results": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Moscow end-user ISP probe (Globalping eyeball-network)", "",
        "An absent eyeball probe or provider error is **not** evidence that the website failed.",
        "HTTP 200 from such a node is **not** a Russian Chromium browser rendering check.", "",
        "| Website | State | Confirmed eyeball probes | Network | HTTP |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        summary.append(f"| {row['site']} | {row['state']} | {row['returned_confirmed_eyeball_probes']} | "
                       f"{', '.join(str(p.get('network', '')) for p in row['probes']) or '—'} | "
                       f"{', '.join(str(p.get('http_code')) for p in row['probes']) or '—'} |")
    (args.output / "report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    # A *real observed failure* fails CI. Missing community probes are a separate
    # outcome, not a claimed outage nor silently reported as successful coverage.
    return 2 if any(r["state"] == "observed_http_failure" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
