#!/usr/bin/env python3
"""VVPN Moscow follow-up: discriminate recurrent AS50867 TCP timeouts.

Probes a small, fixed set of public URLs from the *specific* Moscow AS that
previously timed out and a known-good comparison AS. This is HTTP testing,
not Chromium, not an end-user ISP test, and not a website configuration change.
Never infer that a request ran from an ASN merely because it was requested.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from vvpn import GLOBALPING, city_key, globalping_payload, parse_globalping_result, request_json
from moscow_media import verify_mime

CITY = "Moscow"
# HOSTKEY had partial TCP-connect timeouts in two independent VVPN runs;
# Timeweb was healthy for these same assets. These numbers are observed ASNs.
NETWORKS = {50867: "HOSTKEY", 9123: "Timeweb"}
# No POSTs, submissions, authentication, or user data.
TARGETS = [
    ("https://btc2g.com/", "html"),
    ("https://btc2g.com/assets/css/index.css?v=20261110-1", "stylesheet"),
    ("https://bmychance.com/storybook-fast.webp", "image"),
    ("https://bmychance.com/site.css?v=mobile-header-4px", "stylesheet"),
    ("https://findbbest.com/workbook.js", "script"),
]


def payload(url: str, asn: int) -> dict:
    """Globalping locations support explicit asn + city + country filters."""
    if asn not in NETWORKS:
        raise ValueError("Only the documented comparison ASNs are allowed")
    value = globalping_payload(url, CITY)
    value.pop("limit", None)
    value["locations"] = [{"country": "RU", "city": CITY, "asn": asn, "limit": 1}]
    return value


def classify(url: str, kind: str, asn: int, response: dict) -> dict:
    rows = response.get("results") or []
    valid = []
    for raw in rows:
        probe = raw.get("probe") or {}
        try:
            observed_asn = int(str(probe.get("asn", "")).removeprefix("AS"))
        except ValueError:
            continue
        if (str(probe.get("country", "")).upper() == "RU"
                and city_key(str(probe.get("city", ""))) == city_key(CITY)
                and observed_asn == asn):
            valid.append(raw)
    state = ("pending" if response.get("status") == "in-progress"
             else "no_probe" if not rows
             else "mismatched_probe" if not valid
             else "provider_incomplete" if response.get("status") != "finished"
             else None)
    row = {"url": url, "asset_kind": kind, "city": CITY, "requested_asn": asn,
           "requested_network": NETWORKS[asn], "provider_status": response.get("status"),
           "measurement_id": response.get("id"), "matching_probes": len(valid),
           "state": state or "unknown", "http_code": None}
    if state is not None:
        return row
    parsed = parse_globalping_result(valid[0], url)
    if kind in ("image", "stylesheet", "script"):
        parsed = verify_mime(parsed, kind)
    probe = valid[0].get("probe") or {}
    row.update({
        "state": parsed.get("state"),
        "http_code": parsed.get("code"),
        "content_type": parsed.get("content_type"),
        "resolved_ip": parsed.get("resolved_ip"),
        "observed_asn": probe.get("asn"),
        "observed_network": probe.get("network"),
        "diagnostic": parsed.get("detail", "")[:180] if parsed.get("state") != "ok" else "",
    })
    # A 204/206 or redirect is not sufficient to demonstrate full-body asset delivery.
    if row["state"] == "ok" and row["http_code"] != 200:
        row["state"] = "unexpected_http_status"
    return row


def query(url: str, kind: str, asn: int, wait: int) -> dict:
    try:
        created = request_json(GLOBALPING, payload=payload(url, asn))
        deadline = time.monotonic() + wait
        while True:
            response = request_json(GLOBALPING + "/" + created["id"])
            if response.get("status") != "in-progress" or time.monotonic() >= deadline:
                return classify(url, kind, asn, response)
            time.sleep(1)
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        return {"url": url, "asset_kind": kind, "city": CITY,
                "requested_asn": asn, "requested_network": NETWORKS[asn],
                "state": "provider_error", "http_code": None,
                "detail": str(exc)[:180]}


def main(argv=None):
    cli = argparse.ArgumentParser(description="Compare HOSTKEY/Timeweb routing of public VVPN assets from Moscow")
    cli.add_argument("--output", type=Path, default=Path("vvpn-results/moscow-route-followup"))
    cli.add_argument("--wait", type=int, default=30)
    args = cli.parse_args(argv)
    if not 5 <= args.wait <= 90:
        cli.error("wait must be 5..90 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for url, kind in TARGETS:
        for asn in NETWORKS:
            row = query(url, kind, asn, args.wait)
            rows.append(row)
            print("MOSCOW AS ROUTE", asn, url, row["state"], row.get("http_code"),
                  row.get("diagnostic", row.get("detail", "")), flush=True)
    report = {
        "tool": "VVPN", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Targeted Moscow AS50867 vs AS9123 HTTP GET; NOT a full browser",
        "results": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Moscow HOSTKEY vs Timeweb route follow-up", "",
          "This measures two specific datacenter ASNs. A failed TCP connection is not an HTTP error code.",
          "Missing / mismatched probes are coverage gaps, not website failures.", "",
          "| URL | Requested ASN | State | HTTP | Observed ASN/network |",
          "|---|---:|---|---:|---|"]
    for row in rows:
        md.append(f"| {row['url'].replace('|', '%7C')} | AS{row['requested_asn']} "
                  f"| {row['state']} | {row.get('http_code') or '—'} | "
                  f"{row.get('observed_asn', '—')} {row.get('observed_network', '')} |")
    (args.output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # An incomplete diagnostic must not silently be classified as proof of reachability.
    return 0 if all(x["state"] == "ok" for x in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
