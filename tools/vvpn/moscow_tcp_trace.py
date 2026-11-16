#!/usr/bin/env python3
"""VVPN: bounded TCP/443 traceroute from two verified Moscow ASNs.

Use only after repeated TCP-connect timeouts in Moscow's AS50867. Traceroute
can identify route asymmetry; timeouts or missing hops are NOT proof of a
website ban or broken page. Does not test Chromium, JS, CSS or browser egress.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from vvpn import GLOBALPING, city_key, request_json

ASNS = (50867, 9123)  # HOSTKEY and Timeweb, respectively
TARGETS = ("btc2g.com", "findbbest.com")
CITY = "Moscow"


def payload(hostname: str, asn: int) -> dict:
    if hostname not in TARGETS or asn not in ASNS:
        raise ValueError("Only approved VVPN domains and observed ASNs are allowed")
    return {"type": "traceroute", "target": hostname,
            "measurementOptions": {"protocol": "TCP", "port": 443, "ipVersion": 4},
            "locations": [{"country": "RU", "city": CITY, "asn": asn, "limit": 1}]}


def classify(hostname: str, asn: int, result: dict) -> dict:
    records = result.get("results") or []
    matching = []
    for raw in records:
        meta = raw.get("probe") or {}
        try:
            actual_asn = int(str(meta.get("asn", "")).removeprefix("AS"))
        except ValueError:
            continue
        if (actual_asn == asn and str(meta.get("country", "")).upper() == "RU"
                and city_key(str(meta.get("city", ""))) == city_key(CITY)):
            matching.append(raw)
    status = ("pending" if result.get("status") == "in-progress"
              else "no_probe" if not records
              else "unverified_probe" if not matching
              else "provider_incomplete" if result.get("status") != "finished"
              else "measurement_complete")
    out = {"hostname": hostname, "requested_asn": asn, "city": CITY,
           "state": status, "measurement_id": result.get("id"), "provider_status": result.get("status")}
    if matching and status == "measurement_complete":
        row = matching[0]["result"]
        meta = matching[0]["probe"]
        out.update({"observed_asn": meta.get("asn"), "network": meta.get("network"),
                    "trace_status": row.get("status"),
                    "resolved_ip": row.get("resolvedAddress"),
                    "raw_excerpt": str(row.get("rawOutput", ""))[:6000],
                    "hops": (row.get("hops") or [])[:50]})
        if row.get("status") != "finished":
            out["state"] = "trace_failed_or_offline"
    return out


def query(hostname: str, asn: int, wait: int) -> dict:
    try:
        started = request_json(GLOBALPING, payload=payload(hostname, asn))
        deadline = time.monotonic() + wait
        while True:
            response = request_json(GLOBALPING + "/" + started["id"])
            if response.get("status") != "in-progress" or time.monotonic() >= deadline:
                return classify(hostname, asn, response)
            time.sleep(1)
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {"hostname": hostname, "requested_asn": asn, "city": CITY,
                "state": "provider_error", "detail": str(exc)[:160]}


def main(argv=None):
    cli = argparse.ArgumentParser(description="VVPN diagnostic TCP/443 trace from two Moscow ASNs")
    cli.add_argument("--output", type=Path, default=Path("vvpn-results/moscow-route-trace"))
    cli.add_argument("--wait", type=int, default=45)
    args = cli.parse_args(argv)
    if not 5 <= args.wait <= 90:
        cli.error("wait must be 5..90 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for host in TARGETS:
        for asn in ASNS:
            row = query(host, asn, args.wait)
            rows.append(row)
            print("MOSCOW TRACE", host, "AS" + str(asn), row["state"],
                  "resolved", row.get("resolved_ip"), "network", row.get("network"), flush=True)
    report = {"tool": "VVPN", "created_utc": datetime.now(timezone.utc).isoformat(),
              "meaning": "Moscow TCP/443 traceroute from HOSTKEY vs Timeweb; not browser rendering, not HTTP",
              "results": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Moscow TCP/443 traceroutes after HOSTKEY HTTP-connect timeouts", "",
          "Incomplete hops do NOT prove filtering or an HTTP website failure.", "",
          "| Target | Requested ASN | Probe result | Resolved IP |",
          "|---|---:|---|---|"]
    for row in rows:
        md.append(f"| {row['hostname']} | AS{row['requested_asn']} | {row['state']} | {row.get('resolved_ip') or '—'} |")
    (args.output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0 if all(r["state"] == "measurement_complete" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
