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
    parse_globalping_result, request_json,
)

CITY = "Moscow"


def measurement_payload(site: str, kind: str, probes: int = 2) -> dict:
    if kind not in ("http", "dns") or not (1 <= probes <= 5):
        raise ValueError("Expected kind http/dns and 1-5 probes")
    if kind == "http":
        payload = globalping_payload("https://" + site, CITY)
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
    return "ok" if result.get("answers") else "dns_no_answers"


def analyze_measurement(site: str, kind: str, response: dict, expected: int = 2) -> dict:
    rows = []
    for raw in response.get("results", []):
        probe = raw.get("probe") or {}
        if str(probe.get("country", "")).upper() != "RU" or city_key(str(probe.get("city", ""))) != city_key(CITY):
            continue
        result = raw.get("result") or {}
        if kind == "http":
            parsed = parse_globalping_result(raw, "https://" + site)
            state, code = parsed["state"], parsed.get("code")
        else:
            state, code = classify_dns(result), result.get("statusCode")
        rows.append({
            "city": probe.get("city"), "network": probe.get("network", ""),
            "asn": probe.get("asn", ""), "probe_tags": probe.get("tags", []),
            "state": state, "code": code,
            "resolved_ip": result.get("resolvedAddress"),
            "resolver": result.get("resolver") if kind == "dns" else None,
            "answer_count": len(result.get("answers") or []) if kind == "dns" else None,
            "error_excerpt": str(result.get("rawOutput", ""))[:220] if state != "ok" else "",
        })
    networks = sorted({str(r["asn"] or r["network"]).strip() for r in rows if r["asn"] or r["network"]})
    return {
        "site": site, "kind": kind, "city": CITY,
        "requested_probes": expected, "returned_probes": len(rows),
        "independent_network_count": len(networks),
        "networks": networks,
        "all_observed_ok": bool(rows) and all(r["state"] == "ok" for r in rows),
        "two_networks_confirmed": len(networks) >= 2,
        "measurement_id": response.get("id"),
        "status": response.get("status"),
        "probes": rows,
    }


def query(site: str, kind: str, wait: int = 30) -> dict:
    try:
        started = request_json(GLOBALPING, payload=measurement_payload(site, kind))
        result_url = GLOBALPING + "/" + started["id"]
        deadline = time.monotonic() + wait
        while True:
            response = request_json(result_url)
            if response.get("status") != "in-progress" or time.monotonic() >= deadline:
                break
            time.sleep(1)
        report = analyze_measurement(site, kind, response)
        if response.get("status") == "in-progress":
            report["status"] = "pending"
            report["all_observed_ok"] = False
        return report
    except (RuntimeError, KeyError, ValueError, TypeError) as exc:
        return {"site": site, "kind": kind, "city": CITY, "status": "provider_error",
                "requested_probes": 2, "returned_probes": 0,
                "all_observed_ok": False, "two_networks_confirmed": False,
                "error": str(exc)[:240], "probes": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check Moscow HTTP/DNS using up to two Globalping probes per site")
    parser.add_argument("--output", type=Path, default=Path("vvpn-results/moscow-diversity"))
    parser.add_argument("--wait", type=int, default=30)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in SITES:
        for kind in ("http", "dns"):
            data = query(site, kind, args.wait)
            rows.append(data)
            observed = [(r["network"], r["state"], r["code"]) for r in data.get("probes", [])]
            print(site, kind, data["status"], "probes", data["returned_probes"],
                  "two distinct ASNs/networks", data.get("two_networks_confirmed"),
                  "observed", observed, "error", data.get("error", ""), flush=True)
    report = {"tool": "VVPN", "measurement": "Moscow public-node comparison",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "meaning": "Community HTTP/DNS probes, NOT browser rendering",
              "results": rows}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Moscow multi-probe HTTP / DNS", "",
             "A missing second probe is **not** proof of website failure.",
             "HTTP 200 and DNS NOERROR **do not** prove Chromium rendering from Moscow.", "",
             "| Site | Kind | Probes | Distinct networks | All observed OK | Codes |",
             "|---|---|---:|---:|---|---|"]
    for r in rows:
        codes = ", ".join(str(n.get("code")) for n in r.get("probes", [])) or "no probes"
        lines.append(f"| {r['site']} | {r['kind']} | {r['returned_probes']} | "
                     f"{r.get('independent_network_count', 0)} | {r['all_observed_ok']} | {codes} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # An API outage is not a site outage, but it must not become green silently.
    return 0 if all(x["all_observed_ok"] for x in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
