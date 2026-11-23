#!/usr/bin/env python3
"""Repeat failed Vladivostok checks via system DNS and explicit 1.1.1.1."""
from pathlib import Path
import time
import vvpn

FAILURES = [
    "https://aikickai.com/",
    "https://bmychance.com/account-entry.css",
    "https://btc2g.com/assets/css/index.css?v=20261110-1",
    "https://vitdart.com/static/platform.css?v=4f64b24ebc3b",
    "https://findbbest.com/workbook.css",
    "https://findbbest.com/account-entry.css",
]


def retest():
    rows = []
    for url in FAILURES:
        for resolver, attempts in (("system", 2), ("1.1.1.1", 1)):
            for attempt in range(1, attempts + 1):
                row = {"provider": "Globalping", "city": "Vladivostok", "url": url,
                       "resolver": resolver, "attempt": attempt}
                try:
                    p = vvpn.globalping_payload(url, "Vladivostok")
                    if resolver != "system":
                        p["measurementOptions"]["resolver"] = resolver
                    started = vvpn.request_json(vvpn.GLOBALPING, payload=p)
                    row["measurement_id"] = started["id"]
                    deadline = time.monotonic() + 30
                    while True:
                        report = vvpn.request_json(f"{vvpn.GLOBALPING}/{started['id']}")
                        if report.get("status") != "in-progress" or time.monotonic() >= deadline:
                            break
                        time.sleep(2)
                    matches = [x for x in report.get("results", []) if x.get("probe", {}).get("country") == "RU" and vvpn.city_key(x.get("probe", {}).get("city", "")) == "vladivostok"]
                    if matches:
                        row["probe"] = matches[0].get("probe", {}).get("network")
                        row.update(vvpn.parse_globalping_result(matches[0], url))
                    else:
                        row["state"] = "no_probe"
                    if report.get("status") == "in-progress":
                        row["state"] = "pending"
                except Exception as exc:
                    row.update({"state": "provider_error", "detail": str(exc)[:300]})
                rows.append(row)
                print(resolver, attempt, row["state"], row.get("code"), url, row.get("detail", "")[:110], flush=True)
                time.sleep(1)
    vvpn.write_report(rows, Path("vvpn-dns-retest"))
    return 0


if __name__ == "__main__":
    raise SystemExit(retest())
