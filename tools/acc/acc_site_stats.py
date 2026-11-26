"""Fixed aggregate-only Cloudflare queries. No SQL or credentials from inbox messages."""
import json
import os
from pathlib import Path
import subprocess
import tomllib
from datetime import datetime, timezone, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from acc_github_bridge import BridgeError, NoRedirect

ACCOUNT = "c61ff13e32298bb52a103be80e07e1ac"
SITES = {
    "vitdart.com": ("vitdart-accounts", "vitdart-users", "12e521d1-5707-4ec0-bd91-a26dced308cf"),
    "btc2g.com": ("btc2g-accounts", "btc2g-users", "c84b5fe0-b86e-4a79-bfaf-097d424e0a87"),
    "bmychance.com": ("bmychance-accounts", "bmychance-users", "1c814f77-2a66-46f8-b122-3b08f832e6ad"),
    "bit2g.com": ("bit2g-ideas", "bit2g-ideas-users", "4763dce4-a400-4ef1-9222-868bc70ac22c"),
    "findbbest.com": ("findbbest-accounts", "findbbest-users", "bebe2a27-7c68-40a7-957c-dff0cbd81e3c"),
    "aikickai.com": ("aikickai-accounts", "aikickai-users", "3992b6f5-bf02-4d7d-8540-2fa6032a3fef"),
}
COUNT_SQL = """SELECT COUNT(*) AS registered_accounts,
COALESCE(SUM(CASE WHEN email_verified_at IS NOT NULL THEN 1 ELSE 0 END),0) AS verified_email,
COALESCE(SUM(CASE WHEN role='admin' THEN 1 ELSE 0 END),0) AS administrators,
COALESCE(SUM(CASE WHEN disabled_at IS NOT NULL THEN 1 ELSE 0 END),0) AS disabled_accounts
FROM users"""


def cloudflare_token():
    if os.environ.get("CLOUDFLARE_API_TOKEN"):
        return os.environ["CLOUDFLARE_API_TOKEN"].strip()
    path = Path.home() / "AppData/Roaming/xdg.config/.wrangler/config/default.toml"
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(config["expiration_time"].replace("Z", "+00:00"))
        if expires <= datetime.now(timezone.utc) + timedelta(minutes=2):
            wrangler = Path.home() / "PycharmProjects/bmychance/node_modules/wrangler/bin/wrangler.js"
            result = subprocess.run(["node", str(wrangler), "whoami"], capture_output=True,
                                    timeout=40, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise BridgeError("Existing Cloudflare login could not refresh; no new login/key was created.")
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        return config["oauth_token"]
    except (OSError, KeyError, ValueError, subprocess.TimeoutExpired):
        raise BridgeError("Existing Cloudflare credentials are unavailable or expired.") from None


class SiteStats:
    def __init__(self, token=None):
        self._token = token or cloudflare_token()
        self.opener = build_opener(NoRedirect())

    def request(self, path, data=None):
        request = Request("https://api.cloudflare.com/client/v4/accounts/" + ACCOUNT + path,
                          headers={"Authorization": "Bearer " + self._token,
                                   "Content-Type": "application/json", "User-Agent": "ACC-Readonly-Stats"},
                          data=None if data is None else json.dumps(data).encode())
        try:
            with self.opener.open(request, timeout=20) as response:
                result = json.load(response)
            if not result.get("success"):
                raise BridgeError("Cloudflare did not confirm the read request.")
            return result["result"]
        except HTTPError as exc:
            raise BridgeError(f"Cloudflare HTTP {exc.code}; no count available.") from None
        except (URLError, OSError, ValueError, KeyError):
            raise BridgeError("Cloudflare read failed; no count available.") from None

    def collect(self, cancel=None):
        rows = []
        for site, (worker, database, database_id) in SITES.items():
            if cancel and cancel.is_set():
                raise BridgeError("Statistics request cancelled.")
            row = {"site": site, "source": f"Cloudflare D1 / {database} / users",
                   "checked_at": datetime.now(timezone.utc).isoformat()}
            try:
                settings = self.request(f"/workers/scripts/{worker}/settings")
                bindings = settings.get("bindings", [])
                db = any(x.get("type") == "d1" and x.get("name") == "DB" and x.get("id") == database_id for x in bindings)
                origin = next((x.get("text") for x in bindings if x.get("name") == "PUBLIC_SITE_ORIGIN"), None)
                if not db or origin != "https://" + site:
                    raise BridgeError("Deployed site/database binding differs from the reviewed registry; no query executed.")
                result = self.request(f"/d1/database/{database_id}/query", {"sql": COUNT_SQL})
                counts = result[0]["results"][0]
                keys = ("registered_accounts", "verified_email", "administrators", "disabled_accounts")
                if not result[0].get("success") or any(type(counts.get(k)) is not int or counts[k] < 0 for k in keys):
                    raise BridgeError("Invalid aggregate response; no count available.")
                row.update(status="OK", counts={k: counts[k] for k in keys})
            except (IndexError, KeyError, TypeError):
                row.update(status="UNAVAILABLE", error="Unexpected aggregate response; no count available.")
            except BridgeError as exc:
                row.update(status="UNAVAILABLE", error=str(exc))
            rows.append(row)
        return rows


def statistics_report(cancel=None):
    rows = SiteStats().collect(cancel)
    lines = ["Registration counts — read-only Cloudflare query", "",
             "Total = all existing rows in users, including admin/disabled/unverified accounts; not visits, subscribers or submissions.",
             "No automatic exclusion of test accounts; no individual records were retrieved.", "",
             "| Site | Total | Verified email | Admin | Disabled | Checked UTC |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["status"] == "OK":
            c = row["counts"]
            lines.append(f"| {row['site']} | {c['registered_accounts']} | {c['verified_email']} | {c['administrators']} | {c['disabled_accounts']} | {row['checked_at']} |")
        else:
            lines.append(f"| {row['site']} | unavailable | — | — | — | {row['checked_at']} |")
    lines.append("")
    for row in rows:
        lines.append(f"Source {row['site']}: {row['source']}. " + row.get("error", ""))
    return {"state": "COMPLETED" if all(r["status"] == "OK" for r in rows) else "FAILED",
            "result": "\n".join(lines)}
