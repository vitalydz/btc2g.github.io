# VVPN — Russia website availability tester

VVPN checks whether our six websites answer **HTTPS requests from remote Russian probes**. It also offers an **optional full Chromium browser test through an authorized proxy or remote CDP endpoint** to detect failed CSS/JavaScript and save screenshots.

**Important:** VVPN is a diagnostic utility, **not a VPN provider or bypass**. Running a check from a provider's probe does not route your own traffic through Russia. For a browser session with a Russian exit IP, supply an authorized SOCKS5/HTTP proxy or a remote Chromium CDP endpoint whose actual browser exit is in the target city. A city label is not geolocation proof.

## Quick start — no VPN and no paid domain required

Python 3.10+; basic HTTP checks use only the Python standard library.

```bash
python vvpn.py bmychance.com --backend checkhost
python vvpn.py bmychance.com --backend globalping --cities 'Moscow,Vladivostok'
python vvpn.py --all --backend both --cities 'Moscow,Novosibirsk,Vladivostok'
python -m unittest discover -s tests -v
```

The program creates `vvpn-report/report.md` and `vvpn-report/report.json`. The `--all` flag checks bmychance.com, btc2g.com, bit2g.com, vitdart.com, findbbest.com, and aikickai.com.

`no_probe` means the provider cannot check that **specific city** — not that a site is blocked. Provider API errors are similarly separate from site errors. Redirect 301/302/308 proves reachability but may lead to a different host.

Check-Host API: https://check-host.net/about/api  
Globalping API: https://globalping.io/docs/api.globalping.io

Both providers have usage limits and node availability varies over time. VVPN makes no guarantee that an active probe exists in every large Russian city. Respect their fair-use policies.

## Full browser/CSS test from a Moscow IP

Requires a **proxy or remote Chromium endpoint you control or are authorized to use**. Do not use untrusted public proxy lists for personal logins. The code checks actual browser egress through two independent geo-IP services and marks the browser `unverified_proxy` if they do not agree on **both IP and Moscow, Russia**. A fake `--proxy-city Moscow` label cannot pass verification.

```bash
pip install playwright
python -m playwright install chromium
python vvpn.py bmychance.com --backend browser --proxy-server socks5://127.0.0.1:1080 --proxy-city Moscow
```

### Authorized remote Chromium through CDP (alternative to a proxy)

If a browser service offers a remote Chromium DevTools endpoint (CDP), VVPN can attach without purchasing a VPS or setting up a private VPN. Set the endpoint **only in the local environment or a private secret** — never commit it or print it in logs:

```bash
export VVPN_CDP_ENDPOINT='wss://your-authorized-browser-provider.example/secret-session'
python vvpn.py --all --backend browser --cdp-from-env --proxy-city Moscow --output vvpn-moscow-browser
```

The example URL is only a placeholder, **not an existing service**. Do not combine `--cdp-from-env` with `--proxy-server` / `VVPN_PROXY_SERVER`. If remote CDP is unavailable, no browser verification occurs. If CDP is reachable but its egress is outside Moscow, result is **`unverified_proxy`** even if every page responds HTTP 200. Existing provider contexts are used when possible so that the configured city is not silently lost.

A candidate for no-card testing is [browserstorm](https://browserstorm.com/pricing/), which publicly advertises remote CDP and a Free plan, **but the Free plan allows only three regions**. We have **not confirmed** whether Moscow is available to that plan or whether a usable CDP endpoint can be obtained. [LocaBrowser](https://www.locabrowser.com/) exposes Russian Federation in its location selector, but does not on its public page prove Moscow-city exit or provide a VVPN-compatible CDP endpoint. Do not report either provider as a successful Moscow browser test until the egress verification and actual page screenshots pass.

Browser reports include passive form summaries: form methods, counts of required/unlabeled controls and submit buttons. **No inputs are filled and no forms are submitted.** The test does not authorize account creation, payments or messages.

For a proxy requiring authentication, set `VVPN_PROXY_USER` and `VVPN_PROXY_PASSWORD` as environment variables, never save them in GitHub. An SSH SOCKS tunnel can be created outside VVPN with `ssh -N -D 127.0.0.1:1080 your-server`, but the server's **real** location must be independently confirmed.

Browser mode records response statuses for stylesheet/script/font resources, network failures, final URL, and a full-page PNG. Modern web apps may have ongoing network connections; screenshots are taken shortly after DOM loading, so some delayed UI may require a manual review. No registration or login is attempted.

## Reading results

| Result | Meaning |
|---|---|
| `ok` (HTTP 2xx) | The tested URL answered from that probe; not full visual proof |
| `redirect` (HTTP 3xx) | Reachable URL redirected, destination not necessarily checked |
| `http_error` (HTTP 4xx/5xx) | Server returned an error |
| `network_error` | Connection/DNS/TLS failure or probe error |
| `css_failure` | Browser saw a failed stylesheet request |
| `no_probe` | No matching usable remote node in that city |
| `provider_error` | Problem with test service, not verified site outage |

**Priority: Moscow.** First obtain Moscow-confirmed Chromium screenshots and browser DOM/JS/CSS/network/form results for all six sites, ideally on two independent local networks. Only after that compare Vladivostok and other cities. If a Moscow browser exit is unavailable, use HTTP probes as a partial check, **not** as proof that the pages render identically for residents.

## Data handling

Reports can contain website URLs, response metadata and screenshots. Keep private pages out of browser checks; reports are saved locally only. Do not commit reports, credentials, or proxy addresses to the repository.