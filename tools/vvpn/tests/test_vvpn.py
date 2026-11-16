import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import vvpn


class VVPNTests(unittest.TestCase):
    def test_site_url(self):
        self.assertEqual(vvpn.site_url("bmychance.com"), "https://bmychance.com")
        self.assertEqual(vvpn.site_url("https://bmychance.com/path"), "https://bmychance.com/path")
        for invalid in ("file:///tmp/a", "https://user:pass@site.com", ""):
            with self.assertRaises(ValueError):
                vvpn.site_url(invalid)

    def test_http_states(self):
        self.assertEqual(vvpn.categorize_http(200), "ok")
        self.assertEqual(vvpn.categorize_http(308), "redirect")
        self.assertEqual(vvpn.categorize_http(403), "http_error")
        self.assertEqual(vvpn.categorize_http(None), "network_error")
        self.assertEqual(vvpn.categorize_http(None, False), "pending")

    def test_checkhost_result(self):
        self.assertEqual(vvpn.parse_checkhost_result([[1, 0.15, "OK", "200"]])["code"], 200)
        self.assertEqual(vvpn.parse_checkhost_result([[1, 0.15, "Moved", "301"]])["state"], "redirect")
        self.assertEqual(vvpn.parse_checkhost_result(None)["state"], "pending")
        self.assertEqual(vvpn.parse_checkhost_result([[0, 0.3, "No DNS", None]])["state"], "network_error")

    @patch("vvpn.request_json")
    def test_checkhost_city_filter(self, req):
        req.side_effect = [
            {"nodes": {"ru1.node": {"location": ["ru", "Russia", "Moscow"]}, "us1.node": {"location": ["us", "USA", "Miami"]}}},
            {"request_id": "abcd", "permanent_link": "https://check-host.net/check-report/abcd"},
            {"ru1.node": [[1, 0.10, "OK", "200", "1.2.3.4"]]},
        ]
        rows = vvpn.checkhost("https://bmychance.com", ["Moscow", "Vladivostok"])
        self.assertEqual(rows[0]["state"], "ok")
        self.assertEqual(rows[1]["state"], "no_probe")
        self.assertEqual(req.call_count, 3)

    @patch("vvpn.request_json")
    def test_globalping_rejects_mismatched_city(self, req):
        req.side_effect = [
            {"id": "one"},
            {"status": "finished", "results": [{"probe": {"country": "RU", "city": "Moscow"}, "result": {"statusCode": 200}}]},
        ]
        row = vvpn.globalping("https://bmychance.com", ["Vladivostok"])[0]
        self.assertEqual(row["state"], "location_mismatch")

    def test_globalping_payload_uses_hostname_not_url(self):
        p = vvpn.globalping_payload("https://bmychance.com/hello?q=1", "Moscow")
        self.assertEqual(p["target"], "bmychance.com")
        self.assertEqual(p["measurementOptions"]["protocol"], "HTTPS")
        self.assertEqual(p["measurementOptions"]["request"], {"method": "GET", "path": "/hello", "query": "q=1"})

    def test_globalping_extracts_css_and_js_sources(self):
        rec = {"result": {"statusCode": 200, "rawBody": '<html><head><link rel="stylesheet" href="/style.css"><script src="https://cdn.example.org/app.js"></script></head></html>'}}
        result = vvpn.parse_globalping_result(rec, "https://bmychance.com")
        self.assertEqual(result["state"], "ok")
        self.assertEqual(result["asset_candidates"], [
            {"type": "stylesheet", "url": "https://bmychance.com/style.css"},
            {"type": "script", "url": "https://cdn.example.org/app.js"},
        ])

    def test_invalid_css_mime(self):
        response = {"result": {"statusCode": 200, "headers": {"Content-Type": "text/html"}}}
        self.assertEqual(vvpn.parse_globalping_result(response, "https://bmychance.com/site.css")["state"], "invalid_mime")
        response["result"]["headers"]["Content-Type"] = "text/css; charset=utf-8"
        self.assertEqual(vvpn.parse_globalping_result(response, "https://bmychance.com/site.css")["state"], "ok")

    def test_proxy_geo_requires_public_ip_country_and_city(self):
        valid = {"success": True, "ip": "5.45.60.70", "country_code": "RU", "city": "Moscow"}
        self.assertTrue(vvpn.evaluate_proxy_location(valid, "Moscow")["matches"])
        self.assertFalse(vvpn.evaluate_proxy_location({**valid, "city": "Saint Petersburg"}, "Moscow")["matches"])
        self.assertFalse(vvpn.evaluate_proxy_location({**valid, "country_code": "DE"}, "Moscow")["matches"])
        self.assertFalse(vvpn.evaluate_proxy_location({**valid, "ip": "127.0.0.1"}, "Moscow")["matches"])
        self.assertFalse(vvpn.evaluate_proxy_location({**valid, "success": False}, "Moscow")["matches"])
        self.assertFalse(vvpn.evaluate_proxy_location({}, "Moscow")["matches"])

    @patch("vvpn.globalping")
    def test_all_sites_must_succeed_not_just_one(self, probe):
        probe.side_effect = [
            [{"provider": "Globalping", "city": "Moscow", "url": "https://bmychance.com", "state": "ok", "code": 200}],
            [{"provider": "Globalping", "city": "Moscow", "url": "https://btc2g.com", "state": "network_error", "code": None}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = vvpn.main(["bmychance.com", "btc2g.com", "--backend", "globalping",
                                "--cities", "Moscow", "--output", directory])
            self.assertEqual(result, 2)
            self.assertEqual(probe.call_count, 2)

    def test_provider_get_retries_transient_tls_failure(self):
        with patch("vvpn.urlopen", side_effect=[
            URLError("[SSL: CERTIFICATE_VERIFY_FAILED] hostname mismatch"),
            BytesIO(b'{"status":"finished"}'),
        ]) as open_url, patch("vvpn.time.sleep") as pause:
            self.assertEqual(vvpn.request_json("https://api.globalping.io/v1/measurements/abc"),
                             {"status": "finished"})
        self.assertEqual(open_url.call_count, 2)
        pause.assert_called_once_with(0.5)

    def test_provider_get_stops_after_three_transport_failures(self):
        with patch("vvpn.urlopen", side_effect=URLError("connection reset")) as open_url, \
             patch("vvpn.time.sleep") as pause:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempt"):
                vvpn.request_json("https://api.globalping.io/v1/measurements/abc")
        self.assertEqual(open_url.call_count, 3)
        self.assertEqual(pause.call_count, 2)

    def test_provider_post_is_never_retried(self):
        with patch("vvpn.urlopen", side_effect=URLError("TLS error")) as open_url, \
             patch("vvpn.time.sleep") as pause:
            with self.assertRaisesRegex(RuntimeError, "after 1 attempt"):
                vvpn.request_json("https://api.globalping.io/v1/measurements",
                                  payload={"type": "http", "target": "example.org"})
        self.assertEqual(open_url.call_count, 1)
        pause.assert_not_called()

    def test_provider_429_is_never_retried(self):
        error = HTTPError("https://api.globalping.io/v1/measurements/abc", 429,
                          "Too Many Requests", {}, BytesIO(b"rate limit"))
        with patch("vvpn.urlopen", side_effect=error) as open_url, \
             patch("vvpn.time.sleep") as pause:
            with self.assertRaisesRegex(RuntimeError, "Provider HTTP 429"):
                vvpn.request_json("https://api.globalping.io/v1/measurements/abc")
        self.assertEqual(open_url.call_count, 1)
        pause.assert_not_called()

    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            vvpn.write_report([{"provider": "Check-Host", "city": "Moscow", "url": "https://bmychance.com", "state": "ok", "code": 200}], Path(d))
            self.assertIn("Moscow", (Path(d) / "report.md").read_text())
            self.assertIn('"tool": "VVPN"', (Path(d) / "report.json").read_text())


if __name__ == "__main__":
    unittest.main()
