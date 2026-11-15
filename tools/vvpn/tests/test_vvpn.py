import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            vvpn.write_report([{"provider": "Check-Host", "city": "Moscow", "url": "https://bmychance.com", "state": "ok", "code": 200}], Path(d))
            self.assertIn("Moscow", (Path(d) / "report.md").read_text())
            self.assertIn('"tool": "VVPN"', (Path(d) / "report.json").read_text())


if __name__ == "__main__":
    unittest.main()
