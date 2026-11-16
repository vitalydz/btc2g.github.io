"""No-network regression tests for Moscow probe coverage and DNS classification."""
import unittest

from moscow_diversity import measurement_payload, classify_dns, analyze_measurement, critical_assets


class MoscowDiversityTests(unittest.TestCase):
    def test_uses_city_limit_not_conflicting_global_limit(self):
        for kind in ("http", "dns"):
            p = measurement_payload("bmychance.com", kind)
            self.assertEqual(p["locations"][0]["limit"], 2)
            self.assertNotIn("limit", p)
            self.assertEqual(p["locations"][0]["country"], "RU")
            self.assertEqual(p["locations"][0]["city"], "Moscow")
        self.assertEqual(measurement_payload("bmychance.com", "http")["measurementOptions"]["request"]["path"], "/")
        self.assertEqual(measurement_payload("https://btc2g.com/app.js?v=1", "http")["measurementOptions"]["request"], {"method": "GET", "path": "/app.js", "query": "v=1"})

    def test_dns_noerror_with_answer(self):
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 0,
                                      "answers": [{"type": "A", "value": "8.8.8.8"}]}), "ok")
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 3, "answers": []}), "dns_error")
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 0, "answers": []}), "dns_no_answers")
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 0, "answers": [{"type": "CNAME", "value": "foo"}]}), "dns_no_a_record")
        self.assertEqual(classify_dns({"status": "failed", "rawOutput": "timeout"}), "network_error")

    def test_does_not_invent_second_network(self):
        a = {"probe": {"country": "RU", "city": "Moscow", "network": "Example ISP", "asn": "AS123"},
             "result": {"statusCode": 200}}
        r = analyze_measurement("bmychance.com", "http", {"status": "finished", "results": [a]})
        self.assertTrue(r["all_observed_ok"])
        self.assertEqual(r["returned_probes"], 1)
        self.assertFalse(r["two_networks_confirmed"])
        self.assertFalse(r["full_coverage_ok"])

    def test_same_asn_is_not_independent(self):
        probes = [{"probe": {"country": "RU", "city": "Moscow", "asn": "AS123"},
                   "result": {"statusCode": 200}} for _ in range(2)]
        r = analyze_measurement("btc2g.com", "http", {"status": "finished", "results": probes})
        self.assertTrue(r["all_observed_ok"])
        self.assertFalse(r["two_networks_confirmed"])
        self.assertFalse(r["full_coverage_ok"])

    def test_rejects_wrong_country_city(self):
        probes = [{"probe": {"country": "DE", "city": "Berlin"}, "result": {"statusCode": 200}},
                  {"probe": {"country": "RU", "city": "Saint Petersburg"}, "result": {"statusCode": 200}}]
        r = analyze_measurement("btc2g.com", "http", {"status": "finished", "results": probes})
        self.assertFalse(r["all_observed_ok"])
        self.assertEqual(r["returned_probes"], 0)


    def test_two_distinct_good_probes_required_for_coverage(self):
        probes = [
            {"probe": {"country": "RU", "city": "Moscow", "asn": "AS123"},
             "result": {"statusCode": 200}},
            {"probe": {"country": "RU", "city": "Moscow", "asn": "AS987"},
             "result": {"statusCode": 200}},
        ]
        result = analyze_measurement("bmychance.com", "http",
                                     {"status": "finished", "results": probes})
        self.assertTrue(result["full_coverage_ok"])
        probes[1]["result"]["statusCode"] = 403
        self.assertFalse(analyze_measurement("bmychance.com", "http",
                         {"status": "finished", "results": probes})["full_coverage_ok"])

    def test_critical_assets_discovery_is_passive(self):
        from unittest.mock import patch
        html = ('<html><head><link rel="stylesheet" href="/style.css">'
                '<script src="/site.js"></script></head>'
                '<body><img src="/hero.jpg"></body></html>')
        with patch("moscow_diversity.discover_html", return_value=html):
            result = critical_assets("bmychance.com")
        self.assertEqual(result, [
            ("stylesheet", "https://bmychance.com/style.css"),
            ("script", "https://bmychance.com/site.js"),
            ("image", "https://bmychance.com/hero.jpg"),
        ])


    def test_critical_assets_prefer_site_css_and_painting(self):
        from unittest.mock import patch
        html = ('<html><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto">'
                '<link rel="stylesheet" href="/static/style.css">'
                '<script src="https://cdn.example.com/vendor.js"></script>'
                '<script src="/static/platform.js"></script>'
                '<img src="/static/brand-logo.png">'
                '<img src="/images/painting-id"></html>')
        with patch("moscow_diversity.discover_html", return_value=html):
            result = critical_assets("vitdart.com")
        self.assertEqual(result, [
            ("stylesheet", "https://vitdart.com/static/style.css"),
            ("script", "https://vitdart.com/static/platform.js"),
            ("image", "https://vitdart.com/images/painting-id"),
        ])

    def test_critical_assets_missing_are_not_silently_ignored(self):
        from unittest.mock import patch
        with patch("moscow_diversity.discover_html", return_value="<html></html>"):
            with self.assertRaisesRegex(ValueError, "No stylesheet"):
                critical_assets("bmychance.com")

    def test_asset_mime_failure_from_moscow_probe(self):
        probes = [
            {"probe": {"country": "RU", "city": "Moscow", "asn": "AS123"},
             "result": {"statusCode": 200, "headers": {"Content-Type": "text/html"}}},
            {"probe": {"country": "RU", "city": "Moscow", "asn": "AS987"},
             "result": {"statusCode": 200, "headers": {"Content-Type": "text/html"}}},
        ]
        result = analyze_measurement("https://bmychance.com/app.js", "http",
                                     {"status": "finished", "results": probes},
                                     asset_kind="script")
        self.assertFalse(result["full_coverage_ok"])
        self.assertTrue(all(x["state"] != "ok" for x in result["probes"]))


if __name__ == "__main__":
    unittest.main()
