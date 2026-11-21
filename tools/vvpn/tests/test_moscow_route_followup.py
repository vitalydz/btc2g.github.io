"""Unit tests for pinpointing Moscow AS50867 route failures without false positives."""
import unittest
from unittest.mock import patch

import moscow_route_followup as m


def response(asn=50867, city="Moscow", status="finished", code=200,
             content_type="text/html", result_status="finished"):
    return {
        "id": "dummy", "status": status,
        "results": [{
            "probe": {"country": "RU", "city": city, "asn": str(asn), "network": "test"},
            "result": {"status": result_status, "statusCode": code,
                       "headers": {"content-type": content_type}, "rawOutput": ""}
        }]
    }


class MoscowRouteFollowupTests(unittest.TestCase):
    def test_requested_network_and_city_are_explicit(self):
        p = m.payload("https://btc2g.com/a.css?v=1", 50867)
        self.assertEqual(p["locations"], [{"country": "RU", "city": "Moscow", "asn": 50867, "limit": 1}])
        self.assertNotIn("limit", p)
        self.assertEqual(p["measurementOptions"]["request"]["query"], "v=1")

    def test_unknown_asn_rejected(self):
        with self.assertRaises(ValueError):
            m.payload("https://btc2g.com", 12345)

    def test_verified_200(self):
        r = m.classify("https://btc2g.com", "html", 50867, response())
        self.assertEqual(r["state"], "ok")
        self.assertEqual(r["http_code"], 200)
        self.assertEqual(r["matching_probes"], 1)

    def test_matching_asn_but_wrong_city_not_accepted(self):
        r = m.classify("https://btc2g.com", "html", 50867, response(city="Saint Petersburg"))
        self.assertEqual(r["state"], "mismatched_probe")

    def test_wrong_asn_not_accepted(self):
        r = m.classify("https://btc2g.com", "html", 9123, response(asn=50867))
        self.assertEqual(r["state"], "mismatched_probe")

    def test_probe_timeout_has_no_http_code(self):
        r = m.classify("https://btc2g.com", "html", 50867,
                       response(code=None, result_status="failed"))
        self.assertEqual(r["state"], "network_error")
        self.assertIsNone(r["http_code"])

    def test_no_results_means_no_probe(self):
        r = m.classify("https://btc2g.com", "html", 50867,
                       {"status": "finished", "results": []})
        self.assertEqual(r["state"], "no_probe")

    def test_incomplete_provider_result_not_counted(self):
        r = m.classify("https://btc2g.com", "html", 50867,
                       response(status="in-progress"))
        self.assertEqual(r["state"], "pending")

    def test_wrong_image_mime(self):
        r = m.classify("https://bmychance.com/storybook-fast.webp", "image", 50867,
                       response(content_type="text/html"))
        self.assertEqual(r["state"], "invalid_image_mime")

    def test_http_204_not_sufficient(self):
        r = m.classify("https://btc2g.com", "html", 50867, response(code=204))
        self.assertEqual(r["state"], "unexpected_http_status")

    def test_provider_error_is_not_web_failure(self):
        with patch.object(m, "request_json", side_effect=RuntimeError("HTTP 429 from API")):
            r = m.query("https://btc2g.com", "html", 50867, 5)
        self.assertEqual(r["state"], "provider_error")
        self.assertIsNone(r["http_code"])


if __name__ == "__main__":
    unittest.main()
