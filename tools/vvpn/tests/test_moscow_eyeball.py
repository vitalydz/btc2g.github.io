"""Tests without network access: Moscow eyeball-network filtering."""
import unittest
from unittest.mock import patch
from moscow_eyeball import eyeball_payload, classify, query


class MoscowEyeballTests(unittest.TestCase):
    def test_requests_moscow_eyeball_only(self):
        p = eyeball_payload("btc2g.com")
        self.assertEqual(p["locations"], [{"country": "RU", "city": "Moscow",
                                            "limit": 1, "tags": ["eyeball-network"]}])
        self.assertEqual(p["target"], "btc2g.com")

    def test_datacenter_is_never_misreported_as_home_isp(self):
        raw = {"status": "finished", "results": [
            {"probe": {"country": "RU", "city": "Moscow", "asn": 9123,
                       "network": "Timeweb", "tags": ["datacenter-network"]},
             "result": {"statusCode": 200}}]}
        report = classify("btc2g.com", raw)
        self.assertEqual(report["state"], "unverified_location_or_type")
        self.assertEqual(report["returned_confirmed_eyeball_probes"], 0)

    def test_actual_eyeball_http200_is_confirmed(self):
        raw = {"status": "finished", "results": [
            {"probe": {"country": "RU", "city": "Moscow", "asn": 123,
                       "network": "Home ISP", "tags": ["eyeball-network"]},
             "result": {"statusCode": 200}}]}
        report = classify("btc2g.com", raw)
        self.assertEqual(report["state"], "http_200")
        self.assertEqual(report["probes"][0]["http_code"], 200)

    def test_eyeball_http_error_is_a_real_error(self):
        raw = {"status": "finished", "results": [
            {"probe": {"country": "RU", "city": "Moscow",
                       "network": "Home ISP", "tags": ["eyeball-network"]},
             "result": {"statusCode": 403}}]}
        self.assertEqual(classify("btc2g.com", raw)["state"], "observed_http_failure")

    def test_missing_probe_is_not_a_site_failure(self):
        self.assertEqual(classify("btc2g.com", {"status": "finished", "results": []})["state"],
                         "no_eyeball_probe")

    @patch("moscow_eyeball.request_json", side_effect=RuntimeError("HTTP 429"))
    def test_rate_limit_is_provider_error(self, mocked):
        row = query("btc2g.com")
        self.assertEqual(row["state"], "provider_error")
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
