"""No-network regression tests for Moscow probe coverage and DNS classification."""
import unittest

from moscow_diversity import measurement_payload, classify_dns, analyze_measurement


class MoscowDiversityTests(unittest.TestCase):
    def test_uses_city_limit_not_conflicting_global_limit(self):
        for kind in ("http", "dns"):
            p = measurement_payload("bmychance.com", kind)
            self.assertEqual(p["locations"][0]["limit"], 2)
            self.assertNotIn("limit", p)
            self.assertEqual(p["locations"][0]["country"], "RU")
            self.assertEqual(p["locations"][0]["city"], "Moscow")
        self.assertEqual(measurement_payload("bmychance.com", "http")["measurementOptions"]["request"]["path"], "/")

    def test_dns_noerror_with_answer(self):
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 0,
                                      "answers": [{"type": "A", "value": "8.8.8.8"}]}), "ok")
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 3, "answers": []}), "dns_error")
        self.assertEqual(classify_dns({"status": "finished", "statusCode": 0, "answers": []}), "dns_no_answers")
        self.assertEqual(classify_dns({"status": "failed", "rawOutput": "timeout"}), "network_error")

    def test_does_not_invent_second_network(self):
        a = {"probe": {"country": "RU", "city": "Moscow", "network": "Example ISP", "asn": "AS123"},
             "result": {"statusCode": 200}}
        r = analyze_measurement("bmychance.com", "http", {"status": "finished", "results": [a]})
        self.assertTrue(r["all_observed_ok"])
        self.assertEqual(r["returned_probes"], 1)
        self.assertFalse(r["two_networks_confirmed"])

    def test_same_asn_is_not_independent(self):
        probes = [{"probe": {"country": "RU", "city": "Moscow", "asn": "AS123"},
                   "result": {"statusCode": 200}} for _ in range(2)]
        r = analyze_measurement("btc2g.com", "http", {"status": "finished", "results": probes})
        self.assertTrue(r["all_observed_ok"])
        self.assertFalse(r["two_networks_confirmed"])

    def test_rejects_wrong_country_city(self):
        probes = [{"probe": {"country": "DE", "city": "Berlin"}, "result": {"statusCode": 200}},
                  {"probe": {"country": "RU", "city": "Saint Petersburg"}, "result": {"statusCode": 200}}]
        r = analyze_measurement("btc2g.com", "http", {"status": "finished", "results": probes})
        self.assertFalse(r["all_observed_ok"])
        self.assertEqual(r["returned_probes"], 0)


if __name__ == "__main__":
    unittest.main()
