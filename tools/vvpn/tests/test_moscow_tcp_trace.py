import unittest
from moscow_tcp_trace import payload, classify


def sample(asn=50867, city="Moscow", status="finished", trace_status="finished"):
    return {"id": "test", "status": status, "results": [{
        "probe": {"country": "RU", "city": city, "asn": f"AS{asn}", "network": "example"},
        "result": {"status": trace_status, "resolvedAddress": "203.0.113.55",
                   "rawOutput": "hop 1", "hops": []},
    }]}


class MoscowTCPTraceTests(unittest.TestCase):
    def test_payload_is_ipv4_tcp_443_with_explicit_asn(self):
        p = payload("btc2g.com", 50867)
        self.assertEqual(p["measurementOptions"], {"protocol": "TCP", "port": 443, "ipVersion": 4})
        self.assertEqual(p["locations"], [{"country": "RU", "city": "Moscow", "asn": 50867, "limit": 1}])

    def test_unapproved_host_or_network_rejected(self):
        with self.assertRaises(ValueError):
            payload("example.org", 50867)
        with self.assertRaises(ValueError):
            payload("btc2g.com", 12345)

    def test_confirm_asn_city_before_success(self):
        r = classify("btc2g.com", 50867, sample())
        self.assertEqual(r["state"], "measurement_complete")
        self.assertEqual(r["observed_asn"], "AS50867")
        self.assertEqual(classify("btc2g.com", 50867, sample(asn=9123))["state"], "unverified_probe")
        self.assertEqual(classify("btc2g.com", 50867, sample(city="Kazan"))["state"], "unverified_probe")

    def test_missing_probe_is_not_site_failure(self):
        self.assertEqual(classify("btc2g.com", 50867, {"status": "finished", "results": []})["state"], "no_probe")

    def test_trace_failure_has_distinct_state(self):
        self.assertEqual(classify("btc2g.com", 50867, sample(trace_status="failed"))["state"], "trace_failed_or_offline")

    def test_pending_trace_cannot_be_reported_complete(self):
        self.assertEqual(classify("btc2g.com", 50867, sample(status="in-progress"))["state"], "pending")


if __name__ == "__main__":
    unittest.main()
