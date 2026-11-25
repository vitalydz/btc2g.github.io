import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_multi_asset.py"
SPEC = importlib.util.spec_from_file_location("update_multi_asset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MultiAssetPayloadTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-08-01", periods=40, freq="D")
        self.prices = pd.DataFrame({
            "Bitcoin": [80000 + day * 100 for day in range(40)],
            "Gold": [4000 + day * 2 for day in range(40)],
            "Silver": [50 + day * 0.1 for day in range(40)],
            "SPX": [7000 + day * 5 for day in range(40)],
        }, index=dates)

    def test_payload_uses_only_common_positive_observations(self):
        payload = MODULE.build_payload(
            self.prices,
            generated_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(payload["updated_for_market_date"], "2026-09-09")
        self.assertEqual(len(payload["series"]), 40)
        latest = payload["series"][-1]
        self.assertAlmostEqual(latest["btc_gold"], latest["btc_spx"] / latest["gold_spx"], places=4)
        self.assertIn(payload["current"]["strongest_btc_30d"], {"btc_gold", "btc_silver", "btc_spx"})

    def test_payload_rejects_missing_and_nonpositive_sources(self):
        with self.assertRaises(ValueError):
            MODULE.build_payload(self.prices.drop(columns=["Silver"]))
        invalid = self.prices.copy()
        invalid.loc[invalid.index[-1], "SPX"] = 0
        with self.assertRaises(ValueError):
            MODULE.build_payload(invalid)


if __name__ == "__main__":
    unittest.main()
