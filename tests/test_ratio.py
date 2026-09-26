import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.update_btc_gold_chart import build_chart, build_signal


class RatioTests(unittest.TestCase):
    def prices(self, days=366):
        return pd.DataFrame(
            {"Bitcoin": 100000.0, "Gold": 2500.0},
            index=pd.date_range("2025-01-01", periods=days),
        )

    def test_ratio_is_gold_ounces_per_bitcoin(self):
        signal = build_signal(self.prices())
        self.assertEqual(signal["ratio"], 40.0)
        self.assertEqual(signal["signal"], "HOLD")
        self.assertEqual(signal["confidence"], 0.5)
        json.dumps(signal, allow_nan=False)

    def test_date_uses_latest_common_quote_even_if_unsorted(self):
        prices = self.prices()
        prices.loc[prices.index[-1], "Gold"] = np.nan
        signal = build_signal(prices.iloc[::-1])
        self.assertEqual(signal["last_updated"], "2025-12-31")
        self.assertEqual(signal["ratio"], 40.0)

    def test_insufficient_common_history(self):
        with self.assertRaisesRegex(RuntimeError, "Not enough"):
            build_signal(self.prices(364))

    def test_invalid_prices_never_publish_a_signal(self):
        for column in ("Bitcoin", "Gold"):
            for value in (0.0, -1.0, np.inf, -np.inf):
                for position in (0, -1):
                    with self.subTest(column=column, value=value, position=position):
                        prices = self.prices()
                        prices.loc[prices.index[position], column] = value
                        with self.assertRaises(RuntimeError):
                            build_signal(prices)

    def test_signal_direction_uses_ratio(self):
        for latest_price, expected in ((50000.0, "BUY"), (150000.0, "SELL")):
            with self.subTest(expected=expected):
                prices = self.prices()
                prices.loc[prices.index[-1], "Bitcoin"] = latest_price
                signal = build_signal(prices)
                self.assertEqual(signal["ratio"], latest_price / 2500)
                self.assertEqual(signal["signal"], expected)
                self.assertTrue(0.5 <= signal["confidence"] <= 0.95)

    def test_rounded_ratio_cannot_be_zero(self):
        prices = self.prices()
        prices["Bitcoin"] = 0.001
        with self.assertRaisesRegex(RuntimeError, "too small"):
            build_signal(prices)

    def test_invalid_input_is_rejected_before_chart_writes(self):
        prices = self.prices()
        prices.loc[prices.index[-1], "Gold"] = 0
        with patch("scripts.update_btc_gold_chart.download_data", side_effect=[
            prices["Gold"], prices["Bitcoin"]
        ]), patch("scripts.update_btc_gold_chart.plt.savefig") as save_chart:
            with self.assertRaises(RuntimeError):
                build_chart()
            save_chart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
