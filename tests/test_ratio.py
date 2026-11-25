import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.update_btc_gold_chart import build_chart, build_history, build_signal, main, validate_market_date


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
        self.assertEqual(signal["btc_price_usd"], 100000.0)
        self.assertEqual(signal["gold_price_usd"], 2500.0)
        self.assertEqual(signal["ratio_change_7d_pct"], 0.0)
        json.dumps(signal, allow_nan=False)

    def test_history_publishes_only_finite_source_values(self):
        history = build_history(self.prices(), limit=30)
        self.assertEqual(history["source"], "yfinance")
        self.assertEqual(len(history["series"]), 30)
        self.assertEqual(history["series"][-1]["ratio"], 40.0)
        self.assertEqual(history["series"][-1]["model_score"], 50)
        json.dumps(history, allow_nan=False)

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

    def test_market_freshness_allows_weekend_but_rejects_stale_or_future_quotes(self):
        with patch("scripts.update_btc_gold_chart.utc_today", return_value=datetime(2026, 10, 30, tzinfo=timezone.utc)):
            for date, accepted in (("2026-10-29", True), ("2026-10-26", True),
                                   ("2026-10-25", False), ("2026-10-31", False)):
                with self.subTest(date=date):
                    prices = pd.DataFrame({"Gold": [2500], "Bitcoin": [100000]}, index=pd.to_datetime([date]))
                    if accepted:
                        validate_market_date(prices)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "stale"):
                            validate_market_date(prices)

    def test_download_failure_is_reported_even_when_previous_chart_exists(self):
        with patch("scripts.update_btc_gold_chart.build_chart", side_effect=RuntimeError("source unavailable")), \
             patch("scripts.update_btc_gold_chart.CHART_PATH") as chart, \
             patch("scripts.update_btc_gold_chart.SIGNAL_PATH") as signal:
            chart.exists.return_value = True
            signal.exists.return_value = True
            self.assertEqual(main(), 1)
            chart.write_text.assert_not_called()
            signal.write_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
