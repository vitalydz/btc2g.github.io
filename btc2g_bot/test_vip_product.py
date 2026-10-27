import unittest
from datetime import date
from decimal import Decimal

from vip_product import PRICE_USD, PRICE_USD_CENTS, offer_text, validate_forecast


class ProductTests(unittest.TestCase):
    def test_exact_price(self):
        self.assertEqual(PRICE_USD, Decimal('4.95'))
        self.assertEqual(PRICE_USD * 100, PRICE_USD_CENTS)
        self.assertIn('$4.95', offer_text())

    def forecast(self):
        return dict(issued_on='2026-10-19', valid_until='2026-10-26',
                    horizon_days=7, text='Test forecast')

    def test_valid_forecast(self):
        data = self.forecast()
        self.assertEqual(validate_forecast(data, date(2026, 10, 19)), data)
        self.assertEqual(validate_forecast(data, date(2026, 10, 25)), data)

    def test_future_and_expired(self):
        for today in (date(2026, 10, 18), date(2026, 10, 26)):
            with self.assertRaises(ValueError):
                validate_forecast(self.forecast(), today)

    def test_missing_weekly_content(self):
        for change in ({'text': ''}, {'text': None}, {'horizon_days': 30},
                       {'valid_until': '2026-10-27'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_forecast(self.forecast() | change, date(2026, 10, 19))


if __name__ == '__main__':
    unittest.main()
