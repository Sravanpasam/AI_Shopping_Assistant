"""
Unit tests for backend/price_parser.parse_price_constraint.

Run:  .venv\\Scripts\\python.exe tests\\test_price_parsing.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from price_parser import parse_price_constraint  # noqa: E402

AED = 3.67


class TestParsePriceConstraint(unittest.TestCase):

    def check(self, text, cleaned, lo, hi):
        got_cleaned, got_lo, got_hi = parse_price_constraint(text)
        self.assertEqual(got_cleaned, cleaned, f"cleaned for {text!r}")
        self._assert_money(got_lo, lo, f"min for {text!r}")
        self._assert_money(got_hi, hi, f"max for {text!r}")

    def _assert_money(self, got, expected, label):
        if expected is None:
            self.assertIsNone(got, label)
        else:
            self.assertIsNotNone(got, label)
            self.assertAlmostEqual(got, expected, places=2, msg=label)

    # ---- max variants -------------------------------------------------
    def test_under(self):
        self.check("running shoes under 100", "running shoes", None, 100.0)

    def test_below_with_dollar(self):
        self.check("wireless headphones below $200", "wireless headphones", None, 200.0)

    def test_less_than(self):
        self.check("laptops less than 1500", "laptops", None, 1500.0)

    def test_up_to(self):
        self.check("up to 50 usd", "", None, 50.0)

    def test_no_more_than(self):
        self.check("shoes no more than 75", "shoes", None, 75.0)

    def test_max_keyword(self):
        self.check("gift ideas max 300", "gift ideas", None, 300.0)

    # ---- min variants -------------------------------------------------
    def test_over(self):
        self.check("gaming phones over 500", "gaming phones", 500.0, None)

    def test_at_least(self):
        self.check("at least 100", "", 100.0, None)

    def test_more_than(self):
        self.check("smartwatches more than 300", "smartwatches", 300.0, None)

    def test_from(self):
        self.check("monitors from 200", "monitors", 200.0, None)

    # ---- ranges -------------------------------------------------------
    def test_between_and(self):
        self.check("between 50 and 100", "", 50.0, 100.0)

    def test_between_with_noun(self):
        self.check("shoes between 50 and 100 dollars", "shoes", 50.0, 100.0)

    def test_x_to_y(self):
        self.check("laptops 800 to 1200 usd", "laptops", 800.0, 1200.0)

    def test_from_x_to_y(self):
        self.check("from 100 to 200", "", 100.0, 200.0)

    def test_hyphen_range(self):
        self.check("headphones 50-150 dollars", "headphones", 50.0, 150.0)

    def test_to_with_currency(self):
        self.check("headphones 50 to 100 dollars", "headphones", 50.0, 100.0)

    def test_between_plain(self):
        self.check("between 20 and 60", "", 20.0, 60.0)

    def test_budget_word_range(self):
        self.check("budget laptops 800 to 1200", "budget laptops", 800.0, 1200.0)

    # ---- min AND max together ----------------------------------------
    def test_min_and_max(self):
        self.check("over 50 under 200", "", 50.0, 200.0)

    # ---- bare ranges / alnum tokens that must NOT be prices -----------
    def test_bare_range_no_signal(self):
        self.check("compare 2 to 3 products", "compare 2 to 3 products", None, None)

    def test_bare_hyphen_range_no_signal(self):
        self.check("headphones 50-150", "headphones 50-150", None, None)

    def test_alnum_token_hyphen(self):
        self.check(
            "sony WH-1000-XM5 headphones", "sony WH-1000-XM5 headphones", None, None
        )

    # ---- non-price units ---------------------------------------------
    def test_unit_ml(self):
        self.check("shampoo 500 ml", "shampoo 500 ml", None, None)

    def test_unit_pack_of(self):
        self.check("pack of 2", "pack of 2", None, None)

    def test_unit_spf(self):
        self.check("SPF 50 sunscreen", "SPF 50 sunscreen", None, None)

    def test_unit_under_minutes(self):
        self.check("under 5 minutes", "under 5 minutes", None, None)

    def test_unit_over_grams(self):
        self.check("over 100 g", "over 100 g", None, None)

    def test_unit_at_least_gb(self):
        self.check("at least 8 gb", "at least 8 gb", None, None)

    # ---- currency AED -------------------------------------------------
    def test_aed_dirhams(self):
        self.check("less than 100 dirhams", "", None, round(100 / AED, 2))

    def test_aed_word(self):
        self.check("under 367 aed", "", None, round(367 / AED, 2))

    def test_aed_dhs(self):
        self.check("over 367 dhs", "", round(367 / AED, 2), None)

    def test_under_aed(self):
        self.check("under 300 AED", "", None, round(300 / AED, 2))

    def test_aed_translated_phrasing(self):
        # e.g. Arabic "أريد هاتفاً بأقل من 100 درهم" translated to English
        self.check(
            "I want a phone for less than 100 dirhams",
            "I want a phone",
            None,
            round(100 / AED, 2),
        )

    def test_aed_range(self):
        self.check(
            "between 367 and 734 dirhams", "", round(367 / AED, 2), round(734 / AED, 2)
        )

    # ---- no price -----------------------------------------------------
    def test_no_price(self):
        self.check("wireless noise cancelling headphones", "wireless noise cancelling headphones", None, None)

    def test_no_price_with_number(self):
        self.check("iphone 15 charger", "iphone 15 charger", None, None)

    def test_empty(self):
        self.check("", "", None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
