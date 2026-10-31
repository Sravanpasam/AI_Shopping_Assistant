"""
Tests for fix #6: no invented fallback text and no "$$" price bug.

Run:  .venv\\Scripts\\python.exe tests\\test_grounding.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import comparison  # noqa: E402
from comparison import _generate_comparison_summary  # noqa: E402
from policy_qa import answer_policy  # noqa: E402


def product(asin, price, rating=4.5, count=10):
    return {
        "parent_asin": asin,
        "title": f"Product {asin}",
        "price": price,
        "average_rating": rating,
        "rating_number": count,
    }


class TestComparisonSummary(unittest.TestCase):

    def test_no_double_dollar_for_equal_prices(self):
        products = [product("A1", 10.0), product("A2", 10.0)]
        with mock.patch.object(comparison, "get_reviews_for_product", return_value=[]):
            summary = _generate_comparison_summary(products)
        self.assertIn("$10.00", summary)
        self.assertNotIn("$$", summary)

    def test_no_invented_review_text(self):
        products = [product("A1", 10.0), product("A2", 12.0)]
        with mock.patch.object(comparison, "get_reviews_for_product", return_value=[]):
            summary = _generate_comparison_summary(products)
        self.assertNotIn("Reviewers consistently praise", summary)
        self.assertIn("No customer review text is available", summary)

    def test_uses_real_review_when_present(self):
        products = [product("A1", 10.0), product("A2", 12.0)]
        review = {"helpful_vote": 5, "text": "Works great for daily use.", "title": "Good"}
        with mock.patch.object(comparison, "get_reviews_for_product", return_value=[review]):
            summary = _generate_comparison_summary(products)
        self.assertIn("Works great for daily use.", summary)


class TestPolicyNoInventedText(unittest.TestCase):

    def test_return_policy_without_info(self):
        fake = {"features_text": "A nice water bottle.", "description_text": "Keeps drinks cold."}
        with mock.patch("policy_qa.get_product_by_asin", return_value=fake):
            answer = answer_policy("A1", "return")
        self.assertIn("No return, refund, or exchange information", answer)
        self.assertNotIn("Standard 30-day", answer)

    def test_return_policy_with_info(self):
        fake = {
            "features_text": "Free returns within 30 days.",
            "description_text": "Contact support to start a return.",
        }
        with mock.patch("policy_qa.get_product_by_asin", return_value=fake):
            answer = answer_policy("A1", "return")
        self.assertIn("Return & Exchange Policy", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
