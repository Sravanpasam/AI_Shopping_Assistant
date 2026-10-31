"""
Tests for backend.comparison.resolve_comparison_products (fix #4).

Run:  .venv\\Scripts\\python.exe tests\\test_comparison.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from comparison import resolve_comparison_products  # noqa: E402


def P(asin):
    return {"parent_asin": asin, "title": asin}


class TestResolveComparisonProducts(unittest.TestCase):

    def setUp(self):
        self.last = [P("A1"), P("A2"), P("A3")]

    # -- reference to previously shown products -------------------------
    def test_these_uses_last_products(self):
        self.assertEqual(resolve_comparison_products("compare these", self.last), self.last)

    def test_them_uses_last_products(self):
        self.assertEqual(resolve_comparison_products("compare them", self.last), self.last)

    def test_both_returns_two(self):
        self.assertEqual(resolve_comparison_products("compare both", self.last), self.last[:2])

    def test_first_two_returns_two(self):
        self.assertEqual(
            resolve_comparison_products("compare the first two", self.last), self.last[:2]
        )

    def test_reference_without_session_returns_empty(self):
        self.assertEqual(resolve_comparison_products("compare these", []), [])

    # -- split named products -------------------------------------------
    def test_split_on_and(self):
        with mock.patch(
            "product_search.search_products",
            side_effect=lambda q, top_k=1: [P(f"ASIN::{q}")],
        ):
            result = resolve_comparison_products("compare Sony and Bose", [])
        self.assertEqual([p["parent_asin"] for p in result], ["ASIN::Sony", "ASIN::Bose"])

    def test_split_on_vs(self):
        with mock.patch(
            "product_search.search_products",
            side_effect=lambda q, top_k=1: [P(f"ASIN::{q}")],
        ):
            result = resolve_comparison_products("Sony vs Bose", [])
        self.assertEqual([p["parent_asin"] for p in result], ["ASIN::Sony", "ASIN::Bose"])

    def test_split_on_versus(self):
        with mock.patch(
            "product_search.search_products",
            side_effect=lambda q, top_k=1: [P(f"ASIN::{q}")],
        ):
            result = resolve_comparison_products("compare shampoo versus conditioner", [])
        self.assertEqual(
            [p["parent_asin"] for p in result], ["ASIN::shampoo", "ASIN::conditioner"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
