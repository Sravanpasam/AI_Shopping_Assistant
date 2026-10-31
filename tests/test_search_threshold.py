"""
Tests for floor-plus-margin search behaviour and fallbacks (scores are mocked).

Run:  .venv\\Scripts\\python.exe tests\\test_search_threshold.py
"""

import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "legacy"))

import app_baseline  # noqa: E402
import product_search  # noqa: E402
from product_search import search_products_filtered  # noqa: E402
from config import SEARCH_APPROX_FLOOR, SEARCH_FLOOR, SEARCH_MARGIN  # noqa: E402


def P(asin, score, price):
    return {"parent_asin": asin, "title": asin, "similarity_score": score, "price": price}


class TestThresholdFiltering(unittest.TestCase):

    def test_below_floor_is_no_match(self):
        candidates = [P("lo", SEARCH_APPROX_FLOOR - 0.01, 12.0)]
        with mock.patch.object(product_search, "_search_raw", return_value=candidates), \
             mock.patch.object(product_search, "category_suggestions", return_value=[{"value": "hair"}]):
            result = search_products_filtered("shampoo")
        self.assertEqual(result["products"], [])
        self.assertEqual(result["category_suggestions"][0]["value"], "hair")

    def test_margin_keeps_close_scores_and_drops_far_ones(self):
        candidates = [
            P("top", 0.90, 10.0),
            P("close", 0.90 - SEARCH_MARGIN + 0.01, 12.0),
            P("far", 0.90 - SEARCH_MARGIN - 0.01, 14.0),
        ]
        with mock.patch.object(product_search, "_search_raw", return_value=candidates):
            result = search_products_filtered("shampoo")
        self.assertEqual([p["parent_asin"] for p in result["products"]], ["top", "close"])

    def test_price_filter_does_not_bring_back_subthreshold(self):
        candidates = [P("hi", 0.90, 100.0), P("sub", SEARCH_APPROX_FLOOR - 0.01, 5.0)]
        with mock.patch.object(product_search, "_search_raw", return_value=candidates):
            result = search_products_filtered("shampoo under 50")
        ids = [p["parent_asin"] for p in result["products"]]
        ids += [p["parent_asin"] for p in result["fallback_products"]]
        self.assertNotIn("sub", ids)

    def test_fallback_is_cheapest_relevant(self):
        candidates = [
            P("a", 0.90, 40.0),
            P("b", 0.90 - SEARCH_MARGIN + 0.01, 20.0),
            P("c", 0.90 - SEARCH_MARGIN + 0.02, 30.0),
            P("sub", 0.20, 1.0),
        ]
        with mock.patch.object(product_search, "_search_raw", return_value=candidates):
            result = search_products_filtered("shampoo under 10")
        self.assertEqual(result["products"], [])
        self.assertEqual([p["parent_asin"] for p in result["fallback_products"]], ["b", "c", "a"])

    def test_weak_results_are_flagged_approximate(self):
        weak = (SEARCH_FLOOR + SEARCH_APPROX_FLOOR) / 2
        candidates = [P("weak", weak, 12.0)]
        with mock.patch.object(product_search, "_search_raw", return_value=candidates):
            result = search_products_filtered("gift for my mom")
        self.assertTrue(result["approximate"])
        self.assertEqual(result["products"][0]["parent_asin"], "weak")
        self.assertTrue(result["products"][0]["approximate"])

    def test_scores_below_approx_floor_get_suggestions(self):
        candidates = [P("nonsense", SEARCH_APPROX_FLOOR - 0.01, 12.0)]
        with mock.patch.object(product_search, "_search_raw", return_value=candidates), \
             mock.patch.object(product_search, "category_suggestions", return_value=[{"value": "hair"}]):
            result = search_products_filtered("quantum flux capacitor")
        self.assertEqual(result["products"], [])
        self.assertFalse(result["approximate"])
        self.assertEqual(len(result["category_suggestions"]), 1)


_EMPTY_RESULT = {
    "cleaned_query": "x",
    "min_usd": None,
    "max_usd": None,
    "filter_active": False,
    "products": [],
    "fallback_products": [],
}


class TestChatNoMatch(unittest.TestCase):

    def test_no_match_message(self):
        client = app_baseline.app.test_client()
        with mock.patch("app_baseline.ensure_nlp"), \
             mock.patch("product_search.search_products_filtered", return_value=_EMPTY_RESULT):
            data = client.post(
                "/api/chat",
                json={"message": "show me shampoo under 50", "session_id": str(uuid.uuid4())},
            ).get_json()
        self.assertEqual(data["response_type"], "text")
        self.assertEqual(data["products"], [])
        self.assertIn("beauty and personal care", data["text"])
        self.assertIn("rephras", data["text"].lower())

    def test_compare_these_fresh_session_does_not_search(self):
        client = app_baseline.app.test_client()
        with mock.patch("app_baseline.ensure_nlp"), \
             mock.patch("product_search.search_products") as sp, \
             mock.patch("product_search.search_products_filtered") as spf:
            data = client.post(
                "/api/chat",
                json={"message": "compare these", "session_id": str(uuid.uuid4())},
            ).get_json()
        sp.assert_not_called()
        spf.assert_not_called()
        self.assertEqual(data["response_type"], "text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
