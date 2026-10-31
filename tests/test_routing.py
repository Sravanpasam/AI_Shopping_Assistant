"""
Tests for backend message routing: price parsing + intent + low-confidence handling.

These assert intent and price filters only (never specific catalog products), so
they stay valid when the catalog is swapped for the real ~1,000-product data.

Run:  .venv\\Scripts\\python.exe tests\\test_routing.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import nlp_engine  # noqa: E402
from nlp_engine import route_message, is_short_noun_phrase  # noqa: E402


class TestRouteMessage(unittest.TestCase):

    def assert_route(self, message, intent, price_filter=False, min_usd=None, max_usd=None):
        result = route_message(message)
        self.assertEqual(result["intent"], intent, f"intent for {message!r}")
        self.assertEqual(result["price_filter"], price_filter, f"price_filter for {message!r}")
        if max_usd is not None:
            self.assertIsNotNone(result["max_usd"], f"max_usd for {message!r}")
            self.assertAlmostEqual(result["max_usd"], max_usd, places=2)
        if min_usd is not None:
            self.assertIsNotNone(result["min_usd"], f"min_usd for {message!r}")
            self.assertAlmostEqual(result["min_usd"], min_usd, places=2)

    def test_headphones_under_100(self):
        self.assert_route("headphones under 100", "search_product", True, max_usd=100.0)

    def test_laptop_under_500_dollars(self):
        self.assert_route("laptop under 500 dollars", "search_product", True, max_usd=500.0)

    def test_shampoo(self):
        self.assert_route("shampoo", "search_product")

    def test_waterproof_hiking_boots(self):
        self.assert_route("waterproof hiking boots", "search_product")

    def test_can_i_return_this(self):
        self.assert_route("can I return this", "policy_return")

    def test_cash_on_delivery(self):
        self.assert_route("cash on delivery", "policy_cod")

    def test_compare_these(self):
        self.assert_route("compare these", "compare_products")

    # -- Shopping requests must reach product search, not clarify -----------
    def test_need_a_hair_dryer(self):
        self.assert_route("I need a hair dryer", "search_product")

    def test_do_you_have_curling_wands(self):
        self.assert_route("do you have curling wands", "search_product")

    def test_something_for_frizzy_hair(self):
        self.assert_route("something for frizzy hair", "search_product")

    def test_show_me_shampoo_under_50(self):
        self.assert_route("show me shampoo under 50", "search_product", True, max_usd=50.0)

    def test_want_a_brush_for_hair(self):
        self.assert_route("I want a brush for hair", "search_product")


class TestRouteMessageLowConfidence(unittest.TestCase):
    """Deterministic checks of the low-confidence logic (classifier is mocked)."""

    def test_price_forces_search_even_if_classifier_disagrees(self):
        with mock.patch.object(nlp_engine, "classify_intent", return_value=("policy_cod", 0.99)):
            result = route_message("headphones under 100")
        self.assertEqual(result["intent"], "search_product")
        self.assertTrue(result["price_filter"])

    def test_low_conf_noun_phrase_defaults_to_search(self):
        with mock.patch.object(nlp_engine, "classify_intent", return_value=("compare_products", 0.2)):
            result = route_message("waterproof boots")
        self.assertEqual(result["intent"], "search_product")

    def test_low_conf_shopping_request_defaults_to_search(self):
        with mock.patch.object(nlp_engine, "classify_intent", return_value=("compare_products", 0.2)):
            result = route_message("I need a gift for her")
        self.assertEqual(result["intent"], "search_product")

    def test_low_conf_sentence_asks_to_clarify(self):
        with mock.patch.object(nlp_engine, "classify_intent", return_value=("compare_products", 0.2)):
            result = route_message("tell me more about it")
        self.assertEqual(result["intent"], "clarify")


class TestIsShortNounPhrase(unittest.TestCase):

    def test_noun_phrases(self):
        self.assertTrue(is_short_noun_phrase("headphones"))
        self.assertTrue(is_short_noun_phrase("waterproof hiking boots"))
        self.assertTrue(is_short_noun_phrase("waterproof boots"))

    def test_sentences_and_questions(self):
        self.assertFalse(is_short_noun_phrase("can I return this"))
        self.assertFalse(is_short_noun_phrase("how do I pay"))
        self.assertFalse(is_short_noun_phrase("should i buy this now"))

    def test_empty(self):
        self.assertFalse(is_short_noun_phrase(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
