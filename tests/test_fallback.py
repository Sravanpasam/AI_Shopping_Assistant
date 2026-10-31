"""Tests for the rule-based limited-mode fallback (no LLM)."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import fallback  # noqa: E402
from translation import TranslationError  # noqa: E402


class TestFallbackSearch(unittest.TestCase):

    def test_searches_the_real_catalog(self):
        result = fallback.fallback_turn("moisturizer for dry skin under $25", "en")
        self.assertTrue(result.reply)
        self.assertTrue(result.products)
        self.assertTrue(all(float(p["price"]) <= 25 for p in result.products))
        self.assertIn("search_products", [t["name"] for t in result.tools_used])
        self.assertTrue(any(a["page"] == "search" for a in result.actions))

    def test_remembers_products_for_a_follow_up(self):
        session = {}
        fallback.fallback_turn("find me a mascara", "en", session)
        result = fallback.fallback_turn("compare these two", "en", session)
        self.assertIsNotNone(result.table)
        self.assertIn("compare_products", [t["name"] for t in result.tools_used])

    def test_cheaper_options_use_the_previous_search(self):
        session = {}
        first = fallback.fallback_turn("find me a mascara", "en", session)
        cheapest = min(float(p["price"]) for p in first.products)
        result = fallback.fallback_turn("show me cheaper options", "en", session)
        self.assertTrue(result.products)
        self.assertTrue(all(float(p["price"]) < cheapest for p in result.products))

    def test_details_open_the_product(self):
        session = {}
        fallback.fallback_turn("find me a mascara", "en", session)
        result = fallback.fallback_turn("tell me about the first one", "en", session)
        self.assertIn("Price:", result.reply)
        self.assertTrue(any(a["page"] == "product" for a in result.actions))


class TestFallbackPoliciesAndOrders(unittest.TestCase):

    def test_policy_answer_has_a_source(self):
        result = fallback.fallback_turn("What is your return policy?", "en")
        self.assertIn("Source:", result.reply)
        self.assertIn("returns_and_refunds.md", result.reply)

    def test_warranty_for_viewed_product(self):
        search = fallback.fallback_turn("find me a mascara", "en")
        asin = search.products[0]["parent_asin"]
        result = fallback.fallback_turn(
            "Does this product have a warranty?", "en", {},
            {"page": "product", "asin": asin},
        )
        self.assertIn("warrant", result.reply.lower())
        self.assertIn("get_product_details", [t["name"] for t in result.tools_used])

    def test_order_status_and_return_policy(self):
        result = fallback.fallback_turn(
            "What is the status of ORD-10025 and can I return it?", "en"
        )
        self.assertIn("ORD-10025", result.reply)
        self.assertIn("return", result.reply.lower())
        self.assertIn("check_order", [t["name"] for t in result.tools_used])

    def test_unknown_information_is_not_invented(self):
        result = fallback.fallback_turn("Do you ship to Mars?", "en")
        self.assertIn("couldn't find", result.reply.lower())
        self.assertNotIn("yes", result.reply.lower())

    def test_never_raises_on_a_tool_error(self):
        result = fallback.fallback_turn("tell me about the weather", "en")
        self.assertTrue(result.reply)
        self.assertNotIn("Traceback", result.reply)

    def test_arabic_routes_policy_when_translation_is_down(self):
        with mock.patch.object(
            fallback, "translate_to_english", side_effect=TranslationError("down")
        ), mock.patch.object(
            fallback, "translate_to_arabic", side_effect=TranslationError("down")
        ):
            result = fallback.fallback_turn("ما هي سياسة الإرجاع؟", "ar")
        self.assertEqual(result.intent, "policy_return")
        self.assertIn("Source:", result.reply)
        self.assertTrue(any("\u0600" <= ch <= "\u06ff" for ch in result.reply))


if __name__ == "__main__":
    unittest.main()
