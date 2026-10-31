"""
Tests for agent_tools (System B tool wrappers).

These use the real catalog, reviews, policy docs and orders database, so they
check grounding and the never-raise / JSON-safe contract. No network.

Run:  .venv\\Scripts\\python.exe tests\\test_agent_tools.py
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import agent_tools  # noqa: E402
from agent_tools import TOOL_SCHEMAS, TOOL_FUNCTIONS, call_tool  # noqa: E402
from config import SEARCH_FLOOR  # noqa: E402


EXPECTED_TOOLS = {
    "search_products",
    "get_product_details",
    "compare_products",
    "search_reviews",
    "search_policies",
    "check_order",
    "build_routine",
    "gift_finder",
    "best_in_category",
    "escalate_to_human",
}


class TestSchemas(unittest.TestCase):

    def test_all_tools_declared(self):
        names = {s["name"] for s in TOOL_SCHEMAS}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_schemas_are_valid_json_and_described(self):
        for schema in TOOL_SCHEMAS:
            self.assertTrue(schema["description"].strip())
            params = schema["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertIsInstance(params.get("properties"), dict)
            json.dumps(schema)  # must be serializable

    def test_every_schema_has_a_function(self):
        self.assertEqual(set(TOOL_FUNCTIONS), EXPECTED_TOOLS)

    def test_descriptions_mention_english(self):
        for schema in TOOL_SCHEMAS:
            self.assertIn("English", schema["description"])


class TestSearchProducts(unittest.TestCase):

    def test_finds_relevant_products(self):
        result = agent_tools.search_products("waterproof mascara", limit=3)
        self.assertEqual(result["count"], len(result["products"]))
        self.assertGreater(result["count"], 0)
        self.assertGreaterEqual(result["products"][0]["similarity_score"], SEARCH_FLOOR)

    def test_total_matches_counts_before_the_limit(self):
        result = agent_tools.search_products("mascara", limit=1)
        self.assertIn("total_matches", result)
        self.assertEqual(len(result["products"]), result["count"])
        self.assertGreaterEqual(result["total_matches"], result["count"])

    def test_total_matches_respects_the_price_filter(self):
        result = agent_tools.search_products("mascara", max_price_usd=8, limit=10)
        self.assertGreaterEqual(result["total_matches"], result["count"])
        for product in result["products"]:
            self.assertLessEqual(product["price"], 8)

    def test_respects_max_price(self):
        result = agent_tools.search_products("mascara", max_price_usd=8, limit=10)
        for product in result["products"]:
            self.assertGreater(product["price"], 0)
            self.assertLessEqual(product["price"], 8)

    def test_empty_query_has_note(self):
        result = agent_tools.search_products("")
        self.assertEqual(result["count"], 0)
        self.assertIn("note", result)

    def test_normalizes_a_typo(self):
        result = agent_tools.search_products("JINRI hair dyer", limit=1)
        self.assertGreater(result["count"], 0)
        self.assertIn("hair dryer", result["products"][0]["title"].lower())

    def test_nonsense_gives_real_category_suggestions(self):
        result = agent_tools.search_products("quantum flux capacitor")
        self.assertEqual(result["count"], 0)
        self.assertFalse(result["approximate"])
        self.assertEqual(len(result["category_suggestions"]), 3)
        for suggestion in result["category_suggestions"]:
            self.assertNotEqual(suggestion["value"], "other")

    def test_result_is_json_serializable(self):
        json.dumps(agent_tools.search_products("mascara", limit=2))


class TestGetProductDetails(unittest.TestCase):

    def test_found(self):
        hit = agent_tools.search_products("mascara", limit=1)["products"][0]
        result = agent_tools.get_product_details(hit["parent_asin"])
        self.assertTrue(result["found"])
        self.assertIn("title", result["product"])

    def test_missing_id(self):
        result = agent_tools.get_product_details("NOT-A-REAL-ASIN")
        self.assertFalse(result["found"])
        self.assertIn("note", result)


class TestCompare(unittest.TestCase):

    def test_compares_two_products(self):
        products = agent_tools.search_products("mascara", limit=3)["products"]
        asins = [p["parent_asin"] for p in products[:2]]
        result = agent_tools.compare_products(asins)
        self.assertTrue(result["compared"])
        self.assertEqual(len(result["table"]["headers"]), 3)
        json.dumps(result)

    def test_needs_two(self):
        result = agent_tools.compare_products(["ONLY-ONE"])
        self.assertFalse(result["compared"])
        self.assertIn("note", result)

    def test_caps_at_three(self):
        products = agent_tools.search_products("mascara", limit=5)["products"]
        asins = [p["parent_asin"] for p in products]
        result = agent_tools.compare_products(asins)
        self.assertLessEqual(len(result["parent_asins"]), 3)


class TestSearchReviews(unittest.TestCase):

    def test_returns_reviews(self):
        hit = agent_tools.search_products("mascara", limit=1)["products"][0]
        result = agent_tools.search_reviews(hit["parent_asin"], limit=3)
        self.assertGreater(result["count"], 0)
        for review in result["reviews"]:
            self.assertIn("text", review)
        json.dumps(result)

    def test_query_ranking(self):
        hit = agent_tools.search_products("mascara", limit=1)["products"][0]
        result = agent_tools.search_reviews(hit["parent_asin"], query="smudge", limit=2)
        self.assertLessEqual(result["count"], 2)

    def test_unknown_product(self):
        result = agent_tools.search_reviews("NOT-A-REAL-ASIN")
        self.assertEqual(result["count"], 0)
        self.assertIn("note", result)


class TestSearchPolicies(unittest.TestCase):

    def test_return_question_grounds(self):
        result = agent_tools.search_policies("What is your return window?")
        self.assertGreater(result["count"], 0)
        self.assertEqual(result["policies"][0]["source"], "returns_and_refunds.md")

    def test_empty_query(self):
        result = agent_tools.search_policies("")
        self.assertEqual(result["count"], 0)


class TestCheckOrder(unittest.TestCase):

    def test_found_with_ord_id(self):
        result = agent_tools.check_order("where is my order ORD-10025?")
        self.assertTrue(result["found"])
        self.assertEqual(result["order"]["order_id"], "ORD-10025")

    def test_bare_number(self):
        result = agent_tools.check_order("10025")
        self.assertTrue(result["found"])

    def test_not_found(self):
        result = agent_tools.check_order("ORD-99999999")
        self.assertFalse(result["found"])
        self.assertIn("note", result)

    def test_no_id_in_text(self):
        result = agent_tools.check_order("no id here")
        self.assertFalse(result["found"])
        self.assertIsNone(result["order_id"])


class TestEscalation(unittest.TestCase):

    def test_escalates(self):
        result = agent_tools.escalate_to_human("user is furious")
        self.assertTrue(result["escalated"])
        self.assertTrue(result["reason"])


class TestDispatch(unittest.TestCase):

    def test_unknown_tool(self):
        self.assertIn("error", call_tool("does_not_exist", {}))

    def test_bad_arguments(self):
        self.assertIn("error", call_tool("search_products", {"wrong": 1}))

    def test_dispatch_works(self):
        result = call_tool("search_products", {"query": "mascara", "limit": 1})
        self.assertEqual(result["count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
