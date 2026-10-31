"""
Tests for legacy.app_baseline._resolve_policy_product (fix #5).

Run:  .venv\\Scripts\\python.exe tests\\test_policy_resolution.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "legacy"))

from app_baseline import _resolve_policy_product  # noqa: E402


def P(asin):
    return {"parent_asin": asin, "title": asin}


class TestResolvePolicyProduct(unittest.TestCase):

    def setUp(self):
        self.last = [P("A1"), P("A2"), P("A3")]

    # -- ordinal references ---------------------------------------------
    def test_second_one(self):
        session = {"last_products": self.last, "focused_product": None}
        result = _resolve_policy_product("what is the return policy for the second one", session)
        self.assertEqual(result, self.last[1])

    def test_third(self):
        session = {"last_products": self.last, "focused_product": None}
        self.assertEqual(_resolve_policy_product("warranty on the third", session), self.last[2])

    def test_first(self):
        session = {"last_products": self.last, "focused_product": None}
        self.assertEqual(_resolve_policy_product("cod for the first", session), self.last[0])

    def test_last(self):
        session = {"last_products": self.last, "focused_product": None}
        self.assertEqual(_resolve_policy_product("return policy of the last one", session), self.last[-1])

    # -- named product --------------------------------------------------
    def test_named_product_searches(self):
        session = {"last_products": self.last, "focused_product": None}
        with mock.patch("product_search.search_products", return_value=[P("NAMED")]) as m:
            result = _resolve_policy_product("what is the warranty on the Sony headphones", session)
        self.assertEqual(result["parent_asin"], "NAMED")
        m.assert_called_once()
        query = m.call_args.args[0]
        self.assertIn("sony", query)
        self.assertIn("headphones", query)

    # -- fallbacks ------------------------------------------------------
    def test_reference_uses_focused_product(self):
        session = {"last_products": self.last, "focused_product": P("FOCUS")}
        self.assertEqual(
            _resolve_policy_product("can I return this", session)["parent_asin"], "FOCUS"
        )

    def test_falls_back_to_first_shown_product(self):
        session = {"last_products": self.last, "focused_product": None}
        self.assertEqual(_resolve_policy_product("can I return this", session), self.last[0])

    def test_no_context_returns_none(self):
        session = {"last_products": [], "focused_product": None}
        self.assertIsNone(_resolve_policy_product("can I return this", session))


if __name__ == "__main__":
    unittest.main(verbosity=2)
