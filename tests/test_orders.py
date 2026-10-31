"""
Tests for orders (System B knowledge tool).

Run:  .venv\\Scripts\\python.exe tests\\test_orders.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from orders import get_order, extract_order_id, DB_PATH, FIELDS  # noqa: E402


class TestExtractOrderId(unittest.TestCase):

    def test_finds_order_id_in_sentence(self):
        self.assertEqual(
            extract_order_id("Where is my order ORD-10025 please?"), "ORD-10025"
        )

    def test_case_insensitive_and_uppercased(self):
        self.assertEqual(extract_order_id("track ord-10042"), "ORD-10042")

    def test_returns_none_when_absent(self):
        self.assertIsNone(extract_order_id("where is my order"))
        self.assertIsNone(extract_order_id(""))
        self.assertIsNone(extract_order_id(None))

    def test_first_id_wins(self):
        self.assertEqual(extract_order_id("ORD-10001 and ORD-10002"), "ORD-10001")


@unittest.skipUnless(os.path.exists(DB_PATH), "data/orders.db not present")
class TestGetOrder(unittest.TestCase):

    def test_existing_order_returns_full_row(self):
        order = get_order("ORD-10000")
        self.assertIsNotNone(order)
        self.assertEqual(set(order.keys()), set(FIELDS))
        self.assertEqual(order["order_id"], "ORD-10000")

    def test_lookup_is_case_insensitive(self):
        self.assertIsNotNone(get_order("ord-10000"))

    def test_unknown_order_returns_none(self):
        self.assertIsNone(get_order("ORD-999999"))

    def test_blank_input_returns_none(self):
        self.assertIsNone(get_order(""))
        self.assertIsNone(get_order("   "))
        self.assertIsNone(get_order(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
