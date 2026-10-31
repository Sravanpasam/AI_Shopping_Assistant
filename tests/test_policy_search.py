"""
Tests for policy_search (System B knowledge tool).

These use the real embedding model (all-MiniLM-L6-v2) so they verify grounding:
a policy question must retrieve the correct source document.

Run:  .venv\\Scripts\\python.exe tests\\test_policy_search.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from policy_search import get_chunks, search_policies, DOCS_DIR  # noqa: E402

EXPECTED_DOCS = {
    "warranty_and_guarantee.md",
    "returns_and_refunds.md",
    "shipping_and_delivery.md",
    "payment_and_cash_on_delivery.md",
    "order_tracking_and_cancellation.md",
}


class TestChunking(unittest.TestCase):

    def test_all_docs_are_chunked(self):
        sources = {c["source"] for c in get_chunks()}
        self.assertTrue(EXPECTED_DOCS.issubset(sources), sources)

    def test_chunks_have_required_fields(self):
        chunks = get_chunks()
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertTrue(chunk["text"].strip())
            self.assertTrue(chunk["heading"].strip())
            self.assertTrue(chunk["source"].endswith(".md"))


class TestSearch(unittest.TestCase):

    def test_return_window_grounds_in_returns_doc(self):
        hits = search_policies("What is the return window?", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["source"], "returns_and_refunds.md")
        self.assertIn("14 days", hits[0]["text"])

    def test_warranty_question_grounds_in_warranty_doc(self):
        hits = search_policies("How long is the warranty on electronics?", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["source"], "warranty_and_guarantee.md")

    def test_cod_question_grounds_in_payment_doc(self):
        hits = search_policies("Can I pay cash on delivery?", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["source"], "payment_and_cash_on_delivery.md")

    def test_result_shape_and_order(self):
        hits = search_policies("How do I track my order?", k=3)
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), 3)
        for hit in hits:
            self.assertEqual(set(hit.keys()), {"text", "source", "heading", "score"})
            self.assertIsInstance(hit["score"], float)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_blank_query_returns_empty(self):
        self.assertEqual(search_policies(""), [])
        self.assertEqual(search_policies("   "), [])

    def test_k_limits_results(self):
        self.assertLessEqual(len(search_policies("return and refund", k=1)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
