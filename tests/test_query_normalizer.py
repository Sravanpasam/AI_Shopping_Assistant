"""
Tests for query_normalizer (filler removal, typo repair, price protection).

Run:  .venv\\Scripts\\python.exe tests\\test_query_normalizer.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from price_parser import parse_price_constraint  # noqa: E402
from query_normalizer import normalize_query  # noqa: E402


class TestNormalizeQuery(unittest.TestCase):

    def test_removes_english_filler(self):
        self.assertEqual(normalize_query("I want a JINRI hair dryer"), "jinri hair dryer")
        self.assertEqual(normalize_query("show me shampoo please"), "shampoo")

    def test_removes_use_description_filler(self):
        self.assertEqual(
            normalize_query("something to moisturize my body after a shower"),
            "moisturize body after shower",
        )

    def test_repairs_letter_typos(self):
        self.assertEqual(normalize_query("shampo"), "shampoo")
        self.assertEqual(normalize_query("sandalwod perfume"), "sandalwood perfume")

    def test_repairs_transposition_using_catalog_context(self):
        self.assertEqual(normalize_query("curling wnad"), "curling wand")
        self.assertEqual(normalize_query("nail polsih"), "nail polish")

    def test_does_not_rewrite_unknown_nonsense_words(self):
        self.assertEqual(normalize_query("dragon scale armor"), "dragon scale armor")
        self.assertEqual(normalize_query("lunar rover tires"), "lunar rover tires")

    def test_price_phrase_survives_normalization(self):
        normalized = normalize_query("show me shampoo under 20")
        self.assertIn("under 20", normalized)
        _, _, max_usd = parse_price_constraint(normalized)
        self.assertEqual(max_usd, 20.0)

    def test_arabic_filler_removed(self):
        self.assertEqual(normalize_query("أريد شامبو"), "شامبو")


if __name__ == "__main__":
    unittest.main(verbosity=2)
