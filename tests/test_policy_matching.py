"""
Tests for policy matching (fix #9): newline sentence splitting and
false-positive resistance for 'guaranteed' and bare 'replacement'.

Run:  .venv\\Scripts\\python.exe tests\\test_policy_matching.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from policy_qa import (  # noqa: E402
    _split_into_sentences, _WARRANTY_REGEX, _RETURN_REGEX,
)


class TestNewlineSplitting(unittest.TestCase):

    def test_newlines_split_features_into_separate_sentences(self):
        text = "Ergonomic handle for comfort\nGuarantee & Return Policy: 30 Days No-reason to return."
        sentences = _split_into_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("Ergonomic handle for comfort", sentences)

    def test_sentence_punctuation_still_splits(self):
        sentences = _split_into_sentences("First line. Second line! Third line?")
        self.assertEqual(sentences, ["First line.", "Second line!", "Third line?"])

    def test_blank_lines_dropped(self):
        self.assertEqual(_split_into_sentences("  \n \n only one \n"), ["only one"])


class TestWarrantyFalsePositives(unittest.TestCase):

    def _match(self, s: str) -> bool:
        return bool(_WARRANTY_REGEX.search(s))

    def test_quality_adjective_not_a_warranty(self):
        self.assertFalse(self._match("a guaranteed hit with all of the little dinosaurs"))
        self.assertFalse(self._match("velvet fabric guarantee its durability and wearability"))

    def test_real_warranty_statements_still_match(self):
        self.assertTrue(self._match("The Ultra Plus model includes a three-year warranty."))
        self.assertTrue(self._match("We offer a 30-day money back guarantee."))
        self.assertTrue(self._match("100% manufacturer guarantee."))


class TestReturnFalsePositives(unittest.TestCase):

    def _match(self, s: str) -> bool:
        return bool(_RETURN_REGEX.search(s))

    def test_spare_part_replacement_not_a_return_policy(self):
        self.assertFalse(self._match("Replacement foil and cutter block for 360 Complete models 8995."))
        self.assertFalse(self._match("The dental tool kit won't be the replacement of all you dental cleaning."))

    def test_real_return_statements_still_match(self):
        self.assertTrue(self._match("If you're not fully satisfied, you can return it for a full refund."))
        self.assertTrue(self._match("We provide 15-days return/replacement service."))
        self.assertTrue(self._match("30 Days No-reason to return."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
