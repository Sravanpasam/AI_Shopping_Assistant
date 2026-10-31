"""
Tests for greetings.py (no-LLM greeting short-circuit).

Run:  .venv\\Scripts\\python.exe tests\\test_greetings.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from greetings import is_greeting  # noqa: E402


class TestGreetings(unittest.TestCase):

    def test_english_greetings(self):
        for text in ["hello", "Hi", "hey", "Hello!", "hi there", "good morning",
                     "good evening", "how are you?", "hello 👋", "  HELLO  "]:
            self.assertTrue(is_greeting(text), text)

    def test_arabic_greetings(self):
        for text in ["مرحبا", "أهلا", "السلام عليكم", "صباح الخير", "مساء الخير",
                     "كيف حالك", "مرحبا!"]:
            self.assertTrue(is_greeting(text), text)

    def test_transliterated_greetings(self):
        for text in ["salam", "marhaba", "ahlan", "as-salamu alaykum"]:
            self.assertTrue(is_greeting(text), text)

    def test_not_greetings(self):
        for text in ["", "   ", "hi, find me a mascara", "hello can I return this",
                     "what is the return policy", "thanks", "hello there my friend",
                     "I want to speak to a human", "متى يوصل الطلب"]:
            self.assertFalse(is_greeting(text), text)

    def test_request_after_greeting_is_not_swallowed(self):
        self.assertFalse(is_greeting("hi, do you have this in blue?"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
