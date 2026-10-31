"""
Tests for escalation (System B knowledge tool).

Run:  .venv\\Scripts\\python.exe tests\\test_escalation.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from escalation import detect_escalation  # noqa: E402


class TestExplicitHumanRequest(unittest.TestCase):

    def test_talk_to_human(self):
        escalate, reason = detect_escalation("I want to talk to a human")
        self.assertTrue(escalate)
        self.assertEqual(reason, "explicit_human_request")

    def test_speak_to_manager(self):
        self.assertTrue(detect_escalation("Can I speak to a manager?")[0])

    def test_customer_support(self):
        self.assertTrue(detect_escalation("please connect me to customer support")[0])

    def test_arabic(self):
        self.assertTrue(detect_escalation("أريد التحدث مع شخص حقيقي")[0])


class TestFrustration(unittest.TestCase):

    def test_useless_and_broken(self):
        escalate, reason = detect_escalation("This is useless, nothing works")
        self.assertTrue(escalate)
        self.assertEqual(reason, "frustration")

    def test_repeated_complaint(self):
        self.assertTrue(detect_escalation("I already told you this")[0])

    def test_arabic(self):
        self.assertTrue(detect_escalation("هذا غير مقبول")[0])


class TestNeutral(unittest.TestCase):

    def test_policy_question(self):
        self.assertFalse(detect_escalation("What is the return window?")[0])

    def test_arabic_policy_question(self):
        self.assertFalse(detect_escalation("ما هي سياسة الإرجاع؟")[0])

    def test_product_search(self):
        self.assertFalse(detect_escalation("show me hair dryers")[0])

    def test_greeting(self):
        self.assertFalse(detect_escalation("hello")[0])

    def test_generic_help_request_is_not_escalation(self):
        self.assertFalse(detect_escalation("I need help with my order")[0])

    def test_blank(self):
        self.assertEqual(detect_escalation(""), (False, None))
        self.assertEqual(detect_escalation(None), (False, None))


class TestSimilarityFallback(unittest.TestCase):

    def test_paraphrased_human_request(self):
        # No rule match; should be caught by similarity to the human phrases.
        escalate, reason = detect_escalation(
            "can you put me through to somebody who can actually help me"
        )
        self.assertTrue(escalate)
        self.assertEqual(reason, "explicit_human_request")

    def test_paraphrased_frustration(self):
        escalate, reason = detect_escalation(
            "you keep telling me the same thing and it never works"
        )
        self.assertTrue(escalate)
        self.assertEqual(reason, "frustration")


if __name__ == "__main__":
    unittest.main(verbosity=2)
