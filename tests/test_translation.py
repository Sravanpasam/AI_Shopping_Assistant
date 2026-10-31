"""
Tests for the translation layer: retry on rate limit and explicit failure.

Run:  .venv\\Scripts\\python.exe tests\\test_translation.py
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "legacy"))

import translation  # noqa: E402
from deep_translator.exceptions import TooManyRequests  # noqa: E402

from translation import translate_to_english, translate_to_arabic, TranslationError  # noqa: E402
from strings_ar import get_string  # noqa: E402

import app_baseline as app_module  # noqa: E402

ARABIC_TEXT = "أريد شامبو للشعر"


class TranslationMockMixin:
    """Patch the network call and the back-off sleep for every test."""

    def setUp(self):
        translate_to_english.cache_clear()
        translate_to_arabic.cache_clear()

        # Replace the real 3/sec limiter with a permissive one so the patched
        # (no-op) sleep below can never make acquire() spin.
        self._real_limiter = translation._rate_limiter
        translation._rate_limiter = translation._RateLimiter(max_calls=10 ** 9, period=1.0)
        self.addCleanup(self._restore_limiter)

        self.sleep_patcher = mock.patch("translation.time.sleep")
        self.mock_sleep = self.sleep_patcher.start()
        self.addCleanup(self.sleep_patcher.stop)

        self.gt_patcher = mock.patch("translation.GoogleTranslator")
        self.mock_gt = self.gt_patcher.start()
        self.addCleanup(self.gt_patcher.stop)

    def _restore_limiter(self):
        translation._rate_limiter = self._real_limiter


class TestRetryOnRateLimit(TranslationMockMixin, unittest.TestCase):

    def test_rate_limit_then_success(self):
        self.mock_gt.return_value.translate.side_effect = [TooManyRequests("429"), "shampoo"]
        result = translate_to_english(ARABIC_TEXT)
        self.assertEqual(result, "shampoo")
        self.assertEqual(self.mock_gt.return_value.translate.call_count, 2)
        self.mock_sleep.assert_called_once_with(1.0)

    def test_rate_limit_on_every_attempt_raises(self):
        self.mock_gt.return_value.translate.side_effect = TooManyRequests("429")
        with self.assertRaises(TranslationError):
            translate_to_english(ARABIC_TEXT)
        # 1 initial attempt + 2 retries
        self.assertEqual(self.mock_gt.return_value.translate.call_count, 3)
        self.assertEqual([c.args[0] for c in self.mock_sleep.call_args_list], [1.0, 2.0])

    def test_non_rate_limit_error_raises_without_retry(self):
        self.mock_gt.return_value.translate.side_effect = ValueError("boom")
        with self.assertRaises(TranslationError):
            translate_to_english(ARABIC_TEXT)
        self.assertEqual(self.mock_gt.return_value.translate.call_count, 1)
        self.mock_sleep.assert_not_called()

    def test_empty_result_raises_instead_of_returning_original(self):
        self.mock_gt.return_value.translate.return_value = "   "
        with self.assertRaises(TranslationError):
            translate_to_arabic("hello")

    def test_never_returns_untranslated_input(self):
        self.mock_gt.return_value.translate.side_effect = TooManyRequests("429")
        with self.assertRaises(TranslationError):
            translate_to_arabic("hello world")


class TestCache(TranslationMockMixin, unittest.TestCase):

    def test_arabic_cache_hit_calls_translator_once(self):
        self.mock_gt.return_value.translate.return_value = "مرحبا"
        first = translate_to_arabic("hello")
        second = translate_to_arabic("hello")
        self.assertEqual(first, "مرحبا")
        self.assertEqual(second, "مرحبا")
        self.assertEqual(self.mock_gt.return_value.translate.call_count, 1)

    def test_english_cache_hit_calls_translator_once(self):
        self.mock_gt.return_value.translate.return_value = "shampoo"
        translate_to_english(ARABIC_TEXT)
        translate_to_english(ARABIC_TEXT)
        self.assertEqual(self.mock_gt.return_value.translate.call_count, 1)


def _route(intent):
    return {
        "intent": intent,
        "confidence": 0.9,
        "price_filter": False,
        "classified_intent": intent,
        "min_usd": None,
        "max_usd": None,
    }


class TestStaticStringsAppWiring(unittest.TestCase):
    """Static reply text must come from strings_ar (never the online translator)."""

    def test_arabic_greeting_uses_static_string_without_translator(self):
        client = app_module.app.test_client()
        with mock.patch("app_baseline.ensure_nlp"), \
                mock.patch("nlp_engine.route_message", return_value=_route("greeting")), \
                mock.patch("translation.detect_language", return_value="ar"), \
                mock.patch("translation.translate_to_english", return_value="hello"), \
                mock.patch("translation.translate_to_arabic") as mock_ar:
            resp = client.post("/api/chat", json={"message": "مرحبا", "session_id": "s-greet"})
            self.assertEqual(resp.status_code, 200)
            body = resp.get_json()
            self.assertEqual(body["text"], get_string("greeting", "ar"))
            self.assertEqual(body["intent"], "greeting")
            mock_ar.assert_not_called()

    def test_dynamic_policy_text_is_translated(self):
        client = app_module.app.test_client()
        with mock.patch("app_baseline.ensure_nlp"), \
                mock.patch("nlp_engine.route_message", return_value=_route("policy_warranty")), \
                mock.patch("translation.detect_language", return_value="ar"), \
                mock.patch("translation.translate_to_english", return_value="warranty"), \
                mock.patch("app_baseline._handle_policy", return_value=("Warranty info: 2 years", True)), \
                mock.patch("translation.translate_to_arabic", return_value="الضمان") as mock_ar:
            resp = client.post("/api/chat", json={"message": "ما الضمان؟", "session_id": "s-pol"})
            body = resp.get_json()
            self.assertEqual(body["text"], "الضمان")
            mock_ar.assert_called_once_with("Warranty info: 2 years")

    def test_input_translation_failure_returns_arabic_notice(self):
        client = app_module.app.test_client()
        with mock.patch("app_baseline.ensure_nlp"), \
                mock.patch("translation.detect_language", return_value="ar"), \
                mock.patch("translation.translate_to_english", side_effect=TranslationError("429")):
            resp = client.post("/api/chat", json={"message": "مرحبا", "session_id": "s-fail"})
            self.assertEqual(resp.status_code, 200)
            body = resp.get_json()
            self.assertEqual(body["intent"], "translation_error")
            self.assertEqual(body["text"], get_string("translation_unavailable", "ar"))


class TestRateLimiter(unittest.TestCase):

    def test_default_is_three_per_second(self):
        self.assertEqual(translation._rate_limiter.max_calls, 3)
        self.assertEqual(translation._rate_limiter.period, 1.0)

    def test_blocks_beyond_budget(self):
        limiter = translation._RateLimiter(max_calls=2, period=0.3)
        start = time.monotonic()
        limiter.acquire()  # 1st - immediate
        limiter.acquire()  # 2nd - immediate
        self.assertLess(time.monotonic() - start, 0.2)
        limiter.acquire()  # 3rd - must wait for the window to roll over
        self.assertGreaterEqual(time.monotonic() - start, 0.25)


class TestTranslationHealth(unittest.TestCase):

    def setUp(self):
        translation._selftest_state["ts"] = None
        translation._selftest_state["status"] = "unknown"

    def test_health_reports_ok(self):
        client = app_module.app.test_client()
        with mock.patch("translation._translate_with_retry", return_value="تم"):
            body = client.get("/api/health").get_json()
        self.assertEqual(body["translation"], "ok")

    def test_health_reports_degraded_on_failure(self):
        client = app_module.app.test_client()
        with mock.patch("translation._translate_with_retry", side_effect=TranslationError("429")):
            body = client.get("/api/health").get_json()
        self.assertEqual(body["translation"], "degraded")

    def test_selftest_is_cached_within_ttl(self):
        with mock.patch("translation._translate_with_retry", return_value="تم") as mocked:
            self.assertEqual(translation.translation_status(force=True), "ok")
            self.assertEqual(translation.translation_status(), "ok")
            self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
