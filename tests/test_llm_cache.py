"""
Tests for llm_cache.py (opt-in evaluation response cache) and its /api/chat use.

No network and no API key: the provider is faked.

Run:  .venv\\Scripts\\python.exe tests\\test_llm_cache.py
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import llm_cache  # noqa: E402


def _reset():
    """Drop any in-memory cache state so each test starts clean."""
    llm_cache._loaded = False
    llm_cache._store = {}


class TestLlmCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="llmcache_")
        self.env = {
            "LLM_CACHE": os.environ.get("LLM_CACHE"),
            "LLM_CACHE_PATH": os.environ.get("LLM_CACHE_PATH"),
        }
        os.environ["LLM_CACHE_PATH"] = os.path.join(self.tmp, "cache.json")
        _reset()

    def tearDown(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enabled_flag(self):
        for value in ("1", "true", "YES", " on "):
            os.environ["LLM_CACHE"] = value
            self.assertTrue(llm_cache.enabled(), value)
        for value in ("0", "false", "", "no"):
            os.environ["LLM_CACHE"] = value
            self.assertFalse(llm_cache.enabled(), value)

    def test_make_key_is_stable_and_sensitive(self):
        a = llm_cache.make_key("hi", {"page": "search"}, [])
        b = llm_cache.make_key("hi", {"page": "search"}, [])
        self.assertEqual(a, b)
        self.assertNotEqual(a, llm_cache.make_key("hello", {"page": "search"}, []))
        self.assertNotEqual(a, llm_cache.make_key("hi", {"page": "product"}, []))

    def test_round_trip_and_persistence(self):
        key = llm_cache.make_key("q")
        self.assertIsNone(llm_cache.get(key))
        llm_cache.put(key, {"text": "answer", "actions": []})
        self.assertEqual(llm_cache.get(key)["text"], "answer")
        self.assertTrue(os.path.exists(llm_cache.path()))

        # A fresh process would reload the file.
        _reset()
        self.assertEqual(llm_cache.get(key)["text"], "answer")

    def test_corrupt_file_is_ignored(self):
        with open(llm_cache.path(), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        _reset()
        self.assertIsNone(llm_cache.get("anything"))
        llm_cache.put("k", {"text": "ok"})  # still usable afterwards
        self.assertEqual(llm_cache.get("k")["text"], "ok")

    def test_clear_removes_entries_and_file(self):
        llm_cache.put("k", {"text": "ok"})
        llm_cache.clear()
        self.assertIsNone(llm_cache.get("k"))
        self.assertFalse(os.path.exists(llm_cache.path()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
