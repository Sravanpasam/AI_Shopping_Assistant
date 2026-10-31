"""
Tests for LLM env override behavior and health model reporting.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""


class TestEnvOverride(unittest.TestCase):
    def test_dotenv_override_wins(self):
        import tempfile
        import importlib
        import llm_client

        # Create a temp .env with a value
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env")
        tmp.write("GEMINI_MODEL=gemini-3.8-flash\n")
        tmp.close()
        old_env = os.environ.get("LLM_ENV_FILE")
        old_shell = os.environ.get("GEMINI_MODEL")
        os.environ["LLM_ENV_FILE"] = tmp.name
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"  # stale shell
        try:
            importlib.reload(llm_client)
            # project .env should override shell
            self.assertEqual(os.environ.get("GEMINI_MODEL"), "gemini-3.8-flash")
        finally:
            if old_env is None:
                os.environ.pop("LLM_ENV_FILE", None)
            else:
                os.environ["LLM_ENV_FILE"] = old_env
            if old_shell is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = old_shell
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


class TestHealthModels(unittest.TestCase):
    def setUp(self):
        os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
        os.environ["GEMINI_API_KEY"] = ""
        import llm_client
        llm_client.reset_client()

    def test_health_reports_models_list(self):
        from app import app
        client = app.test_client()
        resp = client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)
        # no secrets
        self.assertNotIn("api_key", data)
        for m in data["models"]:
            self.assertIn("name", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
