"""
Tests for limits (fix #8): session cap, message length, audio upload size.

Run:  .venv\\Scripts\\python.exe tests\\test_limits.py
"""

import io
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app as app_module  # noqa: E402
from app import app, get_session, MAX_SESSIONS, MAX_MESSAGE_CHARS  # noqa: E402


class TestSessionCap(unittest.TestCase):

    def tearDown(self):
        app_module.sessions.clear()

    def test_caps_sessions_and_evicts_oldest(self):
        app_module.sessions.clear()
        for i in range(MAX_SESSIONS + 5):
            get_session(f"sid{i}")
        self.assertLessEqual(len(app_module.sessions), MAX_SESSIONS)
        self.assertNotIn("sid0", app_module.sessions)
        self.assertIn(f"sid{MAX_SESSIONS + 4}", app_module.sessions)


class TestMessageLength(unittest.TestCase):

    def test_over_long_message_rejected(self):
        client = app.test_client()
        resp = client.post(
            "/api/chat",
            json={"message": "x" * (MAX_MESSAGE_CHARS + 1), "session_id": str(uuid.uuid4())},
        )
        self.assertEqual(resp.status_code, 400)


class TestAudioUploadSize(unittest.TestCase):

    def test_oversized_audio_rejected(self):
        client = app.test_client()
        with mock.patch("app.MAX_AUDIO_BYTES", 10):
            resp = client.post(
                "/api/transcribe",
                data={"audio": (io.BytesIO(b"0" * 100), "clip.wav")},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 413)


if __name__ == "__main__":
    unittest.main(verbosity=2)
