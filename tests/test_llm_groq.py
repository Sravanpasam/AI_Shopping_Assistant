"""
Tests for the Groq (OpenAI-compatible) provider in llm_client.py.

No network and no API key: httpx.MockTransport serves canned responses.

Run:  .venv\\Scripts\\python.exe tests\\test_llm_groq.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import httpx  # noqa: E402

import llm_client  # noqa: E402


def _provider(*args, **kwargs):
    # Resolve through the module so a reload elsewhere (test_llm_env) cannot
    # leave us comparing against a stale exception class object.
    return llm_client.GroqProvider(*args, **kwargs)


def _client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="https://api.groq.com/openai/v1")


def _ok(payload):
    return lambda request: httpx.Response(200, json=payload)


class TestGroqParser(unittest.TestCase):
    def test_text_response(self):
        payload = {
            "model": "llama-3.3-70b-versatile",
            "choices": [{"message": {"role": "assistant", "content": "Hello there"}}],
        }
        provider = _provider("k", "llama-3.3-70b-versatile", 5, http_client=_client(_ok(payload)))
        result = provider.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(result.text, "Hello there")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.model, "llama-3.3-70b-versatile")

    def test_tool_call_arguments_are_parsed(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "search_products",
                                    "arguments": json.dumps({"query": "brush", "max_price": 15}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        provider = _provider("k", "m", 5, http_client=_client(_ok(payload)))
        result = provider.generate([{"role": "user", "content": "brush"}])
        self.assertIsNone(result.text)
        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call.name, "search_products")
        self.assertEqual(call.args, {"query": "brush", "max_price": 15})
        self.assertEqual(call.id, "call_abc")


class TestGroqMessageConversion(unittest.TestCase):
    def test_round_trip_shape(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content.decode("utf-8"))
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = _provider("secret-key", "m", 5, http_client=_client(handler))
        provider.generate(
            [
                {"role": "user", "content": "find brush"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "search_products", "args": {"query": "brush"}, "id": "call_1"}],
                },
                {"role": "tool", "name": "search_products", "content": {"products": []}, "id": "call_1"},
            ],
            system="be grounded",
            tools=[{"name": "search_products", "description": "d", "parameters": {"type": "object", "properties": {}}}],
        )

        body = captured["body"]
        self.assertEqual(captured["auth"], "Bearer secret-key")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "be grounded"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "find brush"})
        assistant = body["messages"][2]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "search_products")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"query": "brush"})
        tool = body["messages"][3]
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], "call_1")
        self.assertEqual(json.loads(tool["content"]), {"products": []})
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["tool_choice"], "auto")


class TestGroqErrors(unittest.TestCase):
    def _provider_for(self, status, text):
        return _provider(
            "k", "m", 5, http_client=_client(lambda request: httpx.Response(status, text=text))
        )

    def test_not_found_retires_model(self):
        with self.assertRaises(llm_client.LLMModelUnavailableError):
            self._provider_for(404, "model_not_found").generate([{"role": "user", "content": "hi"}])

    def test_daily_quota(self):
        with self.assertRaises(llm_client.LLMDailyQuotaError):
            self._provider_for(429, "tokens per day exceeded").generate([{"role": "user", "content": "hi"}])

    def test_minute_rate_limit(self):
        with self.assertRaises(llm_client.LLMRateLimitError):
            self._provider_for(429, "requests per minute exceeded").generate([{"role": "user", "content": "hi"}])

    def test_auth_error(self):
        with self.assertRaises(llm_client.LLMConfigError):
            self._provider_for(401, "invalid api key").generate([{"role": "user", "content": "hi"}])

    def test_server_error(self):
        with self.assertRaises(llm_client.LLMServerError):
            self._provider_for(503, "unavailable").generate([{"role": "user", "content": "hi"}])


class TestGroqClientSelection(unittest.TestCase):
    def test_provider_and_model_from_env(self):
        old = {k: os.environ.get(k) for k in ("LLM_PROVIDER", "GROQ_MODEL", "GROQ_FALLBACK_MODELS")}
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_MODEL"] = "llama-3.1-8b-instant"
        os.environ["GROQ_FALLBACK_MODELS"] = "llama-3.3-70b-versatile"
        try:
            client = llm_client.LLMClient(provider_impl=object())
            self.assertEqual(client.provider_name, "groq")
            state = client.pool_state()
            self.assertEqual(state["models"][0]["name"], "llama-3.1-8b-instant")
            self.assertEqual(state["models"][1]["name"], "llama-3.3-70b-versatile")
            self.assertEqual(llm_client.get_model_name(), "llama-3.1-8b-instant")
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_missing_groq_key_is_config_error(self):
        client = llm_client.LLMClient(provider="groq", model="m", api_key=None)
        client._api_key = None
        with self.assertRaises(llm_client.LLMConfigError):
            client._build_provider()


if __name__ == "__main__":
    unittest.main(verbosity=2)
