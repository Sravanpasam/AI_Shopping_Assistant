"""
Tests for llm_client (System B). The provider SDK is never called over the
network: we inject a fake provider (``provider_impl``) and also exercise the
Gemini conversion helpers directly (constructing a client does not send a
request).

Run:  .venv\\Scripts\\python.exe tests\\test_llm_client.py
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Tests must never read the developer's real .env (it holds a live key): point
# dotenv at a path that does not exist and clear the key. This must happen
# before importing llm_client, which loads .env at import time.
os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import llm_client  # noqa: E402
from llm_client import (  # noqa: E402
    LLMClient,
    LLMResponse,
    ToolCall,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    LLMDailyQuotaError,
    LLMModelUnavailableError,
    LLMQuotaExhaustedError,
    ModelPool,
    _RateLimiter,
    _map_gemini_error,
)

DEMO_TOOL = {
    "name": "get_current_time",
    "description": "Return the current time for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class _FakeProvider:
    """Minimal provider stub: returns queued results or raises queued errors."""

    name = "fake"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate(self, messages, tools=None, system=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestRateLimiter(unittest.TestCase):

    def test_allows_up_to_limit_then_waits(self):
        limiter = _RateLimiter(max_calls=2, period=0.2)
        start = time.monotonic()
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()  # third must wait for the window
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.15)


class TestRetries(unittest.TestCase):

    def test_retries_then_succeeds(self):
        provider = _FakeProvider([
            LLMRateLimitError("busy"),
            LLMServerError("busy"),
            LLMResponse(text="hello"),
        ])
        client = LLMClient(provider_impl=provider, max_requests_per_minute=1000)
        with mock.patch("llm_client.time.sleep") as sleeper:
            response = client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(response.text, "hello")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(sleeper.call_count, 2)

    def test_exhausts_retries_raises(self):
        provider = _FakeProvider([LLMRateLimitError("busy")] * 4)
        client = LLMClient(
            provider_impl=provider, max_requests_per_minute=1000, max_retries=3
        )
        with mock.patch("llm_client.time.sleep"):
            with self.assertRaises(LLMRateLimitError):
                client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(provider.calls, 4)  # 1 try + 3 retries

    def test_non_retryable_raises_immediately(self):
        provider = _FakeProvider([LLMResponseError("bad request")])
        client = LLMClient(provider_impl=provider, max_requests_per_minute=1000)
        with self.assertRaises(LLMResponseError):
            client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(provider.calls, 1)

    def test_latency_is_recorded(self):
        provider = _FakeProvider([LLMResponse(text="ok")])
        client = LLMClient(provider_impl=provider, max_requests_per_minute=1000)
        response = client.generate([{"role": "user", "content": "hi"}])
        self.assertIsNotNone(response.latency_s)
        self.assertGreaterEqual(response.latency_s, 0.0)


class TestToolParsing(unittest.TestCase):

    def test_tool_calls_surface(self):
        provider = _FakeProvider([
            LLMResponse(tool_calls=[ToolCall(name="get_current_time", args={"city": "Tokyo"})]),
        ])
        client = LLMClient(provider_impl=provider, max_requests_per_minute=1000)
        response = client.generate([{"role": "user", "content": "time?"}], tools=[DEMO_TOOL])
        self.assertEqual(response.tool_calls[0].name, "get_current_time")
        self.assertEqual(response.tool_calls[0].args, {"city": "Tokyo"})


class TestConfig(unittest.TestCase):

    def test_missing_key_raises_config_error(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            client = LLMClient(api_key=None)
        with self.assertRaises(LLMConfigError):
            client.generate([{"role": "user", "content": "hi"}])

    def test_unknown_provider_raises(self):
        client = LLMClient(provider="nonsense", api_key="dummy")
        with self.assertRaises(LLMConfigError):
            client.generate([{"role": "user", "content": "hi"}])


class TestHermeticEnvironment(unittest.TestCase):
    """No test may depend on the developer's real .env file."""

    def test_real_dotenv_is_not_loaded(self):
        self.assertTrue(
            llm_client._ENV_FILE.endswith("does-not-exist.env"),
            "tests must point LLM_ENV_FILE at a nonexistent path",
        )
        self.assertFalse(os.getenv("GEMINI_API_KEY"), "real .env key leaked into tests")


class TestKeyNeverLogged(unittest.TestCase):

    def test_secret_not_in_logs(self):
        provider = _FakeProvider([LLMResponseError("boom")])
        client = LLMClient(
            provider_impl=provider, api_key="SECRET-KEY-123", max_requests_per_minute=1000
        )
        with self.assertLogs("llm_client", level="INFO") as captured:
            with self.assertRaises(LLMResponseError):
                client.generate([{"role": "user", "content": "hi"}])
        self.assertNotIn("SECRET-KEY-123", "\n".join(captured.output))


class TestErrorMapping(unittest.TestCase):

    def test_maps_rate_limit(self):
        from google.genai import errors

        exc = errors.APIError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
        mapped = _map_gemini_error(exc)
        self.assertIsInstance(mapped, LLMRateLimitError)
        self.assertTrue(mapped.retryable)

    def test_maps_server_error(self):
        from google.genai import errors

        exc = errors.APIError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
        self.assertIsInstance(_map_gemini_error(exc), LLMServerError)

    def test_maps_auth_error(self):
        from google.genai import errors

        exc = errors.APIError(403, {"error": {"code": 403, "status": "PERMISSION_DENIED"}})
        self.assertIsInstance(_map_gemini_error(exc), LLMConfigError)

    def test_maps_bad_request(self):
        from google.genai import errors

        exc = errors.APIError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
        self.assertIsInstance(_map_gemini_error(exc), LLMResponseError)

    def test_maps_timeout(self):
        self.assertIsInstance(_map_gemini_error(TimeoutError()), LLMTimeoutError)


class TestGeminiConversion(unittest.TestCase):
    """Exercise the neutral -> Gemini conversion without any network call."""

    def setUp(self):
        self.provider = llm_client.GeminiProvider("dummy-key", "gemini-x", 30.0)

    def test_user_message(self):
        contents = self.provider._to_contents([{"role": "user", "content": "hi"}])
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "hi")

    def test_assistant_tool_call(self):
        contents = self.provider._to_contents([
            {"role": "assistant", "content": "", "tool_calls": [{"name": "t", "args": {"x": 1}}]}
        ])
        self.assertEqual(contents[0].role, "model")
        self.assertEqual(contents[0].parts[0].function_call.name, "t")

    def test_tool_response_turn(self):
        contents = self.provider._to_contents([
            {"role": "tool", "name": "t", "content": {"result": 1}}
        ])
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].function_response.name, "t")

    def test_tools_declaration(self):
        tools = self.provider._to_tools([DEMO_TOOL])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].function_declarations[0].name, "get_current_time")


class _RoutedProvider:
    """Provider that returns a per-model outcome and records which model ran.

    ``by_model`` maps a model name to an ``LLMResponse``, an ``Exception``, or a
    ``callable(model, n)`` returning either (where ``n`` is that model's call
    count). The client sets ``provider.model`` before each call, which lets the
    tests observe model fallback.
    """

    name = "fake"

    def __init__(self, by_model):
        self.by_model = by_model
        self.model = None
        self.calls = []

    def generate(self, messages, tools=None, system=None):
        self.calls.append(self.model)
        outcome = self.by_model.get(self.model)
        if callable(outcome):
            outcome = outcome(self.model, self.calls.count(self.model))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestModelPool(unittest.TestCase):

    def test_current_skips_retired_model(self):
        pool = ModelPool(["a", "b"], daily_budget=10)
        self.assertEqual(pool.current(), "a")
        pool.retire("a")
        self.assertEqual(pool.current(), "b")

    def test_budget_moves_to_next_model(self):
        pool = ModelPool(["a", "b"], daily_budget=1)
        pool.record_call("a")
        self.assertEqual(pool.current(), "b")

    def test_all_spent_returns_none(self):
        pool = ModelPool(["a"], daily_budget=1)
        pool.record_call("a")
        self.assertIsNone(pool.current())

    def test_duplicates_removed_and_blank_ignored(self):
        pool = ModelPool(["a", "", "a", "b"], daily_budget=10)
        self.assertEqual(pool.models, ["a", "b"])


class TestModelFallback(unittest.TestCase):

    def _client(self, by_model, models=("a", "b"), max_retries=0, daily_budget=10):
        provider = _RoutedProvider(by_model)
        client = LLMClient(
            provider_impl=provider,
            models=list(models),
            daily_budget=daily_budget,
            max_retries=max_retries,
            max_requests_per_minute=1000,
        )
        return client, provider

    def test_model_not_found_falls_back(self):
        client, provider = self._client({
            "a": LLMModelUnavailableError("retired"),
            "b": LLMResponse(text="from b"),
        })
        resp = client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.text, "from b")
        self.assertEqual(provider.calls, ["a", "b"])
        self.assertEqual(client.active_model(), "b")

    def test_per_day_quota_falls_back(self):
        client, provider = self._client({
            "a": LLMDailyQuotaError("daily"),
            "b": LLMResponse(text="from b"),
        })
        resp = client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.text, "from b")
        self.assertEqual(provider.calls, ["a", "b"])

    def test_switch_remembered_for_the_day(self):
        client, provider = self._client({
            "a": LLMDailyQuotaError("daily"),
            "b": LLMResponse(text="from b"),
        })
        client.generate([{"role": "user", "content": "hi"}])
        client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(provider.calls, ["a", "b", "b"])

    def test_all_models_exhausted_raises(self):
        client, provider = self._client({
            "a": LLMDailyQuotaError("daily"),
            "b": LLMModelUnavailableError("retired"),
        })
        with self.assertRaises(LLMQuotaExhaustedError):
            client.generate([{"role": "user", "content": "hi"}])
        self.assertTrue(client.quota_exhausted())

    def test_budget_exhaustion_raises_when_all_used(self):
        client, provider = self._client(
            {"a": LLMResponse(text="a")}, models=("a",), daily_budget=1
        )
        client.generate([{"role": "user", "content": "hi"}])
        with self.assertRaises(LLMQuotaExhaustedError):
            client.generate([{"role": "user", "content": "hi"}])

    def test_budget_moves_to_next_model(self):
        client, provider = self._client(
            {"a": LLMResponse(text="a"), "b": LLMResponse(text="b")},
            daily_budget=2,
        )
        client.generate([{"role": "user", "content": "hi"}])
        client.generate([{"role": "user", "content": "hi"}])
        client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(provider.calls, ["a", "a", "b"])

    def test_per_minute_rate_limit_retries_same_model(self):
        client, provider = self._client(
            {"a": lambda m, n: LLMRateLimitError("minute") if n == 1 else LLMResponse(text="ok")},
            models=("a",),
            max_retries=2,
        )
        with mock.patch("llm_client.time.sleep"):
            resp = client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.text, "ok")
        self.assertEqual(provider.calls, ["a", "a"])
        self.assertEqual(client.active_model(), "a")

    def test_pool_state_reports_usage(self):
        client, _ = self._client({"a": LLMResponse(text="a")}, models=("a",))
        client.generate([{"role": "user", "content": "hi"}])
        state = client.pool_state()
        self.assertEqual(state["active"], "a")
        self.assertEqual(state["models"][0]["used"], 1)


class TestQuotaHelpers(unittest.TestCase):

    def test_quota_state_ok_before_client(self):
        self.assertEqual(llm_client.quota_state(), "ok")

    def test_quota_state_exhausted_after_failures(self):
        provider = _RoutedProvider({"a": LLMDailyQuotaError("daily")})
        client = LLMClient(
            provider_impl=provider, models=["a"], max_requests_per_minute=1000
        )
        with self.assertRaises(LLMQuotaExhaustedError):
            client.generate([{"role": "user", "content": "hi"}])
        saved = llm_client._client
        llm_client._client = client
        try:
            self.assertEqual(llm_client.quota_state(), "quota_exhausted")
        finally:
            llm_client._client = saved


class TestQuotaMapping(unittest.TestCase):

    def _error(self, message, metric=None):
        from google.genai import errors

        body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message}}
        if metric:
            body["error"]["details"] = [
                {"quotaMetric": metric}
            ]
        return errors.APIError(429, body)

    def test_per_day_429_maps_to_daily_quota(self):
        mapped = _map_gemini_error(
            self._error("quota exceeded", metric="generate_requests_per_day")
        )
        self.assertIsInstance(mapped, LLMDailyQuotaError)

    def test_per_minute_429_maps_to_rate_limit(self):
        mapped = _map_gemini_error(
            self._error("quota exceeded", metric="generate_requests_per_minute")
        )
        self.assertIsInstance(mapped, LLMRateLimitError)

    def test_404_maps_to_model_unavailable(self):
        from google.genai import errors

        exc = errors.APIError(404, {"error": {"code": 404, "status": "NOT_FOUND"}})
        self.assertIsInstance(_map_gemini_error(exc), LLMModelUnavailableError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
