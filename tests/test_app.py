"""
Tests for app.py (the ShopBot LLM agent Flask server).

The LLM is faked with scripted responses, so no network and no API key are
needed. The real tools run against the local catalog/docs/orders.

Run:  .venv\\Scripts\\python.exe tests\\test_app.py
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Never read the developer's real .env: point dotenv at a nonexistent path and
# clear the key before any module loads it.
os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import app  # noqa: E402
import agent_tools  # noqa: E402
from agent import ShopBotAgent  # noqa: E402
from llm_client import (  # noqa: E402
    LLMResponse,
    ToolCall,
    LLMRateLimitError,
    LLMQuotaExhaustedError,
)


class FakeClient:
    """Returns queued responses (or raises queued exceptions); records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, messages, tools=None, system=None):
        self.requests.append({"messages": list(messages), "tools": tools, "system": system})
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RepeatingClient:
    """Always returns the same tool-call response (to test the round cap)."""

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, messages, tools=None, system=None):
        self.calls += 1
        return self.response


def _two_asins():
    products = agent_tools.search_products("mascara", limit=2)["products"]
    if len(products) < 2:
        raise unittest.SkipTest("need at least two products in the local catalog")
    return [p["parent_asin"] for p in products]


class AppTestCase(unittest.TestCase):
    """Shared setup: fresh session store, models marked ready, test client."""

    def setUp(self):
        app.sessions.clear()
        app._agent = None
        app._limited_mode = False
        app._models_ready = True  # skip the explicit startup pre-load
        app.app.config["TESTING"] = True
        self.client = app.app.test_client()

    def chat(self, message, session_id=None):
        payload = {"message": message}
        if session_id is not None:
            payload["session_id"] = session_id
        return self.client.post("/api/chat", json=payload)


# ---------------------------------------------------------------------------
# Static / health / validation
# ---------------------------------------------------------------------------

class TestStaticAndHealth(AppTestCase):

    def test_app_js_is_served_byte_identical_to_source(self):
        # app must not rewrite app.js; it is served exactly as on disk.
        with open(os.path.join(app.FRONTEND_DIR, "app.js"), "rb") as handle:
            source = handle.read()
        resp = self.client.get("/app.js")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_data(), source)

    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["name"], "ShopBot")
        self.assertIn("model", body)
        self.assertNotIn("api_key", body)
        self.assertEqual(body["llm"], "ok")

    def test_empty_message_rejected(self):
        resp = self.chat("   ")
        self.assertEqual(resp.status_code, 400)

    def test_message_too_long_rejected(self):
        resp = self.chat("a" * (app.MAX_MESSAGE_CHARS + 1))
        self.assertEqual(resp.status_code, 400)

    def test_compare_needs_two_asins(self):
        resp = self.client.post("/api/compare", json={"asins": ["B0001"], "language": "en"})
        self.assertEqual(resp.status_code, 400)


# ---------------------------------------------------------------------------
# Chat — happy paths
# ---------------------------------------------------------------------------

class TestSiteEndpoint(AppTestCase):

    def test_serves_widget_fields_without_paths(self):
        resp = self.client.get("/api/site")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        for key in ("name", "assistant_name", "currency", "languages", "pages"):
            self.assertIn(key, body)
        self.assertNotIn("paths", body)
        self.assertEqual(body["pages"]["cart"], "#/cart")
        self.assertIn("en", body["languages"])


class TestChatSearch(AppTestCase):

    def test_fresh_search_returns_product_list(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 3})]),
            LLMResponse(text="Here are some mascaras I found."),
        ]))
        resp = self.chat("find me a mascara")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()

        self.assertEqual(body["response_type"], "product_list")
        self.assertTrue(body["products"])
        self.assertEqual(body["intent"], "agent")
        self.assertFalse(body["escalated"])
        self.assertIsNone(body["comparison"])
        self.assertEqual(body["tools_used"][0]["name"], "search_products")
        self.assertTrue(body["tools_used"][0]["ok"])
        self.assertTrue(body["session_id"])
        self.assertIn("language", body)

    def test_search_response_includes_a_navigate_action(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products",
                args={"query": "hair brushes", "max_price_usd": 15, "limit": 5})]),
            LLMResponse(text="I found 7 products for you — taking you to the results."),
        ]))
        body = self.chat("show me hair brushes under $15").get_json()
        self.assertEqual(len(body["actions"]), 1)
        action = body["actions"][0]
        self.assertEqual(action["type"], "navigate")
        self.assertEqual(action["page"], "search")
        self.assertEqual(action["params"]["q"], "hair brushes")
        self.assertEqual(action["params"]["max_price"], 15)

    def test_no_match_search_has_empty_actions(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "zzzznonsense", "limit": 5})]),
            LLMResponse(text="I couldn't find anything like that."),
        ]))
        body = self.chat("find zzzznonsense").get_json()
        self.assertEqual(body["actions"], [])

    def test_price_filter_is_passed_and_respected(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products",
                args={"query": "mascara", "max_price_usd": 100, "limit": 5})]),
            LLMResponse(text="Some affordable options."),
        ]))
        resp = self.chat("mascara under 100 dollars")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "product_list")
        self.assertTrue(body["products"])
        for product in body["products"]:
            self.assertLessEqual(product["price"], 100)
        self.assertEqual(body["tools_used"][0]["args"].get("max_price_usd"), 100)

    def test_compare_returns_comparison_table(self):
        asins = _two_asins()
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="compare_products", args={"parent_asins": asins})]),
            LLMResponse(text="The second is better rated."),
        ]))
        resp = self.chat("compare these two")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "comparison")
        self.assertIsNotNone(body["comparison"])
        self.assertEqual(len(body["comparison"]["table"]["headers"]), 3)
        self.assertTrue(body["comparison"]["summary"])
        self.assertEqual(body["tools_used"][0]["name"], "compare_products")

    def test_policy_question_is_grounded(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="search_policies", args={"query": "return policy"})]),
            LLMResponse(text="You can return items within 30 days."),
        ]))
        resp = self.chat("what is your return policy?")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "text")
        self.assertEqual(body["products"], [])
        self.assertEqual(body["tools_used"][0]["name"], "search_policies")

    def test_order_lookup(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="check_order", args={"order_id": "ORD-10025"})]),
            LLMResponse(text="Order ORD-10025 is currently Processing."),
        ]))
        resp = self.chat("where is my order ORD-10025?")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "text")
        self.assertEqual(body["tools_used"][0]["name"], "check_order")
        self.assertTrue(body["tools_used"][0]["ok"])
        self.assertIn("ORD-10025", body["text"])

    def test_product_lookup_returns_product_action(self):
        asin = _two_asins()[0]
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="get_product_details", args={"parent_asin": asin})]),
            LLMResponse(text="Here is what I found."),
        ]))
        body = self.chat("tell me about this one").get_json()
        self.assertEqual(body["actions"][0]["page"], "product")
        self.assertEqual(body["actions"][0]["params"]["asin"], asin)

    def test_compare_returns_compare_action(self):
        asins = _two_asins()
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="compare_products", args={"parent_asins": asins})]),
            LLMResponse(text="The second is better rated."),
        ]))
        body = self.chat("compare these two").get_json()
        self.assertEqual(body["actions"][0]["page"], "compare")
        self.assertEqual(body["actions"][0]["params"]["asins"], asins)

    def test_order_returns_orders_action(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="check_order", args={"order_id": "ORD-10025"})]),
            LLMResponse(text="Order ORD-10025 is processing."),
        ]))
        body = self.chat("where is ORD-10025?").get_json()
        self.assertEqual(body["actions"][0]["page"], "orders")
        self.assertEqual(body["actions"][0]["params"]["order_id"], "ORD-10025")

    def test_policy_returns_help_action(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="search_policies", args={"query": "return policy"})]),
            LLMResponse(text="Here is our return policy."),
        ]))
        body = self.chat("what is your return policy?").get_json()
        helps = [a for a in body["actions"] if a["page"] == "help"]
        self.assertEqual(len(helps), 1)
        self.assertIn(helps[0]["params"]["topic"],
                      ("returns", "warranty", "shipping", "payment", "orders"))

    def test_zero_match_search_shows_no_products_and_no_action(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "zzzznonsense", "limit": 5})]),
            LLMResponse(text="I couldn't find anything like that."),
        ]))
        body = self.chat("find zzzznonsense").get_json()
        self.assertEqual(body["products"], [])
        self.assertEqual(body["actions"], [])

    def test_order_not_found(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="check_order", args={"order_id": "ORD-99999"})]),
            LLMResponse(text="I couldn't find an order with that number."),
        ]))
        resp = self.chat("track ORD-99999")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "text")
        self.assertFalse(body["escalated"])
        self.assertEqual(body["tools_used"][0]["name"], "check_order")


# ---------------------------------------------------------------------------
# Chat — escalation, failures, limits
# ---------------------------------------------------------------------------

class TestChatGreetings(AppTestCase):

    def test_greeting_short_circuits_the_llm(self):
        # Empty queue: if the LLM were called, FakeClient would raise.
        client = FakeClient([])
        app._agent = ShopBotAgent(client=client)
        resp = self.chat("hello")
        body = resp.get_json()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body["response_type"], "text")
        self.assertTrue(body["text"].strip())
        self.assertEqual(body["tools_used"], [])
        self.assertEqual(body["actions"], [])
        self.assertEqual(client.requests, [])  # LLM never called

    def test_greeting_replies_in_arabic(self):
        app._agent = ShopBotAgent(client=FakeClient([]))
        resp = self.chat("مرحبا")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "text")
        self.assertFalse(body["escalated"])
        self.assertEqual(body["language"], "ar")
        self.assertTrue(any("\u0600" <= ch <= "\u06ff" for ch in body["text"]))

    def test_greeting_with_a_request_goes_to_the_agent(self):
        app._agent = ShopBotAgent(client=FakeClient([LLMResponse(text="Here you go.")]))
        resp = self.chat("hi, find me a mascara")
        body = resp.get_json()
        self.assertEqual(body["text"], "Here you go.")  # not the cookie-cutter greeting


class TestChatEscalationAndFailures(AppTestCase):

    def test_escalation_short_circuits_the_llm(self):
        client = FakeClient([LLMResponse(text="should not be called")])
        app._agent = ShopBotAgent(client=client)
        resp = self.chat("I want to speak to a human agent right now")
        body = resp.get_json()

        self.assertEqual(body["response_type"], "escalation")
        self.assertTrue(body["escalated"])
        self.assertEqual(body["tools_used"][0]["name"], "escalate_to_human")
        self.assertEqual(body["actions"], [])
        self.assertEqual(client.requests, [])  # LLM never called

    def test_escalation_replies_in_arabic(self):
        app._agent = ShopBotAgent(client=FakeClient([LLMResponse(text="x")]))
        resp = self.chat("أريد التحدث مع موظف خدمة العملاء")
        body = resp.get_json()
        self.assertEqual(body["response_type"], "escalation")
        self.assertEqual(body["language"], "ar")
        self.assertTrue(any("\u0600" <= ch <= "\u06ff" for ch in body["text"]))

    def test_tool_failure_is_marked_not_ok(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="unknown_tool", args={})]),
            LLMResponse(text="Sorry, I can't do that."),
        ]))
        resp = self.chat("do something weird")
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(body["tools_used"][0]["ok"])

    def test_llm_failure_falls_back_to_limited_mode(self):
        app._agent = ShopBotAgent(client=FakeClient([LLMRateLimitError("boom secret")]))
        resp = self.chat("find me a mascara")
        body = resp.get_json()

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(body["degraded"])
        self.assertIsNone(body.get("error"))
        self.assertEqual(body["confidence"], 0.0)
        self.assertTrue(body["products"])
        self.assertIn("found", body["text"].lower())
        self.assertNotIn("limited mode", body["text"].lower())
        self.assertNotIn("boom", resp.get_data(as_text=True))
        self.assertTrue(app._limited_mode)
        health = self.client.get("/api/health").get_json()
        self.assertEqual(health["llm"], "limited")

    def test_quota_exhausted_falls_back_to_limited_mode(self):
        app._agent = ShopBotAgent(
            client=FakeClient([LLMQuotaExhaustedError("all models out of quota")])
        )
        resp = self.chat("find me a mascara")
        body = resp.get_json()

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(body["degraded"])
        self.assertIsNone(body.get("error"))
        self.assertTrue(body["text"].strip())
        self.assertNotIn("quota", body["text"].lower())
        self.assertNotIn("limited mode", body["text"].lower())

    def test_successful_turn_leaves_limited_mode(self):
        app._limited_mode = True
        app._agent = ShopBotAgent(client=FakeClient([LLMResponse(text="Here you go.")]))
        body = self.chat("I need something for my skin").get_json()
        self.assertFalse(body["degraded"])
        self.assertFalse(app._limited_mode)
        self.assertEqual(self.client.get("/api/health").get_json()["llm"], "ok")

    def test_tool_round_limit_is_bounded(self):
        self.assertEqual(app.MAX_TOOL_ROUNDS, 5)
        looping = LLMResponse(tool_calls=[ToolCall(name="search_policies", args={"query": "shipping"})])
        agent = ShopBotAgent(client=RepeatingClient(looping),
                             max_iterations=app.MAX_TOOL_ROUNDS)
        app._agent = agent
        resp = self.chat("tell me everything about shipping")
        body = resp.get_json()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(body["tools_used"]), app.MAX_TOOL_ROUNDS)
        self.assertNotEqual(body["response_type"], "error")


# ---------------------------------------------------------------------------
# Page context
# ---------------------------------------------------------------------------

class TestChatResponseCache(AppTestCase):
    """LLM_CACHE=1 replays a stored /api/chat response (no second LLM call)."""

    def setUp(self):
        super().setUp()
        import llm_cache
        self._llm_cache = llm_cache
        self._tmp = tempfile.mkdtemp(prefix="appb_cache_")
        self._saved = {
            "LLM_CACHE": os.environ.get("LLM_CACHE"),
            "LLM_CACHE_PATH": os.environ.get("LLM_CACHE_PATH"),
        }
        os.environ["LLM_CACHE"] = "1"
        os.environ["LLM_CACHE_PATH"] = os.path.join(self._tmp, "cache.json")
        llm_cache._loaded = False
        llm_cache._store = {}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._llm_cache._loaded = False
        self._llm_cache._store = {}
        shutil.rmtree(self._tmp, ignore_errors=True)
        super().tearDown()

    def test_second_identical_turn_is_served_from_cache(self):
        app._agent = ShopBotAgent(client=FakeClient([LLMResponse(text="cached!")]))
        first = self.chat("find me a mascara").get_json()
        self.assertEqual(first["text"], "cached!")
        # Empty queue: a second LLM call would raise, so a hit is the only way.
        app._agent = ShopBotAgent(client=FakeClient([]))
        second = self.chat("find me a mascara").get_json()
        self.assertEqual(second["text"], "cached!")
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_errors_are_not_cached(self):
        app._agent = ShopBotAgent(client=FakeClient([LLMRateLimitError("boom")]))
        self.chat("find me a mascara")  # provider failure -> not stored
        app._agent = ShopBotAgent(client=FakeClient([LLMResponse(text="recovered")]))
        body = self.chat("find me a mascara").get_json()
        self.assertEqual(body["text"], "recovered")


class TestPageContext(AppTestCase):

    def test_context_is_added_to_the_system_prompt(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="get_product_details", args={"parent_asin": "B0TESTASIN"})]),
            LLMResponse(text="It has a one-year warranty."),
        ])
        app._agent = ShopBotAgent(client=client)
        resp = self.client.post("/api/chat", json={
            "message": "does this have a warranty?",
            "context": {"page": "product", "asin": "B0TESTASIN"},
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn("B0TESTASIN", client.requests[0]["system"])

    def test_request_without_context_works_as_before(self):
        client = FakeClient([LLMResponse(text="Hi!")])
        app._agent = ShopBotAgent(client=client)
        body = self.chat("find me a mascara").get_json()
        self.assertEqual(body["response_type"], "text")
        self.assertNotIn("currently viewing", client.requests[0]["system"])

    def test_long_context_strings_are_truncated(self):
        client = FakeClient([LLMResponse(text="ok")])
        app._agent = ShopBotAgent(client=client)
        self.chat_with_context("find me a mascara", {"page": "search", "query": "a" * 5000})
        system = client.requests[0]["system"]
        self.assertIn("search results page", system)
        self.assertNotIn("a" * (app._CONTEXT_STRING_MAX + 1), system)

    def test_oversized_context_is_ignored(self):
        with mock.patch.object(app, "MAX_CONTEXT_CHARS", 10):
            self.assertEqual(app._clean_context({"page": "search"}), {})

    def test_filters_are_capped(self):
        cleaned = app._clean_context({"filters": {f"k{i}": "v" for i in range(50)}})
        self.assertLessEqual(len(cleaned["filters"]), app._CONTEXT_FILTER_KEYS_MAX)

    def test_non_dict_context_is_ignored(self):
        for bad in (None, [], "text", 5):
            self.assertEqual(app._clean_context(bad), {})

    def chat_with_context(self, message, context):
        return self.client.post("/api/chat", json={"message": message, "context": context})


# ---------------------------------------------------------------------------
# Contract + session memory
# ---------------------------------------------------------------------------

class TestContractAndMemory(AppTestCase):

    def test_response_has_every_field_run_eval_reads(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="Here you go."),
        ]))
        body = self.chat("find me a mascara").get_json()
        for field in ("session_id", "intent", "confidence", "response_type",
                      "text", "products", "comparison", "language",
                      "tools_used", "escalated", "actions"):
            self.assertIn(field, body, f"missing field: {field}")

    def test_session_history_is_kept_for_followups(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="Here are two mascaras."),
        ])
        app._agent = ShopBotAgent(client=client)
        first = self.chat("find me a mascara").get_json()
        session_id = first["session_id"]

        history = app.sessions[session_id]["history"]
        self.assertTrue(history)
        roles = [m["role"] for m in history]
        self.assertEqual(roles[0], "user")
        self.assertIn("assistant", roles)

        # A follow-up must resend the prior turn(s) to the model.
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(text="The second one is the better rated option."),
        ]))
        self.chat("tell me more about the second one", session_id=session_id)
        sent = app._agent.client.requests[0]["messages"]
        self.assertTrue(any("mascara" in str(m.get("content", "")).lower() for m in sent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
