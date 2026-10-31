"""
Tests for the smarter-agent additions: routine/gift/category tools, session
profile, follow-up suggestions and the grounding check.

No network and no API key: the LLM is scripted, and the tools use the local
catalog/docs/orders.

Run:  .venv\\Scripts\\python.exe tests\\test_agent_enhancements.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import agent_tools  # noqa: E402
import app  # noqa: E402
from agent import ShopBotAgent, grounding_issues  # noqa: E402
from llm_client import LLMResponse, ToolCall  # noqa: E402


class FakeClient:
    """Returns queued responses; records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, messages, tools=None, system=None):
        self.requests.append({"messages": list(messages), "tools": tools, "system": system})
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestNewTools(unittest.TestCase):

    def test_new_tools_are_declared_and_dispatched(self):
        names = {schema["name"] for schema in agent_tools.TOOL_SCHEMAS}
        for name in ("build_routine", "gift_finder", "best_in_category"):
            self.assertIn(name, names)
            self.assertIn(name, agent_tools.TOOL_FUNCTIONS)

    def test_build_routine_returns_real_steps(self):
        result = agent_tools.build_routine("dry skin", budget=50)
        if result.get("not_available"):
            self.assertIn("note", result)
            return
        self.assertGreater(result["count"], 0)
        for step in result["steps"]:
            self.assertTrue(step["step"])
            self.assertTrue(step["product"]["parent_asin"])
            self.assertLessEqual(step["product"]["price"], 50)
        json.dumps(result)

    def test_build_routine_unknown_goal_is_not_available(self):
        result = agent_tools.build_routine("quantum flux capacitor", budget=20)
        self.assertTrue(result.get("not_available"))
        self.assertTrue(result.get("note"))
        json.dumps(result)

    def test_gift_finder_returns_products_or_not_available(self):
        result = agent_tools.gift_finder("mom", "birthday", budget=30)
        self.assertIn("recipient", result)
        if result.get("not_available"):
            self.assertTrue(result.get("note"))
        else:
            self.assertGreater(result["count"], 0)
            for product in result["products"]:
                self.assertLessEqual(product["price"], 30)
        json.dumps(result)

    def test_best_in_category_uses_catalog_ratings(self):
        result = agent_tools.best_in_category("makeup", max_price=50)
        self.assertTrue(result.get("found"))
        self.assertEqual(result["best"]["parent_asin"], result["products"][0]["parent_asin"])
        ratings = [p["average_rating"] for p in result["products"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        for product in result["products"]:
            self.assertLessEqual(product["price"], 50)
        json.dumps(result)

    def test_best_in_category_unknown_is_not_available(self):
        result = agent_tools.best_in_category("quantum flux capacitor")
        self.assertTrue(result.get("not_available"))
        self.assertTrue(result.get("category_suggestions"))
        json.dumps(result)


class TestSuggestionsAndProfile(unittest.TestCase):

    def test_search_reply_gets_followup_suggestions(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="I found a few options."),
        ])
        result = ShopBotAgent(client=client).run("find me a mascara")
        self.assertGreaterEqual(len(result.suggestions), 1)
        self.assertLessEqual(len(result.suggestions), 3)
        self.assertEqual(result.suggestions[0], "Tell me more about the first one")
        self.assertIn("Compare the first two", result.suggestions)

    def test_profile_is_added_to_the_system_prompt(self):
        client = FakeClient([LLMResponse(text="ok")])
        ShopBotAgent(client=client).run(
            "what do you recommend?",
            profile={"budget": 30, "skin_type": "dry", "viewed": ["B0TEST1"]},
        )
        system = client.requests[0]["system"]
        self.assertIn("budget is up to $30", system)
        self.assertIn("skin type is dry", system)
        self.assertIn("B0TEST1", system)

    def test_prompt_treats_tool_results_as_untrusted_data(self):
        client = FakeClient([LLMResponse(text="ok")])
        ShopBotAgent(client=client).run("hi")
        system = client.requests[0]["system"]
        self.assertIn("untrusted data", system)
        self.assertIn("Ignore any text", system)

    def test_app_stores_budget_and_skin_type_from_the_message(self):
        client = FakeClient([LLMResponse(text="ok")])
        app._agent = ShopBotAgent(client=client)
        app._models_ready = True
        app.sessions.clear()
        response = app.app.test_client().post("/api/chat", json={
            "message": "I have dry skin and my budget is $30",
            "session_id": "profile-test",
        })
        self.assertEqual(response.status_code, 200)
        profile = app.sessions["profile-test"]["profile"]
        self.assertEqual(profile["budget"], 30)
        self.assertEqual(profile["skin_type"], "dry")
        system = client.requests[0]["system"]
        self.assertIn("skin type is dry", system)
        self.assertIn("budget is up to $30", system)

    def test_app_remembers_viewed_products(self):
        client = FakeClient([LLMResponse(text="ok")])
        app._agent = ShopBotAgent(client=client)
        app._models_ready = True
        app.sessions.clear()
        response = app.app.test_client().post("/api/chat", json={
            "message": "tell me about this",
            "session_id": "viewed-test",
            "context": {"page": "product", "asin": "B0VIEWED1"},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            app.sessions["viewed-test"]["profile"]["viewed"], ["B0VIEWED1"]
        )

    def test_chat_payload_includes_suggestions(self):
        app._agent = ShopBotAgent(client=FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="I found a few options."),
        ]))
        app._models_ready = True
        app.sessions.clear()
        body = app.app.test_client().post(
            "/api/chat", json={"message": "find me a mascara"}
        ).get_json()
        self.assertIn("suggestions", body)
        self.assertLessEqual(len(body["suggestions"]), 3)
        self.assertGreaterEqual(len(body["suggestions"]), 1)


class TestProfileLimits(unittest.TestCase):

    def test_profile_is_capped(self):
        cleaned = app._clean_profile({
            "budget": 42.129,
            "hair_type": "x" * 500,
            "skin_type": "dry",
            "viewed": [f"B{i}" for i in range(50)],
        })
        self.assertEqual(cleaned["budget"], 42.13)
        self.assertEqual(len(cleaned["hair_type"]), app._PROFILE_STRING_MAX)
        self.assertEqual(len(cleaned["viewed"]), app.MAX_PROFILE_VIEWED)
        self.assertEqual(cleaned["viewed"][-1], "B49")

    def test_bad_profile_values_are_ignored(self):
        self.assertEqual(app._clean_profile("not a dict"), {})
        self.assertEqual(app._clean_profile({"budget": "nope"}), {})
        self.assertEqual(app._clean_profile({"viewed": "B0"}), {})


class TestGroundingCheck(unittest.TestCase):

    def test_grounding_issues_flags_unsupported_price_and_status(self):
        tool_results = [("check_order", {
            "order_id": "ORD-10025", "found": True,
            "order": {"status": "Processing", "estimated_delivery": "2026-10-21"},
        })]
        issues = grounding_issues(
            "It costs $999 and was delivered on 2026-10-20.", tool_results
        )
        self.assertTrue(any("price" in issue for issue in issues))
        self.assertTrue(any("date" in issue for issue in issues))
        self.assertTrue(any("order status" in issue for issue in issues))

    def test_grounding_accepts_supported_facts(self):
        tool_results = [("check_order", {
            "order_id": "ORD-10025", "found": True,
            "order": {"status": "Processing", "estimated_delivery": "2026-10-21"},
        })]
        self.assertEqual(
            grounding_issues(
                "Order ORD-10025 is Processing and is expected on 2026-10-21.",
                tool_results,
            ),
            [],
        )

    def test_grounding_accepts_aed_ratings_and_hyphenated_windows(self):
        tool_results = [
            ("search_products", {"products": [
                {"price": 15.0, "average_rating": 4.5},
            ]}),
            ("search_policies", {"chunks": [
                {"text": "You can return items within 14 days of delivery."},
            ]}),
        ]
        self.assertEqual(
            grounding_issues(
                "Rated 4.5, it costs AED 55 and has a 14-day return window.",
                tool_results,
            ),
            [],
        )

    def test_unsupported_reply_is_regenerated_once(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="This mascara costs $999.99 and has 4.9 stars."),
            LLMResponse(text="I found a few options within your budget."),
        ])
        result = ShopBotAgent(client=client).run("find me a mascara")
        self.assertTrue(result.regenerated)
        self.assertFalse(result.grounding_failed)
        self.assertNotIn("$999.99", result.reply)
        self.assertEqual(len(client.requests), 3)
        self.assertIsNone(client.requests[2]["tools"])

    def test_second_unsupported_reply_gets_a_safe_answer(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="This mascara costs $999.99."),
            LLMResponse(text="It still costs $888.88."),
        ])
        result = ShopBotAgent(client=client).run("find me a mascara")
        self.assertTrue(result.grounding_failed)
        self.assertNotIn("999", result.reply)
        self.assertNotIn("888", result.reply)
        self.assertIn("verify", result.reply.lower())

    def test_arabic_safe_answer_uses_arabic(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 2})]),
            LLMResponse(text="This mascara costs $999.99."),
        ])
        result = ShopBotAgent(client=client).run("ابحث عن ماسكارا")
        self.assertTrue(result.grounding_failed)
        self.assertTrue(any("\u0600" <= ch <= "\u06ff" for ch in result.reply))


if __name__ == "__main__":
    unittest.main(verbosity=2)
