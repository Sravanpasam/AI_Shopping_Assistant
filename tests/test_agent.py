"""
Tests for agent.py (System B tool-calling loop).

The LLM is faked with scripted responses, so no network and no API key are
needed. The real tools run against the local catalog/docs/orders.

Run:  .venv\\Scripts\\python.exe tests\\test_agent.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Never read the developer's real .env: point dotenv at a nonexistent path and
# clear the key before any module loads it.
os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import agent_tools  # noqa: E402
from agent import ShopBotAgent, AgentResult, run_agent  # noqa: E402
from llm_client import LLMResponse, ToolCall, LLMRateLimitError  # noqa: E402


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


def _two_asins():
    products = agent_tools.search_products("mascara", limit=2)["products"]
    if len(products) < 2:
        raise unittest.SkipTest("need at least two products in the local catalog")
    return [p["parent_asin"] for p in products]


class TestPlainAnswer(unittest.TestCase):

    def test_plain_answer(self):
        client = FakeClient([LLMResponse(text="Hello! How can I help?")])
        result = ShopBotAgent(client=client).run("hi")
        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.reply, "Hello! How can I help?")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls, [])
        self.assertIsNone(result.error)

    def test_empty_text_falls_back(self):
        client = FakeClient([LLMResponse(text="")])
        result = ShopBotAgent(client=client).run("hi")
        self.assertTrue(result.reply.strip())

    def test_tools_are_passed_to_model(self):
        client = FakeClient([LLMResponse(text="ok")])
        ShopBotAgent(client=client).run("hi")
        sent_names = {t["name"] for t in client.requests[0]["tools"]}
        self.assertIn("search_products", sent_names)
        self.assertIn("check_order", sent_names)
        self.assertIsNotNone(client.requests[0]["system"])


class TestToolLoop(unittest.TestCase):

    def test_search_then_answer(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="search_products", args={"query": "mascara", "limit": 1})]),
            LLMResponse(text="Here is a good mascara."),
        ])
        result = ShopBotAgent(client=client).run("find me a mascara")
        self.assertEqual(result.reply, "Here is a good mascara.")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.products), 1)
        self.assertEqual(result.tool_calls[0]["name"], "search_products")

        # Second request must carry the tool call + tool result back to the model.
        second = client.requests[1]["messages"]
        roles = [m["role"] for m in second]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(second[2]["name"], "search_products")
        self.assertIn("products", second[2]["content"])

    def test_compare_captures_table(self):
        asins = _two_asins()
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="compare_products", args={"parent_asins": asins})]),
            LLMResponse(text="The first is cheaper; the second is better rated."),
        ])
        result = ShopBotAgent(client=client).run("compare these two")
        self.assertIsNotNone(result.table)
        self.assertEqual(len(result.table["headers"]), 3)
        self.assertTrue(result.reply)

    def test_compare_summary_used_when_no_text(self):
        asins = _two_asins()
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="compare_products", args={"parent_asins": asins})]),
            LLMResponse(text=""),
        ])
        result = ShopBotAgent(client=client).run("compare these")
        self.assertTrue(result.reply.strip())
        self.assertNotEqual(result.reply.strip(), "")  # summary or fallback

    def test_escalation_captured(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="escalate_to_human", args={"reason": "user asked"})]),
            LLMResponse(text="Connecting you to a person."),
        ])
        result = ShopBotAgent(client=client).run("I want a human")
        self.assertTrue(result.escalated)

    def test_history_passed_through(self):
        client = FakeClient([LLMResponse(text="ok")])
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
        ShopBotAgent(client=client).run("now", history=history)
        sent = client.requests[0]["messages"]
        self.assertEqual(sent[0]["content"], "earlier")
        self.assertEqual(sent[-1]["content"], "now")


class TestLimitsAndErrors(unittest.TestCase):

    def test_iteration_cap(self):
        loop = LLMResponse(tool_calls=[ToolCall(name="search_policies", args={"query": "returns"})])
        client = FakeClient([loop] * 20)
        result = ShopBotAgent(client=client, max_iterations=3).run("loop please")
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.error, "max_iterations")
        self.assertTrue(result.reply.strip())

    def test_provider_error_is_captured(self):
        client = FakeClient([LLMRateLimitError("quota")])
        result = ShopBotAgent(client=client).run("hello")
        self.assertIsNotNone(result.error)
        self.assertIn("LLMRateLimitError", result.error)
        self.assertTrue(result.reply.strip())  # safe, non-factual fallback


class TestToolCallDedupe(unittest.TestCase):

    def test_identical_repeated_tool_call_runs_once(self):
        search = ToolCall(name="search_products", args={"query": "mascara", "limit": 3})
        client = FakeClient([
            LLMResponse(tool_calls=[search]),
            LLMResponse(tool_calls=[search]),  # identical repeat
            LLMResponse(text="Here you go."),
        ])
        result = ShopBotAgent(client=client).run("find me a mascara")
        expected = agent_tools.search_products("mascara", limit=3)["products"]
        # The second identical call reuses the first result: results not doubled.
        self.assertEqual(len(result.products), len(expected))
        self.assertEqual(len(result.tool_calls), 2)  # both rounds recorded


class TestPageContext(unittest.TestCase):

    def test_no_context_uses_the_base_prompt(self):
        client = FakeClient([LLMResponse(text="ok")])
        ShopBotAgent(client=client).run("hi")
        self.assertNotIn("currently viewing", client.requests[0]["system"])

    def test_product_context_is_added_to_the_system_prompt(self):
        client = FakeClient([LLMResponse(text="It has a warranty.")])
        ShopBotAgent(client=client).run(
            "does this have a warranty?",
            context={"page": "product", "asin": "B0TESTASIN"},
        )
        system = client.requests[0]["system"]
        self.assertIn("currently viewing", system)
        self.assertIn("B0TESTASIN", system)

    def test_search_context_includes_query_and_filters(self):
        client = FakeClient([LLMResponse(text="ok")])
        ShopBotAgent(client=client).run(
            "compare it with a cheaper one",
            context={"page": "search", "query": "mascara", "filters": {"max_price": 15}},
        )
        system = client.requests[0]["system"]
        self.assertIn("search results page", system)
        self.assertIn("mascara", system)
        self.assertIn("max_price=15", system)


class TestSystemPromptFromSite(unittest.TestCase):

    def test_prompt_uses_site_assistant_name_and_scope(self):
        from agent import build_system_prompt
        prompt = build_system_prompt(
            {"assistant_name": "Aria", "scope_description": "a toy store"}
        )
        self.assertIn("Aria", prompt)
        self.assertIn("a toy store", prompt)
        self.assertNotIn("__ASSISTANT_NAME__", prompt)
        self.assertNotIn("__SCOPE__", prompt)

    def test_default_prompt_loads_the_shopstore_site(self):
        from agent import SYSTEM_PROMPT
        self.assertIn("ShopBot", SYSTEM_PROMPT)


class TestRunAgent(unittest.TestCase):

    def test_exports_callable(self):
        self.assertTrue(callable(run_agent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
