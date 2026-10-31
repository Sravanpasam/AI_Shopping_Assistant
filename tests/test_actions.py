"""
Tests for System B navigate actions (agent.build_action + the agent loop).

Actions are built from the tool calls the agent made; a search with no matches
(or any failed lookup) must not produce a navigate action. No network.

Run:  .venv\\Scripts\\python.exe tests\\test_actions.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Never read the developer's real .env.
os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import agent_tools  # noqa: E402
from agent import ShopBotAgent, build_action  # noqa: E402
from llm_client import LLMResponse, ToolCall  # noqa: E402


class FakeClient:
    """Returns queued responses; records the requests it received."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, messages, tools=None, system=None):
        self.requests.append({"messages": list(messages), "tools": tools, "system": system})
        return self.responses.pop(0)


def _two_asins():
    products = agent_tools.search_products("mascara", limit=2)["products"]
    if len(products) < 2:
        raise unittest.SkipTest("need at least two products in the local catalog")
    return [p["parent_asin"] for p in products]


class TestBuildActionSearch(unittest.TestCase):

    def test_search_with_matches_uses_tool_arguments(self):
        args = {"query": "hair brushes", "max_price_usd": 15, "limit": 5}
        result = {"count": 3, "total_matches": 7, "products": [{"parent_asin": "X"}]}
        action = build_action("search_products", args, result)
        self.assertEqual(action["type"], "navigate")
        self.assertEqual(action["page"], "search")
        self.assertEqual(action["params"]["q"], "hair brushes")
        self.assertEqual(action["params"]["max_price"], 15)
        self.assertNotIn("min_price", action["params"])   # not provided
        self.assertEqual(action["label"], "Hair brushes under $15")

    def test_search_label_between_and_over(self):
        self.assertEqual(
            build_action("search_products", {"query": "mascara", "min_price_usd": 5,
                                             "max_price_usd": 20},
                         {"count": 1})["label"],
            "Mascara between $5 and $20",
        )
        self.assertEqual(
            build_action("search_products", {"query": "mascara", "min_price_usd": 5},
                         {"count": 1})["label"],
            "Mascara over $5",
        )

    def test_search_without_price_label_is_just_query(self):
        action = build_action("search_products", {"query": "lipstick"}, {"count": 2})
        self.assertEqual(action["label"], "Lipstick")

    def test_search_zero_matches_gives_no_action(self):
        self.assertIsNone(build_action("search_products", {"query": "unicorn"}, {"count": 0}))

    def test_search_error_gives_no_action(self):
        self.assertIsNone(
            build_action("search_products", {"query": "x"}, {"error": "search_products failed"})
        )


class TestBuildActionOthers(unittest.TestCase):

    def test_product_found(self):
        action = build_action(
            "get_product_details", {"parent_asin": "B1"}, {"found": True, "parent_asin": "B1"}
        )
        self.assertEqual(action["page"], "product")
        self.assertEqual(action["params"]["asin"], "B1")

    def test_product_not_found(self):
        self.assertIsNone(build_action("get_product_details", {"parent_asin": "B1"},
                                       {"found": False}))

    def test_compare(self):
        action = build_action(
            "compare_products", {"parent_asins": ["A", "B"]},
            {"compared": True, "parent_asins": ["A", "B"]},
        )
        self.assertEqual(action["page"], "compare")
        self.assertEqual(action["params"]["asins"], ["A", "B"])

    def test_compare_not_compared(self):
        self.assertIsNone(build_action("compare_products", {"parent_asins": ["A"]},
                                       {"compared": False}))

    def test_order_found(self):
        action = build_action("check_order", {"order_id": "ORD-1"},
                              {"found": True, "order_id": "ORD-1"})
        self.assertEqual(action["page"], "orders")
        self.assertEqual(action["params"]["order_id"], "ORD-1")

    def test_order_not_found(self):
        self.assertIsNone(build_action("check_order", {"order_id": "ORD-9"}, {"found": False}))

    def test_unknown_tool(self):
        self.assertIsNone(build_action("search_reviews", {"parent_asin": "A"},
                                       {"count": 2, "reviews": []}))


class TestPolicyTopic(unittest.TestCase):

    def test_returns_question_maps_to_returns_topic(self):
        result = agent_tools.search_policies("what is your return policy?")
        action = build_action("search_policies", {"query": "return policy"}, result)
        self.assertEqual(action["page"], "help")
        self.assertEqual(action["params"]["topic"], "returns")

    def test_shipping_question_maps_to_shipping_topic(self):
        result = agent_tools.search_policies("how long does shipping take?")
        if not result.get("policies"):
            self.skipTest("no policy chunks indexed")
        action = build_action("search_policies", {"query": "shipping"}, result)
        self.assertEqual(action["params"]["topic"], "shipping")

    def test_policy_zero_hits_gives_no_action(self):
        self.assertIsNone(build_action("search_policies", {"query": "x"},
                                       {"count": 0, "policies": []}))


class TestAgentLoopActions(unittest.TestCase):

    def test_search_turn_carries_one_action(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "mascara", "limit": 3})]),
            LLMResponse(text="I found some mascaras; taking you to the results."),
        ])
        result = ShopBotAgent(client=client).run("find me a mascara")
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0]["page"], "search")
        self.assertEqual(result.actions[0]["params"]["q"], "mascara")

    def test_no_match_search_produces_no_action(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "zzzznonsense", "limit": 5})]),
            LLMResponse(text="I couldn't find anything like that."),
        ])
        result = ShopBotAgent(client=client).run("find nonsense")
        self.assertEqual(result.actions, [])

    def test_duplicate_tool_calls_dedupe_action(self):
        args = {"query": "mascara", "limit": 2}
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="search_products", args=args)]),
            LLMResponse(tool_calls=[ToolCall(name="search_products", args=args)]),
            LLMResponse(text="Here you go."),
        ])
        result = ShopBotAgent(client=client).run("mascara")
        self.assertEqual(len(result.actions), 1)

    def test_plain_answer_has_no_actions(self):
        client = FakeClient([LLMResponse(text="Hi! How can I help?")])
        result = ShopBotAgent(client=client).run("hello")
        self.assertEqual(result.actions, [])

    def test_zero_match_search_keeps_products_empty(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="search_products", args={"query": "zzzznonsense", "limit": 5})]),
            LLMResponse(text="I couldn't find anything."),
        ])
        result = ShopBotAgent(client=client).run("nonsense")
        self.assertEqual(result.products, [])
        self.assertEqual(result.actions, [])


class TestAgentLoopActionPerTool(unittest.TestCase):
    """Each tool type drives the matching navigate action in the agent loop."""

    def test_product_lookup_navigates_to_product(self):
        asin = agent_tools.search_products("mascara", limit=1)["products"][0]["parent_asin"]
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(
                name="get_product_details", args={"parent_asin": asin})]),
            LLMResponse(text="Here is what I found about it."),
        ])
        result = ShopBotAgent(client=client).run("tell me about this one")
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0]["page"], "product")
        self.assertEqual(result.actions[0]["params"]["asin"], asin)

    def test_compare_navigates_to_compare(self):
        asins = _two_asins()
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="compare_products", args={"parent_asins": asins})]),
            LLMResponse(text="The second is better rated."),
        ])
        result = ShopBotAgent(client=client).run("compare these")
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0]["page"], "compare")
        self.assertEqual(result.actions[0]["params"]["asins"], asins)

    def test_order_lookup_navigates_to_orders(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="check_order", args={"order_id": "ORD-10025"})]),
            LLMResponse(text="Order ORD-10025 is processing."),
        ])
        result = ShopBotAgent(client=client).run("where is ORD-10025?")
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0]["page"], "orders")
        self.assertEqual(result.actions[0]["params"]["order_id"], "ORD-10025")

    def test_policy_question_navigates_to_help(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="search_policies", args={"query": "return policy"})]),
            LLMResponse(text="Here is our return policy."),
        ])
        result = ShopBotAgent(client=client).run("what is your return policy?")
        help_actions = [a for a in result.actions if a["page"] == "help"]
        self.assertEqual(len(help_actions), 1)
        self.assertIn(help_actions[0]["params"]["topic"],
                      ("returns", "warranty", "shipping", "payment", "orders"))

    def test_escalation_tool_has_no_navigate_action(self):
        client = FakeClient([
            LLMResponse(tool_calls=[ToolCall(name="escalate_to_human", args={"reason": "asked"})]),
            LLMResponse(text="I'll connect you with a human agent."),
        ])
        result = ShopBotAgent(client=client).run("I want to speak to a human")
        self.assertTrue(result.escalated)
        self.assertEqual(result.actions, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
