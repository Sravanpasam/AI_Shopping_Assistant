"""
Tests for the store-adapter layer (adapters/base.py, adapters/local_files.py)
and the fact that agent_tools now reads data through the adapter.

Uses the real local catalog/docs/orders. No network.

Run:  .venv\\Scripts\\python.exe tests\\test_adapters.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["LLM_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "does-not-exist.env")
os.environ["GEMINI_API_KEY"] = ""

import agent_tools  # noqa: E402
from adapters import LocalFilesAdapter, StoreAdapter, get_adapter, reset_adapter, set_adapter  # noqa: E402


class TestLocalFilesAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = LocalFilesAdapter()

    def test_is_a_store_adapter(self):
        self.assertIsInstance(self.adapter, StoreAdapter)

    def test_site_info_has_name(self):
        self.assertTrue(self.adapter.site_info.get("name"))

    def test_search_products_returns_products(self):
        products = self.adapter.search_products("mascara", top_k=3)
        self.assertIsInstance(products, list)
        self.assertGreater(len(products), 0)
        self.assertIn("parent_asin", products[0])

    def test_get_product_found_and_missing(self):
        asin = self.adapter.search_products("mascara", top_k=1)[0]["parent_asin"]
        self.assertEqual(self.adapter.get_product(asin)["parent_asin"], asin)
        self.assertIsNone(self.adapter.get_product("NOT-A-REAL-ASIN"))

    def test_get_reviews_returns_list(self):
        asin = self.adapter.search_products("mascara", top_k=1)[0]["parent_asin"]
        self.assertIsInstance(self.adapter.get_reviews(asin), list)

    def test_search_knowledge_finds_policies(self):
        hits = self.adapter.search_knowledge("return policy", k=3)
        self.assertGreater(len(hits), 0)
        self.assertIn("source", hits[0])

    def test_get_order_found_and_missing(self):
        self.assertEqual(self.adapter.get_order("ORD-10025")["order_id"], "ORD-10025")
        self.assertIsNone(self.adapter.get_order("ORD-99999"))


class TestDefaultPathsAreUnchanged(unittest.TestCase):

    def test_constructing_the_adapter_does_not_change_default_paths(self):
        import data_loader
        import policy_search
        import orders
        before = (
            data_loader.CATALOG_PATH, data_loader.REVIEWS_PATH,
            policy_search.DOCS_DIR, orders.DB_PATH,
        )
        LocalFilesAdapter()  # default site == the modules' built-in defaults
        after = (
            data_loader.CATALOG_PATH, data_loader.REVIEWS_PATH,
            policy_search.DOCS_DIR, orders.DB_PATH,
        )
        self.assertEqual(before, after)


class TestAdapterSelection(unittest.TestCase):

    def tearDown(self):
        reset_adapter()

    def test_default_adapter_is_a_store_adapter(self):
        reset_adapter()
        self.assertIsInstance(get_adapter(), StoreAdapter)

    def test_get_adapter_is_a_singleton(self):
        reset_adapter()
        self.assertIs(get_adapter(), get_adapter())

    def test_set_and_reset_adapter(self):
        custom = LocalFilesAdapter()
        set_adapter(custom)
        self.assertIs(get_adapter(), custom)
        reset_adapter()
        self.assertIsNot(get_adapter(), custom)


class SpyAdapter(StoreAdapter):
    """Records which adapter methods the tools call, and returns empty data."""

    def __init__(self):
        self.calls = {}

    @property
    def site_info(self):
        return {"name": "spy"}

    def search_products(self, query, top_k=5):
        self.calls["search_products"] = (query, top_k)
        return []

    def get_product(self, asin):
        self.calls["get_product"] = asin
        return None

    def get_reviews(self, asin):
        self.calls["get_reviews"] = asin
        return []

    def search_knowledge(self, query, k=3):
        self.calls["search_knowledge"] = (query, k)
        return []

    def get_order(self, order_id):
        self.calls["get_order"] = order_id
        return None


class TestAgentToolsUseTheAdapter(unittest.TestCase):
    """agent_tools must reach store data only through get_adapter()."""

    def setUp(self):
        self.spy = SpyAdapter()
        set_adapter(self.spy)

    def tearDown(self):
        reset_adapter()

    def test_search_products_goes_through_the_adapter(self):
        agent_tools.search_products("mascara")
        self.assertIn("search_products", self.spy.calls)

    def test_get_product_details_goes_through_the_adapter(self):
        agent_tools.get_product_details("X")
        self.assertIn("get_product", self.spy.calls)

    def test_search_reviews_goes_through_the_adapter(self):
        agent_tools.search_reviews("X")
        self.assertIn("get_reviews", self.spy.calls)

    def test_search_policies_goes_through_the_adapter(self):
        agent_tools.search_policies("returns")
        self.assertIn("search_knowledge", self.spy.calls)

    def test_check_order_goes_through_the_adapter(self):
        agent_tools.check_order("ORD-10025")
        self.assertIn("get_order", self.spy.calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
