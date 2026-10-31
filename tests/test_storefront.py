"""
Tests for the storefront product API (backend/storefront.py), which is served
by the ShopBot app and by the legacy baseline.

These use the real local catalog/reviews (no models, no network).

Run:  .venv\\Scripts\\python.exe tests\\test_storefront.py
"""

import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "legacy"))

import app_baseline  # noqa: E402
import sqlite3  # noqa: E402

import pandas as pd  # noqa: E402

import app as app_module  # noqa: E402
import orders  # noqa: E402
import storefront as storefront_module  # noqa: E402
from app import app  # noqa: E402
from data_loader import load_catalog  # noqa: E402
from storefront import categorize_title, _clean_price, _to_card  # noqa: E402

# Product dict keys returned by /api/chat (System A contract). The storefront
# work must not add or remove any of them.
CHAT_PRODUCT_KEYS = {
    "parent_asin", "title", "store", "price", "average_rating",
    "rating_number", "category_path", "features_text", "description_text",
    "similarity_score",
}
CHAT_TOP_KEYS = {
    "session_id", "intent", "confidence", "response_type", "text",
    "products", "comparison", "language",
}


class TestCategories(unittest.TestCase):

    def test_rule_is_deterministic_and_ordered(self):
        self.assertEqual(categorize_title("Hair Dryer 1800W"), "hair")
        self.assertEqual(categorize_title("Makeup Brush Set"), "makeup")  # makeup before tools
        self.assertEqual(categorize_title("Nail Polish Remover"), "nails")
        self.assertEqual(categorize_title("Eau de Parfum for Women"), "fragrance")
        self.assertEqual(categorize_title("Moisturizing Face Cream"), "skin care")
        self.assertEqual(categorize_title("Body Lotion with Shea"), "bath and body")
        self.assertEqual(categorize_title("Stainless Steel Tweezers"), "tools and accessories")
        self.assertEqual(categorize_title("Completely Unrelated Widget"), "other")
        self.assertEqual(categorize_title(""), "other")

    def test_every_title_gets_a_category(self):
        catalog = load_catalog()
        assigned = {categorize_title(t) for t in catalog["title"].fillna("")}
        self.assertTrue(assigned)
        self.assertIn("other", assigned)


class TestMissingPrice(unittest.TestCase):

    def test_clean_price(self):
        self.assertIsNone(_clean_price(0.0))
        self.assertIsNone(_clean_price(None))
        self.assertIsNone(_clean_price(float("nan")))
        self.assertEqual(_clean_price(12.5), 12.5)

    def test_card_price_is_null_when_unavailable(self):
        row = pd.Series({
            "parent_asin": "A1", "title": "Thing", "store": "S",
            "price": 0.0, "average_rating": 4.0, "rating_number": 3,
            "category": "other", "image_url": "",
        })
        self.assertIsNone(_to_card(row)["price"])


class StorefrontTestCase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.catalog = load_catalog()


class TestListEndpoint(StorefrontTestCase):

    def test_shape_and_legacy_fields(self):
        d = self.client.get("/api/products?page_size=5").get_json()
        self.assertEqual(
            set(d.keys()),
            {"items", "total", "page", "page_size", "pages", "facets", "products", "per_page"},
        )
        self.assertEqual(d["total"], len(self.catalog))
        self.assertEqual(len(d["items"]), 5)
        self.assertEqual(d["products"], d["items"])  # legacy alias
        self.assertEqual(d["per_page"], d["page_size"])  # legacy alias
        self.assertIn("categories", d["facets"])
        self.assertIn("prices", d["facets"])
        self.assertIn("ratings", d["facets"])

    def test_card_fields(self):
        d = self.client.get("/api/products?page_size=1").get_json()
        card = d["items"][0]
        self.assertEqual(
            set(card.keys()),
            {"parent_asin", "title", "store", "price", "average_rating",
             "rating_number", "category", "image_url"},
        )

    def test_pagination(self):
        first = self.client.get("/api/products?page=1&page_size=10").get_json()
        second = self.client.get("/api/products?page=2&page_size=10").get_json()
        self.assertEqual(first["page"], 1)
        self.assertEqual(second["page"], 2)
        self.assertEqual(len(second["items"]), 10)
        first_asins = [p["parent_asin"] for p in first["items"]]
        second_asins = [p["parent_asin"] for p in second["items"]]
        self.assertFalse(set(first_asins) & set(second_asins), "pages must not overlap")

    def test_page_size_capped(self):
        d = self.client.get("/api/products?page_size=1000").get_json()
        self.assertEqual(d["page_size"], 60)

    def test_text_filter_requires_all_tokens(self):
        d = self.client.get("/api/products?q=hair%20dryer&page_size=60").get_json()
        for card in d["items"]:
            title = card["title"].lower()
            self.assertIn("hair", title)
            self.assertIn("dryer", title)

    def test_category_filter(self):
        d = self.client.get("/api/products?category=nails&page_size=60").get_json()
        self.assertTrue(d["items"])
        for card in d["items"]:
            self.assertEqual(card["category"], "nails")

    def test_price_filter(self):
        d = self.client.get("/api/products?min_price=10&max_price=20&page_size=60").get_json()
        for card in d["items"]:
            self.assertIsNotNone(card["price"])
            self.assertTrue(10 <= card["price"] <= 20)

    def test_rating_filter(self):
        d = self.client.get("/api/products?min_rating=4.5&page_size=60").get_json()
        for card in d["items"]:
            self.assertGreaterEqual(card["average_rating"], 4.5)

    def test_sort_price_asc(self):
        d = self.client.get("/api/products?sort=price_asc&page_size=60").get_json()
        prices = [p["price"] for p in d["items"]]
        self.assertEqual(prices, sorted(prices))

    def test_sort_price_desc(self):
        d = self.client.get("/api/products?sort=price_desc&page_size=60").get_json()
        prices = [p["price"] for p in d["items"]]
        self.assertEqual(prices, sorted(prices, reverse=True))

    def test_sort_reviews(self):
        d = self.client.get("/api/products?sort=reviews&page_size=20").get_json()
        counts = [p["rating_number"] for p in d["items"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_sort_rating(self):
        d = self.client.get("/api/products?sort=rating&page_size=20").get_json()
        ratings = [p["average_rating"] for p in d["items"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))


class TestFacets(StorefrontTestCase):

    def test_category_counts_ignore_category_filter(self):
        base = self.client.get("/api/products").get_json()
        filtered = self.client.get("/api/products?category=nails").get_json()
        base_counts = {f["value"]: f["count"] for f in base["facets"]["categories"]}
        filtered_counts = {f["value"]: f["count"] for f in filtered["facets"]["categories"]}
        # Choosing a category must not zero out the other category counts.
        self.assertEqual(base_counts, filtered_counts)

    def test_category_counts_sum_to_total(self):
        d = self.client.get("/api/products?q=hair").get_json()
        total = sum(f["count"] for f in d["facets"]["categories"])
        self.assertEqual(total, d["total"])

    def test_price_facet_counts_consistent(self):
        d = self.client.get("/api/products").get_json()
        # Sum of price buckets equals the number of priced products in the base set.
        priced = sum(f["count"] for f in d["facets"]["prices"])
        self.assertGreater(priced, 0)
        self.assertLessEqual(priced, d["total"])


class TestDetailEndpoint(StorefrontTestCase):

    def _first_asin(self):
        return self.client.get("/api/products?page_size=1").get_json()["items"][0]["parent_asin"]

    def test_detail_fields(self):
        asin = self._first_asin()
        resp = self.client.get(f"/api/products/{asin}")
        self.assertEqual(resp.status_code, 200)
        d = resp.get_json()
        self.assertEqual(
            set(d.keys()),
            {"parent_asin", "title", "store", "price", "average_rating",
             "rating_number", "category", "image_url", "features", "description",
             "n_reviews_sampled", "reviews"},
        )
        self.assertEqual(d["parent_asin"], asin)
        self.assertIsInstance(d["features"], list)
        self.assertIsInstance(d["reviews"], list)
        self.assertLessEqual(len(d["reviews"]), 5)

    def test_reviews_sorted_by_helpfulness(self):
        asin = self._first_asin()
        reviews = self.client.get(f"/api/products/{asin}").get_json()["reviews"]
        votes = [r["helpful_vote"] for r in reviews]
        self.assertEqual(votes, sorted(votes, reverse=True))

    def test_unknown_asin_returns_404(self):
        self.assertEqual(self.client.get("/api/products/NOPE123").status_code, 404)


class TestOrderEndpoint(StorefrontTestCase):

    def _any_order_id(self):
        connection = sqlite3.connect(orders.DB_PATH)
        try:
            row = connection.execute("SELECT order_id FROM orders LIMIT 1").fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(row, "orders.db has no rows to test against")
        return row[0]

    def test_known_order_returns_status_and_tracking(self):
        order_id = self._any_order_id()
        resp = self.client.get(f"/api/orders/{order_id}")
        self.assertEqual(resp.status_code, 200)
        d = resp.get_json()
        self.assertEqual(
            set(d.keys()),
            {"order_id", "parent_asin", "product_title", "quantity", "status",
             "order_date", "estimated_delivery", "delivered_date", "tracking_number"},
        )
        self.assertEqual(d["order_id"], order_id)
        self.assertTrue(d["status"])

    def test_order_never_exposes_personal_fields(self):
        order_id = self._any_order_id()
        d = self.client.get(f"/api/orders/{order_id}").get_json()
        for personal in ("customer_name", "city", "payment_method"):
            self.assertNotIn(personal, d)

    def test_product_title_matches_catalog(self):
        order_id = self._any_order_id()
        d = self.client.get(f"/api/orders/{order_id}").get_json()
        match = self.catalog[self.catalog["parent_asin"].astype(str) == d["parent_asin"]]
        if not match.empty:
            self.assertEqual(d["product_title"], str(match.iloc[0]["title"]))

    def test_lookup_is_case_insensitive(self):
        order_id = self._any_order_id()
        lower = self.client.get(f"/api/orders/{order_id.lower()}")
        self.assertEqual(lower.status_code, 200)

    def test_unknown_order_returns_404(self):
        self.assertEqual(self.client.get("/api/orders/ORD-9999999").status_code, 404)


class TestHelpEndpoint(StorefrontTestCase):

    def test_every_topic_returns_markdown(self):
        for topic in storefront_module.TOPIC_DOCS:
            resp = self.client.get(f"/api/help/{topic}")
            self.assertEqual(resp.status_code, 200, topic)
            d = resp.get_json()
            self.assertEqual(d["topic"], topic)
            self.assertTrue(d["title"])
            self.assertTrue(d["source"].endswith(".md"))
            self.assertTrue(d["markdown"].strip())

    def test_markdown_matches_the_source_document(self):
        d = self.client.get("/api/help/returns").get_json()
        path = os.path.join(storefront_module.DOCS_DIR, d["source"])
        with open(path, "r", encoding="utf-8") as handle:
            self.assertEqual(d["markdown"], handle.read())

    def test_topic_is_case_insensitive(self):
        self.assertEqual(self.client.get("/api/help/Warranty").status_code, 200)

    def test_unknown_topic_returns_404(self):
        self.assertEqual(self.client.get("/api/help/bananas").status_code, 404)


class TestBlueprintRegistration(unittest.TestCase):

    def test_products_route_registered_once_on_app(self):
        rules = [r for r in app.url_map.iter_rules() if str(r) == "/api/products"]
        self.assertEqual(len(rules), 1)

    def test_orders_and_help_routes_registered(self):
        rules = {str(r) for r in app.url_map.iter_rules()}
        self.assertIn("/api/orders/<order_id>", rules)
        self.assertIn("/api/help/<topic>", rules)


class TestChatContractUnchanged(unittest.TestCase):
    """Registering the storefront blueprint must not alter /api/chat."""

    def test_chat_response_and_product_keys_unchanged(self):
        product = {
            "parent_asin": "A1", "title": "Test", "store": "S", "price": 9.99,
            "average_rating": 4.0, "rating_number": 10, "category_path": "",
            "features_text": "f", "description_text": "d", "similarity_score": 0.7,
        }
        route = {
            "intent": "search_product", "confidence": 0.9, "price_filter": False,
            "min_usd": None, "max_usd": None, "classified_intent": "search_product",
        }
        search = {
            "cleaned_query": "shampoo", "min_usd": None, "max_usd": None,
            "filter_active": False, "products": [product], "fallback_products": [],
        }
        client = app_baseline.app.test_client()
        with mock.patch("app_baseline.ensure_nlp"), \
                mock.patch("nlp_engine.route_message", return_value=route), \
                mock.patch("product_search.search_products_filtered", return_value=search):
            resp = client.post(
                "/api/chat",
                json={"message": "shampoo", "session_id": str(uuid.uuid4())},
            )

        self.assertEqual(resp.status_code, 200)
        d = resp.get_json()
        self.assertEqual(set(d.keys()), CHAT_TOP_KEYS)
        self.assertEqual(set(resp.get_json()["products"][0].keys()), CHAT_PRODUCT_KEYS)
        self.assertNotIn("image_url", resp.get_json()["products"][0])

    def test_health_still_ok(self):
        with mock.patch("translation.translation_status", return_value="ok"):
            resp = app_baseline.app.test_client().get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")


class TestAgentAppBlueprint(unittest.TestCase):

    def test_serves_products(self):
        client = app.test_client()
        resp = client.get("/api/products?page_size=2")
        self.assertEqual(resp.status_code, 200)
        d = resp.get_json()
        self.assertIn("items", d)
        self.assertIn("facets", d)

    def test_has_products_route_once(self):
        rules = [r for r in app.url_map.iter_rules() if str(r) == "/api/products"]
        self.assertEqual(len(rules), 1)

    def test_serves_orders_and_help(self):
        client = app.test_client()
        self.assertEqual(client.get("/api/help/returns").status_code, 200)
        connection = sqlite3.connect(orders.DB_PATH)
        try:
            order_id = connection.execute("SELECT order_id FROM orders LIMIT 1").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(client.get(f"/api/orders/{order_id}").status_code, 200)




if __name__ == "__main__":
    unittest.main(verbosity=2)
