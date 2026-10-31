"""
Tests for site_config (loading and validating sites/<id>/site.json).

No network; reads the real sites/shopstore/site.json.

Run:  .venv\\Scripts\\python.exe tests\\test_site_config.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import site_config  # noqa: E402


class TestDefaultSite(unittest.TestCase):

    def setUp(self):
        site_config.clear_cache()

    def tearDown(self):
        os.environ.pop("SITE", None)
        site_config.clear_cache()

    def test_loads_the_shopstore_site(self):
        site = site_config.load_site()
        self.assertEqual(site["site_id"], "shopstore")
        self.assertTrue(site["name"])
        self.assertTrue(site["assistant_name"])
        self.assertTrue(site["scope_description"])

    def test_currency_and_secondary_currency(self):
        site = site_config.load_site()
        self.assertEqual(site["currency"], "USD")
        self.assertEqual(site["secondary_currency"], "AED")
        self.assertEqual(site["secondary_currency_rate"], 3.67)

    def test_languages_include_english_and_arabic(self):
        site = site_config.load_site()
        self.assertIn("en", site["languages"])
        self.assertIn("ar", site["languages"])

    def test_pages_map_has_the_expected_routes(self):
        pages = site_config.load_site()["pages"]
        for key in ("home", "search", "category", "product", "compare", "cart", "orders", "help"):
            self.assertIn(key, pages, f"missing page: {key}")
        self.assertEqual(pages["home"], "#/")
        self.assertEqual(pages["product"], "#/product/{asin}")
        self.assertEqual(pages["compare"], "#/compare?asins={asins}")
        self.assertEqual(pages["cart"], "#/cart")
        self.assertEqual(pages["orders"], "#/orders?order_id={order_id}")
        self.assertEqual(pages["help"], "#/help/{topic}")
        self.assertEqual(pages["category"], "#/category/{category}")
        self.assertIn("{q}", pages["search"])
        self.assertIn("{min_price}", pages["search"])

    def test_paths_resolve_to_existing_targets(self):
        site = site_config.load_site()
        for key in ("catalog", "reviews", "docs", "orders_db"):
            resolved = site_config.resolve_path(site, key)
            self.assertTrue(resolved, f"no path for {key}")
            self.assertTrue(os.path.exists(resolved), f"path does not exist: {resolved}")

    def test_load_is_cached(self):
        first = site_config.load_site()
        second = site_config.load_site()
        self.assertIs(first, second)


class TestPublicSiteInfo(unittest.TestCase):

    def setUp(self):
        site_config.clear_cache()

    def tearDown(self):
        os.environ.pop("SITE", None)
        site_config.clear_cache()

    def test_excludes_filesystem_paths(self):
        info = site_config.public_site_info()
        self.assertNotIn("paths", info)

    def test_includes_widget_fields(self):
        info = site_config.public_site_info()
        for key in ("name", "assistant_name", "currency", "languages", "pages"):
            self.assertIn(key, info)
        self.assertEqual(info["pages"]["cart"], "#/cart")


class TestSiteSelection(unittest.TestCase):

    def setUp(self):
        site_config.clear_cache()

    def tearDown(self):
        os.environ.pop("SITE", None)
        site_config.clear_cache()

    def test_site_env_var_selects_the_site(self):
        os.environ["SITE"] = "shopstore"
        self.assertEqual(site_config.load_site()["site_id"], "shopstore")

    def test_missing_site_raises(self):
        os.environ["SITE"] = "no-such-site"
        with self.assertRaises(site_config.SiteConfigError):
            site_config.load_site()

    def test_safe_loader_returns_empty_on_error(self):
        os.environ["SITE"] = "no-such-site"
        self.assertEqual(site_config.load_site_safe(), {})

    def test_validate_rejects_incomplete_config(self):
        with self.assertRaises(site_config.SiteConfigError):
            site_config._validate({"site_id": "x"}, "x")

    def test_validate_rejects_empty_pages(self):
        config = {
            "site_id": "x", "name": "n", "assistant_name": "a",
            "scope_description": "s", "currency": "USD",
            "languages": ["en"], "pages": {},
        }
        with self.assertRaises(site_config.SiteConfigError):
            site_config._validate(config, "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
