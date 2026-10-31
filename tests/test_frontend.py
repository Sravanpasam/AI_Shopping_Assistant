"""
test_frontend.py
----------------
Structural tests for the static storefront frontend (no JS runtime needed).

The frontend is split into three scripts:
  * common.js          - shared helpers, API base, cart/compare storage, event bus
  * shopbot/shopbot.js - the self-contained Shadow-DOM ShopBot widget (EN/AR, voice)
  * app.js             - the hash-routed storefront (home/search/category/product/
                         compare/cart/orders/help)

These tests guard against the mistakes that are easy to make when hand-editing
index.html / *.js / style.css:
  * an id referenced by the JS that does not exist in index.html,
  * losing the file:// fallback literal in the API base,
  * dropping one of the API endpoints the UI depends on,
  * the chat shipping in only one language.
"""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")

# Ids the JS creates dynamically (so they are expected NOT to be in index.html).
DYNAMIC_IDS = {"homeTiles", "listingCount", "compareProducts", "compareTable"}

JS_FILES = ("common.js", "shopbot/shopbot.js", "app.js")
SHOPBOT_DIR = os.path.join(FRONTEND, "shopbot")


def read(name: str) -> str:
    with open(os.path.join(FRONTEND, name), encoding="utf-8") as handle:
        return handle.read()


class TestFrontendFiles(unittest.TestCase):
    def setUp(self):
        self.html = read("index.html")
        self.css = read("style.css")
        self.widget_css = read(os.path.join("shopbot", "shopbot.css"))
        self.scripts = {name: read(name) for name in JS_FILES}
        self.js = "\n".join(self.scripts.values())
        self.widget = self.scripts["shopbot/shopbot.js"]

    def test_core_regions_present(self):
        # Store chrome that app.js drives. The assistant is a Shadow-DOM
        # widget with no markup in index.html, so it is not listed here.
        for token in (
            'id="brandLink"', 'id="headerSearchForm"', 'id="headerSearchInput"',
            'id="headerChatBtn"', 'id="langSwitcher"', 'id="cartLink"',
            'id="cartCount"', 'id="categoryTabs"', 'id="view"',
            'id="compareBar"', 'id="toastContainer"',
        ):
            self.assertIn(token, self.html, f"index.html is missing {token}")

    def test_widget_script_is_loaded_before_app(self):
        # The widget mounts itself and app.js then configures it, so the widget
        # script tag must come before app.js in index.html.
        order = [self.html.find('src="%s"' % name) for name in JS_FILES]
        self.assertEqual(order, sorted(order),
                         "scripts must load common.js, shopbot/shopbot.js, app.js in order")
        self.assertNotIn(-1, order, "index.html is missing a frontend script tag")

    def test_widget_is_self_contained(self):
        # The widget must not depend on the host page's globals/helpers.
        for forbidden in ("window.ShopStore", "window.SS", "SS."):
            self.assertNotIn(forbidden, self.widget,
                             "shopbot.js must stay self-contained (no %s)" % forbidden)
        # It must use a Shadow DOM root and inject its own stylesheet.
        self.assertIn("attachShadow", self.widget)
        self.assertIn("shopbot.css", self.widget)

    def test_scripts_reference_expected_endpoints(self):
        for endpoint in ("/api/products", "/api/chat", "/api/compare",
                         "/api/transcribe", "/api/health"):
            self.assertIn(endpoint, self.js, f"no script calls {endpoint}")
        # The widget also reads the store config from the active backend.
        self.assertIn("/api/site", self.widget)

    def test_api_base_is_computed_client_side(self):
        # common.js chooses its own API base: window.location.origin when served
        # over http/https, http://localhost:5000 only when opened from file://.
        # No server-side rewriting is involved.
        common = self.scripts["common.js"]
        self.assertIn("http://localhost:5000", common)
        self.assertIn("window.location.origin", common)

    def test_every_referenced_id_exists(self):
        referenced = set()
        for source in self.scripts.values():
            referenced |= set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", source))
            referenced |= set(re.findall(r"getElementById\('([^']+)'\)", source))
        present = set(re.findall(r'id="([^"]+)"', self.html))
        missing = sorted(referenced - present - DYNAMIC_IDS)
        self.assertEqual([], missing, f"JS references unknown id(s): {missing}")

    def test_widget_i18n_has_both_languages_with_same_keys(self):
        en_block = re.search(r"\ben:\s*\{(.*?)\n    \},", self.widget, re.S)
        ar_block = re.search(r"\bar:\s*\{(.*?)\n    \}", self.widget, re.S)
        self.assertIsNotNone(en_block, "English i18n block not found in shopbot.js")
        self.assertIsNotNone(ar_block, "Arabic i18n block not found in shopbot.js")

        def keys(block: str):
            return set(re.findall(r"^\s{6}([a-z0-9_]+):", block, re.M))

        en_keys, ar_keys = keys(en_block.group(1)), keys(ar_block.group(1))
        self.assertTrue(en_keys, "no English translation keys parsed")
        self.assertEqual(en_keys, ar_keys, "EN and AR translation keys diverge")

    def test_widget_renders_followup_suggestions(self):
        self.assertIn("data.suggestions", self.widget)
        self.assertIn("message.suggestions", self.widget)
        self.assertIn("sb-followups", self.widget)
        self.assertIn("sb-followup", self.widget)
        self.assertIn(".sb-followups", self.widget_css)
        self.assertIn(".sb-followup", self.widget_css)

    def test_widget_shows_quota_state(self):
        # /api/health can report the daily LLM quota as exhausted; the widget
        # then shows an amber status dot with a translated label.
        self.assertIn("quota_exhausted", self.widget)
        self.assertIn("status_quota", self.widget)
        self.assertIn(".sb-dot.is-quota", self.widget_css)
        self.assertIn("--sb-amber", self.widget_css)

    def test_widget_shows_limited_mode(self):
        self.assertIn("status_limited", self.widget)
        self.assertIn("degraded", self.widget)
        self.assertIn("llm === 'limited'", self.widget)
        self.assertIn(".sb-dot.is-limited", self.widget_css)
        self.assertIn(".sb-badge.is-limited", self.widget_css)

    def test_widget_navigates_on_products_without_actions(self):
        # System A sends products but no actions; the widget still opens the
        # search results page using the user's (cleaned) query.
        self.assertIn("data.products", self.widget)
        self.assertIn("lastUser", self.widget)
        self.assertIn("page: 'search'", self.widget)

    def test_css_defines_global_hidden_rule(self):
        # The UI toggles visibility with the `hidden` attribute everywhere.
        self.assertRegex(self.css, r"\[hidden\]\s*\{\s*display:\s*none\s*!important")
        self.assertRegex(self.widget_css,
                         r"\[hidden\]\s*\{\s*display:\s*none\s*!important")

    def test_arabic_rtl_support_present(self):
        # The store and the widget both support RTL: the store via style.css,
        # the widget via its own scoped dir attribute + shopbot.css.
        self.assertIn("rtl", self.css)
        self.assertIn("rtl", self.widget)
        self.assertIn("dir", self.widget)
        self.assertIn("rtl", self.widget_css)

    def test_no_amazon_branding(self):
        for text in [self.html.lower(), self.js.lower()]:
            self.assertNotIn("amazon", text)

    def test_braces_balanced(self):
        self._assert_balanced(self.css, "{}")
        for name, source in self.scripts.items():
            self._assert_balanced(source, "{}", skip_strings=True, label=name)

    @staticmethod
    def _assert_balanced(source: str, pairs: str, skip_strings: bool = False,
                         label: str = ""):
        opening, closing = pairs[0], pairs[1]
        depth = 0
        previous = ""
        index = 0
        length = len(source)
        while index < length:
            char = source[index]
            if skip_strings and char in "'\"":
                quote = char
                index += 1
                while index < length and source[index] != quote:
                    index += 2 if source[index] == "\\" else 1
                index += 1
                continue
            if skip_strings and char == "/" and index + 1 < length and source[index + 1] == "*":
                end = source.find("*/", index + 2)
                index = length if end < 0 else end + 2
                previous = "/"
                continue
            if skip_strings and char == "/" and index + 1 < length and source[index + 1] == "/":
                while index < length and source[index] != "\n":
                    index += 1
                continue
            if skip_strings and char == "/" and previous in "(,=:[!&|?{};+-*%~^<>" :
                index += 1
                while index < length and source[index] != "\n":
                    if source[index] == "\\":
                        index += 1
                    elif source[index] == "/":
                        break
                    index += 1
                previous = "/"
                index += 1
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth < 0:
                    raise AssertionError(f"unbalanced closing brace in {label or 'source'}")
            if not char.isspace():
                previous = char
            index += 1
        if depth != 0:
            raise AssertionError(f"unbalanced braces (depth={depth}) in {label or 'source'}")


if __name__ == "__main__":
    unittest.main()
