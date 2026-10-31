"""
e2e visual check for the storefront and the ShopBot widget (System A server).

Starts app.py (System A, port 5000) locally, drives the hash-routed page with
Playwright Chromium, saves screenshots to docs/screenshots/ and fails if any
page reports a console error or an uncaught exception.

The ShopBot widget is exercised with its built-in mock replies (`?mock=1`):
this exercises the real widget UI with zero network calls, so no LLM and no
Gemini key is ever used. System B is never started here. .env is never read,
printed or captured.

The storefront is a hash-routed SPA:
    #/                      home
    #/search?q=...          search results
    #/category/<name>       category listing
    #/product/<asin>        product detail
    #/compare?asins=a,b,c    comparison
    #/cart                   cart
    #/orders?order_id=...    order tracking
    #/help/<topic>           policy help

The assistant is a self-contained Shadow DOM widget ([data-shopbot-host]);
Playwright's CSS engine pierces the open shadow root, so plain selectors like
.sb-launcher reach inside it.

This is NOT part of the default unit test run (it lives in tests/e2e/ and is
not named test_*.py). Run it explicitly:

    .venv\\Scripts\\python.exe tests\\e2e\\screenshot_journey.py

Prerequisites (dev-only):  pip install -r requirements-dev.txt
                           .venv\\Scripts\\python.exe -m playwright install chromium
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
SHOTS = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:5000"
SEARCH_HASH = "#/search?q=hair+brushes"
SEARCH_QUERY = "show me hair brushes"
HELP_TOPIC = "returns"
VIEWPORTS = [(1440, 900), (390, 844)]


def first_order_id():
    """A real order id from data/orders.db so the orders page has data."""
    connection = sqlite3.connect(str(SHOTS.parent.parent / "data" / "orders.db"))
    try:
        row = connection.execute("SELECT order_id FROM orders LIMIT 1").fetchone()
    finally:
        connection.close()
    return row[0] if row else "ORD-10000"


ORDER_ID = first_order_id()


def start_server():
    """Start app.py (System A) and return (process, log_path, log_handle)."""
    log_path = Path(tempfile.gettempdir()) / "e2e_system_a_server.log"
    handle = open(log_path, "w", encoding="utf-8")
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PORT": "5000"})
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "app.py"],
        cwd=str(BACKEND), stdout=handle, stderr=subprocess.STDOUT, env=env,
    )
    return proc, log_path, handle


def wait_health(timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            time.sleep(2)
    return False


def capture(page, name):
    path = SHOTS / name
    page.screenshot(path=str(path))
    print("    saved docs/screenshots/" + name)


def _grid_facts(page):
    return page.evaluate("""() => {
      const cards = [...document.querySelectorAll('.product-card')];
      const imgs = [...document.querySelectorAll('.product-card img')];
      const loaded = imgs.filter(i => i.complete && i.naturalWidth > 0).length;
      const clipped = cards.filter(c => {
        const t = c.querySelector('.card-title');
        return t && t.scrollWidth > t.clientWidth + 1;
      }).length;
      const first = cards[0];
      const pick = (sel) => (first && first.querySelector(sel)) ? first.querySelector(sel).textContent.trim() : '';
      const count = document.querySelector('#listingCount');
      return {
        cards: cards.length,
        imageElements: imgs.length,
        imagesLoaded: loaded,
        placeholders: document.querySelectorAll('.product-card .card-media-placeholder').length,
        clippedTitles: clipped,
        results: count ? count.textContent.trim() : '',
        firstTitle: pick('.card-title'),
        firstStore: pick('.card-store'),
        firstPrice: pick('.price-usd') || 'n/a',
        tabs: [...document.querySelectorAll('#categoryTabs .category-tab')].map(b => b.textContent.trim()),
        dir: document.documentElement.dir,
        lang: document.documentElement.lang,
      };
    }""")


def _detail_facts(page):
    return page.evaluate("""() => {
      const t = document.querySelector('.detail-title');
      const img = document.querySelector('.detail-media img');
      const pick = (sel) => { const n = document.querySelector(sel); return n ? n.textContent.trim() : ''; };
      return {
        title: t ? t.textContent.trim() : null,
        store: pick('.detail-store'),
        price: pick('.detail-price .price-usd'),
        features: document.querySelectorAll('.detail-features li').length,
        hasDescription: !!document.querySelector('.detail-description'),
        sections: document.querySelectorAll('.detail-section').length,
        reviews: document.querySelectorAll('.review-item').length,
        mediaImageLoaded: img ? (img.complete && img.naturalWidth > 0) : false,
        hasPlaceholder: !!document.querySelector('.detail-media .card-media-placeholder'),
        crumb: (document.querySelector('.crumb-current') || {}).textContent || '',
      };
    }""")


def _order_facts(page):
    return page.evaluate("""() => {
      const pick = (sel) => { const n = document.querySelector(sel); return n ? n.textContent.trim() : ''; };
      return {
        orderId: pick('.order-id'),
        status: pick('.order-status'),
        fields: document.querySelectorAll('.order-field').length,
        hasProduct: !!document.querySelector('.order-field'),
      };
    }""")


def _help_facts(page):
    return page.evaluate("""() => {
      const content = document.querySelector('.help-content');
      return {
        tabs: document.querySelectorAll('.help-tab').length,
        current: (document.querySelector('.help-tab.is-current') || {}).textContent || '',
        headings: content ? content.querySelectorAll('h3, h4').length : 0,
        bullets: content ? content.querySelectorAll('li').length : 0,
        chars: content ? content.textContent.trim().length : 0,
      };
    }""")


def _widget_facts(page):
    return page.evaluate("""() => {
      const host = document.querySelector('[data-shopbot-host]');
      const sr = host && host.shadowRoot;
      const q = (sel) => (sr ? sr.querySelector(sel) : null);
      const text = (sel) => { const n = q(sel); return n ? n.textContent.trim() : ''; };
      const root = q('.sb-root');
      const win = q('.sb-window');
      const backdrop = q('.sb-backdrop');
      const bubbles = sr ? [...sr.querySelectorAll('.sb-msg.bot .sb-bubble')] : [];
      const last = bubbles.length ? bubbles[bubbles.length - 1].textContent.trim() : '';
      return {
        mode: root ? root.getAttribute('data-mode') : null,
        lang: root ? root.getAttribute('data-lang') : null,
        dir: root ? root.getAttribute('dir') : null,
        mockBadge: text('.sb-badge.is-mock'),
        status: text('.sb-status-text'),
        userMessages: sr ? sr.querySelectorAll('.sb-msg.user').length : 0,
        botMessages: sr ? sr.querySelectorAll('.sb-msg.bot').length : 0,
        searchChips: sr ? sr.querySelectorAll('.sb-action:not(.sb-action-primary)').length : 0,
        compareButtons: sr ? sr.querySelectorAll('.sb-action-primary').length : 0,
        productCards: sr ? sr.querySelectorAll('.sb-product, .sb-product-card, .chat-product-card').length : 0,
        windowOpen: (win && !win.hidden) ? true : false,
        backdropOpen: (backdrop && !backdrop.hidden) ? true : false,
        lastBubble: last,
      };
    }""")


def _store_layout(page):
    return page.evaluate("""() => {
      const host = document.querySelector('[data-shopbot-host]');
      const launcher = host && host.shadowRoot ? host.shadowRoot.querySelector('.sb-launcher') : null;
      const lb = launcher ? launcher.getBoundingClientRect() : null;
      const lang = document.querySelector('#langSwitcher');
      const langBox = lang ? lang.getBoundingClientRect() : null;
      const overlaps = (a, b) => (a && b)
        ? !(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y)
        : false;
      return {
        horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
        scrollWidth: document.documentElement.scrollWidth,
        innerWidth: window.innerWidth,
        launcherVisible: launcher ? !launcher.hidden : false,
        launcherInViewport: lb ? (lb.x >= -1 && lb.y >= -1 &&
                                  lb.x + lb.width <= window.innerWidth + 1 &&
                                  lb.y + lb.height <= window.innerHeight + 1) : false,
        launcherOverlapsLang: overlaps(lb, langBox),
      };
    }""")


def run_viewport(browser, width, height):
    print(f"  viewport {width}x{height}")
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()

    console_errors, page_errors, failed_requests = [], [], []

    def on_console(msg):
        if msg.type != "error":
            return
        # Browser logs the expected /api/site 404 (see on_response below) as a
        # console error with only the status in its text; skip it by source URL.
        if (msg.location or {}).get("url", "").endswith("/api/site"):
            return
        console_errors.append(msg.text)

    page.on("console", on_console)
    page.on("pageerror", lambda e: page_errors.append(str(e)))

    def on_failed(request):
        failed_requests.append((request.url, str(request.failure or "")))

    def on_response(response):
        if response.status < 400:
            return
        # The widget probes GET /api/site for the active store's page templates.
        # System A (app.py) has no such endpoint; the widget is built to fall
        # back to its own templates, so this one 404 is expected here.
        if response.status == 404 and response.url.endswith("/api/site"):
            return
        failed_requests.append((response.url, f"HTTP {response.status}"))

    page.on("requestfailed", on_failed)
    page.on("response", on_response)

    # Home
    page.goto(BASE, wait_until="load", timeout=60000)
    page.wait_for_selector(".product-card", timeout=60000)
    page.wait_for_timeout(1500)
    capture(page, f"store_home_{width}.png")
    print("    home:  " + json.dumps(_grid_facts(page), ensure_ascii=False))

    # Search results
    page.goto(BASE + "/" + SEARCH_HASH, wait_until="load", timeout=60000)
    page.wait_for_selector("#listingCount", timeout=60000)
    page.wait_for_selector(".product-card", timeout=60000)
    page.wait_for_timeout(1200)
    capture(page, f"store_grid_{width}.png")
    print("    grid:  " + json.dumps(_grid_facts(page), ensure_ascii=False))

    # Product detail (route)
    page.locator(".card-details-link").first.click()
    page.wait_for_selector(".detail-title", timeout=30000)
    page.wait_for_timeout(1200)
    capture(page, f"product_detail_{width}.png")
    print("    detail: " + json.dumps(_detail_facts(page), ensure_ascii=False))

    # Order tracking
    page.goto(BASE + "/#/orders?order_id=" + ORDER_ID, wait_until="load", timeout=60000)
    page.wait_for_selector(".order-card", timeout=30000)
    page.wait_for_timeout(800)
    capture(page, f"order_tracking_{width}.png")
    print("    order: " + json.dumps(_order_facts(page), ensure_ascii=False))

    # Help / policy page
    page.goto(BASE + "/#/help/returns", wait_until="load", timeout=60000)
    page.wait_for_selector(".help-content h3, .help-content h4", timeout=30000)
    page.wait_for_timeout(800)
    capture(page, f"help_returns_{width}.png")
    print("    help:  " + json.dumps(_help_facts(page), ensure_ascii=False))

    # --- ShopBot widget (mock mode: real UI, no network, no LLM) ----------
    page.goto(BASE + "/?mock=1", wait_until="load", timeout=60000)
    page.wait_for_selector(".product-card", timeout=60000)

    page.click(".sb-launcher")
    page.wait_for_selector(".sb-window", state="visible", timeout=15000)
    page.wait_for_timeout(700)
    capture(page, f"assistant_large_{width}.png")
    print("    widget_large: " + json.dumps(_widget_facts(page), ensure_ascii=False))

    # Search action: one chip + automatic navigation, then Docked mode.
    page.fill(".sb-input", SEARCH_QUERY)
    page.press(".sb-input", "Enter")
    page.wait_for_selector(".sb-action", timeout=15000)
    page.wait_for_selector("#listingCount", timeout=30000)
    page.wait_for_timeout(900)
    capture(page, f"assistant_search_{width}.png")
    print("    widget_search: " + json.dumps(_widget_facts(page), ensure_ascii=False))

    # Compare action: a summary + an "Open comparison" button (no auto-nav).
    page.fill(".sb-input", "compare the first two")
    page.press(".sb-input", "Enter")
    page.wait_for_selector(".sb-action-primary", timeout=15000)
    page.wait_for_timeout(700)
    capture(page, f"assistant_compare_{width}.png")
    print("    widget_compare: " + json.dumps(_widget_facts(page), ensure_ascii=False))
    page.click(".sb-action-primary")
    page.wait_for_timeout(700)

    # Order and policy questions: text only (no chip, no navigation).
    page.fill(".sb-input", "where is ORD-10025?")
    page.press(".sb-input", "Enter")
    page.wait_for_timeout(600)
    page.fill(".sb-input", "what is your return policy?")
    page.press(".sb-input", "Enter")
    page.wait_for_timeout(600)
    capture(page, f"assistant_text_{width}.png")
    print("    widget_text: " + json.dumps(_widget_facts(page), ensure_ascii=False))

    # Arabic / RTL
    page.click(".sb-lang-btn")
    page.wait_for_timeout(600)
    capture(page, f"assistant_arabic_{width}.png")
    print("    widget_ar: " + json.dumps(_widget_facts(page), ensure_ascii=False))

    # Mode switches and Escape-to-close.
    page.click(".sb-modes button:nth-child(2)")   # docked
    page.wait_for_timeout(500)
    capture(page, f"assistant_docked_{width}.png")
    page.click(".sb-modes button:nth-child(1)")   # compact
    page.wait_for_timeout(500)
    print("    widget_modes: " + json.dumps(_widget_facts(page), ensure_ascii=False))
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    print("    widget_closed: " + json.dumps(_widget_facts(page), ensure_ascii=False))
    print("    layout:  " + json.dumps(_store_layout(page), ensure_ascii=False))

    context.close()
    return console_errors, page_errors, failed_requests


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)

    server, log_path, log_handle = start_server()
    all_console, all_page, all_failed = [], [], []
    try:
        if not wait_health():
            print("FAILED: System A did not become healthy on", BASE)
            return 2

        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                for width, height in VIEWPORTS:
                    try:
                        c, p, f = run_viewport(browser, width, height)
                    except Exception as exc:  # noqa: BLE001 — keep the report complete
                        all_page.append((width, f"driver error: {exc}"))
                        continue
                    all_console += [(width, x) for x in c]
                    all_page += [(width, x) for x in p]
                    all_failed += [(width, *x) for x in f]
            finally:
                browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()
        log_handle.close()

    print("\n" + "=" * 70)
    print("Console errors:", len(all_console))
    for width, text in all_console:
        print(f"  [{width}] {text}")
    print("Uncaught page exceptions:", len(all_page))
    for width, text in all_page:
        print(f"  [{width}] {text}")
    print("Failed / error network responses:", len(all_failed))
    seen = set()
    for width, url, reason in all_failed:
        key = (url, reason)
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{width}] {reason}  {url}")

    if all_console or all_page:
        print("\nRESULT: FAIL (console/page errors present)")
        return 1
    print("\nRESULT: PASS (no console or page errors)")
    print("server log:", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
