# Project context: E-commerce NLP chatbot (ShopBot)

University NLP course project (Master of Applied AI). Make small, focused changes. Do not restructure the project or rewrite working modules unless asked. Work on one fix at a time, show what changed, test it, and wait for approval.

## Goal
A chatbot for an e-commerce site that:
- finds products from natural language, with price filters
- compares 2 to 3 products side by side
- answers service questions (warranty, returns, cash on delivery, shipping, orders)
- works in English and Arabic
- accepts voice input only (mic), no text-to-speech; answers are shown on screen
- every answer must be grounded in data. Never invent facts. If the information is not in the data, say so

## Decisions already made
- Runs locally on a laptop (Flask, no hosting). Bind to 127.0.0.1, debug off
- Comparison is an inline table in the chat built from structured data (not LLM text), with a short summary under it. Max 3 products
- Voice: browser records audio, Whisper transcribes, transcript is shown in the input box so the user can edit it, then sent through the same pipeline as typed text
- Data: Amazon Reviews 2023, about 1,000 products with reviews, in data/catalog.parquet and data/reviews.parquet (built by 01_get_data.py; a mock sample with the same schema is used until the real data is in place)
- Catalog columns: parent_asin, title, store, price, average_rating, rating_number, category_path, features_text, description_text, n_reviews_sampled
- Review columns: parent_asin, rating, title, text, helpful_vote, verified_purchase, timestamp
- Prices are USD. AED conversion uses a fixed rate of 3.67
- Store policies are fictional and live in docs/ (returns, warranty, shipping, payment/COD, order tracking). Orders are mock data in data/orders.db, created by seed_orders.py

## Current backend (backend/)
app.py (Flask routes, in-memory sessions), data_loader.py, nlp_engine.py (intent classification by cosine similarity), product_search.py (FAISS), comparison.py, policy_qa.py (regex policy answers), transcription.py (Whisper), translation.py (langdetect + Google Translate), price_parser.py (new). Frontend is in frontend/. Tests are in tests/.

## Fix list (all 10 done)
1. DONE — Flask host 127.0.0.1, debug off, models load at startup
2. DONE — FAISS fingerprint (data/faiss_meta.json: row count + SHA-256 of sorted ASINs + text hash), index rebuilds on mismatch
3. DONE — price filter and similarity threshold
   - price_parser.py parse_price_constraint(text) returns (cleaned_query, min_usd, max_usd); bare ranges need a price signal; numbers followed by units (ml, g, pack, spf...) are not prices
   - used in search only on search intents: embed the cleaned query, take top 50 from FAISS, drop hits below MIN_SIMILARITY, filter by price, return top k; if nothing is left, say so and show the 3 cheapest relevant products; the applied filter is stated in the reply
   - MIN_SIMILARITY = 0.45 (backend/config.py), chosen from similarity_report.py on the real catalog: sensible queries 0.51–0.69 (except "gift for my mom" 0.420), nonsense ceiling 0.396
4. DONE — "Compare these" resolves the session's last products for these/them/both/first two; otherwise splits on and/vs/versus and searches each named product (top 1 each)
5. DONE — session tracks a "focused product"; policy answers resolve "the second one" or a named product before falling back to last_products[0]
6. DONE — removed invented fallback text (policy_qa return sentence, comparison "Reviewers consistently praise..."); unavailable info is stated as unavailable. Fixed the "$$" price bug in comparison.py
7. DONE — intent confidence is used: below CLARIFY_THRESHOLD the bot asks a clarifying question; the unreachable search fallback was removed
8. DONE — sessions capped (MAX_SESSIONS=200, oldest evicted); message length (2000 chars) and audio upload size (25 MB) limited; catalog lookup precomputed once instead of per query
9. DONE — policy text splits on newlines (features are newline-joined) and on sentence punctuation, so matches no longer span unrelated lines; tightened warranty (bare "guaranteed") and return (bare "replacement") regexes
10. DONE — requirements.txt present; README no longer claims spaCy NER; removed unused safe_str. scikit-learn kept (sentence-transformers imports sklearn via util)

## System A (current pipeline — FROZEN, tagged system-a)
The rule-based/retrieval pipeline above is System A. It is frozen for evaluation: no changes to its behavior, routing, or replies. Git tag `system-a` marks this baseline.

## System B (LLM agent) — now the one active system
The agent was merged into `backend/app.py` (port 5000, 127.0.0.1, debug off, models preloaded). The old rule-based server moved to `legacy/app_baseline.py` and `backend/app_b.py` was deleted. Built modules:
- backend/policy_search.py — DONE. Chunks docs/ by heading/paragraph, embeds with the existing model, search_policies(query, k=3) -> chunk text + source file + score. MIN_POLICY_SIMILARITY = 0.40, chosen from policy_scores.py (policy questions 0.437–0.793 vs off-topic ceiling 0.360)
- backend/orders.py — DONE. get_order(order_id) reads data/orders.db; extract_order_id(text) for ORD-\d+; structured data or None; never invents anything
- backend/escalation.py — DONE. detect_escalation(text) -> (bool, reason) for explicit human requests and strong frustration / repeated complaints; rule-based + similarity to example phrases; English and Arabic-translated phrasing
- LLM client + tools — DONE (step 1 of 2). Provider-neutral backend/llm_client.py (Google Gemini via google-genai, Ollama later) with rate limiting, retries, latency logging; backend/agent_tools.py wraps the search/comparison/policy/orders/escalation modules as never-raising JSON tools + TOOL_SCHEMAS; scripts/llm_smoke_test.py and scripts/list_models.py. Gemini key + model name come from .env (never logged). A model that is retired (404 NOT_FOUND) or over its daily quota rotates to the next model in GEMINI_FALLBACK_MODELS and is retired for the day; per-model daily budget LLM_DAILY_BUDGET guards free-tier limits.
- backend/agent.py — DONE (step 2 of 2). ShopBotAgent: bounded tool-calling loop (MAX_TOOL_ITERATIONS=6) over the 7 tools; grounded system prompt (answer only from tool data, say when unavailable, never claims an action was performed, tool args in English, reply in the user's language); captures a comparison table, the products shown, and escalation as structured results; provider errors never raise. Identical tool calls within one turn reuse the first result.
- backend/app.py — DONE. The single ShopBot Flask server on 127.0.0.1:5000 (debug off, models preloaded), serving the frontend/ folder. Endpoints: POST /api/chat, POST /api/compare, POST /api/transcribe, GET /api/health, GET /api/site, plus the shared storefront blueprint. /api/chat keeps the full contract (session_id, intent="agent", confidence, response_type, text, products, comparison, language) plus tools_used, escalated and actions, so eval/run_eval.py works unchanged. Per-session history (last 20 messages incl. tool results), greeting short-circuit and escalation pre-check before the LLM, max 3 tool rounds (MAX_TOOL_ROUNDS), friendly bilingual error on provider failure, 2000-char / 25 MB limits. GET /api/health reports the ShopBot name and active model (never keys). Tests: tests/test_app.py (LLM mocked); the old rule-based server is legacy/app_baseline.py.

### System B — resilience (DONE)
- Model fallback + budget in `backend/llm_client.py`: GEMINI_MODEL (first choice) plus GEMINI_FALLBACK_MODELS (comma-separated). A 404/NOT_FOUND (model retired) or a per-day 429 retires that model for the day and rotates to the next; LLM_DAILY_BUDGET caps calls per model per day (default 20). Errors are mapped to LLMDailyQuotaError / LLMModelUnavailableError / LLMQuotaExhaustedError; `get_active_model_name()` / `quota_state()` report the current model.
- Groq provider (DONE): `GroqProvider` in `backend/llm_client.py` speaks Groq's OpenAI-compatible chat-completions API via `httpx` (function calling supported; messages/tool calls converted to the OpenAI shape). Select with `LLM_PROVIDER=groq`, key in `GROQ_API_KEY`, first model in `GROQ_MODEL` (default `llama-3.3-70b-versatile`), backups in `GROQ_FALLBACK_MODELS`. Same rotation/budget/error mapping as Gemini. `get_model_name()` is provider-aware; `scripts/show_llm_config.py` reports both providers. Tests: `tests/test_llm_groq.py` (mocked transport, no network).
- Quota surfacing (DONE): GET /api/health returns an `llm` field (`"ok"` or `"quota_exhausted"`); a /api/chat turn that exhausts every model replies with a bilingual message and `response_type:"error"`, `error:"quota_exhausted"`. The widget shows an amber status dot + label (see below).
- Fewer LLM calls (DONE): backend/greetings.py answers a bare bilingual greeting (EN/AR + transliteration) with a canned reply and never calls the LLM — matched **before** the escalation check (the fuzzy escalation matcher mis-fires on Arabic greetings like مرحبا). MAX_TOOL_ROUNDS is 3.
- Optional eval cache (DONE): backend/llm_cache.py stores whole /api/chat responses keyed by sha256(message, context, history) in data/llm_cache.json (gitignored) when LLM_CACHE=1; only successful turns are cached. Off by default.

### System B — navigate actions (DONE)
- Every /api/chat response includes an `actions` list (empty when none); all other fields are unchanged. The agent turns each successful tool call into at most one action so the frontend can navigate instead of listing products in the chat.
- Action shapes: `{type:"navigate", page:"search", params:{q,min_price,max_price,category,sort}, label}`; `page:"product" params:{asin}`; `page:"compare" params:{asins:[...]}`; `page:"orders" params:{order_id}`; `page:"help" params:{topic}` where topic ∈ returns|warranty|shipping|payment|orders. `page:"cart"` is defined but never emitted (no cart tool). Actions come only from successful calls: a zero-match search or a failed lookup yields no action. Built in agent.py (`build_action`), deduped in the tool loop. `search_products` also returns `total_matches` (matches passing the similarity + price filter, before the limit) which the prompt uses to say how many results there are.

### System B — page context (DONE)
- POST /api/chat accepts an optional `context` object: `{page, asin, query, filters, cart_count, order_id}`. It is validated and size-capped (MAX_CONTEXT_CHARS=1000; strings ≤200; ≤10 filters) and added to the system prompt for that one turn ("The customer is currently viewing …"). Absent, malformed or oversized context behaves exactly as before.

### System B — per-site config and adapter (DONE)
- `sites/<site_id>/site.json` describes one store: site_id, name, assistant_name, scope_description, currency (+ secondary_currency and rate, e.g. AED 3.67), languages, the frontend `pages` hash-route templates (home, search, category, product, compare, cart, orders, help), and paths to the catalog, reviews, docs folder and orders db. The active site is chosen by the `SITE` env var or `app.py --site` (default `shopstore`).
- backend/site_config.py loads/validates/caches site.json; `public_site_info()` returns the non-sensitive parts served by GET /api/site (name, assistant_name, currency, languages, pages — never filesystem paths).
- backend/adapters/ is the store-data boundary: `base.StoreAdapter` (site_info, search_products, get_product, get_reviews, search_knowledge, get_order) and `local_files.LocalFilesAdapter` (reads the local parquet/docs/orders via the existing modules). `agent_tools` reads all store data through `get_adapter()`; the agent's system prompt is built from the site config. The default site's paths equal the modules' built-in defaults, so behavior is identical to before. `get_adapter`/`set_adapter`/`reset_adapter` allow swapping the adapter in tests.

The agent is now wired into the Flask app: `backend/app.py` is the one running system; `legacy/app_baseline.py` stays for the System A comparison. Next: run the evaluation of System A vs System B on ~100 questions (English, Arabic, unanswerable). Multilingual embeddings also still open.

## Storefront (frontend + product API)
- The frontend is a vanilla, dependency-free single-page app with hash routing, served by app.py (ShopBot, port 5000); the legacy baseline serves the same files. It is split into three files loaded in order: `frontend/common.js` (helpers, API base, cart/compare in localStorage, event bus → `window.ShopStore`/`SS`), `frontend/shopbot/shopbot.js` (+ `shopbot.css`, the ShopBot widget → `window.ShopBot`, Shadow-DOM isolated, EN/AR, voice), `frontend/app.js` (the router + pages). The old in-page `frontend/chat.js` was replaced by the widget and removed. `frontend/index.html` is the shell; `frontend/style.css` holds the store styles.
- Routes match `sites/<site>/site.json` pages: `#/`, `#/search?q=&min_price=&max_price=&category=&sort=&page=`, `#/category/{category}`, `#/product/{asin}`, `#/compare?asins=a,b,c`, `#/cart`, `#/orders?order_id=`, `#/help/{topic}`.
- `backend/storefront.py` is the shared product/order/help blueprint, registered by both the ShopBot app and the baseline (chat untouched): `GET /api/products` (filter/sort/paginate + facets: categories, prices, ratings), `GET /api/products/<asin>` (features, description, top reviews), `GET /api/orders/<order_id>` (status, dates, product title, quantity, tracking number from data/orders.db — personal fields such as customer name/city/payment method are never returned; unknown ids are a clean 404), `GET /api/help/<topic>` (a docs/*.md policy document as Markdown for topics returns|warranty|shipping|payment|orders).
- Product categories are derived deterministically from the title by the single commented `CATEGORY_RULES` list in storefront.py (first matching keyword wins; order matters, e.g. "makeup brush" → makeup). Current counts on the 1,000-product catalog: hair 167, skin care 154, makeup 133, tools and accessories 112, bath and body 105, fragrance 87, nails 69, other 173 (the remaining "other" titles are genuinely non-beauty items).
- Tests: `tests/test_storefront.py` covers the endpoints; `tests/test_frontend.py` is a structural check of the static files (no JS runtime). `tests/e2e/screenshot_journey.py` is a dev-only Playwright visual check (not in the default unit run); it drives the real pages and fails on any console/page/network error.

### ShopBot widget (frontend/shopbot/shopbot.js + shopbot.css)
- Self-contained, Shadow-DOM-isolated chat widget. Any page embeds it with `ShopBot.init({apiBase, host:{navigate, getContext}})` (idempotent); it never touches `window.ShopStore`/`SS` — the storefront passes those in through `host` and reads `window.ShopBot`. Server-side it is served as static assets by `app.py` (and the baseline); it computes its own API base from `location.origin`. The status badge shows the ShopBot name plus the active model from `/api/health` (never keys).
- Modes: Large (default, centred over a dimmed backdrop), Docked (fixed right-hand full-height panel, page still visible), Compact (small panel above the launcher); fullscreen on phones (max-width 640px); Esc closes; focus trap + aria. The conversation sidebar auto-opens only in the wide Large window on desktop — on phones it is an overlay and stays closed so it never covers the messages (the header toggle can still open it).
- Same endpoints as the old panel: `GET /api/site` (optional — the widget falls back to built-in `site.pages` templates when the backend has none, e.g. System A), `GET /api/health`, `POST /api/chat`, `POST /api/transcribe`, `POST /api/compare`. Health's `llm:"quota_exhausted"` (or a quota chat reply) turns the status dot amber with a translated label (EN/AR `status_quota`); offline still wins. Each `/api/chat` `action` becomes a `site.pages` deep link navigated via `host.navigate` (search → the results page, compare → summary + "Open comparison"); product/order/help answers stay text-only. It never renders product cards. Each turn it sends `host.getContext()` (`{page,asin,query,filters,cart_count,order_id}`) as the optional `context` and keeps its session id in `sessionStorage`.
- Dev preview: append `?mock=1` to the URL and the widget runs fully client-side (fake health/site/chat replies, no network, no LLM) with a visible "MOCK DATA" badge. The Playwright e2e journey drives this so it never touches the real provider.


## Whisper notes
Needs ffmpeg on PATH. The base model is weak on Arabic; try small.

## Working rules
- Small, additive changes; explain what you changed and why in plain language
- Keep chatbot logic in backend modules and the Flask app as a thin layer
- Keep column names exactly as listed above
- Ask before adding new dependencies or changing the architecture
- Commit to git after each fix
