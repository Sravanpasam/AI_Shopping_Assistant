"""
app_baseline.py
---------------
Legacy System A Flask server, kept for baseline comparison only.

The active server is backend/app.py (the ShopBot LLM agent). This file is not
started by the frontend; it is preserved so the System A evaluation baseline
can still be run.

Endpoints:
  GET  /                  -- Serve frontend index.html
  GET  /api/health        -- Health check
  POST /api/chat          -- Main conversational endpoint
  POST /api/compare       -- Direct product comparison
  POST /api/transcribe    -- Whisper speech-to-text
  GET  /api/products      -- Storefront list/filter/sort/paginate (shared blueprint)
  GET  /api/products/<asin> -- Single product detail (shared blueprint)
"""

import os
import re
import sys
import uuid
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup so backend modules import cleanly
# ---------------------------------------------------------------------------
LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LEGACY_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from strings_ar import get_string, TABLE_LABELS  # noqa: E402
from storefront import storefront_bp  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Storefront product endpoints (shared with app_b.py) — additive, chat untouched.
app.register_blueprint(storefront_bp)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_SESSIONS = 200          # oldest sessions are evicted beyond this
MAX_MESSAGE_CHARS = 2000    # reject over-long chat messages
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB max upload for voice input
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES

# ---------------------------------------------------------------------------
# In-memory session store  {session_id -> session_dict}
# ---------------------------------------------------------------------------
sessions: dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    """Return or create a session dict, evicting the oldest if at capacity."""
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            sessions.pop(oldest, None)
        sessions[session_id] = {
            "history": [],          # list of {role, content}
            "last_products": [],    # last search results
            "last_intent": None,
            "focused_product": None,  # product a follow-up ("its", "the second one") refers to
        }
    return sessions[session_id]

# ---------------------------------------------------------------------------
# Lazy imports — heavy NLP modules load only when first needed
# ---------------------------------------------------------------------------
_nlp_loaded = False

def ensure_nlp():
    """Load NLP modules once and build the FAISS index."""
    global _nlp_loaded
    if _nlp_loaded:
        return
    logger.info("Loading NLP modules (first request)…")
    import nlp_engine          # noqa — triggers model load
    import product_search
    product_search.build_index()
    _nlp_loaded = True
    logger.info("NLP modules ready.")

# ---------------------------------------------------------------------------
# Helper: human-readable price filter phrase
# ---------------------------------------------------------------------------
def _price_filter_phrase(result: dict) -> str:
    """Human-readable price filter (e.g. 'under $100'), or '' if none applied."""
    lo = result.get("min_usd")
    hi = result.get("max_usd")

    def fmt(value) -> str:
        value = float(value)
        return f"${value:,.0f}" if value.is_integer() else f"${value:,.2f}"

    if lo is not None and hi is not None:
        return f"between {fmt(lo)} and {fmt(hi)}"
    if hi is not None:
        return f"under {fmt(hi)}"
    if lo is not None:
        return f"over {fmt(lo)}"
    return ""


def _remember_products(session: dict, products) -> None:
    """Store the products just shown and focus the top one for follow-ups."""
    if products:
        session["last_products"] = products
        session["focused_product"] = products[0]


def _localize_table(table: dict) -> dict:
    """Translate the static row/column labels of a comparison table to Arabic.

    Product titles and cell values stay as-is; only the fixed labels
    ("Feature", "Price", "Brand / Store", "Rating", "Category", "Key Features")
    are replaced from the pre-translated dictionary.
    """
    if not table:
        return table

    headers = list(table.get("headers", []))
    if headers:
        headers[0] = TABLE_LABELS.get(headers[0], headers[0])

    rows = []
    for row in table.get("rows", []):
        row = list(row)
        if row:
            row[0] = TABLE_LABELS.get(row[0], row[0])
        rows.append(row)

    return {"headers": headers, "rows": rows}


# ---------------------------------------------------------------------------
# Routes — static files
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    from translation import translation_status
    return jsonify({
        "status": "ok",
        "message": "ShopBot API is running 🛒",
        "translation": translation_status(),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main conversational endpoint.

    Request JSON:
      { "message": str, "session_id": str (optional) }

    Response JSON:
      { session_id, intent, response_type, text, products, comparison, language }
    """
    try:
        ensure_nlp()

        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        session_id = data.get("session_id") or str(uuid.uuid4())

        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        if len(message) > MAX_MESSAGE_CHARS:
            from translation import detect_language as _detect
            err_lang = "ar" if _detect(message) == "ar" else "en"
            return jsonify({
                "error": get_string("error_message_too_long", err_lang, max=MAX_MESSAGE_CHARS)
            }), 400

        session = get_session(session_id)

        # ── 1. Language detection ──────────────────────────────────────────
        from translation import (
            detect_language, translate_to_english, translate_to_arabic, TranslationError,
        )
        detected_lang = detect_language(message)
        is_arabic = detected_lang == "ar"
        lang = "ar" if is_arabic else "en"

        # ── 2. Translate to English for all NLP processing ─────────────────
        if is_arabic:
            try:
                en_message = translate_to_english(message)
            except TranslationError:
                logger.warning("[%s] Arabic->English translation failed; returning notice.", session_id)
                return jsonify({
                    "session_id": session_id,
                    "intent": "translation_error",
                    "confidence": 0.0,
                    "response_type": "text",
                    "text": get_string("translation_unavailable", "ar"),
                    "products": [],
                    "comparison": None,
                    "language": detected_lang,
                }), 200
        else:
            en_message = message
        logger.info("[%s] msg='%s' lang=%s", session_id, en_message[:80], detected_lang)

        # ── 3. Route: price parse + intent classification ──────────────────
        from nlp_engine import route_message
        route = route_message(en_message)
        intent = route["intent"]
        confidence = route["confidence"]
        logger.info(
            "[%s] intent=%s conf=%.2f price_filter=%s (classified=%s)",
            session_id, intent, confidence, route["price_filter"], route["classified_intent"],
        )

        # ── 4. Intent handling ─────────────────────────────────────────────
        response_type = "text"
        text = ""
        products = []
        comparison_data = None
        needs_translation = False  # True only for dynamic (not pre-translated) text

        # -- Greeting --------------------------------------------------------
        if intent == "greeting":
            text = get_string("greeting", lang)
            response_type = "text"

        # -- Product search (clear or vague) ---------------------------------
        elif intent in ("search_product", "vague_search"):
            from product_search import search_products_filtered
            result = search_products_filtered(en_message, top_k=5)
            filter_note = _price_filter_phrase(result)

            if result["products"]:
                products = result["products"]
                if filter_note:
                    text = get_string("search_filtered", lang, filter=filter_note)
                else:
                    count = len(products)
                    text = get_string(
                        "search_found", lang, count=count, plural="" if count == 1 else "s"
                    )
                response_type = "product_list"
                _remember_products(session, products)
            elif result["fallback_products"]:
                products = result["fallback_products"]
                text = get_string(
                    "search_fallback", lang, filter=filter_note, count=len(products)
                )
                response_type = "product_list"
                _remember_products(session, products)
            else:
                text = get_string("no_match", lang)
                response_type = "text"

        # -- Compare products ------------------------------------------------
        elif intent == "compare_products":
            from comparison import compare_products, resolve_comparison_products

            # Resolve what to compare: session products for "compare these",
            # otherwise search each named product (split on and / vs / versus).
            source_products = resolve_comparison_products(
                en_message, session.get("last_products", [])
            )

            if len(source_products) >= 2:
                asins = [p["parent_asin"] for p in source_products[:3]]
                table, summary = compare_products(asins)
                if table:
                    if is_arabic:
                        table = _localize_table(table)
                    comparison_data = {"table": table, "summary": summary}
                    text = get_string("compare_intro", lang)
                    response_type = "comparison"
                    # Update session with the compared products
                    _remember_products(session, source_products[:3])
                else:
                    text = summary  # compare_products returns error msg in summary
                    response_type = "text"
                    needs_translation = is_arabic
            else:
                text = get_string("compare_ask", lang)
                response_type = "text"

        # -- Policy: Warranty ------------------------------------------------
        elif intent == "policy_warranty":
            text, needs_translation = _handle_policy(en_message, session, "warranty", lang)
            response_type = "text"

        # -- Policy: Cash on Delivery ----------------------------------------
        elif intent == "policy_cod":
            text, needs_translation = _handle_policy(en_message, session, "cod", lang)
            response_type = "text"

        # -- Policy: Return / Exchange ----------------------------------------
        elif intent == "policy_return":
            text, needs_translation = _handle_policy(en_message, session, "return", lang)
            response_type = "text"

        # -- Low confidence: ask a clarifying question -----------------------
        elif intent == "clarify":
            text = get_string("clarify", lang)
            response_type = "text"

        # -- Safety net: only reachable if a new intent is added unhandled ---
        else:
            text = get_string("no_match", lang)
            response_type = "text"

        # ── 5. Translate dynamic response text to Arabic if needed ─────────
        # Static replies were already localized from strings_ar; only dynamic
        # text (policy answers, generated summaries) goes through the translator.
        if is_arabic and text and needs_translation:
            try:
                text = translate_to_arabic(text)
            except TranslationError:
                logger.warning("[%s] English->Arabic translation failed; showing English.", session_id)
                text = text + get_string("reply_not_translated", "ar")

        # ── 6. Update session history ──────────────────────────────────────
        session["last_intent"] = intent
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": text, "type": response_type})
        # Keep history bounded
        if len(session["history"]) > 40:
            session["history"] = session["history"][-40:]

        return jsonify({
            "session_id": session_id,
            "intent": intent,
            "confidence": round(confidence, 3),
            "response_type": response_type,
            "text": text,
            "products": products,
            "comparison": comparison_data,
            "language": detected_lang,
        })

    except Exception as exc:
        logger.exception("Chat endpoint error")
        return jsonify({
            "error": str(exc),
            "response_type": "error",
            "text": get_string("error_server", "en"),
        }), 500


# Ordinal words -> index into the session's last products.
_ORDINAL_INDEX = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "last": -1,
}

# Words that carry no product identity in a policy question.
_POLICY_STOPWORDS = {
    "warranty", "warranties", "guarantee", "guarantees", "guaranteed",
    "cover", "covered", "coverage",
    "return", "returns", "returned", "returning", "exchange", "exchanges",
    "refund", "refunds", "refunded", "replacement", "replacements",
    "money", "back", "cod", "cash", "delivery", "deliver", "delivered",
    "pay", "payment", "paying",
    "policy", "policies", "info", "information", "details", "detail",
    "term", "terms", "shipping", "ship", "returnable", "eligible",
    "available", "apply", "applies", "applied",
    "this", "that", "these", "those", "it", "them", "the", "a", "an",
    "of", "for", "to", "on", "my", "your",
    "is", "are", "am", "was", "were", "be", "been",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "you", "i", "me", "we", "us", "he", "she", "they",
    "have", "has", "had",
    "what", "whats", "what's", "how", "hows", "how's", "when", "where",
    "which", "who", "why", "about", "with", "please", "tell",
    "first", "second", "third", "fourth", "1st", "2nd", "3rd", "4th",
    "last", "one", "two", "three",
}


def _resolve_policy_product(en_message: str, session: dict):
    """
    Pick the product a policy question refers to:
      1. an ordinal ("the second one") -> that item from the last products;
      2. a named product ("warranty on the Sony headphones") -> fresh search;
      3. otherwise the focused product, then the first previously shown item.

    Returns a product dict or None.
    """
    last_products = session.get("last_products", [])
    tokens = re.findall(r"[a-z0-9']+", (en_message or "").lower())

    # 1. Ordinal reference into the previously shown products.
    if last_products:
        for token in tokens:
            if token in _ORDINAL_INDEX:
                idx = _ORDINAL_INDEX[token]
                if -len(last_products) <= idx < len(last_products):
                    return last_products[idx]
                break  # ordinal out of range -> fall through

    # 2. Named product: any content words left after removing policy wording.
    content = [t for t in tokens if t not in _POLICY_STOPWORDS]
    if content:
        from product_search import search_products
        hits = search_products(" ".join(content), top_k=1)
        if hits:
            return hits[0]

    # 3. Focused product, then the first previously shown product.
    focused = session.get("focused_product")
    if focused:
        return focused
    return last_products[0] if last_products else None


def _handle_policy(en_message: str, session: dict, policy_type: str, lang: str = "en"):
    """
    Look up a policy (warranty / cod / return) for the resolved product.

    Returns ``(text, needs_translation)``. ``needs_translation`` is True only
    for the dynamic answer (which contains the product title and the policy
    sentence); the static "search first" prompt is already pre-translated.
    """
    from policy_qa import answer_policy

    product = _resolve_policy_product(en_message, session)
    label = {"warranty": "Warranty", "cod": "Cash on Delivery", "return": "Return & Exchange"}[policy_type]

    if product:
        session["focused_product"] = product
        asin = product["parent_asin"]
        product_name = product.get("title", "this product")
        answer = answer_policy(asin, policy_type)
        return f"**{label} info for {product_name}:**\n\n{answer}", True

    prompt_key = {
        "warranty": "policy_need_warranty",
        "cod": "policy_need_cod",
        "return": "policy_need_return",
    }.get(policy_type, "policy_need_warranty")
    return get_string(prompt_key, lang), False


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Direct product comparison endpoint (called from frontend compare bar).

    Request JSON: { "asins": ["B001", "B002"], "language": "en" }
    Response JSON: { "table": {...}, "summary": "..." }
    """
    try:
        ensure_nlp()
        from comparison import compare_products
        from translation import translate_to_arabic, TranslationError

        data = request.get_json(force=True)
        asins = data.get("asins", [])
        language = data.get("language", "en")

        if len(asins) < 2:
            return jsonify({"error": get_string("error_compare_asins", language)}), 400

        asins = asins[:3]  # max 3
        table, summary = compare_products(asins)

        if table is None:
            return jsonify({"error": summary}), 404

        if language == "ar":
            table = _localize_table(table)
            try:
                summary = translate_to_arabic(summary)
            except TranslationError:
                logger.warning("Compare summary translation failed; returning English summary.")

        return jsonify({"table": table, "summary": summary})

    except Exception as exc:
        logger.exception("Compare endpoint error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Whisper audio transcription endpoint.

    Request: multipart/form-data with 'audio' file field.
    Optional form field: 'model_size' (tiny | base | small), default 'base'.
    Response JSON: { "text": "...", "language": "en", "detected_language": "en" }
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No 'audio' file in request."}), 400

        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return jsonify({"error": f"Audio file is too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)."}), 413

        model_size = request.form.get("model_size", "base")

        from transcription import transcribe_audio
        result = transcribe_audio(audio_bytes, model_size=model_size)
        return jsonify(result)

    except Exception as exc:
        logger.exception("Transcribe endpoint error")
        return jsonify({"error": str(exc)}), 500


# NOTE: GET /api/products now lives in the shared storefront blueprint
# (backend/storefront.py), registered above. It keeps the legacy "products",
# "per_page" and "pages" fields for backward compatibility and adds filtering,
# sorting and pagination facets for the storefront page.


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Safe-by-default config, overridable via environment variables.
    # Debug and public binding are opt-in, never the default.
    debug = os.environ.get("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes", "on")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))

    logger.info("=" * 60)
    logger.info("  ShopBot — E-Commerce NLP Chatbot")
    logger.info("  Starting Flask server on http://%s:%d (debug=%s)", host, port, debug)
    if debug:
        logger.warning("Debug mode is ON — do not expose this server to untrusted networks.")

    # Pre-load NLP models and build/load the search index at startup so the
    # first user request isn't delayed (and failures surface in the logs).
    logger.info("Pre-loading NLP models and building the search index...")
    try:
        ensure_nlp()
    except Exception as exc:
        logger.exception("Failed to pre-load NLP models at startup: %s", exc)

    # One live translation self-test at startup so /api/health has a status
    # immediately (and re-checks are throttled to once per TTL afterwards).
    try:
        from translation import translation_status
        logger.info("Translation self-test: %s", translation_status(force=True))
    except Exception as exc:
        logger.exception("Translation self-test failed at startup: %s", exc)
    logger.info("=" * 60)
    app.run(debug=debug, port=port, host=host, use_reloader=False)
