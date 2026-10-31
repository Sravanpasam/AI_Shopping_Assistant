"""
app.py
------
ShopBot Flask server: the single LLM agent application.

It serves the frontend/ folder and exposes the chat contract used by the widget
and eval/run_eval.py, plus the storefront endpoints.

Endpoints:
  GET  /                 -- Serve frontend index.html
  GET  /api/health       -- Health check (ShopBot + active model name)
  POST /api/chat         -- Agent chat (session_id, intent, confidence, response_type,
                            text, products, comparison, language, tools_used,
                            escalated, actions)
  POST /api/compare      -- Direct comparison (used by the frontend compare tray)
  POST /api/transcribe   -- Whisper speech-to-text (reuses transcription.py)
  GET  /api/products     -- Storefront list/filter/sort/paginate (shared blueprint)
  GET  /api/products/<asin> -- Single product detail (shared blueprint)

Run it with:  .venv\\Scripts\\python.exe backend\\app.py
"""

import os
import sys
import re
import json
import uuid
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup so backend modules import cleanly
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
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

# Storefront product endpoints (shared blueprint).
app.register_blueprint(storefront_bp)

# ---------------------------------------------------------------------------
# Limits (same as System A)
# ---------------------------------------------------------------------------
MAX_SESSIONS = 200
MAX_MESSAGE_CHARS = 2000
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB max upload for voice input
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES

# Agent behaviour
MAX_TOOL_ROUNDS = 5          # max tool-call rounds per user message
MAX_HISTORY_MESSAGES = 20    # per-session cap (messages, incl. tool results)

# Session-only profile (budget, hair/skin type, recently viewed product ids).
MAX_PROFILE_VIEWED = 8
_PROFILE_STRING_MAX = 80
MAX_PROFILE_CHARS = 600
_BUDGET_HINT_RE = re.compile(
    r"\bbudget(?:\s*(?:is|of|:))?\s*(?:about|around|up\s+to|under)?\s*\$?\s*"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Optional page context sent with a chat turn (validated + capped, see below).
MAX_CONTEXT_CHARS = 1000     # hard cap on the serialized context size
_CONTEXT_STRING_MAX = 200    # cap on each free-text context field
_CONTEXT_FILTER_KEYS_MAX = 10
_CONTEXT_FILTER_KEY_MAX = 40
_CONTEXT_FILTER_VALUE_MAX = 80

# ---------------------------------------------------------------------------
# In-memory session store  {session_id -> session_dict}
# ---------------------------------------------------------------------------
sessions: dict = {}

# Agent + model singletons
_agent = None
_models_ready = False
_limited_mode = False


def get_session(session_id: str) -> dict:
    """Return or create a session dict, evicting the oldest if at capacity."""
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            sessions.pop(oldest, None)
        sessions[session_id] = {"history": [], "profile": {}}
    return sessions[session_id]


def ensure_models() -> None:
    """Load the embedding model and build/load the FAISS index once."""
    global _models_ready
    if _models_ready:
        return
    logger.info("Loading ShopBot models (embedding model + search index)…")
    import nlp_engine            # noqa: F401 — triggers model load
    import product_search
    product_search.build_index()
    import policy_search
    policy_search.get_chunks()   # warm the policy index
    _models_ready = True
    logger.info("ShopBot models ready.")


def get_agent():
    """Return the shared ShopBotAgent (building it on first use)."""
    global _agent
    if _agent is None:
        from agent import ShopBotAgent
        _agent = ShopBotAgent(max_iterations=MAX_TOOL_ROUNDS)
    return _agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _trim_history(messages) -> list:
    """
    Keep only the last MAX_HISTORY_MESSAGES messages, starting at a user turn
    so the sequence (user -> assistant/tool ...) stays valid for the API.
    """
    msgs = list(messages or [])
    if len(msgs) > MAX_HISTORY_MESSAGES:
        msgs = msgs[-MAX_HISTORY_MESSAGES:]
    for index, message in enumerate(msgs):
        if message.get("role") == "user":
            return msgs[index:]
    return []


def _clean_context(context) -> dict:
    """
    Validate and shrink the optional page context, or return {} when unusable.

    Accepts {"page", "asin", "query", "filters", "cart_count", "order_id"},
    keeps only the allowed keys, coerces their types and caps every field. An
    oversized or malformed context is ignored (the turn still works).
    """
    if not isinstance(context, dict) or not context:
        return {}

    cleaned: dict = {}
    for key in ("page", "asin", "query", "order_id"):
        value = context.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned[key] = text[:_CONTEXT_STRING_MAX]

    cart_count = context.get("cart_count")
    if cart_count is not None:
        try:
            count = int(cart_count)
            if count >= 0:
                cleaned["cart_count"] = count
        except (TypeError, ValueError):
            pass

    filters = context.get("filters")
    if isinstance(filters, dict) and filters:
        slim: dict = {}
        for raw_key, value in list(filters.items())[:_CONTEXT_FILTER_KEYS_MAX]:
            if value is None:
                continue
            key = str(raw_key).strip()[:_CONTEXT_FILTER_KEY_MAX]
            if not key:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                slim[key] = value
            else:
                text = str(value).strip()[:_CONTEXT_FILTER_VALUE_MAX]
                if text:
                    slim[key] = text
        if slim:
            cleaned["filters"] = slim

    if not cleaned:
        return {}

    # Hard cap on the serialized size: ignore anything still too big.
    if len(json.dumps(cleaned, ensure_ascii=False)) > MAX_CONTEXT_CHARS:
        logger.warning("Ignoring oversized chat context (%d keys).", len(cleaned))
        return {}
    return cleaned


def _clean_profile(profile) -> dict:
    """Validate and cap an optional client-supplied session profile."""
    if not isinstance(profile, dict) or not profile:
        return {}
    cleaned: dict = {}
    budget = profile.get("budget")
    if budget is not None:
        try:
            value = float(budget)
            if 0 < value <= 1_000_000:
                cleaned["budget"] = round(value, 2)
        except (TypeError, ValueError):
            pass
    for key in ("hair_type", "skin_type"):
        value = profile.get(key)
        if value is None:
            continue
        text = str(value).strip()[:_PROFILE_STRING_MAX]
        if text:
            cleaned[key] = text
    viewed = profile.get("viewed")
    if isinstance(viewed, list):
        asins = []
        for value in viewed:
            text = str(value).strip()[:_PROFILE_STRING_MAX]
            if text and text not in asins:
                asins.append(text)
        if asins:
            cleaned["viewed"] = asins[-MAX_PROFILE_VIEWED:]
    if not cleaned:
        return {}
    if len(json.dumps(cleaned, ensure_ascii=False)) > MAX_PROFILE_CHARS:
        cleaned.pop("viewed", None)
    return cleaned


_HAIR_TYPES = ("dry", "oily", "curly", "wavy", "straight", "fine", "damaged", "color-treated")
_SKIN_TYPES = ("dry", "oily", "sensitive", "combination", "normal", "acne-prone")
_HAIR_WORDS = ("hair", "scalp", "curl", "dandruff", "شعر", "فروة")
_SKIN_WORDS = ("skin", "face", "acne", "بشرة", "وجه")


def _update_profile(session: dict, message: str, context=None, incoming=None,
                    products=None) -> dict:
    """Merge explicit profile data, message hints and viewed products (capped)."""
    profile = _clean_profile(session.get("profile"))
    profile.update(_clean_profile(incoming))

    text = (message or "").lower()
    try:
        from price_parser import parse_price_constraint
        _, _, max_usd = parse_price_constraint(message or "")
        if max_usd is not None:
            profile["budget"] = max_usd
    except Exception:  # noqa: BLE001 - profile hints are optional
        pass
    hint = _BUDGET_HINT_RE.search(message or "")
    if hint:
        try:
            value = float(hint.group(1))
            if 0 < value <= 1_000_000:
                profile["budget"] = round(value, 2)
        except ValueError:
            pass

    if any(word in text for word in _HAIR_WORDS):
        for hair_type in _HAIR_TYPES:
            if hair_type in text:
                profile["hair_type"] = hair_type
                break
    if any(word in text for word in _SKIN_WORDS):
        for skin_type in _SKIN_TYPES:
            if skin_type in text:
                profile["skin_type"] = skin_type
                break

    viewed = list(profile.get("viewed") or [])
    if isinstance(context, dict) and context.get("asin"):
        viewed.append(str(context["asin"]))
    for product in products or []:
        asin = str(product.get("parent_asin") or "").strip()
        if asin:
            viewed.append(asin)
    deduped = []
    for asin in viewed:
        if asin not in deduped:
            deduped.append(asin)
    if deduped:
        profile["viewed"] = deduped[-MAX_PROFILE_VIEWED:]

    session["profile"] = _clean_profile(profile)
    return session["profile"]


def _append_history(session: dict, user_message: str, assistant_text: str) -> None:
    """Append a simple user/assistant pair to the session history (capped)."""
    session["history"] = _trim_history(
        list(session.get("history", []))
        + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ]
    )


def _localize_table(table: dict) -> dict:
    """Translate the fixed row/column labels of a comparison table to Arabic."""
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


def _chat_payload(session_id, response_type, text, products, comparison, language,
                  tools_used, escalated, actions=None, confidence=1.0, error=None,
                  suggestions=None, degraded=False):
    """Build a JSON-safe payload with the full chat contract fields."""
    payload = {
        "session_id": session_id,
        "intent": "agent",
        "confidence": confidence,
        "response_type": response_type,
        "text": text,
        "products": products or [],
        "comparison": comparison,
        "language": language,
        "tools_used": [
            {"name": t.get("name"), "args": t.get("args", {}), "ok": bool(t.get("ok", True))}
            for t in (tools_used or [])
        ],
        "escalated": bool(escalated),
        "actions": list(actions or []),
        "suggestions": [
            str(item).strip()[:120]
            for item in (suggestions or [])[:3]
            if str(item).strip()
        ],
        "degraded": bool(degraded),
    }
    if error:
        payload["error"] = error
    return payload


# ---------------------------------------------------------------------------
# Routes — static files
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    resp = send_from_directory(FRONTEND_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/<path:filename>")
def static_files(filename):
    resp = send_from_directory(FRONTEND_DIR, filename)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    try:
        from llm_client import get_active_model_name, quota_state, get_client
        model = get_active_model_name()
        llm = quota_state()
        models_state = []
        try:
            client = get_client()
            ps = client.pool_state()
            for m in ps.get("models") or []:
                models_state.append({
                    "name": m.get("name"),
                    "used": m.get("used"),
                    "budget": ps.get("daily_budget"),
                    "retired": bool(m.get("retired")),
                    "available": (not bool(m.get("retired"))) and (m.get("used", 0) < ps.get("daily_budget", 0)),
                })
        except Exception:
            models_state = []
    except Exception:  # never let a health check fail
        model = ""
        llm = "ok"
        models_state = []
    if _limited_mode:
        llm = "limited"
    return jsonify({
        "status": "ok",
        "name": "ShopBot",
        "message": f"ShopBot is running ({model})" if model else "ShopBot is running",
        "model": model,  # active provider model name only — never the API key
        "llm": llm,      # "ok" or "quota_exhausted"
        "models": models_state,
    })


@app.route("/api/site")
def site():
    """
    Non-sensitive site config for the widget: name, assistant name, currency,
    languages and the hash-route 'pages' map. Never exposes filesystem paths.
    """
    try:
        from site_config import public_site_info
        return jsonify(public_site_info())
    except Exception as exc:  # never let a config problem crash the widget
        logger.exception("Site endpoint error: %s", exc)
        return jsonify({"error": "site_config_unavailable"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Agent chat endpoint.

    Request JSON:  { "message": str, "session_id": str (optional),
                     "context": {...} (optional, see _clean_context),
                     "profile": {...} (optional, see _clean_profile) }
    Response JSON: session_id, intent ("agent"), confidence, response_type,
                   text, products, comparison, language, tools_used, escalated,
                   actions (list of navigate actions; empty when none),
                   suggestions (up to 3 follow-up prompts; empty when none)
    """
    global _limited_mode
    try:
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        session_id = data.get("session_id") or str(uuid.uuid4())
        context = _clean_context(data.get("context"))
        incoming_profile = _clean_profile(data.get("profile"))

        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        from translation import detect_language
        detected_lang = detect_language(message)
        lang = "ar" if detected_lang == "ar" else "en"

        if len(message) > MAX_MESSAGE_CHARS:
            return jsonify({
                "error": get_string("error_message_too_long", lang, max=MAX_MESSAGE_CHARS)
            }), 400

        session = get_session(session_id)
        _update_profile(session, message, context, incoming=incoming_profile)
        logger.info("[%s] msg='%s' lang=%s", session_id, message[:80], detected_lang)

        # ── 1. Cheap short-circuits (no LLM call) ──────────────────────────
        # Greetings first: a bare "مرحبا"/"hello" is never a human-handoff, and
        # the escalation matcher is fuzzy enough to catch Arabic greetings.
        from greetings import is_greeting
        if is_greeting(message):
            logger.info("[%s] greeting short-circuit (no LLM call)", session_id)
            text = get_string("greeting", lang)
            _append_history(session, message, text)
            return jsonify(_chat_payload(
                session_id, "text", text, [], None, detected_lang, [], False,
                confidence=1.0,
            ))

        from escalation import detect_escalation
        should_escalate, reason = detect_escalation(message)
        if should_escalate:
            logger.info("[%s] escalation pre-check fired: %s", session_id, reason)
            text = get_string("escalation_handoff", lang)
            _append_history(session, message, text)
            return jsonify(_chat_payload(
                session_id, "escalation", text, [], None, detected_lang,
                [{"name": "escalate_to_human", "args": {"reason": reason}, "ok": True}],
                True,
            ))

        # ── 2. Optional response cache (so eval re-runs cost no quota) ─────
        import llm_cache
        cache_key = None
        if llm_cache.enabled():
            cache_key = llm_cache.make_key(
                message,
                context,
                {"history": session.get("history", []),
                 "profile": session.get("profile", {})},
            )
            cached = llm_cache.get(cache_key)
            if cached is not None:
                logger.info("[%s] llm cache hit", session_id)
                _append_history(session, message, cached.get("text", ""))
                payload = dict(cached)
                payload["session_id"] = session_id
                return jsonify(payload)

        # ── 3. Run the agent ───────────────────────────────────────────────
        ensure_models()
        try:
            result = get_agent().run(
                message,
                history=session.get("history", []),
                context=context,
                profile=session.get("profile", {}),
            )
        except Exception as exc:  # never let the endpoint crash
            logger.exception("[%s] agent raised: %s", session_id, exc)
            text = get_string("llm_error", lang)
            _append_history(session, message, text)
            return jsonify(_chat_payload(
                session_id, "error", text, [], None, detected_lang, [], False,
                confidence=0.0, error="agent_error",
            ))

        tools_used = result.tool_calls

        # ── 4. Provider failure after retries -> friendly error ────────────
        if result.error and result.error != "max_iterations":
            logger.warning("[%s] agent provider failure: %s", session_id, result.error)
            _limited_mode = True
            try:
                import fallback
                fb = fallback.fallback_turn(message, lang, session, context)
                products = fb.products or []
                comparison = None
                if fb.table:
                    table = _localize_table(fb.table) if lang == "ar" else fb.table
                    comparison = {"table": table, "summary": fb.reply}
                if fb.escalated:
                    response_type = "escalation"
                elif comparison is not None:
                    response_type = "comparison"
                elif products:
                    response_type = "product_list"
                else:
                    response_type = "text"
                history = list(session.get("history") or []) + list(fb.messages)
                session["history"] = _trim_history(history)
                _update_profile(session, message, products=products)
                logger.info(
                    "[%s] limited mode intent=%s tools=%s",
                    session_id, fb.intent, [t["name"] for t in fb.tools_used],
                )
                return jsonify(_chat_payload(
                    session_id, response_type, fb.reply, products, comparison,
                    detected_lang, fb.tools_used, fb.escalated,
                    actions=fb.actions, suggestions=fb.suggestions,
                    confidence=0.0, degraded=True,
                ))
            except Exception as exc:  # fallback must never expose a raw error
                logger.exception("[%s] fallback failed: %s", session_id, exc)
                text = get_string("llm_error", lang)
                _append_history(session, message, text)
                return jsonify(_chat_payload(
                    session_id, "text", text, [], None, detected_lang, [],
                    False, confidence=0.0, degraded=True,
                ))

        _limited_mode = False

        # ── 5. Success (or tool-round cap: reply is a safe fallback) ───────
        products = result.products or []
        comparison = None
        if result.table:
            table = _localize_table(result.table) if lang == "ar" else result.table
            comparison = {"table": table, "summary": result.reply}

        if result.escalated:
            response_type = "escalation"
        elif comparison is not None:
            response_type = "comparison"
        elif products:
            response_type = "product_list"
        else:
            response_type = "text"

        # Keep the full turn (incl. tool results) for follow-ups like "the second one".
        session["history"] = _trim_history(result.messages)
        _update_profile(session, message, products=products)

        logger.info(
            "[%s] type=%s tools=%s escalated=%s error=%s",
            session_id, response_type,
            [t["name"] for t in tools_used], result.escalated, result.error,
        )
        payload = _chat_payload(
            session_id, response_type, result.reply, products, comparison,
            detected_lang, tools_used, result.escalated,
            actions=result.actions,
            suggestions=result.suggestions,
            confidence=0.0 if result.error else 1.0,
        )
        # Cache successful turns only (never a quota/provider error).
        if cache_key is not None and not result.error:
            stored = dict(payload)
            stored.pop("session_id", None)
            llm_cache.put(cache_key, stored)
        return jsonify(payload)

    except Exception as exc:
        logger.exception("Chat endpoint error")
        return jsonify({
            "error": "internal_error",
            "response_type": "error",
            "text": get_string("error_server", "en"),
        }), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Direct comparison endpoint used by the frontend compare tray.

    Request JSON: { "asins": ["B001", "B002"], "language": "en" }
    Response JSON: { "table": {...}, "summary": "..." }
    """
    try:
        from comparison import compare_products
        from translation import translate_to_arabic, TranslationError

        data = request.get_json(force=True)
        asins = data.get("asins", [])
        language = data.get("language", "en")

        if len(asins) < 2:
            return jsonify({"error": get_string("error_compare_asins", language)}), 400

        asins = asins[:3]
        ensure_models()
        table, summary = compare_products(asins)
        if table is None:
            return jsonify({"error": summary}), 404

        if language == "ar":
            table = _localize_table(table)
            try:
                summary = translate_to_arabic(summary)
            except TranslationError:
                logger.warning("Compare summary translation failed; returning English.")
        return jsonify({"table": table, "summary": summary})

    except Exception as exc:
        logger.exception("Compare endpoint error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Whisper audio transcription endpoint (same contract as System A).

    Request: multipart/form-data with 'audio' file field.
    Response JSON: { "text": "...", "language": "en", "detected_language": "en" }
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No 'audio' file in request."}), 400

        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return jsonify({
                "error": f"Audio file is too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)."
            }), 413

        model_size = request.form.get("model_size", "base")

        from transcription import transcribe_audio
        return jsonify(transcribe_audio(audio_bytes, model_size=model_size))

    except Exception as exc:
        logger.exception("Transcribe endpoint error")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShopBot — LLM agent")
    parser.add_argument(
        "--site",
        default=os.environ.get("SITE", "shopstore"),
        help="site id under sites/ (default: shopstore, or the SITE env var)",
    )
    args, _ = parser.parse_known_args()
    os.environ["SITE"] = args.site  # select the site for this process

    host = "127.0.0.1"
    port = int(os.environ.get("PORT", "5000"))

    logger.info("=" * 60)
    logger.info("  ShopBot — LLM agent")
    logger.info("  Site: %s", args.site)
    logger.info("  Starting Flask server on http://%s:%d (debug=False)", host, port)
    logger.info("=" * 60)

    try:
        ensure_models()
    except Exception as exc:
        logger.exception("Failed to pre-load models at startup: %s", exc)

    app.run(debug=False, port=port, host=host, use_reloader=False)
