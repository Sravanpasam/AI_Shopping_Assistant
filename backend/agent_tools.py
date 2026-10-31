"""
agent_tools.py
--------------
Plain Python tools for the System B LLM agent. Each function reads store data
through the store adapter (see adapters/), returns a JSON-serializable dict, and
never raises (errors come back as {"error": ...}).

Tool arguments are always expected in English; the model translates the user's
question itself. Every description below tells the model when to use the tool.

Tools:
    search_products(query, min_price_usd, max_price_usd, limit)
    get_product_details(parent_asin)
    compare_products(parent_asins)
    search_reviews(parent_asin, query, limit)
    search_policies(query)
    check_order(order_id)
    build_routine(goal, budget)
    gift_finder(recipient, occasion, budget)
    best_in_category(category, max_price)
    escalate_to_human(reason)

TOOL_SCHEMAS holds the JSON-schema declarations passed to the LLM client, and
call_tool(name, arguments) dispatches a model tool call to the right function.

This module is additive (System B). It does not touch System A or app.py.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from config import SEARCH_APPROX_FLOOR
from comparison import compare_products as _compare_products
from orders import extract_order_id
from nlp_engine import get_model
from adapters import get_adapter
from product_search import (
    category_suggestions,
    select_relevant,
)
from query_normalizer import normalize_query

logger = logging.getLogger(__name__)

# How many FAISS candidates to pull before thresholding/price-filtering.
_SEARCH_CANDIDATES = 50
_MAX_LIMIT = 20


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slim_product(product: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the JSON-safe product fields a reply may quote."""
    def _text(value: Any) -> str:
        if value is None:
            return ""
        try:
            if value != value:  # NaN
                return ""
        except Exception:  # noqa: BLE001 - exotic values fall through to str
            pass
        return str(value)

    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return {
        "parent_asin": _text(product.get("parent_asin")),
        "title": _text(product.get("title")),
        "store": _text(product.get("store")),
        "price": _number(product.get("price")),
        "average_rating": _number(product.get("average_rating")),
        "rating_number": int(_number(product.get("rating_number"))),
        "similarity_score": (
            _number(product.get("similarity_score"))
            if product.get("similarity_score") is not None else None
        ),
    }


# ── Tools ───────────────────────────────────────────────────────────────────
def search_products(
    query: str,
    min_price_usd: Any = None,
    max_price_usd: Any = None,
    limit: Any = 5,
) -> Dict[str, Any]:
    """
    Find products matching a natural-language description, optionally filtered
    by price (USD). Uses hybrid search and the floor-plus-margin acceptance
    rule; weak results are returned only when flagged ``approximate``.

    Args:
        query: What the user is looking for, in English (e.g. "waterproof mascara").
        min_price_usd: Optional minimum price in US dollars.
        max_price_usd: Optional maximum price in US dollars.
        limit: Maximum number of products (default 5).

    Returns:
        {"query", "min_price_usd", "max_price_usd", "count", "total_matches",
        "products", "approximate", "category_suggestions"} plus a "note" when
        nothing matched. ``total_matches`` is the number of accepted products
        passing the price filter (before ``limit``), so the reply can state the
        real count. Never raises.
    """
    try:
        query = normalize_query((query or "").strip())
        limit = max(1, min(_as_int(limit, 5), _MAX_LIMIT))
        low = _as_float(min_price_usd)
        high = _as_float(max_price_usd)

        result: Dict[str, Any] = {
            "query": query,
            "min_price_usd": low,
            "max_price_usd": high,
            "count": 0,
            "total_matches": 0,
            "products": [],
            "approximate": False,
            "category_suggestions": [],
        }

        if not query:
            result["note"] = "No search query was provided."
            return result

        candidates = get_adapter().search_products(query, top_k=_SEARCH_CANDIDATES)
        relevant, top_score = select_relevant(candidates)

        def priced(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for product in products:
                price = product.get("price", 0.0)
                if low is not None or high is not None:
                    if not price or price <= 0:
                        continue
                    if low is not None and price < low:
                        continue
                    if high is not None and price > high:
                        continue
                out.append(product)
            return out

        if relevant:
            matches = priced(relevant)
        elif candidates and top_score >= SEARCH_APPROX_FLOOR:
            weak = [
                p for p in candidates
                if float(p.get("similarity_score", 0.0)) >= SEARCH_APPROX_FLOOR
            ]
            matches = [dict(p, approximate=True) for p in priced(weak)]
            result["approximate"] = bool(matches)
            if matches:
                result["note"] = "These are close matches, not exact matches, for that description."
            else:
                result["category_suggestions"] = category_suggestions(query)
                result["note"] = "No product matched that description."
        else:
            matches = []
            result["category_suggestions"] = category_suggestions(query)
            result["note"] = "No product matched that description."

        products = matches[:limit]
        result["count"] = len(products)
        result["total_matches"] = len(matches)
        result["products"] = products
        if not products and "note" not in result:
            result["note"] = (
                "No product matched that description"
                + ("" if low is None and high is None else " within the requested price range")
                + "."
            )
        return result
    except Exception as exc:  # noqa: BLE001 - tools must never raise
        logger.exception("search_products failed: %s", exc)
        return {"error": f"search_products failed: {exc}"}


def get_product_details(parent_asin: str) -> Dict[str, Any]:
    """
    Get the full catalog record for one product by its id (parent_asin), usually
    after search_products. Use it to read features and description before
    answering detail questions or comparing products.

    Returns:
        {"parent_asin", "found", "product"} or {"parent_asin", "found": false,
        "note"}. Never raises.
    """
    try:
        asin = (parent_asin or "").strip()
        product = get_adapter().get_product(asin) if asin else None
        if not product:
            return {
                "parent_asin": asin,
                "found": False,
                "note": "No product with that id exists in the catalog.",
            }
        return {"parent_asin": asin, "found": True, "product": product}
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_product_details failed: %s", exc)
        return {"error": f"get_product_details failed: {exc}"}


def compare_products(parent_asins: List[str]) -> Dict[str, Any]:
    """
    Build a side-by-side comparison table plus a short summary for 2 or 3
    products (by their parent_asin ids). Use it only when the user names or
    refers to specific products; never to compare more than three.

    Returns:
        {"compared": true, "parent_asins", "table", "summary"} or
        {"compared": false, "parent_asins", "note"}. Never raises.
    """
    try:
        asins = [str(a).strip() for a in (parent_asins or []) if str(a).strip()]
        # Preserve order, drop duplicates, cap at 3.
        unique: List[str] = []
        for asin in asins:
            if asin not in unique:
                unique.append(asin)
        unique = unique[:3]

        if len(unique) < 2:
            return {
                "compared": False,
                "parent_asins": unique,
                "note": "At least 2 product ids are required to compare.",
            }

        table, summary = _compare_products(unique)
        if table is None:
            return {"compared": False, "parent_asins": unique, "note": summary}
        return {
            "compared": True,
            "parent_asins": unique,
            "table": table,
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("compare_products failed: %s", exc)
        return {"error": f"compare_products failed: {exc}"}


def _rank_by_relevance(reviews: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Order reviews by cosine similarity to the query (embedding model)."""
    model = get_model()
    texts = [f"{r.get('title', '')} {r.get('text', '')}".strip() for r in reviews]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    query_vec = model.encode(query, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def _norm(matrix: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(matrix)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return matrix / norms

    sims = (_norm(vectors) @ _norm(query_vec)[0])
    order = np.argsort(-sims)
    return [reviews[int(i)] for i in order]


def search_reviews(parent_asin: str, query: Any = None, limit: Any = 3) -> Dict[str, Any]:
    """
    Read customer reviews for a product (by parent_asin). Without a query the
    most helpful reviews are returned; with a query the reviews most relevant to
    it are returned. Use it to ground claims about what customers said.

    Args:
        parent_asin: The product id.
        query: Optional English phrase to rank reviews against (e.g. "battery life").
        limit: Maximum number of reviews (default 3).

    Returns:
        {"parent_asin", "count", "reviews"} plus a "note" when none exist. Never raises.
    """
    try:
        asin = (parent_asin or "").strip()
        limit = max(1, min(_as_int(limit, 3), _MAX_LIMIT))
        if not asin:
            return {"parent_asin": asin, "count": 0, "reviews": [],
                    "note": "No product id was provided."}

        reviews = get_adapter().get_reviews(asin)
        if not reviews:
            return {"parent_asin": asin, "count": 0, "reviews": [],
                    "note": "No reviews are available for this product."}

        if query and str(query).strip():
            try:
                reviews = _rank_by_relevance(reviews, str(query).strip())
            except Exception as exc:  # fall back to helpfulness order
                logger.warning("Review relevance ranking failed: %s", exc)

        trimmed = [
            {
                "rating": r.get("rating", 0.0),
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "helpful_vote": r.get("helpful_vote", 0),
                "verified_purchase": r.get("verified_purchase", False),
            }
            for r in reviews[:limit]
        ]
        return {"parent_asin": asin, "count": len(trimmed), "reviews": trimmed}
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_reviews failed: %s", exc)
        return {"error": f"search_reviews failed: {exc}"}


def search_policies(query: str) -> Dict[str, Any]:
    """
    Look up the store's own policy documents (shipping, returns/refunds,
    warranty, payment/cash on delivery, order tracking) for a service question.
    Use it for questions about store rules, not about a specific product.

    Returns:
        {"query", "count", "policies": [{"text", "source", "heading", "score"}]}.
        The list is empty when nothing relevant is in the documents. Never raises.
    """
    try:
        query = (query or "").strip()
        if not query:
            return {"query": query, "count": 0, "policies": [],
                    "note": "No policy question was provided."}
        hits = get_adapter().search_knowledge(query, k=3)
        return {"query": query, "count": len(hits), "policies": hits}
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_policies failed: %s", exc)
        return {"error": f"search_policies failed: {exc}"}


def check_order(order_id: str) -> Dict[str, Any]:
    """
    Look up the status and details of a customer order by its id (format
    ORD-12345). Use it only when the user gives or refers to an order id.

    Returns:
        {"order_id", "found", "order"} or {"order_id", "found": false, "note"}.
        Never invents an order. Never raises.
    """
    try:
        raw = order_id or ""
        oid = extract_order_id(raw)
        if not oid:
            # Accept a bare number too ("10025" -> "ORD-10025"), but not text.
            stripped = raw.strip()
            oid = f"ORD-{stripped}" if stripped.isdigit() else None
        if not oid:
            return {"order_id": None, "found": False,
                    "note": "No order id (like ORD-10025) was provided."}

        order = get_adapter().get_order(oid)
        if not order:
            return {"order_id": oid, "found": False,
                    "note": "No order with that id was found."}
        return {"order_id": oid, "found": True, "order": order}
    except Exception as exc:  # noqa: BLE001
        logger.exception("check_order failed: %s", exc)
        return {"error": f"check_order failed: {exc}"}


_ROUTINE_STEPS = {
    "skin": (
        ("cleanser", "face cleanser"),
        ("moisturizer", "face moisturizer"),
        ("sunscreen", "sunscreen"),
    ),
    "hair": (
        ("shampoo", "shampoo"),
        ("conditioner", "conditioner"),
        ("treatment", "hair treatment"),
    ),
    "body": (
        ("body wash", "body wash"),
        ("body lotion", "body lotion"),
        ("scrub", "body scrub"),
    ),
}


def _routine_domain(goal: str) -> Optional[str]:
    text = (goal or "").lower()
    if any(word in text for word in ("skin", "acne", "face", "wrinkle", "aging", "anti-aging")):
        return "skin"
    if any(word in text for word in ("hair", "scalp", "curl", "dandruff")):
        return "hair"
    if any(word in text for word in ("body", "bath", "shower")):
        return "body"
    return None


def build_routine(goal: str, budget: Any = None) -> Dict[str, Any]:
    """
    Build a short, data-backed routine for a goal (for example dry skin, acne or
    hair care) within an optional USD budget. Each step is filled with a real
    catalog product; a missing step is omitted rather than invented.

    Returns:
        {"goal", "budget_usd", "count", "steps", "products"} or
        {"goal", "not_available": true, "note", "category_suggestions"}.
        Never raises.
    """
    try:
        goal = normalize_query((goal or "").strip())
        high = _as_float(budget)
        domain = _routine_domain(goal)
        steps: List[Dict[str, Any]] = []

        if domain:
            for label, query in _ROUTINE_STEPS[domain]:
                found = search_products(query, max_price_usd=high, limit=1)
                products = found.get("products") or []
                if products:
                    steps.append({
                        "step": label,
                        "query": query,
                        "product": _slim_product(products[0]),
                    })
        else:
            found = search_products(goal, max_price_usd=high, limit=3)
            for index, product in enumerate(found.get("products") or [], start=1):
                steps.append({
                    "step": f"option {index}",
                    "query": goal,
                    "product": _slim_product(product),
                })

        if not steps:
            fallback = search_products(goal, max_price_usd=high, limit=1)
            return {
                "goal": goal,
                "budget_usd": high,
                "not_available": True,
                "note": "No products in the catalog match that routine goal.",
                "category_suggestions": fallback.get("category_suggestions", []),
            }

        return {
            "goal": goal,
            "budget_usd": high,
            "count": len(steps),
            "steps": steps,
            "products": [step["product"] for step in steps],
            "note": (
                "These are matching catalog products for each step, not medical advice. "
                "Skip any step that is not in the catalog."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - tools must never raise
        logger.exception("build_routine failed: %s", exc)
        return {"error": f"build_routine failed: {exc}"}


def gift_finder(
    recipient: str, occasion: Any = None, budget: Any = None
) -> Dict[str, Any]:
    """
    Find gift products for a recipient and/or occasion, optionally under a USD
    budget. It searches the real catalog and never invents a gift idea.

    Returns:
        {"recipient", "occasion", "budget_usd", "count", "products", "approximate"}
        or {"not_available": true, "note", ...}. Never raises.
    """
    try:
        recipient = normalize_query((recipient or "").strip())
        occasion = normalize_query(str(occasion or "").strip())
        high = _as_float(budget)

        queries: List[str] = []
        if recipient and occasion:
            queries.extend([
                f"gift for {recipient} {occasion}",
                f"{recipient} {occasion}",
                f"gift {occasion}",
            ])
        elif recipient:
            queries.extend([f"gift for {recipient}", recipient])
        elif occasion:
            queries.append(f"gift {occasion}")

        chosen = None
        for query in queries:
            found = search_products(query, max_price_usd=high, limit=5)
            if found.get("products"):
                chosen = found
                break

        # No direct match: offer real, popular gift categories instead of
        # inventing an item. The note makes the fallback explicit.
        fallback_used = False
        if not chosen:
            for query in ("perfume", "body lotion", "makeup"):
                found = search_products(query, max_price_usd=high, limit=5)
                if found.get("products"):
                    chosen = found
                    fallback_used = True
                    break

        base = {
            "recipient": recipient,
            "occasion": occasion,
            "budget_usd": high,
            "count": 0,
            "products": [],
            "approximate": False,
        }
        if not chosen:
            base.update({
                "not_available": True,
                "note": "No products in the catalog match that gift request.",
                "category_suggestions": category_suggestions(
                    f"{recipient} {occasion}".strip()
                ),
            })
            return base

        base.update({
            "count": len(chosen["products"]),
            "products": [_slim_product(p) for p in chosen["products"]],
            "approximate": bool(chosen.get("approximate")),
            "query": chosen.get("query"),
            "fallback": fallback_used,
        })
        if fallback_used:
            base["note"] = (
                "No exact gift match was found; these are real products from "
                "popular gift categories within the budget."
            )
        return base
    except Exception as exc:  # noqa: BLE001 - tools must never raise
        logger.exception("gift_finder failed: %s", exc)
        return {"error": f"gift_finder failed: {exc}"}


_CATEGORY_ALIASES = {
    "skincare": "skin care",
    "haircare": "hair",
    "body care": "bath and body",
    "body": "bath and body",
    "tools": "tools and accessories",
    "accessories": "tools and accessories",
    "perfume": "fragrance",
    "nail": "nails",
}


def _match_category(category: str) -> Optional[str]:
    from storefront import CATEGORY_RULES

    text = (category or "").strip().lower()
    if not text:
        return None
    text = _CATEGORY_ALIASES.get(text, text)
    for name, keywords in CATEGORY_RULES:
        if name == "other":
            continue
        if text == name or text in name:
            return name
        for keyword in keywords:
            if re.search(rf"(?<!\w){re.escape(keyword.strip())}(?!\w)", text):
                return name
    return None


def best_in_category(category: str, max_price: Any = None) -> Dict[str, Any]:
    """
    Pick the highest-rated real catalog product in a category, optionally under
    a USD price. The "best" product is the one with the highest average rating
    (ties broken by the number of ratings). Never invents products.

    Returns:
        {"category", "max_price_usd", "found", "best", "products", "note"} or
        {"category", "not_available": true, "note", "category_suggestions"}.
        Never raises.
    """
    try:
        from data_loader import load_catalog
        from storefront import categorize_title

        raw = (category or "").strip()
        matched = _match_category(raw)
        high = _as_float(max_price)

        if not matched:
            return {
                "category": raw,
                "max_price_usd": high,
                "not_available": True,
                "note": "That category is not in the catalog.",
                "category_suggestions": category_suggestions(raw),
            }

        frame = load_catalog()
        if frame is None or frame.empty:
            return {
                "category": matched,
                "max_price_usd": high,
                "not_available": True,
                "note": "The catalog is not available.",
            }

        rows: List[Dict[str, Any]] = []
        for record in frame.to_dict("records"):
            if categorize_title(str(record.get("title") or "")) != matched:
                continue
            price = float(record.get("price") or 0.0)
            if price <= 0:
                continue
            if high is not None and price > high:
                continue
            rows.append(record)

        if not rows:
            return {
                "category": matched,
                "max_price_usd": high,
                "not_available": True,
                "note": "No products in that category match the price limit.",
                "category_suggestions": category_suggestions(matched),
            }

        rows.sort(
            key=lambda r: (-float(r.get("average_rating") or 0.0),
                           -int(r.get("rating_number") or 0),
                           float(r.get("price") or 0.0))
        )
        products = [_slim_product(r) for r in rows[:3]]
        return {
            "category": matched,
            "max_price_usd": high,
            "found": True,
            "count": len(products),
            "best": products[0],
            "products": products,
            "note": (
                "Best means the highest average catalog rating in this category; "
                "review counts vary between products."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - tools must never raise
        logger.exception("best_in_category failed: %s", exc)
        return {"error": f"best_in_category failed: {exc}"}


def escalate_to_human(reason: str) -> Dict[str, Any]:
    """
    Hand the conversation to a human agent. Use it when the user explicitly asks
    for a human, or is clearly frustrated and the bot cannot help.

    Returns:
        {"escalated": true, "reason", "note"}. Never raises.
    """
    try:
        clean = (reason or "").strip() or "user requested a human agent"
        return {
            "escalated": True,
            "reason": clean,
            "note": "The conversation should be escalated to a human agent.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("escalate_to_human failed: %s", exc)
        return {"error": f"escalate_to_human failed: {exc}"}


# ── Schemas passed to the LLM ────────────────────────────────────────────────
_ENGLISH_NOTE = "All string arguments must be in English; translate the user's wording yourself."

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "search_products",
        "description": (
            "Find products in the catalog from a natural-language description, "
            "optionally filtered by price in US dollars. Use this whenever the "
            "user is shopping for or asks about products. Returns a list of "
            "products with their ids, prices and ratings, plus 'total_matches' "
            "(how many products matched before the limit); call "
            "get_product_details for full details. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, in English, e.g. 'waterproof mascara'.",
                },
                "min_price_usd": {
                    "type": "number",
                    "description": "Optional minimum price in US dollars.",
                },
                "max_price_usd": {
                    "type": "number",
                    "description": "Optional maximum price in US dollars.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of products to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_product_details",
        "description": (
            "Return the full catalog record (features, description, price, "
            "rating) for one product by its parent_asin id. Use it after "
            "search_products when you need details to answer a question about a "
            "specific product. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parent_asin": {
                    "type": "string",
                    "description": "The product's parent_asin id from a search result.",
                }
            },
            "required": ["parent_asin"],
        },
    },
    {
        "name": "compare_products",
        "description": (
            "Compare 2 or 3 specific products side by side using their ids. Use "
            "it only when the user names or refers to products to compare, never "
            "for more than three. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parent_asins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The 2 or 3 product ids (parent_asin) to compare.",
                }
            },
            "required": ["parent_asins"],
        },
    },
    {
        "name": "search_reviews",
        "description": (
            "Read customer reviews for one product (by parent_asin). Without a "
            "query it returns the most helpful reviews; with a query it returns "
            "the reviews most relevant to it. Use it to ground statements about "
            "what customers said. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parent_asin": {
                    "type": "string",
                    "description": "The product's parent_asin id.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional English phrase to rank reviews against, e.g. 'battery life'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of reviews to return (default 3).",
                },
            },
            "required": ["parent_asin"],
        },
    },
    {
        "name": "search_policies",
        "description": (
            "Search the store's own policy documents for answers about shipping, "
            "returns and refunds, warranty, payment including cash on delivery, "
            "and order tracking. Use it for store-rule questions, not for product "
            "details. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The policy question, in English.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_order",
        "description": (
            "Look up a customer order's status and details by its id (format "
            "ORD-12345). Use it only when the user provides or refers to an order "
            "id. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order id, e.g. 'ORD-10025'.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "build_routine",
        "description": (
            "Build a short data-backed routine for a goal (for example dry skin, "
            "acne or hair care) within an optional budget in US dollars. Each "
            "step is filled with a real catalog product, and missing steps are "
            "omitted rather than invented. Use it when the customer asks for a "
            "routine, regimen or set of steps. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The routine goal in English, e.g. 'dry skin' or 'hair care'.",
                },
                "budget": {
                    "type": "number",
                    "description": "Optional maximum total budget in US dollars.",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "gift_finder",
        "description": (
            "Find real catalog products suitable as a gift for a recipient and/or "
            "occasion, optionally under a budget in US dollars. Use it when the "
            "customer asks for gift ideas. It never invents products. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Who the gift is for, in English, e.g. 'mom' or 'a friend'.",
                },
                "occasion": {
                    "type": "string",
                    "description": "Optional occasion in English, e.g. 'birthday'.",
                },
                "budget": {
                    "type": "number",
                    "description": "Optional maximum price in US dollars.",
                },
            },
            "required": ["recipient"],
        },
    },
    {
        "name": "best_in_category",
        "description": (
            "Return the highest-rated real catalog product in a category, "
            "optionally under a price in US dollars. Use it for questions like "
            "'what is the best mascara?' or 'best skin care under $30'. Never "
            "invents a product. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "A catalog category or keyword, e.g. 'mascara' or 'skin care'.",
                },
                "max_price": {
                    "type": "number",
                    "description": "Optional maximum price in US dollars.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate the conversation to a human agent. Use it when the user "
            "explicitly asks for a human, or is clearly frustrated and the bot "
            "cannot help. " + _ENGLISH_NOTE
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short reason for the escalation, in English.",
                }
            },
            "required": ["reason"],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "compare_products": compare_products,
    "search_reviews": search_reviews,
    "search_policies": search_policies,
    "check_order": check_order,
    "build_routine": build_routine,
    "gift_finder": gift_finder,
    "best_in_category": best_in_category,
    "escalate_to_human": escalate_to_human,
}


def call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch a model tool call to the matching function. Never raises."""
    arguments = arguments or {}
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return func(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for '{name}': {exc}"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("tool '%s' failed: %s", name, exc)
        return {"error": f"tool '{name}' failed: {exc}"}
