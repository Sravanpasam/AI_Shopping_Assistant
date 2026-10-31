"""
storefront.py
-------------
Product-browsing API for the storefront page. It is registered as a blueprint
so the ShopBot app and the legacy baseline expose the exact same product
endpoints without duplicating code, and it never touches the chat pipeline.

Endpoints
---------
GET /api/products            List / filter / sort / paginate the catalog and
                             return facet counts for the filter panel.
GET /api/products/<asin>     One product, with its features, description and
                             five most helpful reviews.
GET /api/orders/<order_id>   One order's status/delivery details (no personal
                             fields) from data/orders.db.
GET /api/help/<topic>        One store policy document (docs/) as Markdown.

Grounding rules (same spirit as the chatbot):
  * Every value comes from ``data/catalog.parquet`` / ``data/reviews.parquet``,
    ``data/orders.db`` or ``docs/*.md``. Nothing is invented.
  * A missing or non-positive price is returned as ``null`` so the UI can show
    "Price unavailable" instead of a fake $0.
"""

import os
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from flask import Blueprint, jsonify, request

from data_loader import load_catalog, load_reviews
from orders import get_order

storefront_bp = Blueprint("storefront", __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# Help topics -> the policy document that answers them. Keep in sync with the
# frontend's HELP_TOPICS (frontend/app.js).
TOPIC_DOCS: Dict[str, str] = {
    "returns": "returns_and_refunds.md",
    "warranty": "warranty_and_guarantee.md",
    "shipping": "shipping_and_delivery.md",
    "payment": "payment_and_cash_on_delivery.md",
    "orders": "order_tracking_and_cancellation.md",
}

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
# The catalog's ``category_path`` column is empty for every All_Beauty product,
# so categories are derived deterministically from the product title. The rule
# is: scan the title in lower case and assign the FIRST category whose keyword
# appears (so "hair brush" -> hair, "makeup brush" -> makeup). Titles matching
# no keyword fall into "other".
#
# Measured counts on the current 1,000-product catalog:
#   other 173, hair 167, skin care 154, makeup 133, tools and accessories 112,
#   bath and body 105, fragrance 87, nails 69.
# The remaining "other" titles are genuinely not beauty items (party goods,
# tattoo sheets, electronics accessories), so they stay unclassified.
CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("nails", ["nail", "manicure", "pedicure", "cuticle", "polish"]),
    ("fragrance", [
        "fragrance", "perfume", "cologne", "eau de", "body mist", "mist",
        "scent", "parfum", "edt", "edp", "aftershave", "body spray",
    ]),
    ("makeup", [
        "makeup", "lipstick", "lip gloss", "lip balm", "eyeliner", "mascara",
        "eyeshadow", "blush", "foundation", "concealer", "eyebrow", "brow",
        "contour", "highlighter", "primer", "lip ", "palette", "bronzer",
        "cheek", "lip liner",
    ]),
    ("skin care", [
        "skin", "serum", "moistur", "acne", "cleanser", "exfoli", "facial",
        "face", "sunscreen", "spf", "toner", "retinol", "vitamin c", "eye",
        "wrinkle", "anti-aging", "anti aging", "tanner", "self tan",
        "self-tanner",
    ]),
    ("hair", [
        "hair", "shampoo", "conditioner", "scalp", "wig", "blow", "curl",
        "straighten", "bobby", "barber", "dandruff", "ponytail", "braid",
        "durag", "du-rag", "weave", "styling", "styler", "spiking", "pomade",
        "mousse",
    ]),
    ("bath and body", [
        "bath", "body", "soap", "lotion", "shower", "deodorant", "scrub",
        "hand", "foot", "wash", "oil", "talcum", "talc", "callus",
        "cracked heel", "shave", "shaving",
    ]),
    ("tools and accessories", [
        "brush", "tool", "kit", "scissor", "tweezer", "clipper", "applicator",
        "sponge", "mirror", "comb", "clip", "roller", "massager", "set",
        "headband", "bonnet", "cap", "scarf", "bottle", "pump", "dispenser",
        "jar", "band", "wrap", "crown", "razor", "epilator", "strop",
        "salon", "sterilizer", "massage", "hot stone",
    ]),
]

# Price buckets (low inclusive, high exclusive; None = open ended).
PRICE_BUCKETS: List[Tuple[float, Optional[float], str]] = [
    (0, 10, "Under $10"),
    (10, 25, "$10 – $25"),
    (25, 50, "$25 – $50"),
    (50, 100, "$50 – $100"),
    (100, None, "$100 and up"),
]

# Rating buckets (minimum average rating -> label).
RATING_FACETS: List[Tuple[float, str]] = [
    (4.0, "4 stars & up"),
    (3.0, "3 stars & up"),
    (2.0, "2 stars & up"),
    (1.0, "1 star & up"),
]

SORT_OPTIONS = {"relevance", "price_asc", "price_desc", "rating", "reviews"}

DEFAULT_PAGE_SIZE = 24
MAX_PAGE_SIZE = 60

# Enriched-catalog cache, invalidated when the catalog row count changes.
_frame_cache: Optional[pd.DataFrame] = None
_frame_fingerprint: Optional[int] = None


def categorize_title(title: str) -> str:
    """Derive a coarse category from a product title (see CATEGORY_RULES)."""
    text = (title or "").lower()
    for category, keywords in CATEGORY_RULES:
        for keyword in keywords:
            if keyword in text:
                return category
    return "other"


def _clean_price(value) -> Optional[float]:
    """Return a positive price as float, else None (0/NaN -> unavailable)."""
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _get_frame() -> pd.DataFrame:
    """Return the catalog enriched with derived category / helper columns, cached."""
    global _frame_cache, _frame_fingerprint

    catalog = load_catalog()
    if catalog is None or catalog.empty:
        return pd.DataFrame(
            columns=["parent_asin", "title", "store", "price", "average_rating",
                     "rating_number", "image_url", "category", "_title_lower"]
        )

    fingerprint = len(catalog)
    if _frame_cache is not None and _frame_fingerprint == fingerprint:
        return _frame_cache

    df = catalog.copy()
    for col in ("title", "store", "image_url"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0.0)
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["_title_lower"] = df["title"].str.lower()
    features = (
        df["features_text"].fillna("").astype(str).str.lower()
        if "features_text" in df.columns
        else pd.Series("", index=df.index)
    )
    desc = (
        df["description_text"].fillna("").astype(str).str.lower()
        if "description_text" in df.columns
        else pd.Series("", index=df.index)
    )
    df["_search_text"] = df["_title_lower"] + " " + features + " " + desc
    df["category"] = df["title"].map(categorize_title)

    _frame_cache = df
    _frame_fingerprint = fingerprint
    return df


def _to_card(row: pd.Series) -> Dict[str, Any]:
    """Serialize a catalog row into a compact storefront card dict."""
    return {
        "parent_asin": str(row.get("parent_asin", "")),
        "title": str(row.get("title", "")) or "N/A",
        "store": str(row.get("store", "")) or "N/A",
        "price": _clean_price(row.get("price")),
        "average_rating": round(float(row.get("average_rating", 0.0)), 2),
        "rating_number": int(row.get("rating_number", 0)),
        "category": str(row.get("category", "other")),
        "image_url": str(row.get("image_url", "")) or "",
    }


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tokenize(text: str) -> List[str]:
    cleaned = "".join(c if c.isalnum() else " " for c in (text or "").lower())
    return [t for t in cleaned.split() if t]


_SEARCH_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "should", "now",
    "need", "want", "looking", "find", "show", "get", "something", "product", "products",
    "please", "give", "suggest", "recommend"
}


def _apply_text(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter rows matching query tokens across title and product details, using semantic search when available."""
    if not query or not query.strip():
        return df

    cleaned = query.strip()
    raw_tokens = _tokenize(cleaned)
    if not raw_tokens:
        return df

    tokens = [t for t in raw_tokens if t not in _SEARCH_STOP_WORDS]
    if not tokens:
        tokens = raw_tokens

    # 1. Try strict title match with all meaningful tokens first (preserves exact keyword contract)
    title_mask = pd.Series(True, index=df.index)
    for token in tokens:
        title_mask &= df["_title_lower"].str.contains(token, regex=False)

    matched = df[title_mask]
    if len(matched) > 0:
        return matched

    # 2. Try semantic search from product_search (FAISS + TF-IDF) when strict title match finds nothing
    try:
        from product_search import search_products as _search_sem
        sem_products = _search_sem(cleaned, top_k=60)
        asins = [p["parent_asin"] for p in sem_products if p.get("parent_asin")]
        if asins:
            sem_matched = df[df["parent_asin"].isin(asins)].copy()
            if len(sem_matched) > 0:
                order_map = {asin: i for i, asin in enumerate(asins)}
                sem_matched["_sem_order"] = sem_matched["parent_asin"].map(order_map).fillna(999)
                return sem_matched.sort_values("_sem_order").drop(columns=["_sem_order"])
    except Exception as exc:
        logger.warning("Storefront semantic search fallback: %s", exc)

    search_col = "_search_text" if "_search_text" in df.columns else "_title_lower"

    # 3. Try full search text (title + features + description) with all tokens
    full_mask = pd.Series(True, index=df.index)
    for token in tokens:
        full_mask &= df[search_col].str.contains(token, regex=False)

    matched = df[full_mask]
    if len(matched) > 0:
        return matched

    # 4. Fallback: match ANY token (OR semantics) across title or search text
    or_mask = pd.Series(False, index=df.index)
    for token in tokens:
        if len(token) > 2:
            or_mask |= df[search_col].str.contains(token, regex=False)

    matched = df[or_mask]
    if len(matched) > 0:
        return matched

    return df.iloc[0:0]


def _apply_price(df: pd.DataFrame, min_price: Optional[float], max_price: Optional[float]) -> pd.DataFrame:
    if min_price is None and max_price is None:
        return df
    priced = df[df["price"] > 0]
    if min_price is not None:
        priced = priced[priced["price"] >= min_price]
    if max_price is not None:
        priced = priced[priced["price"] <= max_price]
    return priced


def _apply_rating(df: pd.DataFrame, min_rating: Optional[float]) -> pd.DataFrame:
    if min_rating is None:
        return df
    return df[df["average_rating"] >= min_rating]


def _apply_category(df: pd.DataFrame, category: Optional[str]) -> pd.DataFrame:
    if not category or category.lower() in ("all", ""):
        return df
    return df[df["category"] == category]


def _sort(df: pd.DataFrame, sort: str, query: str) -> pd.DataFrame:
    if sort not in SORT_OPTIONS:
        sort = "relevance"

    if sort == "price_asc":
        priced = df[df["price"] > 0].sort_values("price", ascending=True)
        unpriced = df[df["price"] <= 0]
        return pd.concat([priced, unpriced])
    if sort == "price_desc":
        priced = df[df["price"] > 0].sort_values("price", ascending=False)
        unpriced = df[df["price"] <= 0]
        return pd.concat([priced, unpriced])
    if sort == "rating":
        return df.sort_values(["average_rating", "rating_number"], ascending=[False, False])
    if sort == "reviews":
        return df.sort_values("rating_number", ascending=False)

    # relevance (default): rank by how many query words appear in the title.
    tokens = set(_tokenize(query))
    if tokens:
        relevance = df["_title_lower"].map(lambda title: sum(1 for t in tokens if t in title))
        ranked = df.assign(_rel=relevance).sort_values(
            ["_rel", "rating_number"], ascending=[False, False]
        )
        return ranked.drop(columns=["_rel"])
    return df.sort_values("rating_number", ascending=False)


def _price_facets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    facets = []
    priced = df[df["price"] > 0]
    for low, high, label in PRICE_BUCKETS:
        subset = priced[priced["price"] >= low]
        if high is not None:
            subset = subset[subset["price"] < high]
        facets.append({"value": label, "min": low, "max": high, "count": int(len(subset))})
    return facets


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@storefront_bp.route("/api/products", methods=["GET"])
def list_products():
    """
    List products for the storefront grid.

    Query params:
      q          text search on the title (all words must appear)
      category   one of the derived categories (or "all")
      min_price  minimum price in USD
      max_price  maximum price in USD
      min_rating minimum average rating
      sort       relevance | price_asc | price_desc | rating | reviews
      page       1-based page number (default 1)
      page_size  items per page (default 24, max 60)
      per_page   legacy alias for page_size
    """
    try:
        df = _get_frame()

        query = (request.args.get("q") or "").strip()
        asins_param = (request.args.get("asins") or "").strip()
        category = (request.args.get("category") or "").strip()
        min_price = _to_float(request.args.get("min_price"))
        max_price = _to_float(request.args.get("max_price"))
        min_rating = _to_float(request.args.get("min_rating"))
        sort = (request.args.get("sort") or "relevance").strip()

        try:
            page = max(1, int(request.args.get("page", 1)))
        except (TypeError, ValueError):
            page = 1
        raw_size = request.args.get("page_size") or request.args.get("per_page") or DEFAULT_PAGE_SIZE
        try:
            page_size = int(raw_size)
        except (TypeError, ValueError):
            page_size = DEFAULT_PAGE_SIZE
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))

        # Explicit ASINs requested (e.g. from chatbot recommendations)
        if asins_param:
            target_asins = [a.strip() for a in asins_param.split(",") if a.strip()]
            if target_asins:
                matched = df[df["parent_asin"].isin(target_asins)].copy()
                if len(matched) > 0:
                    order_map = {asin: i for i, asin in enumerate(target_asins)}
                    matched["_order"] = matched["parent_asin"].map(order_map).fillna(999)
                    filtered = matched.sort_values("_order").drop(columns=["_order"])
                    if category and category.lower() not in ("all", ""):
                        cat_filtered = _apply_category(filtered, category)
                        if len(cat_filtered) > 0:
                            filtered = cat_filtered
                    filtered = _apply_price(filtered, min_price, max_price)
                    filtered = _apply_rating(filtered, min_rating)
                    total = int(len(filtered))
                    start = (page - 1) * page_size
                    window = filtered.iloc[start:start + page_size]
                    items = [_to_card(row) for _, row in window.iterrows()]
                    pages = (total + page_size - 1) // page_size if page_size else 0
                    return jsonify({
                        "items": items,
                        "total": total,
                        "page": page,
                        "page_size": page_size,
                        "per_page": page_size,
                        "pages": pages,
                        "products": items,
                        "facets": {"categories": [], "prices": [], "ratings": []},
                    })

        # Facets are counted over the set filtered by everything EXCEPT the
        # facet's own dimension (standard shopping-UX behaviour).
        text_df = _apply_text(df, query)
        cat_source = _apply_rating(_apply_price(text_df, min_price, max_price), min_rating)
        price_source = _apply_rating(_apply_category(text_df, category), min_rating)
        rating_source = _apply_price(_apply_category(text_df, category), min_price, max_price)

        category_facets = [
            {"value": name, "label": name, "count": int((cat_source["category"] == name).sum())}
            for name, _ in CATEGORY_RULES
        ]
        category_facets.append(
            {"value": "other", "label": "other", "count": int((cat_source["category"] == "other").sum())}
        )

        facets = {
            "categories": category_facets,
            "prices": _price_facets(price_source),
            "ratings": [
                {"value": label, "min": low, "count": int((rating_source["average_rating"] >= low).sum())}
                for low, label in RATING_FACETS
            ],
        }

        # Full filter set + sort + paginate.
        text_df = _apply_text(df, query)
        filtered = _apply_category(text_df, category)
        if len(filtered) == 0 and len(text_df) > 0 and query:
            filtered = text_df
        filtered = _apply_price(filtered, min_price, max_price)
        filtered = _apply_rating(filtered, min_rating)
        filtered = _sort(filtered, sort, query)

        total = int(len(filtered))
        start = (page - 1) * page_size
        window = filtered.iloc[start:start + page_size]
        items = [_to_card(row) for _, row in window.iterrows()]
        pages = (total + page_size - 1) // page_size if page_size else 0

        return jsonify({
            # New storefront shape
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "facets": facets,
            # Legacy fields kept so the old /api/products contract still works
            "products": items,
            "per_page": page_size,
        })
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@storefront_bp.route("/api/products/<asin>", methods=["GET"])
def product_detail(asin: str):
    """Return one product with its features, description and top reviews."""
    try:
        df = _get_frame()
        matches = df[df["parent_asin"].astype(str) == str(asin).strip()]
        if matches.empty:
            return jsonify({"error": "Product not found"}), 404

        row = matches.iloc[0]
        card = _to_card(row)

        features_text = str(row.get("features_text", "")) if pd.notna(row.get("features_text")) else ""
        description = str(row.get("description_text", "")) if pd.notna(row.get("description_text")) else ""
        features = [line.strip() for line in features_text.splitlines() if line.strip()]

        reviews: List[Dict[str, Any]] = []
        reviews_df = load_reviews()
        if reviews_df is not None and not reviews_df.empty and "parent_asin" in reviews_df.columns:
            match_rows = reviews_df[reviews_df["parent_asin"].astype(str) == str(asin).strip()]
            if not match_rows.empty:
                match_rows = match_rows.sort_values("helpful_vote", ascending=False).head(5)
                for _, r in match_rows.iterrows():
                    reviews.append({
                        "rating": round(float(r.get("rating")), 1) if pd.notna(r.get("rating")) else None,
                        "title": str(r.get("title", "")) if pd.notna(r.get("title")) else "",
                        "text": str(r.get("text", "")) if pd.notna(r.get("text")) else "",
                        "helpful_vote": int(r.get("helpful_vote", 0)) if pd.notna(r.get("helpful_vote")) else 0,
                        "verified_purchase": bool(r.get("verified_purchase")) if pd.notna(r.get("verified_purchase")) else False,
                    })

        n_sampled = row.get("n_reviews_sampled")
        card.update({
            "features": features,
            "description": description,
            "n_reviews_sampled": int(n_sampled) if pd.notna(n_sampled) else len(reviews),
            "reviews": reviews,
        })
        return jsonify(card)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@storefront_bp.route("/api/orders/<order_id>", methods=["GET"])
def order_detail(order_id: str):
    """
    Return one order's status and delivery details.

    Grounded in ``data/orders.db`` via orders.get_order(). Personal fields
    (customer name, city, payment method) are deliberately not returned — only
    the order/product/tracking facts the storefront needs. Unknown ids are a
    clean 404 so the page can say the order was not found.
    """
    try:
        order = get_order(order_id)
        if order is None:
            return jsonify({"error": "Order not found"}), 404

        asin = str(order.get("parent_asin") or "")
        product_title = ""
        df = _get_frame()
        if asin and not df.empty:
            match = df[df["parent_asin"].astype(str) == asin]
            if not match.empty:
                product_title = str(match.iloc[0].get("title", ""))

        return jsonify({
            "order_id": order.get("order_id"),
            "parent_asin": asin,
            "product_title": product_title or "N/A",
            "quantity": order.get("quantity"),
            "status": order.get("status"),
            "order_date": order.get("order_date"),
            "estimated_delivery": order.get("estimated_delivery"),
            "delivered_date": order.get("delivered_date"),
            "tracking_number": order.get("tracking_number"),
        })
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@storefront_bp.route("/api/help/<topic>", methods=["GET"])
def help_topic(topic: str):
    """Return a store policy document as Markdown for the help page."""
    key = (topic or "").strip().lower()
    filename = TOPIC_DOCS.get(key)
    if filename is None:
        return jsonify({"error": "Unknown help topic"}), 404

    path = os.path.join(DOCS_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "Help topic not available"}), 404

    try:
        with open(path, "r", encoding="utf-8") as handle:
            markdown = handle.read()
    except OSError as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500

    title = ""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    return jsonify({
        "topic": key,
        "title": title or filename,
        "source": filename,
        "markdown": markdown,
    })
