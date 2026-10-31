"""
comparison.py
-------------
Generates side-by-side product comparison tables and AI-driven comparative text summaries
using catalog metadata and top customer reviews.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from data_loader import load_catalog, get_reviews_for_product

logger = logging.getLogger(__name__)


# Reference phrasing that points back to the products just shown.
_REFERENCE_REGEX = re.compile(
    r"\b(these|them|those|both|the\s+first\s+two|first\s+two)\b", re.IGNORECASE
)
# Split "compare A and B" / "A vs B" / "A versus B" into separate products.
_SPLIT_REGEX = re.compile(r"\s+(?:and|vs\.?|versus)\s+", re.IGNORECASE)
_COMPARE_PREFIX_REGEX = re.compile(r"^\s*(?:please\s+)?compare\s+", re.IGNORECASE)


def _references_previous(message: str) -> bool:
    return bool(_REFERENCE_REGEX.search(message or ""))


def resolve_comparison_products(
    message: str, last_products: List[Dict[str, Any]], max_products: int = 3
) -> List[Dict[str, Any]]:
    """
    Decide which products to compare from a user message.

    1. A reference ("these", "them", "both", "first two") -> the session's
       previously shown products.
    2. Otherwise split the message on and/vs/versus and search each named
       product separately (top 1 each).
    3. Otherwise a single semantic search of the whole message.

    Returns a list of product dicts (may be empty or fewer than 2).
    """
    message = (message or "").strip()
    last_products = list(last_products or [])

    if _references_previous(message):
        lowered = message.lower()
        if "both" in lowered or "two" in lowered:
            return last_products[:2]
        return last_products[:max_products]

    cleaned = _COMPARE_PREFIX_REGEX.sub("", message).strip()
    parts = [p.strip(" .,-") for p in _SPLIT_REGEX.split(cleaned)]
    parts = [p for p in parts if p]

    from product_search import search_products  # lazy to avoid import cycles

    if len(parts) >= 2:
        selected: List[Dict[str, Any]] = []
        seen = set()
        for part in parts:
            hits = search_products(part, top_k=1)
            if hits and hits[0]["parent_asin"] not in seen:
                seen.add(hits[0]["parent_asin"])
                selected.append(hits[0])
        return selected[:max_products]

    hits = search_products(cleaned or message, top_k=max_products)
    return hits if len(hits) >= 2 else []



def _format_price(price_val: Any) -> str:
    """Format price value safely into USD currency string."""
    try:
        if pd.notna(price_val) and float(price_val) > 0:
            return f"${float(price_val):.2f}"
    except (ValueError, TypeError):
        pass
    return "N/A"


def _format_rating(rating_val: Any, count_val: Any) -> str:
    """Format average rating and review count with star emoji."""
    try:
        r = float(rating_val) if pd.notna(rating_val) else 0.0
        c = int(count_val) if pd.notna(count_val) else 0
        if r > 0:
            return f"{r:.1f} ⭐ ({c:,} reviews)"
    except (ValueError, TypeError):
        pass
    return "N/A"


def _format_features(features_val: Any, max_len: int = 200) -> str:
    """Format and truncate key features text."""
    if not pd.notna(features_val) or not str(features_val).strip():
        return "N/A"
    clean_text = " ".join(str(features_val).split())
    if len(clean_text) > max_len:
        return f"{clean_text[:max_len].rstrip()}..."
    return clean_text


def _generate_comparison_summary(products: List[Dict[str, Any]]) -> str:
    """
    Generate a concise 2-3 sentence comparative summary:
    1. Compare prices (highlight cheapest option)
    2. Compare ratings (highlight highest rated)
    3. Include real customer insight from the most helpful review of the top-rated item.
    """
    # 1. Price comparison
    valid_prices = [(p, p.get("price", 0.0)) for p in products if p.get("price", 0.0) > 0]
    if valid_prices:
        cheapest_prod, lowest_price = min(valid_prices, key=lambda x: x[1])
        highest_price = max(valid_prices, key=lambda x: x[1])[1]
        if lowest_price == highest_price:
            price_sentence = f"All compared products are priced equally at {_format_price(lowest_price)}."
        else:
            price_sentence = (
                f"Among the options, the {cheapest_prod.get('title', 'item')} is the most affordable at "
                f"{_format_price(lowest_price)}."
            )
    else:
        price_sentence = "Pricing information varies or is not listed for some options."

    # 2. Rating comparison
    valid_ratings = [
        (p, p.get("average_rating", 0.0), p.get("rating_number", 0))
        for p in products
    ]
    top_rated_prod, best_rating, best_count = max(valid_ratings, key=lambda x: (x[1], x[2]))
    if best_rating > 0:
        rating_sentence = (
            f"In terms of customer satisfaction, {top_rated_prod.get('title', 'the leading item')} ranks highest "
            f"with a {best_rating:.1f} ⭐ score across {best_count:,} verified ratings."
        )
    else:
        rating_sentence = "Customer review scores are comparable across the selected products."

    # 3. Top review insight from the highest-rated product
    review_sentence = "No customer review text is available for these products."
    try:
        reviews = get_reviews_for_product(top_rated_prod.get("parent_asin", ""))
        if reviews:
            # Sort by helpfulness
            sorted_reviews = sorted(reviews, key=lambda r: r.get("helpful_vote", 0), reverse=True)
            top_review = sorted_reviews[0]
            review_text = top_review.get("text", "") or top_review.get("title", "")
            clean_review = " ".join(review_text.split())
            if clean_review:
                snippet = clean_review[:130].rstrip()
                if len(clean_review) > 130:
                    snippet += "..."
                review_sentence = f'Top customer feedback highlights: "{snippet}"'
    except Exception as exc:
        logger.debug("Could not fetch top review for comparison summary: %s", exc)

    return f"{price_sentence} {rating_sentence} {review_sentence}"


def compare_products(asins: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Compare multiple products side-by-side by their parent ASINs.

    Args:
        asins: List of product ASIN strings (typically 2 or 3).

    Returns:
        Tuple of (table_dict, summary_str).
        If fewer than 2 valid products are found, returns (None, error_message).
    """
    if not asins:
        return None, "No product ASINs were provided for comparison."

    # Deduplicate while preserving input order
    ordered_asins: List[str] = []
    for a in asins:
        clean_a = str(a).strip()
        if clean_a and clean_a not in ordered_asins:
            ordered_asins.append(clean_a)

    if len(ordered_asins) < 2:
        return None, "Please provide at least 2 distinct product ASINs to compare."

    try:
        catalog = load_catalog()
        if catalog.empty:
            return None, "Catalog data is currently unavailable."

        lookup = catalog.set_index("parent_asin", drop=False)
        found_products: List[Dict[str, Any]] = []

        for asin in ordered_asins:
            if asin in lookup.index:
                row = lookup.loc[asin]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                found_products.append(row.to_dict())

        if len(found_products) < 2:
            return None, (
                f"Found only {len(found_products)} valid product(s) in catalog. "
                "At least 2 valid products are required for comparison."
            )

        # Build comparison table headers and rows
        headers = ["Feature"] + [
            str(p.get("title", "Unknown Product")) if pd.notna(p.get("title")) else "Unknown Product"
            for p in found_products
        ]

        row_prices = ["Price"] + [_format_price(p.get("price")) for p in found_products]
        row_stores = [
            "Brand / Store"
        ] + [str(p.get("store", "N/A")) if pd.notna(p.get("store")) else "N/A" for p in found_products]
        row_ratings = [
            "Rating"
        ] + [_format_rating(p.get("average_rating"), p.get("rating_number")) for p in found_products]
        row_categories = [
            "Category"
        ] + [str(p.get("category_path", "N/A")) if pd.notna(p.get("category_path")) else "N/A" for p in found_products]
        row_features = [
            "Key Features"
        ] + [_format_features(p.get("features_text"), max_len=200) for p in found_products]

        table_dict: Dict[str, Any] = {
            "headers": headers,
            "rows": [
                row_prices,
                row_stores,
                row_ratings,
                row_categories,
                row_features,
            ],
        }

        summary_str = _generate_comparison_summary(found_products)
        return table_dict, summary_str

    except Exception as exc:
        logger.exception("Error executing product comparison for ASINs %s: %s", asins, exc)
        return None, f"An error occurred while generating the comparison: {exc}"
