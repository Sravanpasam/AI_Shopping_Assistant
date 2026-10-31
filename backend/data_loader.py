"""
data_loader.py
--------------
Loads and caches the e-commerce product catalog and customer reviews from Parquet files.
Provides query helper functions by ASIN for catalog items and reviews.
"""

import os
import logging
from typing import Optional, List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Determine the data directory relative to this file
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CATALOG_PATH = os.path.join(DATA_DIR, "catalog.parquet")
REVIEWS_PATH = os.path.join(DATA_DIR, "reviews.parquet")

# Expected schema definitions
CATALOG_COLUMNS = [
    "parent_asin",
    "title",
    "store",
    "price",
    "average_rating",
    "rating_number",
    "category_path",
    "features_text",
    "description_text",
    "n_reviews_sampled",
]

REVIEWS_COLUMNS = [
    "parent_asin",
    "rating",
    "title",
    "text",
    "helpful_vote",
    "verified_purchase",
    "timestamp",
]

# Module-level caches
_catalog_df: Optional[pd.DataFrame] = None
_reviews_df: Optional[pd.DataFrame] = None


def _clean_product_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all fields in a catalog product dict are JSON-safe and properly typed."""
    return {
        "parent_asin": str(record.get("parent_asin", "")),
        "title": str(record.get("title", "N/A")) if pd.notna(record.get("title")) else "N/A",
        "store": str(record.get("store", "N/A")) if pd.notna(record.get("store")) else "N/A",
        "price": float(record.get("price", 0.0)) if pd.notna(record.get("price")) else 0.0,
        "average_rating": float(record.get("average_rating", 0.0)) if pd.notna(record.get("average_rating")) else 0.0,
        "rating_number": int(record.get("rating_number", 0)) if pd.notna(record.get("rating_number")) else 0,
        "category_path": str(record.get("category_path", "N/A")) if pd.notna(record.get("category_path")) else "N/A",
        "features_text": str(record.get("features_text", "")) if pd.notna(record.get("features_text")) else "",
        "description_text": str(record.get("description_text", "")) if pd.notna(record.get("description_text")) else "",
        "n_reviews_sampled": int(record.get("n_reviews_sampled", 0)) if pd.notna(record.get("n_reviews_sampled")) else 0,
    }


def _clean_review_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all fields in a review dict are JSON-safe and properly typed."""
    return {
        "parent_asin": str(record.get("parent_asin", "")),
        "rating": float(record.get("rating", 0.0)) if pd.notna(record.get("rating")) else 0.0,
        "title": str(record.get("title", "")) if pd.notna(record.get("title")) else "",
        "text": str(record.get("text", "")) if pd.notna(record.get("text")) else "",
        "helpful_vote": int(record.get("helpful_vote", 0)) if pd.notna(record.get("helpful_vote")) else 0,
        "verified_purchase": bool(record.get("verified_purchase", False)) if pd.notna(record.get("verified_purchase")) else False,
        "timestamp": int(record.get("timestamp", 0)) if pd.notna(record.get("timestamp")) else 0,
    }


def load_catalog(force_reload: bool = False) -> pd.DataFrame:
    """
    Load catalog.parquet into a pandas DataFrame with module-level caching.

    Args:
        force_reload: If True, reloads the dataset from disk even if cached.

    Returns:
        pd.DataFrame containing the product catalog.
    """
    global _catalog_df
    if _catalog_df is not None and not force_reload:
        return _catalog_df

    try:
        if not os.path.exists(CATALOG_PATH):
            logger.warning("Catalog file not found at '%s'. Returning empty DataFrame.", CATALOG_PATH)
            _catalog_df = pd.DataFrame(columns=CATALOG_COLUMNS)
            return _catalog_df

        logger.info("Loading catalog from %s ...", CATALOG_PATH)
        df = pd.read_parquet(CATALOG_PATH)

        # Enforce column existence
        for col in CATALOG_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Clean types
        df["parent_asin"] = df["parent_asin"].astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0.0)
        df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype(int)

        _catalog_df = df
        logger.info("Loaded catalog successfully (%d products).", len(_catalog_df))
    except Exception as exc:
        logger.exception("Failed to load catalog from %s: %s", CATALOG_PATH, exc)
        _catalog_df = pd.DataFrame(columns=CATALOG_COLUMNS)

    return _catalog_df


def load_reviews(force_reload: bool = False) -> pd.DataFrame:
    """
    Load reviews.parquet into a pandas DataFrame with module-level caching.

    Args:
        force_reload: If True, reloads the dataset from disk even if cached.

    Returns:
        pd.DataFrame containing customer reviews.
    """
    global _reviews_df
    if _reviews_df is not None and not force_reload:
        return _reviews_df

    try:
        if not os.path.exists(REVIEWS_PATH):
            logger.warning("Reviews file not found at '%s'. Returning empty DataFrame.", REVIEWS_PATH)
            _reviews_df = pd.DataFrame(columns=REVIEWS_COLUMNS)
            return _reviews_df

        logger.info("Loading reviews from %s ...", REVIEWS_PATH)
        df = pd.read_parquet(REVIEWS_PATH)

        # Enforce column existence
        for col in REVIEWS_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df["parent_asin"] = df["parent_asin"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

        _reviews_df = df
        logger.info("Loaded reviews successfully (%d reviews).", len(_reviews_df))
    except Exception as exc:
        logger.exception("Failed to load reviews from %s: %s", REVIEWS_PATH, exc)
        _reviews_df = pd.DataFrame(columns=REVIEWS_COLUMNS)

    return _reviews_df


def get_product_by_asin(asin: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve product details by parent ASIN from the cached catalog.

    Args:
        asin: Product identifier.

    Returns:
        Dictionary of product attributes or None if not found.
    """
    if not asin:
        return None

    try:
        catalog = load_catalog()
        matches = catalog[catalog["parent_asin"] == str(asin).strip()]
        if matches.empty:
            logger.debug("Product ASIN '%s' not found in catalog.", asin)
            return None

        record = matches.iloc[0].to_dict()
        return _clean_product_dict(record)
    except Exception as exc:
        logger.exception("Error retrieving product ASIN '%s': %s", asin, exc)
        return None


def get_reviews_for_product(asin: str) -> List[Dict[str, Any]]:
    """
    Retrieve customer reviews for a given product ASIN, sorted by helpfulness.

    Args:
        asin: Product identifier.

    Returns:
        List of review dictionaries.
    """
    if not asin:
        return []

    try:
        reviews_df = load_reviews()
        matches = reviews_df[reviews_df["parent_asin"] == str(asin).strip()]
        if matches.empty:
            logger.debug("No reviews found for product ASIN '%s'.", asin)
            return []

        # Sort by helpful_vote descending
        sorted_matches = matches.sort_values(by="helpful_vote", ascending=False)
        records = sorted_matches.to_dict(orient="records")
        return [_clean_review_dict(r) for r in records]
    except Exception as exc:
        logger.exception("Error retrieving reviews for ASIN '%s': %s", asin, exc)
        return []
