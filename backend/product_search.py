"""
product_search.py
-----------------
FAISS-based dense semantic product search using SentenceTransformer embeddings.
Handles vector index creation, persistence, loading, and nearest-neighbor query execution.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss

from nlp_engine import get_model, get_embedding
from data_loader import load_catalog
from price_parser import parse_price_constraint
from query_normalizer import normalize_query
from config import SEARCH_FLOOR, SEARCH_MARGIN, SEARCH_APPROX_FLOOR

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ASINS_PATH = os.path.join(DATA_DIR, "faiss_asins.json")
META_PATH = os.path.join(DATA_DIR, "faiss_meta.json")

# Reciprocal-rank-fusion settings. Dense similarity carries most of the signal;
# the two lexical indexes mainly repair typos and exact product words.
RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_WORD_WEIGHT = 0.35
RRF_CHAR_WEIGHT = 0.15

# Module-level variables
_index: Optional[faiss.IndexFlatIP] = None
_asins: List[str] = []
_catalog_df: Optional[pd.DataFrame] = None
_catalog_lookup: Optional[pd.DataFrame] = None  # cached parent_asin -> row lookup
_word_vectorizer = None
_char_vectorizer = None
_word_matrix = None
_char_matrix = None
_tfidf_asins: List[str] = []
_category_counts: Optional[Dict[str, int]] = None


def _get_catalog_lookup() -> Optional[pd.DataFrame]:
    """Return the catalog indexed by parent_asin, building it once and caching it."""
    global _catalog_lookup
    if _catalog_lookup is None and _catalog_df is not None and not _catalog_df.empty:
        _catalog_lookup = _catalog_df.set_index("parent_asin", drop=False)
    return _catalog_lookup


def _combined_text(row: pd.Series) -> str:
    """The text embedded for dense search and indexed for TF-IDF."""
    title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    category = str(row.get("category_path", "")) if pd.notna(row.get("category_path")) else ""
    features = str(row.get("features_text", "")) if pd.notna(row.get("features_text")) else ""
    description = str(row.get("description_text", "")) if pd.notna(row.get("description_text")) else ""
    return f"{title} {category} {features} {description}".strip()


def _ensure_tfidf() -> bool:
    """Build (once) word and character TF-IDF matrices aligned with ``_asins``."""
    global _word_vectorizer, _char_vectorizer, _word_matrix, _char_matrix, _tfidf_asins

    if _word_matrix is not None and _tfidf_asins == _asins:
        return True
    if _catalog_df is None or _catalog_df.empty or not _asins:
        return False
    lookup = _get_catalog_lookup()
    if lookup is None:
        return False

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts: List[str] = []
        for asin in _asins:
            if asin not in lookup.index:
                texts.append("product")
                continue
            row = lookup.loc[asin]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            texts.append(_combined_text(row) or "product")

        word = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        char = TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
            sublinear_tf=True, min_df=2,
        )
        _word_matrix = word.fit_transform(texts)
        _char_matrix = char.fit_transform(texts)
        _word_vectorizer = word
        _char_vectorizer = char
        _tfidf_asins = list(_asins)
        logger.info("Built TF-IDF indexes for %d products.", len(texts))
        return True
    except Exception as exc:  # noqa: BLE001 - dense search still works without it
        logger.warning("Could not build TF-IDF index: %s", exc)
        _word_matrix = _char_matrix = None
        _word_vectorizer = _char_vectorizer = None
        _tfidf_asins = []
        return False


def _format_product_record(row: pd.Series, similarity_score: float) -> Dict[str, Any]:
    """Convert a catalog DataFrame row into a clean, typed product dict."""
    return {
        "parent_asin": str(row.get("parent_asin", "")),
        "title": str(row.get("title", "N/A")) if pd.notna(row.get("title")) else "N/A",
        "store": str(row.get("store", "N/A")) if pd.notna(row.get("store")) else "N/A",
        "price": float(row.get("price", 0.0)) if pd.notna(row.get("price")) else 0.0,
        "average_rating": float(row.get("average_rating", 0.0)) if pd.notna(row.get("average_rating")) else 0.0,
        "rating_number": int(row.get("rating_number", 0)) if pd.notna(row.get("rating_number")) else 0,
        "category_path": str(row.get("category_path", "N/A")) if pd.notna(row.get("category_path")) else "N/A",
        "features_text": str(row.get("features_text", "")) if pd.notna(row.get("features_text")) else "",
        "description_text": str(row.get("description_text", "")) if pd.notna(row.get("description_text")) else "",
        "similarity_score": round(float(similarity_score), 4),
    }


def _catalog_fingerprint(df: pd.DataFrame) -> Optional[str]:
    """
    Fingerprint the catalog as:
        "<row_count>:<sha256 of sorted parent_asin list>:<sha256 of row texts>"

    The ASIN hash catches membership changes; the text hash catches edits to the
    fields that are actually embedded (title/category/features/description), so a
    changed catalog (e.g. mock -> real data) forces an index rebuild.
    """
    if df is None or df.empty or "parent_asin" not in df.columns:
        return None

    asins = sorted(str(a) for a in df["parent_asin"].tolist())
    asin_hasher = hashlib.sha256()
    for asin in asins:
        asin_hasher.update(asin.encode("utf-8"))
        asin_hasher.update(b"\n")

    text_hasher = hashlib.sha256()
    for _, row in df.iterrows():
        title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
        category = str(row.get("category_path", "")) if pd.notna(row.get("category_path")) else ""
        features = str(row.get("features_text", "")) if pd.notna(row.get("features_text")) else ""
        description = str(row.get("description_text", "")) if pd.notna(row.get("description_text")) else ""
        combined = f"{title} {category} {features} {description}".strip()
        text_hasher.update(combined.encode("utf-8"))
        text_hasher.update(b"\n")

    return f"{len(df)}:{asin_hasher.hexdigest()}:{text_hasher.hexdigest()}"


def _read_stored_fingerprint() -> Optional[str]:
    """Read the fingerprint recorded alongside the saved index, if any."""
    try:
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("catalog_fingerprint")
    except Exception as exc:
        logger.warning("Could not read index metadata '%s': %s", META_PATH, exc)
    return None


def build_index(force_rebuild: bool = False) -> None:
    """
    Build or load a FAISS IndexFlatIP index for semantic product search.
    If pre-computed index files exist on disk and force_rebuild is False,
    loads the index from disk instead of re-encoding.

    Args:
        force_rebuild: If True, forces re-embedding and overwrites saved files.
    """
    global _index, _asins, _catalog_df, _catalog_lookup, _category_counts

    # Load catalog dataframe
    _catalog_df = load_catalog()
    _category_counts = None

    # Rebuild the cached ASIN lookup for the (possibly new) catalog.
    _catalog_lookup = (
        _catalog_df.set_index("parent_asin", drop=False)
        if _catalog_df is not None and not _catalog_df.empty
        else None
    )

    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(ASINS_PATH):
        current_fingerprint = _catalog_fingerprint(_catalog_df)
        stored_fingerprint = _read_stored_fingerprint()

        if stored_fingerprint is None:
            logger.warning(
                "Saved index is missing a catalog fingerprint (legacy or corrupt metadata). "
                "Rebuilding to avoid serving a stale index..."
            )
        elif current_fingerprint != stored_fingerprint:
            logger.warning(
                "Catalog changed since the index was built "
                "(stored=%s, current=%s). Rebuilding index...",
                str(stored_fingerprint)[:12], str(current_fingerprint)[:12],
            )
        else:
            try:
                logger.info("Loading existing FAISS index from '%s'...", INDEX_PATH)
                loaded_index = faiss.read_index(INDEX_PATH)
                with open(ASINS_PATH, "r", encoding="utf-8") as f:
                    loaded_asins = json.load(f)

                if loaded_index.ntotal == len(loaded_asins) and loaded_index.ntotal > 0:
                    _index = loaded_index
                    _asins = loaded_asins
                    logger.info("FAISS index loaded successfully with %d vectors.", _index.ntotal)
                    return
                else:
                    logger.warning("Index size (%d) does not match ASIN list length (%d). Rebuilding...",
                                   loaded_index.ntotal, len(loaded_asins))
            except Exception as exc:
                logger.warning("Failed to load existing index from disk: %s. Rebuilding...", exc)

    if _catalog_df is None or _catalog_df.empty:
        logger.warning("Catalog is empty. Cannot build FAISS index.")
        _index = None
        _asins = []
        return

    try:
        logger.info("Building FAISS index for %d products...", len(_catalog_df))
        texts: List[str] = []
        asins: List[str] = []

        for _, row in _catalog_df.iterrows():
            title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
            category = str(row.get("category_path", "")) if pd.notna(row.get("category_path")) else ""
            features = str(row.get("features_text", "")) if pd.notna(row.get("features_text")) else ""
            description = str(row.get("description_text", "")) if pd.notna(row.get("description_text")) else ""
            combined = f"{title} {category} {features} {description}".strip()

            texts.append(combined if combined else "product")
            asins.append(str(row.get("parent_asin", "")))

        model = get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        # L2-normalize vectors so inner product equals cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save to disk
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            faiss.write_index(index, INDEX_PATH)
            with open(ASINS_PATH, "w", encoding="utf-8") as f:
                json.dump(asins, f, indent=2)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "catalog_fingerprint": _catalog_fingerprint(_catalog_df),
                        "row_count": int(len(_catalog_df)),
                        "ntotal": int(index.ntotal),
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved FAISS index, ASINs, and metadata to '%s'.", DATA_DIR)
        except Exception as save_exc:
            logger.exception("Failed to persist FAISS index to disk: %s", save_exc)

        _index = index
        _asins = asins
        logger.info("Built FAISS index with %d products.", _index.ntotal)
    except Exception as exc:
        logger.exception("Error while building FAISS index: %s", exc)
        _index = None
        _asins = []


def search_products(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search products with hybrid dense + TF-IDF retrieval (no thresholding).

    Args:
        query: User search string.
        top_k: Maximum number of products to return.

    Returns:
        Product dictionaries ordered by reciprocal-rank fusion. Each has the
        catalog fields plus ``similarity_score`` (dense cosine) and
        ``hybrid_score``.
    """
    return _search_raw(normalize_query(query), top_k)


# How many candidates to retrieve before thresholding/price-filtering.
PRICE_FILTER_CANDIDATES = 50


def _rank_positions(scores: np.ndarray, positive_only: bool = False) -> List[int]:
    """Indices sorted high-to-low, optionally skipping non-positive scores."""
    order = np.argsort(-scores)
    if positive_only:
        return [int(i) for i in order if scores[i] > 0]
    return [int(i) for i in order]


def _search_raw(query: str, k: int) -> List[Dict[str, Any]]:
    """Retrieve up to k products by reciprocal-rank fusion (no price filtering)."""
    global _index, _asins, _catalog_df

    if not query or not query.strip():
        return []

    if _index is None or len(_asins) == 0 or _catalog_df is None:
        build_index()

    if _index is None or _index.ntotal == 0 or _catalog_df is None or _catalog_df.empty:
        logger.warning("FAISS index or catalog unavailable for search.")
        return []

    try:
        # Dense ranking over the whole (small) catalog.
        query_vec = get_embedding(query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        dense_values, dense_indices = _index.search(query_vec, _index.ntotal)
        dense_scores = np.zeros(_index.ntotal, dtype=np.float32)
        for idx, score in zip(dense_indices[0], dense_values[0]):
            if 0 <= idx < len(dense_scores):
                dense_scores[idx] = score

        # Lexical rankings: word unigrams/bigrams and character 3-5 grams.
        rrf = np.zeros(_index.ntotal, dtype=np.float32)
        for rank, pos in enumerate(_rank_positions(dense_scores)):
            rrf[pos] += RRF_DENSE_WEIGHT / (RRF_K + rank + 1)

        if _ensure_tfidf():
            word_vec = _word_vectorizer.transform([query])
            char_vec = _char_vectorizer.transform([query])
            word_scores = (_word_matrix @ word_vec.T).toarray().ravel()
            char_scores = (_char_matrix @ char_vec.T).toarray().ravel()

            for scores, weight in ((word_scores, RRF_WORD_WEIGHT),
                                   (char_scores, RRF_CHAR_WEIGHT)):
                for rank, pos in enumerate(_rank_positions(scores, positive_only=True)):
                    if pos < len(rrf):
                        rrf[pos] += weight / (RRF_K + rank + 1)

        order = np.argsort(-rrf)
        catalog_lookup = _get_catalog_lookup()
        if catalog_lookup is None:
            return []

        results: List[Dict[str, Any]] = []
        for idx in order[:min(k, len(order))]:
            asin = _asins[int(idx)]
            if asin not in catalog_lookup.index:
                continue
            row = catalog_lookup.loc[asin]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            product = _format_product_record(row, dense_scores[int(idx)])
            product["hybrid_score"] = round(float(rrf[int(idx)]), 6)
            results.append(product)
        return results
    except Exception as exc:
        logger.exception("Error during product search for query '%s': %s", query, exc)
        return []


def select_relevant(candidates: List[Dict[str, Any]]):
    """
    Apply the floor-plus-margin rule to already-ranked candidates.

    Returns ``(relevant, top_score)``. The top score must reach SEARCH_FLOOR;
    products scoring within SEARCH_MARGIN of it are kept.
    """
    if not candidates:
        return [], 0.0
    top_score = max(float(p.get("similarity_score", 0.0)) for p in candidates)
    if top_score < SEARCH_FLOOR:
        return [], top_score
    cutoff = max(SEARCH_FLOOR, top_score - SEARCH_MARGIN)
    relevant = [
        p for p in candidates
        if float(p.get("similarity_score", 0.0)) >= cutoff
    ]
    return relevant, top_score


def category_suggestions(query: str = "", limit: int = 3) -> List[Dict[str, Any]]:
    """Return up to ``limit`` real catalog categories, prioritising query keywords."""
    global _category_counts
    try:
        from storefront import CATEGORY_RULES

        catalog = load_catalog()
        if catalog is None or catalog.empty or "title" not in catalog.columns:
            return []
        from storefront import categorize_title

        if _category_counts is None:
            counts: Dict[str, int] = {}
            for title in catalog["title"].dropna().astype(str):
                name = categorize_title(title)
                if name != "other":
                    counts[name] = counts.get(name, 0) + 1
            _category_counts = counts
        counts = _category_counts

        text = (query or "").lower()
        priority = [
            name for name, keywords in CATEGORY_RULES
            if name != "other" and any(keyword in text for keyword in keywords)
        ]
        ordered = priority + sorted(counts, key=lambda n: (-counts[n], n))
        seen = set()
        out: List[Dict[str, Any]] = []
        for name in ordered:
            if name in seen or name == "other":
                continue
            seen.add(name)
            out.append({"value": name, "label": name, "count": counts.get(name, 0)})
            if len(out) >= limit:
                break
        return out
    except Exception as exc:  # noqa: BLE001 - suggestions are optional
        logger.warning("Could not build category suggestions: %s", exc)
        return []


def _apply_price(
    products: List[Dict[str, Any]], min_usd: Optional[float], max_usd: Optional[float]
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for product in products:
        price = product.get("price", 0.0)
        if not price or price <= 0:
            continue
        if min_usd is not None and price < min_usd:
            continue
        if max_usd is not None and price > max_usd:
            continue
        matched.append(product)
    return matched


def search_products_filtered(
    query: str, top_k: int = 5, candidate_k: int = PRICE_FILTER_CANDIDATES
) -> Dict[str, Any]:
    """
    Hybrid search with floor-plus-margin acceptance and an optional price filter.

    Flow:
      1. normalise the non-price part of the query, then retrieve candidates;
      2. accept products when the best score reaches SEARCH_FLOOR, keeping
         products within SEARCH_MARGIN of it;
      3. apply the price filter (missing/zero prices are excluded when active);
      4. if nothing is accepted but the best score reaches SEARCH_APPROX_FLOOR,
         return the closest products flagged ``approximate``;
      5. otherwise return no products plus three real category suggestions.

    If a confident, relevant set exists but the price filter empties it,
    ``fallback_products`` holds the 3 cheapest relevant priced candidates.

    Returns a dict with cleaned_query, min_usd, max_usd, filter_active, products,
    fallback_products, approximate, category_suggestions and top_score.
    """
    cleaned_query, min_usd, max_usd = parse_price_constraint(query)
    filter_active = min_usd is not None or max_usd is not None

    # Normalise only the product words; parse_price_constraint has already
    # separated the price constraint, so the price phrase is never damaged.
    search_text = cleaned_query if cleaned_query.strip() else query
    search_text = normalize_query(search_text)
    candidates = _search_raw(search_text, candidate_k)

    relevant, top_score = select_relevant(candidates)
    products: List[Dict[str, Any]] = []
    fallback_products: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []
    approximate = False

    if relevant:
        if filter_active:
            matched = _apply_price(relevant, min_usd, max_usd)
            products = matched[:top_k]
            if not products:
                priced = [p for p in relevant if p.get("price", 0.0) > 0]
                fallback_products = sorted(priced, key=lambda p: p["price"])[:3]
        else:
            products = relevant[:top_k]
    elif candidates and top_score >= SEARCH_APPROX_FLOOR:
        weak = [
            p for p in candidates
            if float(p.get("similarity_score", 0.0)) >= SEARCH_APPROX_FLOOR
        ]
        approx = _apply_price(weak, min_usd, max_usd) if filter_active else weak
        products = [dict(p, approximate=True) for p in approx[:top_k]]
        approximate = bool(products)
        if not products:
            suggestions = category_suggestions(search_text)
    else:
        suggestions = category_suggestions(search_text)

    return {
        "cleaned_query": cleaned_query,
        "min_usd": min_usd,
        "max_usd": max_usd,
        "filter_active": filter_active,
        "products": products,
        "fallback_products": fallback_products,
        "approximate": approximate,
        "category_suggestions": suggestions,
        "top_score": round(float(top_score), 4),
    }
