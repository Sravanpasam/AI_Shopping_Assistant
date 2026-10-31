"""
Step 1: build the product catalog and review table for the chatbot.

Usage:
    python 01_get_data.py --list-only                 # just show which files exist for a category
    python 01_get_data.py --category All_Beauty       # build catalog + reviews
    python 01_get_data.py --category All_Beauty --max-review-rows 300000

Outputs (in ./data):
    catalog.parquet   one row per product (title, price, rating, description, ...)
    reviews.parquet   reviews for those products only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def find_files(category):
    """Look for meta + review files for a category. Tries main branch, then the parquet PR branch."""
    api = HfApi()
    for rev in ["main", "refs/pr/10"]:
        try:
            files = api.list_repo_files(REPO, repo_type="dataset", revision=rev)
        except Exception as e:
            print(f"  could not list files on '{rev}': {e}")
            continue
        usable = [
            f for f in files
            if category in f
            and "benchmark" not in f
            and f.endswith((".jsonl", ".parquet"))
        ]
        meta = [f for f in usable if "meta" in f.lower()]
        reviews = [f for f in usable if "review" in f.lower() and "meta" not in f.lower()]
        if meta and reviews:
            return rev, meta, reviews
    return None, [], []


def load_files(rev, files, max_rows=None):
    frames = []
    for f in files:
        print(f"  downloading {f} ...")
        path = hf_hub_download(REPO, f, repo_type="dataset", revision=rev)
        if f.endswith(".parquet"):
            df = pd.read_parquet(path)
            if max_rows:
                df = df.head(max_rows)
        else:
            df = pd.read_json(path, lines=True, nrows=max_rows)
        frames.append(df)
        if max_rows and sum(len(x) for x in frames) >= max_rows:
            break
    return pd.concat(frames, ignore_index=True)


def to_text(x):
    """Metadata fields like description/features are lists; flatten them to one string."""
    if x is None:
        return ""
    if isinstance(x, (list, tuple, np.ndarray)):
        return "\n".join(str(i) for i in x)
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--max-review-rows", type=int, default=500_000)
    ap.add_argument("--n-products", type=int, default=1000)
    ap.add_argument("--min-reviews", type=int, default=5)
    args = ap.parse_args()

    print(f"Looking for category: {args.category}")
    rev, meta_files, review_files = find_files(args.category)
    if not meta_files:
        print("No files found. Check the category name (e.g. All_Beauty, Appliances, Gift_Cards)")
        print("or open the dataset's 'Files' tab on Hugging Face and send me what you see.")
        return
    print(f"Found on branch '{rev}':")
    for f in meta_files + review_files:
        print("  ", f)
    if args.list_only:
        return

    print("\nLoading reviews (sampled)...")
    reviews = load_files(rev, review_files, args.max_review_rows)
    print(f"  {len(reviews):,} reviews loaded")

    print("\nLoading product metadata...")
    meta = load_files(rev, meta_files)
    print(f"  {len(meta):,} products loaded")

    # Clean the catalog
    meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
    meta["description_text"] = meta["description"].apply(to_text)
    meta["features_text"] = meta["features"].apply(to_text)
    meta["category_path"] = meta["categories"].apply(lambda c: " > ".join(c) if isinstance(c, (list, np.ndarray)) else "")

    meta = meta[
        meta["price"].notna()
        & meta["title"].notna()
        & ((meta["description_text"].str.len() > 20) | (meta["features_text"].str.len() > 20))
    ]
    print(f"  {len(meta):,} products have price + title + description")

    # Keep products that have enough reviews in our review sample
    counts = reviews["parent_asin"].value_counts()
    good = counts[counts >= args.min_reviews].index
    meta = meta[meta["parent_asin"].isin(good)].copy()
    meta["n_reviews_sampled"] = meta["parent_asin"].map(counts)
    meta = meta.sort_values("n_reviews_sampled", ascending=False)
    before = len(meta)
    meta["_title_key"] = meta["title"].astype(str).str.strip().str.lower()
    meta = meta.drop_duplicates(subset="_title_key").drop(columns="_title_key")
    print(f"  {before - len(meta):,} duplicate-title products dropped")
    meta = meta.head(args.n_products)
    print(f"  {len(meta):,} products kept (>= {args.min_reviews} reviews each)")

    keep_cols = [
        "parent_asin", "title", "store", "price", "average_rating", "rating_number",
        "category_path", "features_text", "description_text", "n_reviews_sampled",
    ]
    catalog = meta[[c for c in keep_cols if c in meta.columns]].reset_index(drop=True)
    reviews = reviews[reviews["parent_asin"].isin(catalog["parent_asin"])].reset_index(drop=True)
    reviews = reviews[[c for c in ["parent_asin", "rating", "title", "text", "helpful_vote", "verified_purchase", "timestamp"] if c in reviews.columns]]

    catalog.to_parquet(OUT / "catalog.parquet")
    reviews.to_parquet(OUT / "reviews.parquet")

    print("\nDone.")
    print(f"  data/catalog.parquet  -> {len(catalog):,} products")
    print(f"  data/reviews.parquet  -> {len(reviews):,} reviews")
    print("\nSample products:")
    print(catalog[["title", "price", "average_rating"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
