"""
add_images.py
-------------
Add an ``image_url`` column to data/catalog.parquet.

The product catalog has no image URLs. This script re-reads the Amazon Reviews
2023 "All_Beauty" *metadata* (the parquet file already cached by
huggingface_hub from 01_get_data.py), picks the first usable large / hi-res
image URL for each product, and writes it back into data/catalog.parquet.

Safety guarantees:
  * Only the EXISTING parent_asin values are touched — the catalog is not
    filtered, re-ordered, or re-indexed, and no other column is modified.
  * Rows keep their original order (the FAISS index and orders.db depend on it).
  - ASINs with no usable image get an empty string (the frontend shows a
    neutral placeholder tile instead).

Usage:
    .venv\\Scripts\\python.exe scripts\\add_images.py
    .venv\\Scripts\\python.exe scripts\\add_images.py --category All_Beauty
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"
ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.parquet"


def find_meta_files(category):
    """Locate the metadata file(s) for a category (same logic as 01_get_data.py)."""
    api = HfApi()
    for rev in ["main", "refs/pr/10"]:
        try:
            files = api.list_repo_files(REPO, repo_type="dataset", revision=rev)
        except Exception as exc:
            print(f"  could not list files on '{rev}': {exc}")
            continue
        usable = [
            f for f in files
            if category in f
            and "benchmark" not in f
            and f.endswith((".jsonl", ".parquet"))
        ]
        meta = [f for f in usable if "meta" in f.lower()]
        if meta:
            return rev, meta
    return None, []


def load_meta(rev, files):
    frames = []
    for f in files:
        print(f"  loading {f} ...")
        path = hf_hub_download(REPO, f, repo_type="dataset", revision=rev)
        if f.endswith(".parquet"):
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_json(path, lines=True))
    return pd.concat(frames, ignore_index=True)


def _first_url(images, key):
    """Return the first non-empty URL from images[key], or ''."""
    if not isinstance(images, dict):
        return ""
    values = images.get(key)
    if values is None:
        return ""
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def pick_image_url(images):
    """Prefer a hi-res image, then a large one. '' when none is available."""
    return _first_url(images, "hi_res") or _first_url(images, "large")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="All_Beauty")
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        print(f"Catalog not found: {CATALOG_PATH}")
        sys.exit(1)

    catalog = pd.read_parquet(CATALOG_PATH)
    original_columns = list(catalog.columns)
    original_order = list(catalog["parent_asin"])
    print(f"Catalog: {len(catalog):,} products, columns: {original_columns}")

    print(f"Locating cached metadata for category '{args.category}' ...")
    rev, meta_files = find_meta_files(args.category)
    if not meta_files:
        print("No metadata file found (are you offline and the file not cached?).")
        sys.exit(1)

    meta = load_meta(rev, meta_files)
    print(f"  {len(meta):,} metadata rows")
    if "images" not in meta.columns:
        print("Metadata has no 'images' column; cannot add image URLs.")
        sys.exit(1)

    url_by_asin = {}
    for asin, images in zip(meta["parent_asin"].astype(str), meta["images"]):
        url = pick_image_url(images)
        if url:
            url_by_asin[asin] = url
    print(f"  {len(url_by_asin):,} products have at least one image URL")

    image_urls = [url_by_asin.get(str(asin), "") for asin in original_order]
    catalog["image_url"] = image_urls

    # Guarantee we did not change row count / order / other columns.
    assert list(catalog.columns)[:len(original_columns)] == original_columns
    assert list(catalog["parent_asin"]) == original_order

    catalog.to_parquet(CATALOG_PATH, index=False)

    have_image = sum(1 for u in image_urls if u)
    print("\nDone.")
    print(f"  data/catalog.parquet  -> {len(catalog):,} products")
    print(f"  with image_url        -> {have_image:,} ({have_image / len(catalog):.1%})")
    print(f"  without image (blank) -> {len(catalog) - have_image:,}")


if __name__ == "__main__":
    main()
