"""
measure_search.py
-----------------
Measure natural-language product search on eval/nl_search_queries.csv.

Metrics
  hit@3              expected product is in the first three confident results
  no-match rejection nonsense query gets no confident match (empty products,
                     or only approximate fallback results)

Usage
  .venv\\Scripts\\python.exe scripts\\measure_search.py
  .venv\\Scripts\\python.exe scripts\\measure_search.py --sweep
  .venv\\Scripts\\python.exe scripts\\measure_search.py --csv path\\to\\queries.csv
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import product_search  # noqa: E402

DEFAULT_CSV = os.path.join(ROOT, "eval", "nl_search_queries.csv")
PHRASINGS = ("bare", "request", "use", "typo", "nonsense")


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_query(query, top_k=3):
    result = product_search.search_products_filtered(query, top_k=top_k)
    return {
        "products": result.get("products") or [],
        "approximate": bool(result.get("approximate")),
        "suggestions": result.get("suggestions") or [],
    }


def evaluate(rows):
    stats = {p: {"n": 0, "hit": 0, "rejected": 0} for p in PHRASINGS}
    for row in rows:
        phrase = (row.get("phrasing") or "").strip().lower()
        if phrase not in stats:
            continue
        result = run_query(row["query"])
        ids = [p.get("parent_asin") for p in result["products"][:3]]
        bucket = stats[phrase]
        bucket["n"] += 1
        if phrase == "nonsense":
            if not result["products"] or result["approximate"]:
                bucket["rejected"] += 1
        elif row.get("expected_asin") in ids:
            bucket["hit"] += 1
    return stats


def report(stats):
    print(f"{'phrasing':<10} {'n':>4} {'hit@3':>7} {'rejection':>10}")
    total_n = total_hit = total_rej = product_n = 0
    for phrase in PHRASINGS:
        bucket = stats[phrase]
        n = bucket["n"]
        if not n:
            continue
        if phrase == "nonsense":
            rate = bucket["rejected"] / n
            print(f"{phrase:<10} {n:>4} {'-':>7} {rate:>9.1%}")
            total_rej += bucket["rejected"]
        else:
            rate = bucket["hit"] / n
            print(f"{phrase:<10} {n:>4} {rate:>6.1%} {'-':>10}")
            total_hit += bucket["hit"]
            product_n += n
        total_n += n
    if product_n:
        print(f"{'overall':<10} {product_n:>4} {total_hit / product_n:>6.1%} "
              f"{(total_rej / 10 if total_rej or stats['nonsense']['n'] else 0):>9.1%}")
    return (total_hit / product_n if product_n else 0.0,
            total_rej / stats["nonsense"]["n"] if stats["nonsense"]["n"] else 0.0)


def _cached_raw(original):
    cache = {}

    def wrapper(query, k):
        key = (query, k)
        if key not in cache:
            cache[key] = original(query, k)
        return cache[key]

    return wrapper


def sweep(rows):
    floors = [round(0.30 + 0.01 * i, 2) for i in range(26)]
    margins = [round(0.02 + 0.01 * i, 2) for i in range(24)]
    original = product_search._search_raw
    original_suggestions = product_search.category_suggestions
    product_search._search_raw = _cached_raw(original)
    product_search.category_suggestions = lambda query="", limit=3: []
    results = []
    try:
        for floor in floors:
            for margin in margins:
                product_search.SEARCH_FLOOR = floor
                product_search.SEARCH_MARGIN = margin
                hit, rejection = report_values(evaluate(rows))
                if rejection >= 0.95:
                    results.append((hit, rejection, floor, margin))
    finally:
        product_search._search_raw = original
        product_search.category_suggestions = original_suggestions
    results.sort(reverse=True)
    print("floor margin hit@3 rejection")
    for hit, rejection, floor, margin in results[:10]:
        print(f"{floor:>5.2f} {margin:>6.2f} {hit:>6.1%} {rejection:>8.1%}")
    if not results:
        print("(no floor/margin pair reached 95% no-match rejection)")


def report_values(stats):
    total_hit = product_n = 0
    for phrase in PHRASINGS[:-1]:
        bucket = stats[phrase]
        total_hit += bucket["hit"]
        product_n += bucket["n"]
    nonsense = stats["nonsense"]
    hit = total_hit / product_n if product_n else 0.0
    rejection = nonsense["rejected"] / nonsense["n"] if nonsense["n"] else 0.0
    return hit, rejection


def main():
    parser = argparse.ArgumentParser(description="Measure natural-language product search.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--sweep", action="store_true",
                        help="sweep SEARCH_FLOOR / SEARCH_MARGIN (requires hybrid search)")
    args = parser.parse_args()
    rows = load_rows(args.csv)
    if args.sweep:
        sweep(rows)
        return
    report(evaluate(rows))


if __name__ == "__main__":
    main()
