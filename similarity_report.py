"""
similarity_report.py
--------------------
Prints the top semantic-similarity scores for 15 sensible beauty/personal-care
queries and 10 nonsense / out-of-scope queries, so we can pick MIN_SIMILARITY.

Run:  .venv\\Scripts\\python.exe similarity_report.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from product_search import search_products, build_index  # noqa: E402

SENSIBLE = [
    "hair brush",
    "curling wand",
    "moisturizer for dry skin",
    "nail polish remover",
    "shampoo for dandruff",
    "lip balm",
    "face cleanser",
    "hair dryer",
    "mascara",
    "sunscreen spf 50",
    "body lotion",
    "electric toothbrush",
    "nail clippers",
    "conditioner for curly hair",
    "makeup remover wipes",
]

NONSENSE = [
    "do you ship to Mars",
    "best pizza recipe",
    "gaming laptop",
    "diesel engine oil",
    "how to file taxes",
    "quantum physics textbook",
    "car tire pressure",
    "mortgage rate today",
    "dog food for pitbull",
    "bitcoin price",
]

# Extra realistic queries, including the English translations of the Arabic
# queries below (the Arabic is translated to English before it is embedded).
EXTRA_SENSIBLE = [
    "something for frizzy hair",
    "gift for my mom",
    "waterproof mascara",
    "hair care",
    "anti-aging serum",
    "nail art",
    "hair brush",                    # فرشاة شعر
    "moisturizer for dry skin",      # كريم مرطب للبشرة الجافة
    "women's perfume",               # عطر نسائي
    "hair dryer",                    # مجفف شعر
]

# The Arabic queries as written (shown for reference only; the pipeline
# translates these to English before embedding).
ARABIC_RAW = [
    "فرشاة شعر",
    "كريم مرطب للبشرة الجافة",
    "عطر نسائي",
    "مجفف شعر",
]


def top_scores(query, k=3):
    hits = search_products(query, top_k=k)
    if not hits:
        return []
    return [(round(h["similarity_score"], 3), h["title"]) for h in hits]


def print_table(title, queries):
    print()
    print(title)
    print("-" * 110)
    print(f"{'#':>2}  {'query':38}  {'top1':>6}  {'top2':>6}  {'top3':>6}  top-1 match")
    print("-" * 110)
    for i, q in enumerate(queries, 1):
        scores = top_scores(q)
        s = [sc for sc, _ in scores] + [None, None, None]
        top_title = (scores[0][1][:38] if scores else "")
        fmt = lambda v: f"{v:6.3f}" if v is not None else "   ---"
        print(f"{i:>2}  {q:38}  {fmt(s[0])}  {fmt(s[1])}  {fmt(s[2])}  {top_title}")
    print("-" * 110)


if __name__ == "__main__":
    build_index()
    print_table("SENSIBLE BEAUTY / PERSONAL-CARE QUERIES (should score high)", SENSIBLE)
    print_table("EXTRA QUERIES + TRANSLATIONS OF ARABIC QUERIES (should clear the threshold)", EXTRA_SENSIBLE)
    print_table("ARABIC QUERIES AS WRITTEN (English model — reference only)", ARABIC_RAW)
    print_table("NONSENSE / OUT-OF-SCOPE QUERIES (should score low)", NONSENSE)
