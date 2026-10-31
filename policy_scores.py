"""
policy_scores.py
----------------
Prints the top semantic-similarity scores for policy_search.search_policies on
10 policy questions and 10 off-topic questions, so MIN_POLICY_SIMILARITY can be
chosen from the gap between them (instead of guessing).

Run:  .venv\\Scripts\\python.exe policy_scores.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from policy_search import search_policies  # noqa: E402

POLICY_QUESTIONS = [
    "What is the return window?",
    "How long is the warranty on electronics?",
    "Can I pay cash on delivery?",
    "How much does shipping cost?",
    "Where do you deliver?",
    "How do I track my order?",
    "Can I cancel my order?",
    "How long do refunds take?",
    "Is express delivery available?",
    "How do I make a warranty claim?",
]

OFF_TOPIC_QUESTIONS = [
    "best pizza recipe",
    "dog food for pitbull",
    "gaming laptop",
    "bitcoin price today",
    "how to file taxes",
    "mortgage rate today",
    "car tire pressure",
    "quantum physics textbook",
    "weather in Paris tomorrow",
    "diesel engine oil",
]


def top_scores(query, k=3):
    hits = search_policies(query, k=k)
    if not hits:
        return []
    return [(round(h["score"], 3), h["source"], h["heading"]) for h in hits]


def print_table(title, queries):
    print()
    print(title)
    print("-" * 112)
    print(f"{'#':>2}  {'query':40}  {'top1':>6}  {'top2':>6}  {'top3':>6}  top-1 (source / heading)")
    print("-" * 112)
    for i, q in enumerate(queries, 1):
        scores = top_scores(q)
        s = [sc for sc, _, _ in scores] + [None, None, None]
        if scores:
            _, src, heading = scores[0]
            top_label = f"{src[:22]} / {heading[:24]}"
        else:
            top_label = ""
        fmt = lambda v: f"{v:6.3f}" if v is not None else "   ---"
        print(f"{i:>2}  {q:40}  {fmt(s[0])}  {fmt(s[1])}  {fmt(s[2])}  {top_label}")
    print("-" * 112)


if __name__ == "__main__":
    print_table("POLICY QUESTIONS (should score high)", POLICY_QUESTIONS)
    print_table("OFF-TOPIC QUESTIONS (should score low)", OFF_TOPIC_QUESTIONS)
