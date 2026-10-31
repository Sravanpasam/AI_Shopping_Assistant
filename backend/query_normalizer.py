"""
query_normalizer.py
-------------------
Lightweight natural-language cleanup for product search.

It removes shopping filler and politeness (English and Arabic), then repairs
typos against the words that actually occur in the catalog. Price phrases are
left intact: digits, currency words and connectors such as "under"/"to" are
never stripped or typo-corrected, so price_parser can still read the query.

No model or network call is made.
"""

import difflib
import functools
import re
import unicodedata
from typing import List, Optional, Set

from data_loader import load_catalog

DEFAULT_TYPO_CUTOFF = 0.72

# Prefixes/shopping phrases are removed as whole phrases (longest first).
_FILLER_PHRASES = [
    "i am looking for", "i'm looking for", "im looking for",
    "i would like to buy", "i would like", "i'd like", "i want to buy",
    "i want", "i need", "we want", "we need", "can you show me",
    "could you show me", "would you show me", "do you have", "do you sell",
    "show me", "find me", "looking for", "searching for", "help me find",
    "something to", "something that", "something for", "products for",
    "product for", "stuff to", "a product to",
    "أبحث عن", "ابحث عن", "أريد شراء", "اريد شراء", "أريد", "اريد",
    "أحتاج", "احتاج", "هل لديك", "هل عندكم", "من فضلك", "لو سمحت",
    "أبحث", "ابحث", "شيئا", "شيء",
]

# Standalone words that add no product meaning. Price connectors and currency
# words are deliberately absent so "under 20 to 30" survives.
_FILLER_WORDS = {
    "a", "an", "the", "some", "any", "please", "plz", "me", "my", "our",
    "for", "with", "of", "about", "kindly", "just", "really", "very",
    "من", "في", "على", "مع", "لي", "لديك", "عندكم",
}

# Never treat these as typos: they carry price/filter meaning.
_PRICE_WORDS = {
    "under", "below", "less", "than", "up", "to", "no", "more", "max",
    "maximum", "cheaper", "over", "above", "at", "least", "greater",
    "from", "starting", "between", "and", "price", "budget", "cost",
    "range", "usd", "aed", "dirham", "dirhams", "dhs", "dollar", "dollars",
    "cheap", "expensive",
}

_NUMBER_RE = re.compile(r"\d")
_TOKEN_RE = re.compile(r"[A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF'-]*")
_vocabulary: Optional[Set[str]] = None
_bigrams: Optional[Set[str]] = None


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip()


def _catalog_words() -> tuple:
    """Catalog unigrams and title bigrams, used for typo correction."""
    global _vocabulary, _bigrams
    if _vocabulary is not None:
        return _vocabulary, _bigrams or set()
    words: Set[str] = set()
    bigrams: Set[str] = set()
    try:
        catalog = load_catalog()
        if catalog is not None and not catalog.empty:
            for column in ("title", "store"):
                if column not in catalog.columns:
                    continue
                for value in catalog[column].dropna().astype(str):
                    tokens = _TOKEN_RE.findall(value.lower())
                    for token in tokens:
                        if len(token) >= 3:
                            words.add(token)
                    if column == "title":
                        for left, right in zip(tokens, tokens[1:]):
                            bigrams.add(f"{left} {right}")
    except Exception:
        words, bigrams = set(), set()
    _vocabulary, _bigrams = words, bigrams
    return words, bigrams


def _catalog_vocabulary() -> Set[str]:
    return _catalog_words()[0]


def _strip_filler(text: str) -> str:
    lowered = text.lower()
    for phrase in sorted(_FILLER_PHRASES, key=len, reverse=True):
        lowered = lowered.replace(phrase, " ")
    tokens = [t for t in lowered.split() if t and t not in _FILLER_WORDS]
    return " ".join(tokens)


def _damerau_distance(left: str, right: str) -> int:
    """Edit distance with adjacent transpositions (for typos like wnad/wand)."""
    rows, cols = len(left) + 1, len(right) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + cost,
            )
            if (
                i > 1 and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                table[i][j] = min(table[i][j], table[i - 2][j - 2] + cost)
    return table[-1][-1]


def _choose_correction(
    token: str, previous: str, following: str, vocabulary: Set[str], bigrams: Set[str]
) -> Optional[str]:
    candidates = difflib.get_close_matches(token, vocabulary, n=6, cutoff=DEFAULT_TYPO_CUTOFF)
    scored = []
    for candidate in candidates:
        distance = _damerau_distance(token, candidate)
        limit = 1 if len(token) <= 7 else 2
        context = bool(
            (previous and f"{previous} {candidate}" in bigrams)
            or (following and f"{candidate} {following}" in bigrams)
        )
        if candidate == token:
            continue
        if len(candidate) > len(token):
            # An omitted letter (shampo -> shampoo) is clear on its own.
            allowed = distance <= limit
        else:
            # Same-length substitutions and deletions are ambiguous with real
            # words (scale -> scalp, tires -> tiles), so require catalog context.
            allowed = context and distance <= limit
        if allowed:
            scored.append((not context, distance, -len(candidate), candidate))
    if not scored:
        return None
    scored.sort()
    return scored[0][-1]


def _fix_typos(text: str) -> str:
    vocabulary, bigrams = _catalog_words()
    if not vocabulary:
        return text
    tokens = text.split()
    plain_tokens = [t.strip(".,!?;:\"'()[]{}").lower() for t in tokens]
    fixed: List[str] = []
    for index, token in enumerate(tokens):
        plain = plain_tokens[index]
        if (
            len(plain) < 3
            or plain in vocabulary
            or plain in _PRICE_WORDS
            or _NUMBER_RE.search(plain)
            or any(ch in plain for ch in "$€£")
        ):
            fixed.append(token)
            continue
        previous = plain_tokens[index - 1] if index > 0 else ""
        following = plain_tokens[index + 1] if index + 1 < len(tokens) else ""
        correction = _choose_correction(plain, previous, following, vocabulary, bigrams)
        fixed.append(correction or token)
    return " ".join(fixed)


@functools.lru_cache(maxsize=1024)
def normalize_query(text: str) -> str:
    """
    Return a search-ready form of ``text`` with filler and typos removed.

    The price part is preserved: only plain product words are corrected, and
    the price connectors in ``_PRICE_WORDS`` are never changed.
    """
    cleaned = _normalize_unicode(text)
    if not cleaned:
        return ""
    cleaned = _strip_filler(cleaned)
    fixes = 0
    while fixes < 2:
        corrected = _fix_typos(cleaned)
        if corrected == cleaned:
            break
        cleaned = corrected
        fixes += 1
    return re.sub(r"\s+", " ", cleaned).strip()
