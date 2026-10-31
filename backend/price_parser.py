"""
price_parser.py
---------------
Pure helper to extract a price constraint from a natural-language query.

``parse_price_constraint(text)`` returns ``(cleaned_query, min_usd, max_usd)``:

* ``cleaned_query`` is the original text with the price phrase removed.
* ``min_usd`` / ``max_usd`` are the bounds normalised to USD (or ``None``).

Supported phrasings:
  * max: under, below, less than, up to, no more than, max, cheaper than
  * min: over, above, at least, more than, greater than, from, starting at
  * range: "between X and Y" (always), "X to Y" / "X-Y" / "from X to Y"
    (the bare forms require a price signal -- a currency token or a price word
    such as price / budget / cost / range -- so "2 to 3 products" is ignored)
  * currency: USD by default; AED / dirham / dirhams / dhs divided by 3.67

Guards:
  * a number glued to a non-price unit (ml, g, kg, oz, gb, tb, inch, cm, pack,
    pcs, pieces, minutes, years, spf) is not treated as a price, unless a
    currency word is present (e.g. "under 5 minutes" is not a price).
  * a hyphen inside a single alphanumeric token (e.g. "WH-1000-XM5") is never
    treated as a range, because numbers must sit on word boundaries.
"""

import re
from typing import Optional, Tuple

# Fixed conversion rate (project decision): 1 USD = 3.67 AED.
AED_PER_USD = 3.67

_NUM = r"\d[\d,]*(?:\.\d+)?"
_CURR = r"(?:aed|usd|dirhams?|dhs|dollars?|\$)"
# Money must sit on a word boundary so digits inside "WH-1000-XM5" are ignored.
_MONEY = rf"(?<![A-Za-z0-9])(?:{_CURR}\s*)?{_NUM}(?:\s*{_CURR})?(?![A-Za-z0-9])"

_MAX_KW = (
    r"(?:under|below|less\s+than|up\s*to|no\s+more\s+than|not\s+more\s+than"
    r"|max(?:imum)?|cheaper\s+than)"
)
_MIN_KW = (
    r"(?:over|above|at\s+least|more\s+than|greater\s+than|from"
    r"|starting\s+(?:at|from))"
)

_RANGE_BETWEEN_RE = re.compile(
    rf"\bbetween\s+({_MONEY})\s+(?:and|to)\s+({_MONEY})", re.IGNORECASE
)
_RANGE_TO_RE = re.compile(
    rf"\b(?:from\s+)?({_MONEY})\s*(?:to|\-|\u2013|\u2014)\s*({_MONEY})", re.IGNORECASE
)
_MAX_RE = re.compile(rf"\b{_MAX_KW}\b\s+(?:price\s+|cost\s+)?({_MONEY})", re.IGNORECASE)
_MIN_RE = re.compile(rf"\b{_MIN_KW}\b\s+(?:price\s+|cost\s+)?({_MONEY})", re.IGNORECASE)

_AED_DETECT = re.compile(r"aed|dirhams?|dhs", re.IGNORECASE)
_NUM_DETECT = re.compile(_NUM)

# A currency token marks a genuine price.
_CURRENCY_DETECT = re.compile(r"\$|usd|aed|dirhams?|dhs|dollars?", re.IGNORECASE)
# A currency token OR a price word counts as a price "signal" (used for ranges).
_PRICE_SIGNAL = re.compile(
    r"\$|usd|aed|dirhams?|dhs|dollars?|price|budget|cost|range", re.IGNORECASE
)

# Non-price units: a number bound to one of these is a measurement, not a price.
_UNIT = r"(?:ml|kg|gb|tb|oz|inch|cm|pack|pcs|pieces|minutes?|years?|spf|g)"
_UNIT_AFTER = re.compile(rf"^\s*{_UNIT}\b", re.IGNORECASE)
_UNIT_BEFORE = re.compile(rf"\b{_UNIT}\s*$", re.IGNORECASE)
_UNIT_BEFORE_OF = re.compile(rf"\b{_UNIT}\s+of\s*$", re.IGNORECASE)


def _money_to_usd(token: str, aed: Optional[bool] = None) -> Optional[float]:
    """
    Convert a matched money token (e.g. '$100', '100 dirhams') to USD.

    ``aed`` overrides currency detection; it is needed for ranges where a single
    currency word (e.g. "between 367 and 734 dirhams") applies to both bounds but
    only appears next to the second one.
    """
    match = _NUM_DETECT.search(token)
    if not match:
        return None
    value = float(match.group(0).replace(",", ""))
    if aed is None:
        aed = bool(_AED_DETECT.search(token))
    if aed:
        value = value / AED_PER_USD
    return round(value, 2)


def _is_price_context(text: str, start: int, end: int) -> bool:
    """
    Decide whether the number at text[start:end] is a price rather than a
    measurement. A currency token nearby wins; otherwise a bound unit rejects it.
    """
    window = text[max(0, start - 20): min(len(text), end + 20)]
    if _CURRENCY_DETECT.search(window):
        return True

    after = text[end: end + 15]
    before = text[max(0, start - 15): start]
    if _UNIT_AFTER.search(after):
        return False
    if _UNIT_BEFORE.search(before) or _UNIT_BEFORE_OF.search(before):
        return False
    return True


def _range_has_price_signal(text: str, start: int, end: int, match) -> bool:
    """
    Bare ranges ("X to Y", "X-Y") are only accepted with a price signal:
    a currency token, a price word, or an explicit "from X to Y" phrasing.
    """
    if match.group(0).strip().lower().startswith("from"):
        return True
    window = text[max(0, start - 25): min(len(text), end + 25)]
    return bool(_PRICE_SIGNAL.search(window))


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _remove_spans(text: str, spans) -> str:
    """Remove the given (start, end) spans from text and tidy the remainder."""
    cleaned = text
    for start, end in sorted(spans, reverse=True):
        cleaned = f"{cleaned[:start]} {cleaned[end:]}"
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" ,.;:-")
    cleaned = re.sub(
        r"\b(for|from|than|of|price|cost)\s*$", "", cleaned, flags=re.IGNORECASE
    ).strip()
    return cleaned


def parse_price_constraint(text: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Parse an optional price constraint out of a query.

    Returns:
        (cleaned_query, min_usd, max_usd). If no price phrase is present,
        returns (text, None, None) with the text unchanged.
    """
    if not text or not text.strip():
        return text, None, None

    # Range: "between X and Y"  (accepted as-is)
    match = _RANGE_BETWEEN_RE.search(text)
    if match:
        aed = bool(_AED_DETECT.search(match.group(0)))
        lo = _money_to_usd(match.group(1), aed)
        hi = _money_to_usd(match.group(2), aed)
        if lo is not None and hi is not None:
            low, high = sorted((lo, hi))
            return _remove_spans(text, [match.span()]), low, high

    # Range: "X to Y", "X-Y", "from X to Y"  (bare forms need a price signal)
    match = _RANGE_TO_RE.search(text)
    if match and _range_has_price_signal(text, *match.span(), match):
        aed = bool(_AED_DETECT.search(match.group(0)))
        lo = _money_to_usd(match.group(1), aed)
        hi = _money_to_usd(match.group(2), aed)
        if lo is not None and hi is not None:
            low, high = sorted((lo, hi))
            return _remove_spans(text, [match.span()]), low, high

    max_val = None
    max_span = None
    max_match = _MAX_RE.search(text)
    if max_match and _is_price_context(text, *max_match.span(1)):
        max_val = _money_to_usd(max_match.group(1))
        max_span = max_match.span()

    min_val = None
    min_span = None
    min_match = _MIN_RE.search(text)
    if min_match and not (max_span and _overlaps(min_match.span(), max_span)):
        if _is_price_context(text, *min_match.span(1)):
            min_val = _money_to_usd(min_match.group(1))
            min_span = min_match.span()

    spans = [span for span in (max_span, min_span) if span]
    if not spans:
        return text, None, None

    return _remove_spans(text, spans), min_val, max_val
