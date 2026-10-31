"""
policy_qa.py
------------
Rule-based question answering for product policies:
- Warranty & Guarantee
- Cash on Delivery (COD) eligibility
- Return, Refund & Exchange policies
"""

import re
import logging
from typing import List, Optional
import pandas as pd

from data_loader import get_product_by_asin

logger = logging.getLogger(__name__)

# Regular expressions for policy extraction
# Warranty: the bare word "guaranteed" is a marketing adjective ("guaranteed hit",
# "guarantees its durability"), so require an explicit warranty term or a
# guarantee used in a policy context (money-back / satisfaction / against a defect).
_WARRANTY_REGEX = re.compile(
    r"\b(?:warrant(?:y|ies))"
    r"|(?:money[\s-]?back|satisfaction|manufacturer|quality|limited|lifetime|full)"
    r"\s+guarantee(?:s|d)?\b"
    r"|\bguarantee(?:s|d)?\s+(?:of|for|against|on|during)\b"
    r"|\b(?:year|month|day|period)[- ]guarantee\b",
    re.IGNORECASE,
)
_NO_WARRANTY_REGEX = re.compile(r"\b(no\s+warranty|not\s+covered|no\s+guarantee)\b", re.IGNORECASE)

_COD_REGEX = re.compile(r"\b(cash\s+on\s+delivery|cod)\b", re.IGNORECASE)
_NO_COD_REGEX = re.compile(r"\b(no\s+(cash\s+on\s+delivery|cod)|(cash\s+on\s+delivery|cod)\s+(not|un)\w*|not\s+available|not\s+accepted)\b", re.IGNORECASE)

# Return: bare "replacement" appears in spare-part descriptions ("Replacement foil and
# cutter block"), so only count it when it is phrased as a policy/coverage promise.
_RETURN_REGEX = re.compile(
    r"\b(?:return|returns|exchange|exchanges|refund|refunds|money-back)\b"
    r"|\breplacement\s+(?:service|policy|plan|for|within|at)\b"
    r"|\b\d{1,3}\s*[-\s]?\s*(?:day|week|month|year)[- ]replacement\b",
    re.IGNORECASE,
)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into clean, non-empty sentences.

    features_text/description_text are newline-joined when the catalog is built,
    so split on newlines first (each feature line is its own unit) and only then
    on sentence punctuation. Splitting on punctuation alone merges every feature
    into one blob, which produced false policy matches across unrelated lines.
    """
    if not text:
        return []
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def answer_policy(asin: str, policy_type: str) -> str:
    """
    Answer customer questions about specific product policies (warranty, cod, return).

    Args:
        asin: Product parent ASIN.
        policy_type: 'warranty', 'cod', or 'return'.

    Returns:
        Formatted policy answer string with emoji indicators (✅, ⚠️, ❌).
    """
    if not asin:
        return "⚠️ Please specify a valid product ASIN to check policies."

    normalized_type = (policy_type or "").strip().lower()

    try:
        product = get_product_by_asin(asin)
        if not product:
            logger.debug("Product with ASIN '%s' not found.", asin)
            return "⚠️ Product details could not be found for the specified ASIN."

        features = str(product.get("features_text", "")) if pd.notna(product.get("features_text")) else ""
        description = str(product.get("description_text", "")) if pd.notna(product.get("description_text")) else ""
        combined_text = f"{features}\n{description}".strip()

        sentences = _split_into_sentences(combined_text)

        # ── 1. WARRANTY POLICY ─────────────────────────────────────────────
        if normalized_type == "warranty":
            matches = [s for s in sentences if _WARRANTY_REGEX.search(s)]
            if not matches:
                return "⚠️ No specific manufacturer warranty information is mentioned in the product description."

            # Check if text explicitly says no warranty
            if any(_NO_WARRANTY_REGEX.search(s) for s in matches):
                return f"❌ Warranty not applicable: {' '.join(matches)}"

            return f"✅ Warranty details:\n{' '.join(matches)}"

        # ── 2. CASH ON DELIVERY (COD) ──────────────────────────────────────
        elif normalized_type == "cod":
            matches = [s for s in sentences if _COD_REGEX.search(s)]
            if not matches:
                return "⚠️ Cash on Delivery (COD) eligibility is not explicitly listed for this product. Check payment options at checkout."

            # Check if negation is present
            if any(_NO_COD_REGEX.search(s) for s in matches):
                return f"❌ Cash on Delivery is unavailable: {' '.join(matches)}"

            return f"✅ Cash on Delivery is available: {' '.join(matches)}"

        # ── 3. RETURN / EXCHANGE / REFUND POLICY ───────────────────────────
        elif normalized_type in ("return", "exchange", "refund"):
            matches = [s for s in sentences if _RETURN_REGEX.search(s)]
            if not matches:
                return "⚠️ No return, refund, or exchange information is available in this product's description."

            return f"✅ Return & Exchange Policy:\n{' '.join(matches)}"

        # ── UNKNOWN POLICY TYPE ────────────────────────────────────────────
        else:
            return f"⚠️ Unknown policy type '{policy_type}'. Supported policies are 'warranty', 'cod', and 'return'."

    except Exception as exc:
        logger.exception("Error checking policy '%s' for ASIN '%s': %s", policy_type, asin, exc)
        return f"⚠️ An error occurred while retrieving policy information: {exc}"
