"""
escalation.py
-------------
Decide when a chat message should be escalated to a human agent.

- detect_escalation(text) -> (bool, reason)

Two triggers:
  1. The user explicitly asks for a human ("talk to a real person", ...).
  2. Strong frustration or repeated complaints ("this is useless", "I already
     told you", "still not working", ...).

Rule-based patterns handle the common phrasings (English and Arabic script);
a similarity check against example phrases catches paraphrases. The similarity
threshold is ESCALATION_SIMILARITY.

This module is additive (System B). It does not touch System A or app.py.
"""

import re
import logging
from typing import List, Optional, Tuple

import numpy as np

from nlp_engine import get_embedding

logger = logging.getLogger(__name__)

# Cosine similarity above which a paraphrase counts as an escalation phrase.
# Chosen from observed scores: paraphrased requests land ~0.54 (human) / ~0.81
# (frustration), while ordinary policy/product/help messages stay <= 0.46.
ESCALATION_SIMILARITY = 0.5

# ── Explicit requests for a human ───────────────────────────────────────────
_HUMAN_REGEX = re.compile(
    r"\b(?:talk|speak|chat)\s+(?:to|with)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:human|real\s+person|person|someone|somebody|agent|representative|manager|support|advisor)\b"
    r"|\b(?:human|real\s+person|live\s+agent|human\s+agent|representative|customer\s+(?:service|support))\b"
    r"|\bconnect\s+me\s+(?:to|with)\b"
    r"|\bget\s+me\s+(?:a\s+)?(?:human|manager|agent)\b"
    r"|\btransfer\s+me\s+to\b",
    re.IGNORECASE,
)

_AR_HUMAN_REGEX = re.compile(
    r"شخص\s+حقيقي|إنسان|التحدث\s+مع|أريد\s+التحدث|خدمة\s+العملاء|موظف|ممثل\s+خدمة|مدير"
)

# ── Strong frustration / repeated complaints ────────────────────────────────
_FRUSTRATION_REGEX = re.compile(
    r"\b(?:this|that|it|you)\s*(?:'s|is|are|re)?\s*"
    r"(?:useless|ridiculous|unacceptable|awful|terrible|horrible|a\s+joke|the\s+worst)\b"
    r"|\bi(?:'m|\s+am)\s+(?:so\s+|really\s+|very\s+)?(?:frustrated|angry|fed\s+up|annoyed|upset)\b"
    r"|\bfed\s+up\b|\bsick\s+and\s+tired\b"
    r"|\bi\s+(?:already|just)\s+told\s+you\b|\bi\s+said\s+(?:this|that)\s+(?:already|before)\b"
    r"|\b(?:stop\s+repeating|you\s+keep\s+(?:saying|asking|repeating)|same\s+(?:answer|thing)\s+again)\b"
    r"|\bnothing\s+(?:works|is\s+working)\b"
    r"|\bstill\s+(?:not\s+working|doesn'?t\s+work|won'?t\s+work|broken|wrong)\b"
    r"|\byou\s+(?:are|'re)\s+not\s+helping\b",
    re.IGNORECASE,
)

_AR_FRUSTRATION_REGEX = re.compile(
    r"غير\s+مقبول|لست\s+راض|غاضب|تعبت|هذا\s+سيء|لا\s+يفيد|لم\s+أحصل\s+على\s+حل"
)

# Example phrases for the similarity fallback.
_HUMAN_PHRASES: List[str] = [
    "talk to a human",
    "speak to a real person",
    "connect me to a human agent",
    "I want to talk to someone who can help",
    "put me through to a representative",
    "let me speak to a manager",
    "customer support please",
    "I need to talk to a real human being",
    "أريد التحدث مع شخص حقيقي",
    "خدمة العملاء",
]

_FRUSTRATION_PHRASES: List[str] = [
    "this is useless",
    "this is completely unacceptable",
    "I am very frustrated with this",
    "you are not helping me at all",
    "I already told you this",
    "nothing works and you keep saying the same thing",
    "I am fed up with this",
    "this is the worst service ever",
    "لم أحصل على حل وأنا غاضب",
    "هذا غير مقبول",
]

_ARABIC_RX = re.compile(r"[\u0600-\u06ff]")

_human_matrix: Optional[np.ndarray] = None
_frustration_matrix: Optional[np.ndarray] = None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def _build_matrices() -> None:
    global _human_matrix, _frustration_matrix
    if _human_matrix is None:
        _human_matrix = _normalize(np.vstack([get_embedding(p) for p in _HUMAN_PHRASES]))
    if _frustration_matrix is None:
        _frustration_matrix = _normalize(
            np.vstack([get_embedding(p) for p in _FRUSTRATION_PHRASES])
        )


def _max_similarity(text: str, matrix: np.ndarray) -> float:
    vec = _normalize(get_embedding(text)[None, :])[0]
    return float(np.max(matrix @ vec))


def detect_escalation(text: str) -> Tuple[bool, Optional[str]]:
    """
    Decide whether a message should be escalated to a human agent.

    Args:
        text: The user's message (English or Arabic; Arabic script is matched
              directly and Arabic is also assumed to be translated upstream).

    Returns:
        (True, reason) where reason is "explicit_human_request" or
        "frustration"; (False, None) when no escalation is warranted.
    """
    if not text or not text.strip():
        return False, None

    if _HUMAN_REGEX.search(text) or _AR_HUMAN_REGEX.search(text):
        return True, "explicit_human_request"

    if _FRUSTRATION_REGEX.search(text) or _AR_FRUSTRATION_REGEX.search(text):
        return True, "frustration"

    # The embedding model is English-centric: Arabic policy questions score
    # above the threshold against every example. Arabic explicit and
    # frustration phrasings are handled by the regexes above.
    if _ARABIC_RX.search(text):
        return False, None

    # Similarity fallback for paraphrases.
    try:
        _build_matrices()
        if _max_similarity(text, _human_matrix) >= ESCALATION_SIMILARITY:
            return True, "explicit_human_request"
        if _max_similarity(text, _frustration_matrix) >= ESCALATION_SIMILARITY:
            return True, "frustration"
    except Exception as exc:  # never let escalation detection crash the caller
        logger.warning("Escalation similarity check failed: %s", exc)

    return False, None
