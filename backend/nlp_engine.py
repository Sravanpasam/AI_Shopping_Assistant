"""
nlp_engine.py
-------------
Semantic intent classification and embedding generation using
SentenceTransformers (all-MiniLM-L6-v2).
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
import re
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from price_parser import parse_price_constraint

logger = logging.getLogger(__name__)

# Predefined intent classes with exemplar phrases
INTENTS: Dict[str, List[str]] = {
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "greetings",
        "سلام",
        "مرحبا",
    ],
    "search_product": [
        "show me products",
        "find products",
        "I want to buy",
        "looking for",
        "search for",
        "do you have",
        "get me",
    ],
    "vague_search": [
        "I need something for",
        "I want something that",
        "what should I use for",
        "help me find something",
        "suggest me",
        "recommend me",
        "I need a container",
        "I need a device",
    ],
    "compare_products": [
        "compare these",
        "difference between",
        "which is better",
        "compare A and B",
        "versus",
        "vs",
        "side by side",
        "contrast",
    ],
    "policy_warranty": [
        "is there warranty",
        "warranty information",
        "how long is warranty",
        "guarantee",
        "does it have warranty",
    ],
    "policy_cod": [
        "cash on delivery",
        "COD",
        "pay on delivery",
        "pay when delivered",
        "payment on delivery",
    ],
    "policy_return": [
        "can I return",
        "return policy",
        "exchange policy",
        "refund",
        "send back",
        "return this",
    ],
}

# Module-level lazy singleton variables
_model: Optional[SentenceTransformer] = None
_intent_embeddings: Optional[Dict[str, torch.Tensor]] = None


def get_model() -> SentenceTransformer:
    """
    Lazy-loaded singleton for the SentenceTransformer model.

    Returns:
        SentenceTransformer instance of 'all-MiniLM-L6-v2'.
    """
    global _model
    if _model is None:
        logger.info("Initializing SentenceTransformer('all-MiniLM-L6-v2') singleton...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer model loaded.")
    return _model


def _precompute_intent_embeddings() -> Dict[str, torch.Tensor]:
    """
    Precompute and cache tensor embeddings for all intent phrases.

    Returns:
        Dictionary mapping intent name to tensor of phrase embeddings.
    """
    global _intent_embeddings
    if _intent_embeddings is None:
        model = get_model()
        logger.info("Pre-computing intent embeddings for %d intents...", len(INTENTS))
        _intent_embeddings = {}
        for intent_name, phrases in INTENTS.items():
            _intent_embeddings[intent_name] = model.encode(
                phrases, convert_to_tensor=True, show_progress_bar=False
            )
        logger.info("Completed pre-computing all intent embeddings.")
    return _intent_embeddings


def get_embedding(text: str) -> np.ndarray:
    """
    Generate a 1D NumPy vector embedding for a single text input.

    Args:
        text: Input string.

    Returns:
        1D float32 NumPy array containing the text embedding.
    """
    if not text:
        text = ""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embedding, dtype=np.float32).flatten()


def classify_intent(text: str) -> Tuple[str, float]:
    """
    Classify the intent of user text by calculating cosine similarities against
    all predefined intent phrases and selecting the intent with the highest maximum similarity.

    Args:
        text: User query in English.

    Returns:
        Tuple of (intent_name, max_cosine_score).
    """
    if not text or not text.strip():
        return "greeting", 1.0

    try:
        model = get_model()
        query_embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        intent_embeddings = _precompute_intent_embeddings()

        best_intent = "search_product"
        best_score = -1.0

        for intent_name, phrase_tensor in intent_embeddings.items():
            # util.cos_sim returns shape (1, num_phrases)
            similarities = util.cos_sim(query_embedding, phrase_tensor)
            max_score_for_intent = float(torch.max(similarities).item())
            if max_score_for_intent > best_score:
                best_score = max_score_for_intent
                best_intent = intent_name

        return best_intent, float(best_score)
    except Exception as exc:
        logger.exception("Intent classification failed for text '%s': %s", text[:40], exc)
        return "search_product", 0.0


# ---------------------------------------------------------------------------
# Routing: price-parsing + intent + low-confidence handling
# ---------------------------------------------------------------------------

# Below this cosine score the intent is treated as unreliable.
CLARIFY_THRESHOLD = 0.5
# A "short noun phrase" is at most this many words and has no verb.
_SHORT_QUERY_MAX_WORDS = 4

# Verbs / auxiliaries / question words that mark a full sentence or request
# rather than a bare product phrase (e.g. "waterproof hiking boots").
_VERB_WORDS = {
    "be", "is", "are", "am", "was", "were", "been", "being",
    "do", "does", "did", "doing", "done",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "have", "has", "had", "having",
    "want", "wants", "wanted", "need", "needs", "needed", "like", "prefer",
    "look", "looking", "find", "show", "buy", "get", "give", "see",
    "help", "suggest", "recommend", "search",
    "return", "returns", "returned", "refund", "refunds", "exchange", "repay",
    "pay", "paid", "ship", "ships", "shipped", "deliver", "delivers", "delivered",
    "order", "orders", "ordered", "track", "tracking",
    "compare", "compares", "contrast", "versus", "vs",
    "guarantee", "guaranteed", "warranty",
    "keep", "keeps", "work", "works", "use", "using", "listen", "play",
    "how", "what", "when", "where", "why", "which", "who", "whose",
}


def is_short_noun_phrase(text: str) -> bool:
    """True for 1-4 word phrases with no verb (e.g. 'headphones', 'waterproof boots')."""
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    if not tokens or len(tokens) > _SHORT_QUERY_MAX_WORDS:
        return False
    return not any(token in _VERB_WORDS for token in tokens)


# Phrasing that signals the user is shopping for a product (so a low-confidence
# request should still go to product search, not a clarifying question).
_SHOPPING_REQUEST_REGEX = re.compile(
    r"\b(want|wants|wanted|need|needs|needed|"
    r"looking\s+for|look\s+for|search(?:ing)?\s+for|"
    r"find|show(?:\s+me)?|buy|purchase|recommend|suggest|"
    r"do\s+you\s+have)\b",
    re.IGNORECASE,
)


def is_shopping_request(text: str) -> bool:
    """True if the text asks for a product (want/need/show me/do you have...)."""
    return bool(_SHOPPING_REQUEST_REGEX.search(text or ""))


def route_message(text: str) -> Dict[str, Any]:
    """
    Decide how to handle a message before dispatch.

    1. Parse any price constraint first.
    2. Classify the *cleaned* query (price phrase removed).
    3. If a price constraint was found, route straight to product search.
    4. Otherwise, low-confidence short noun phrases and shopping requests
       (want/need/show me/do you have...) default to product search; other
       low-confidence messages route to a clarifying question.

    Returns a dict with: intent, confidence, classified_intent, cleaned_query,
    query, min_usd, max_usd, price_filter.
    """
    cleaned, min_usd, max_usd = parse_price_constraint(text)
    has_price = min_usd is not None or max_usd is not None

    # If the price phrase was the entire message, keep the raw text for the
    # embedding so classification still has something to work with.
    query = cleaned if cleaned and cleaned.strip() else text

    intent, confidence = classify_intent(query)

    if has_price:
        effective_intent = "search_product"
    elif confidence < CLARIFY_THRESHOLD:
        if is_short_noun_phrase(query) or is_shopping_request(query):
            effective_intent = "search_product"
        else:
            effective_intent = "clarify"
    else:
        effective_intent = intent

    return {
        "intent": effective_intent,
        "confidence": confidence,
        "classified_intent": intent,
        "cleaned_query": cleaned,
        "query": query,
        "min_usd": min_usd,
        "max_usd": max_usd,
        "price_filter": has_price,
    }


# Pre-compute embeddings for ALL phrases at module load time
try:
    _precompute_intent_embeddings()
except Exception as _init_exc:
    logger.warning("Intent embeddings could not be pre-computed at import time: %s", _init_exc)
