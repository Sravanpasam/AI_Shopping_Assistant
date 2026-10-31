"""Bilingual greeting detection so simple greetings never cost an LLM call.

A message counts as a greeting only when the *whole* message is a greeting
(optionally with punctuation or emoji) — "hello", "hi there", "مرحبا",
"السلام عليكم". Anything that also carries a request ("hi, find me a mascara")
is left for the agent, so we never swallow a real question.
"""

import re
import unicodedata

# Phrases that, on their own, are greetings (English, Arabic, transliterations).
_GREETING_PHRASES = frozenset({
    # English
    "hello", "hi", "hey", "hiya", "heya", "yo", "howdy",
    "hello there", "hi there", "hey there",
    "good morning", "good afternoon", "good evening", "good day",
    "greetings", "nice to meet you", "how are you", "how are you doing",
    # Arabic
    "مرحبا", "مرحبًا", "اهلا", "أهلا", "هلا", "السلام عليكم",
    "صباح الخير", "مساء الخير", "اهلا وسهلا", "أهلا وسهلا", "كيف حالك",
    # transliterations
    "salam", "assalamu alaikum", "as-salamu alaykum",
    "marhaba", "ahlan", "ahlan wa sahlan",
})

# Trailing punctuation / emoji that should not stop a greeting from matching.
_TRAILING = re.compile(r"[\s!.,?؟،…😊👋🙂🤖✨]+$")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    text = _TRAILING.sub("", text)
    return " ".join(text.split())


def is_greeting(text: str, max_words: int = 4) -> bool:
    """True when the whole message is just a greeting (no request inside)."""
    norm = _normalize(text)
    if not norm or len(norm.split()) > max_words:
        return False
    return norm in _GREETING_PHRASES
