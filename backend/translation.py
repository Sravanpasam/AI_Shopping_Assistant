"""
translation.py
--------------
Bilingual language detection and translation services (English and Arabic)
using langdetect and deep_translator.
"""

import re
import time
import logging
import threading
from collections import deque
from functools import lru_cache
from typing import Optional
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests

# Enforce deterministic results for langdetect
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Pre-compiled regex to identify Arabic unicode ranges
_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Back-off delays (seconds) for the two retries after the first attempt.
_RETRY_DELAYS = (1.0, 2.0)

# In-memory translation cache size. Repeated strings (the same product title,
# the same policy sentence) are translated once and then served from memory.
_CACHE_SIZE = 2000

# Global pacing for outgoing translation calls (shared by every thread).
_MAX_CALLS_PER_SEC = 3
_RATE_PERIOD = 1.0  # seconds


class _RateLimiter:
    """Allow at most ``max_calls`` translation requests per ``period`` seconds.

    Shared across all threads so that concurrent chat requests cannot burst the
    online translator and trip its rate limit.
    """

    def __init__(self, max_calls: int = _MAX_CALLS_PER_SEC, period: float = _RATE_PERIOD):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and (now - self._timestamps[0]) >= self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait = self.period - (now - self._timestamps[0])
            time.sleep(max(wait, 0.001))


_rate_limiter = _RateLimiter(max_calls=_MAX_CALLS_PER_SEC, period=_RATE_PERIOD)


class TranslationError(RuntimeError):
    """Raised when text cannot be translated (e.g. the service is rate-limited).

    Callers must treat this as an explicit failure and decide what to show the
    user. We never fall back to returning the untranslated text, which would
    silently present Arabic as if it were a finished English answer (or vice versa).
    """


def _translate_with_retry(text: str, source: str, target: str) -> str:
    """
    Translate ``text`` from ``source`` to ``target``.

    Outgoing requests are paced by the global rate limiter (3 per second) so
    concurrent chat requests cannot burst the online translator. Rate-limit
    responses (``TooManyRequests``) are then retried up to twice with exponential
    back-off (1s, then 2s). Any other error, or a rate limit that survives every
    retry, raises ``TranslationError``.
    """
    attempts = len(_RETRY_DELAYS) + 1
    for attempt in range(attempts):
        _rate_limiter.acquire()
        try:
            translator = GoogleTranslator(source=source, target=target)
            translated = translator.translate(text)
        except TooManyRequests as exc:
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                logger.warning(
                    "Translation rate-limited (%s->%s); retry %d/%d in %.1fs",
                    source, target, attempt + 1, len(_RETRY_DELAYS), delay,
                )
                time.sleep(delay)
                continue
            logger.error("Translation still rate-limited after %d attempts (%s->%s)", attempts, source, target)
            raise TranslationError("translation service rate-limited") from exc
        except Exception as exc:
            logger.error("Translation failed (%s->%s): %s", source, target, exc)
            raise TranslationError(str(exc)) from exc

        if translated and isinstance(translated, str) and translated.strip():
            return translated.strip()
        logger.error("Translation returned an empty result (%s->%s)", source, target)
        raise TranslationError("translation service returned an empty result")

    # Unreachable: the loop either returns or raises.
    raise TranslationError("translation failed")


def detect_language(text: str) -> str:
    """
    Detect the language of the given text using langdetect with Arabic heuristic fallback.

    Args:
        text: Input string to analyze.

    Returns:
        Two-letter ISO 639-1 language code (e.g., 'en', 'ar'). Defaults to 'en' on failure.
    """
    if not text or not text.strip():
        return "en"

    cleaned = text.strip()

    # Fast heuristic check: if Arabic script characters are present
    if _ARABIC_CHAR_PATTERN.search(cleaned):
        return "ar"

    try:
        lang = detect(cleaned)
        return str(lang).lower()
    except Exception as exc:
        logger.debug("Language detection failed for text '%s': %s. Defaulting to 'en'.", cleaned[:30], exc)
        return "en"


@lru_cache(maxsize=_CACHE_SIZE)
def translate_to_english(text: str) -> str:
    """
    Translate text from any language into English.

    Results are cached in memory (see ``_CACHE_SIZE``); a cache hit performs no
    network call. Failures are never cached.

    Args:
        text: Input string.

    Returns:
        English translation. Returns the text unchanged when it is already
        ASCII English (nothing to translate).

    Raises:
        TranslationError: if the text is not English and cannot be translated.
    """
    if not text or not text.strip():
        return text

    # Fast path if already ASCII English -- no network call needed.
    if all(ord(c) < 128 for c in text) and detect_language(text) == "en":
        return text

    return _translate_with_retry(text, source="auto", target="en")


@lru_cache(maxsize=_CACHE_SIZE)
def translate_to_arabic(text: str) -> str:
    """
    Translate text from English into Arabic.

    Results are cached in memory (see ``_CACHE_SIZE``); a cache hit performs no
    network call. Failures are never cached.

    Args:
        text: Input string (typically in English).

    Returns:
        Arabic translation.

    Raises:
        TranslationError: if the text cannot be translated.
    """
    if not text or not text.strip():
        return text

    return _translate_with_retry(text, source="en", target="ar")


# ---------------------------------------------------------------------------
# Health self-test (for /api/health)
# ---------------------------------------------------------------------------
_SELFTEST_TTL = 600.0  # seconds
_selftest_lock = threading.Lock()
_selftest_state = {"ts": None, "status": "unknown"}


def _run_selftest() -> str:
    """One live translation probe; returns 'ok' or 'degraded'. Never raises."""
    try:
        result = _translate_with_retry("ShopBot health check", source="en", target="ar")
        return "ok" if result and result.strip() else "degraded"
    except TranslationError:
        return "degraded"


def translation_status(force: bool = False) -> str:
    """
    Return 'ok' if the translation service is reachable, else 'degraded'.

    The result is cached for ``_SELFTEST_TTL`` seconds; a fresh probe runs only
    when the cache has expired or ``force`` is True. The probe intentionally
    bypasses the translation cache so it reflects the live service.
    """
    now = time.monotonic()
    with _selftest_lock:
        ts = _selftest_state["ts"]
        if not force and ts is not None and (now - ts) < _SELFTEST_TTL:
            return _selftest_state["status"]

    status = _run_selftest()

    with _selftest_lock:
        _selftest_state["ts"] = time.monotonic()
        _selftest_state["status"] = status
    return status
