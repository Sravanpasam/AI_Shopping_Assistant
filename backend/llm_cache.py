"""A small, opt-in disk cache for agent responses (for evaluation runs).

Repeated evaluation runs should not spend LLM quota re-answering the same
questions. When ``LLM_CACHE`` is on, /api/chat stores each response keyed by the
exact ``(message, context, history)`` and replays it on the next identical call.

Off by default: the cache only activates when ``LLM_CACHE`` is truthy, so normal
use and the unit tests are unaffected. The store is a single JSON file
(``LLM_CACHE_PATH`` or ``data/llm_cache.json``) which is gitignored.
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm_cache.json"
)

_lock = threading.RLock()
_loaded = False
_store: Dict[str, Any] = {}


def enabled() -> bool:
    """Whether the cache is switched on (``LLM_CACHE=1`` / true / yes)."""
    return (os.getenv("LLM_CACHE", "") or "").strip().lower() in ("1", "true", "yes", "on")


def path() -> str:
    """The cache file path (``LLM_CACHE_PATH`` or the default)."""
    return os.getenv("LLM_CACHE_PATH") or _DEFAULT_PATH


def _load() -> None:
    global _loaded, _store
    if _loaded:
        return
    _loaded = True
    try:
        with open(path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            _store = data
    except FileNotFoundError:
        _store = {}
    except Exception as exc:  # a corrupt cache must never break a request
        logger.warning("Ignoring unreadable LLM cache (%s): %s", path(), exc)
        _store = {}


def _save() -> None:
    target = path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_store, handle, ensure_ascii=False)
        os.replace(tmp, target)
    except Exception as exc:  # pragma: no cover - best-effort persistence
        logger.warning("Could not write LLM cache (%s): %s", target, exc)


def make_key(message: Any, context: Any = None, history: Any = None) -> str:
    """A stable key for one request (message + page context + conversation)."""
    blob = json.dumps(
        {"message": message, "context": context, "history": history},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(key: str) -> Optional[Dict[str, Any]]:
    with _lock:
        _load()
        value = _store.get(key)
        return dict(value) if isinstance(value, dict) else None


def put(key: str, value: Dict[str, Any]) -> None:
    with _lock:
        _load()
        _store[key] = value
        _save()


def clear() -> None:
    """Forget everything (also removes the file)."""
    global _store
    with _lock:
        _loaded = True
        _store = {}
        try:
            os.remove(path())
        except FileNotFoundError:
            pass


def stats() -> Tuple[int, str]:
    """``(entry_count, path)`` — for health/inspection."""
    with _lock:
        _load()
        return len(_store), path()
