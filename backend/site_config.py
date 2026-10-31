"""
site_config.py
--------------
Loads the per-site configuration from sites/<site_id>/site.json.

A "site" describes everything that makes ShopBot fit one store: its name, the
assistant's name and scope, the currency (and a secondary currency + rate), the
supported languages, the frontend hash-route templates and the paths to the
catalog, reviews, policy docs and orders database.

The active site is chosen by the SITE environment variable (or the --site
option on the server) and defaults to "shopstore". Loading is cached per site id.

This module is additive (System B). It does not touch System A or app.py.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES_DIR = os.path.join(ROOT_DIR, "sites")

DEFAULT_SITE_ID = "shopstore"

# Keys a site.json must define (non-empty).
REQUIRED_KEYS = (
    "site_id",
    "name",
    "assistant_name",
    "scope_description",
    "currency",
    "languages",
    "pages",
)

# Keys returned by the public /api/site endpoint (all non-sensitive).
PUBLIC_KEYS = (
    "site_id",
    "name",
    "assistant_name",
    "scope_description",
    "currency",
    "secondary_currency",
    "secondary_currency_rate",
    "languages",
    "pages",
)

_cache: Dict[str, Dict[str, Any]] = {}


class SiteConfigError(Exception):
    """Raised when a site config is missing, unreadable or invalid."""


def current_site_id() -> str:
    """The active site id (SITE env var, else the default)."""
    return (os.getenv("SITE") or DEFAULT_SITE_ID).strip() or DEFAULT_SITE_ID


def site_path(site_id: Optional[str] = None) -> str:
    """Absolute path of the site.json for a site id (default: the active one)."""
    return os.path.join(SITES_DIR, site_id or current_site_id(), "site.json")


def _validate(site: Any, site_id: str) -> Dict[str, Any]:
    """Check the required keys and value types; raise SiteConfigError."""
    if not isinstance(site, dict):
        raise SiteConfigError(f"site config for '{site_id}' must be a JSON object")

    missing = [key for key in REQUIRED_KEYS if not site.get(key)]
    if missing:
        raise SiteConfigError(
            f"site config for '{site_id}' is missing: {', '.join(missing)}"
        )
    if not isinstance(site["pages"], dict) or not site["pages"]:
        raise SiteConfigError(f"site config for '{site_id}' has an empty 'pages' map")
    if not isinstance(site["languages"], list) or not site["languages"]:
        raise SiteConfigError(f"site config for '{site_id}' has an empty 'languages' list")
    return site


def load_site(site_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate a site config (cached per site id).

    Raises:
        SiteConfigError: if the file is missing, unreadable or invalid.
    """
    sid = site_id or current_site_id()
    if sid in _cache:
        return _cache[sid]

    path = site_path(sid)
    if not os.path.exists(path):
        raise SiteConfigError(f"site config not found: {path}")

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteConfigError(f"could not read site config '{path}': {exc}") from exc

    site = _validate(data, sid)
    _cache[sid] = site
    logger.info("Loaded site config '%s' (%s)", sid, site.get("name"))
    return site


def load_site_safe(site_id: Optional[str] = None) -> Dict[str, Any]:
    """Like load_site but returns {} instead of raising (callers degrade safely)."""
    try:
        return load_site(site_id)
    except Exception as exc:  # noqa: BLE001 - config problems must not crash callers
        logger.warning("Could not load site config: %s", exc)
        return {}


def clear_cache() -> None:
    """Drop the cached site configs (used by tests and site switching)."""
    _cache.clear()


def public_site_info(site: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The non-sensitive parts of site.json served to the widget by /api/site."""
    site = site if site is not None else load_site()
    return {key: site[key] for key in PUBLIC_KEYS if key in site}


def resolve_path(site: Dict[str, Any], key: str) -> Optional[str]:
    """
    Resolve a path from the site's "paths" map to an absolute path.

    Relative paths are resolved against the repository root.
    """
    paths = (site or {}).get("paths") or {}
    value = paths.get(key)
    if not value:
        return None
    return value if os.path.isabs(value) else os.path.join(ROOT_DIR, value)


def site_languages(site: Optional[Dict[str, Any]] = None) -> List[str]:
    """The site's supported language codes (defaults to English)."""
    site = site if site is not None else load_site_safe()
    languages = site.get("languages") or ["en"]
    return [str(lang) for lang in languages]
