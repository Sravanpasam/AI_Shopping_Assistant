"""
adapters/local_files.py
-----------------------
StoreAdapter backed by the local files of this project: the parquet catalog and
reviews (data_loader), the FAISS index (product_search), the policy docs
(policy_search) and the sqlite orders database (orders).

The site's "paths" block in site.json is applied to those modules, so a site can
point at different files. For the default ("shopstore") site the paths equal the
modules' built-in defaults, so this is a no-op and behaviour is unchanged.

This module is additive (System B). It does not touch System A or app.py.
"""

import logging
from typing import Any, Dict, List, Optional

from site_config import load_site_safe, resolve_path

from .base import StoreAdapter

logger = logging.getLogger(__name__)


class LocalFilesAdapter(StoreAdapter):
    """Read from the local catalog / docs / orders files described by site.json."""

    def __init__(self, site: Optional[Dict[str, Any]] = None):
        self._site = site if site is not None else load_site_safe()
        self._apply_paths()

    @property
    def site_info(self) -> Dict[str, Any]:
        return self._site

    def search_products(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        from product_search import search_products as _search
        return _search(query, top_k=top_k)

    def get_product(self, asin: str) -> Optional[Dict[str, Any]]:
        from data_loader import get_product_by_asin
        return get_product_by_asin(asin)

    def get_reviews(self, asin: str) -> List[Dict[str, Any]]:
        from data_loader import get_reviews_for_product
        return get_reviews_for_product(asin)

    def search_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        from policy_search import search_policies
        return search_policies(query, k=k)

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        from orders import get_order as _get_order
        return _get_order(order_id)

    def _apply_paths(self) -> None:
        """
        Point the data modules at the site's configured files.

        Only changes a module when the resolved path actually differs from its
        current setting, so the default site (which matches the defaults) is a
        no-op and existing behaviour/tests are untouched.
        """
        site = self._site or {}
        try:
            catalog = resolve_path(site, "catalog")
            reviews = resolve_path(site, "reviews")
            if catalog or reviews:
                import data_loader
                if catalog and catalog != data_loader.CATALOG_PATH:
                    data_loader.CATALOG_PATH = catalog
                    data_loader._catalog_df = None
                if reviews and reviews != data_loader.REVIEWS_PATH:
                    data_loader.REVIEWS_PATH = reviews
                    data_loader._reviews_df = None

            docs = resolve_path(site, "docs")
            if docs:
                import policy_search
                if docs != policy_search.DOCS_DIR:
                    policy_search.DOCS_DIR = docs
                    policy_search._chunks = None
                    policy_search._chunk_matrix = None

            orders_db = resolve_path(site, "orders_db")
            if orders_db:
                import orders
                if orders_db != orders.DB_PATH:
                    orders.DB_PATH = orders_db
        except Exception as exc:  # noqa: BLE001 - path setup must not crash tools
            logger.warning("Could not apply site paths: %s", exc)
