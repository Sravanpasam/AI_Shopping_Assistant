"""
adapters/base.py
----------------
The data-access interface a ShopBot site must provide.

A StoreAdapter hides "where the data lives" from the rest of the assistant: the
tools in agent_tools.py only ever talk to an adapter, so a different store can be
supported by writing another adapter (a database, an API, a different folder)
without changing the agent, the prompt or the HTTP layer.

All methods return plain JSON-safe Python values (dicts/lists/None); adapters may
raise on hard failures - callers wrap them (the tool layer never raises).

This module is additive (System B). It does not touch System A or app.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StoreAdapter(ABC):
    """Data access for one store (catalog, reviews, knowledge, orders)."""

    @property
    @abstractmethod
    def site_info(self) -> Dict[str, Any]:
        """The loaded site config (see site_config.py)."""
        raise NotImplementedError

    @abstractmethod
    def search_products(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Products semantically matching a natural-language query (English)."""
        raise NotImplementedError

    @abstractmethod
    def get_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """One product record by parent_asin, or None."""
        raise NotImplementedError

    @abstractmethod
    def get_reviews(self, asin: str) -> List[Dict[str, Any]]:
        """Customer reviews for a product id (may be empty)."""
        raise NotImplementedError

    @abstractmethod
    def search_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Store knowledge (policy docs) most relevant to a query."""
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """One order record by id, or None. Never fabricates."""
        raise NotImplementedError
