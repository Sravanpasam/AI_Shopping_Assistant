"""
adapters
--------
The store-adapter layer for System B.

``get_adapter()`` returns the process-wide adapter (a LocalFilesAdapter for the
active site by default). ``set_adapter()`` swaps it (used by tests and by any
future non-file backend).

This package is additive (System B). It does not touch System A or app.py.
"""

from typing import Optional

from .base import StoreAdapter
from .local_files import LocalFilesAdapter

_adapter: Optional[StoreAdapter] = None


def get_adapter() -> StoreAdapter:
    """Return the shared adapter, building the default one on first use."""
    global _adapter
    if _adapter is None:
        _adapter = LocalFilesAdapter()
    return _adapter


def set_adapter(adapter: StoreAdapter) -> None:
    """Replace the shared adapter (for tests or alternate backends)."""
    global _adapter
    _adapter = adapter


def reset_adapter() -> None:
    """Forget the shared adapter so the next get_adapter() rebuilds it."""
    global _adapter
    _adapter = None


__all__ = ["StoreAdapter", "LocalFilesAdapter", "get_adapter", "set_adapter", "reset_adapter"]
