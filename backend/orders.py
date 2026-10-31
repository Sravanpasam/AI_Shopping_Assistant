"""
orders.py
---------
Read-only order lookup for the chatbot's order-status tool.

- extract_order_id(text): pull an order id (ORD-12345) out of a message.
- get_order(order_id): return the order row from data/orders.db, or None.

Structured data or None — this module never invents values. If an order is not
in the database it returns None so the caller can say the order was not found.

This module is additive (System B). It does not touch System A or app.py.
"""

import os
import re
import sqlite3
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
DB_PATH = os.path.join(DATA_DIR, "orders.db")

# Order ids look like ORD-10025.
_ORDER_ID_REGEX = re.compile(r"\bORD-\d+\b", re.IGNORECASE)

# Columns returned from the orders table (kept explicit so the shape is stable).
FIELDS = (
    "order_id",
    "customer_name",
    "parent_asin",
    "quantity",
    "status",
    "order_date",
    "estimated_delivery",
    "delivered_date",
    "city",
    "payment_method",
    "tracking_number",
)


def extract_order_id(text: str) -> Optional[str]:
    """
    Return the first order id found in the text (uppercased), or None.

    Case-insensitive: "ord-10025" and "ORD-10025" both return "ORD-10025".
    """
    if not text:
        return None
    match = _ORDER_ID_REGEX.search(text)
    return match.group(0).upper() if match else None


def get_order(order_id: str) -> Optional[Dict[str, Any]]:
    """
    Look up a single order by id in data/orders.db.

    Args:
        order_id: The order id, e.g. "ORD-10025" (case-insensitive).

    Returns:
        A dict of the order's fields, or None if the id is blank, the database
        is missing, or no such order exists. Never fabricates a result.
    """
    if not order_id or not order_id.strip():
        return None

    normalized = order_id.strip().upper()

    if not os.path.exists(DB_PATH):
        logger.warning("Orders database not found at %s", DB_PATH)
        return None

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (normalized,)
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None
    return {field: row[field] for field in FIELDS}
