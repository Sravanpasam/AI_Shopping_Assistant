"""Rule-based safety net used when every LLM model is unavailable.

The fallback reuses the same local data and tools as the agent (product search,
comparison, policy documents, orders) and fills fixed templates. It never calls
an LLM and never exposes provider errors.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent import build_action, build_suggestions
from agent_tools import (
    check_order,
    compare_products,
    get_product_details,
    search_policies,
    search_products,
)
from comparison import resolve_comparison_products
from nlp_engine import route_message
from orders import extract_order_id
from policy_qa import answer_policy
from translation import TranslationError, translate_to_arabic, translate_to_english

logger = logging.getLogger(__name__)

_POLICY_INTENTS = {
    "policy_warranty": "warranty",
    "policy_return": "return",
    "policy_cod": "payment",
}

_HELP_TOPICS = {
    "policy_return": "returns",
    "policy_warranty": "warranty",
    "policy_cod": "payment",
}

_ARABIC_RX = re.compile(r"[\u0600-\u06ff]")
_ARABIC_TRANSLATION_NOTE = (
    "_(تعذرت ترجمة التفاصيل الآن؛ أعرض النص الإنجليزي أدناه.)_"
)

_SHIPPING_WORDS = ("ship", "shipping", "deliver", "delivery", "arrive", "mars")
_PAYMENT_WORDS = ("cash on delivery", "cod", "payment", "pay", "card")
_SERVICE_WORDS = ("discount", "coupon", "promo", "student", "offer")
_CHEAPER_WORDS = ("cheaper", "lower price", "less expensive", "budget option")
_DETAIL_WORDS = (
    "tell me about", "more about", "details", "how much", "price", "rating",
    "ingredients", "features", "describe", "good for", "worth it", "review",
)
_POLICY_WORDS = (
    "warrant", "guarantee", "return", "refund", "exchange", "ship", "deliver",
    "payment", "cash on delivery", "cod", "order",
)


@dataclass
class FallbackResult:
    reply: str
    products: List[Dict[str, Any]] = field(default_factory=list)
    table: Optional[Dict[str, Any]] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    intent: str = "fallback"
    messages: List[Dict[str, Any]] = field(default_factory=list)


def _money(value: Any) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "price unavailable"


def _rating(value: Any) -> str:
    try:
        return f"{float(value):.1f} out of 5"
    except (TypeError, ValueError):
        return "rating unavailable"


def _record(state, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    state["tools"].append((name, result))
    state["trace"].append({
        "name": name,
        "args": args,
        "ok": isinstance(result, dict) and "error" not in result,
    })
    action = build_action(name, args, result) if isinstance(result, dict) else None
    if action is not None and action not in state["actions"]:
        state["actions"].append(action)


def _product_line(product: Dict[str, Any]) -> str:
    title = str(product.get("title") or "Untitled product")
    return (
        f"- {title} — {_money(product.get('price'))}, "
        f"{_rating(product.get('average_rating'))}"
    )


def _translate_reply(text: str, lang: str) -> str:
    if lang != "ar":
        return text
    try:
        return translate_to_arabic(text)
    except TranslationError:
        return text + "\n\n" + _ARABIC_TRANSLATION_NOTE


def _arabic_hint(message: str) -> str:
    if "ضمان" in message:
        return "warranty policy"
    if any(word in message for word in ("إرجاع", "ارجاع", "استرجاع", "استبدال")):
        return "return policy"
    if any(word in message for word in ("شحن", "توصيل", "التوصيل")):
        return "shipping policy"
    if any(word in message for word in ("دفع", "الدفع", "استلام")):
        return "cash on delivery payment policy"
    if any(word in message for word in ("طلب", "طلبي", "الطلب")):
        return "order tracking"
    return "store policy help"


def _english_message(message: str, lang: str) -> str:
    if lang != "ar":
        return message
    try:
        translated = translate_to_english(message)
        if translated and not _ARABIC_RX.search(translated):
            return translated
    except TranslationError:
        pass
    return _arabic_hint(message)


def _action_topic(intent: str) -> Optional[str]:
    return _HELP_TOPICS.get(intent)


def _policy_reply(query: str, state) -> str:
    result = search_policies(query)
    _record(state, "search_policies", {"query": query}, result)
    hits = result.get("policies") or []
    if not hits:
        return (
            "I couldn't find anything about that in the store policies. "
            "The available policy topics are returns, warranty, shipping, payment and order tracking."
        )
    lines = ["Here is what the store policy says:"]
    for hit in hits[:2]:
        source = str(hit.get("source") or "store policy")
        text = " ".join(str(hit.get("text") or "").split())
        if text:
            lines.append(f"\n{text}\nSource: {source}")
    if len(lines) == 1:
        return "I couldn't find a usable answer in the store policies."
    return "\n".join(lines)


def _order_reply(message: str, state) -> Optional[str]:
    order_id = extract_order_id(message)
    if not order_id:
        return None
    result = check_order(order_id)
    _record(state, "check_order", {"order_id": order_id}, result)
    if not result.get("found"):
        return f"I couldn't find an order with id {order_id} in the store records."
    order = result.get("order") or {}
    lines = [
        f"Order {order.get('order_id') or order_id} is currently "
        f"**{order.get('status') or 'status unavailable'}**."
    ]
    if order.get("estimated_delivery"):
        lines.append(f"Estimated delivery: {order['estimated_delivery']}.")
    if order.get("tracking_number"):
        lines.append(f"Tracking number: {order['tracking_number']}.")
    if "return" in (message or "").lower() or "refund" in (message or "").lower():
        asin = str(order.get("parent_asin") or "")
        if asin:
            product = get_product_details(asin)
            if product.get("found") and product.get("product"):
                state["products"].append(product["product"])
            policy = answer_policy(asin, "return")
            if policy and "No return" not in policy:
                lines.append(f"\nReturn eligibility from the product data: {policy}")
        hits = search_policies("return policy eligibility")
        _record(state, "search_policies", {"query": "return policy eligibility"}, hits)
        for hit in (hits.get("policies") or [])[:1]:
            text = " ".join(str(hit.get("text") or "").split())
            if text:
                lines.append(f"\nStore return policy: {text}\nSource: {hit.get('source') or 'store policy'}")
                break
    return "\n".join(lines)


def _warranty_for_viewed(message: str, context, state) -> Optional[str]:
    asin = ""
    if isinstance(context, dict):
        asin = str(context.get("asin") or "")
    if not asin:
        focused = state.get("focused_product") or {}
        asin = str(focused.get("parent_asin") or "")
    if not asin:
        return None
    product = get_product_details(asin)
    _record(state, "get_product_details", {"parent_asin": asin}, product)
    if not product.get("found"):
        return None
    found = product.get("product") or {}
    title = found.get("title") or asin
    policy = answer_policy(asin, "warranty")
    return f"For **{title}**:\n{policy}\nSource: product details ({asin})."


def _search_reply(message: str, route: Dict[str, Any], state) -> str:
    query = str(route.get("query") or message)
    args = {
        "query": query,
        "min_price_usd": route.get("min_usd"),
        "max_price_usd": route.get("max_usd"),
        "limit": 5,
    }
    result = search_products(
        query,
        min_price_usd=route.get("min_usd"),
        max_price_usd=route.get("max_usd"),
        limit=5,
    )
    _record(state, "search_products", args, result)
    state["last_query"] = query
    products = result.get("products") or []
    state["products"].extend(products)
    if not products:
        suggestions = result.get("category_suggestions") or []
        labels = [str(item.get("label") or item.get("value")) for item in suggestions]
        reply = "I couldn't find a product matching that description in the catalog."
        if labels:
            reply += " Available categories: " + ", ".join(labels) + "."
        return reply
    label = "Close matches" if result.get("approximate") else "I found these products"
    lines = [f"{label}:"]
    lines.extend(_product_line(product) for product in products)
    if route.get("min_usd") is not None or route.get("max_usd") is not None:
        lines.append(
            f"Applied price filter: {_money(route.get('min_usd')) if route.get('min_usd') is not None else 'no minimum'}"
            f" to {_money(route.get('max_usd')) if route.get('max_usd') is not None else 'no maximum'} (USD)."
        )
    return "\n".join(lines)


def _pick_product(message: str, state, context=None) -> Dict[str, Any]:
    products = state.get("last_products") or []
    text = (message or "").lower()
    if isinstance(context, dict) and context.get("asin") and "this" in text:
        for product in products:
            if str(product.get("parent_asin")) == str(context["asin"]):
                return product
        found = get_product_details(str(context["asin"]))
        if found.get("found") and found.get("product"):
            return found["product"]
    ordinals = (("second", 1), ("2nd", 1), ("third", 2), ("3rd", 2), ("first", 0), ("1st", 0))
    for word, index in ordinals:
        if word in text and index < len(products):
            return products[index]
    if "last one" in text and products:
        return products[-1]
    if products and any(word in text for word in _DETAIL_WORDS):
        return products[0]
    return {}


def _detail_reply(message: str, state, context=None) -> Optional[str]:
    product = _pick_product(message, state, context)
    asin = str(product.get("parent_asin") or "")
    if not asin:
        return None
    result = get_product_details(asin)
    _record(state, "get_product_details", {"parent_asin": asin}, result)
    found = result.get("product") if result.get("found") else product
    if not isinstance(found, dict) or not found:
        return None
    state["focused_product"] = found
    lines = [
        f"**{found.get('title') or asin}**",
        f"Price: {_money(found.get('price'))}",
        f"Rating: {_rating(found.get('average_rating'))}",
    ]
    store = str(found.get("store") or "").strip()
    if store:
        lines.append(f"Store: {store}")
    detail = str(found.get("features_text") or found.get("description_text") or "").strip()
    if detail:
        points = [line.strip(" -•\t") for line in detail.splitlines() if line.strip()]
        lines.append("What the listing says:")
        lines.extend(f"- {point[:220]}" for point in points[:3])
    return "\n".join(lines)


def _cheaper_reply(message: str, state) -> Optional[str]:
    if not any(word in (message or "").lower() for word in _CHEAPER_WORDS):
        return None
    products = state.get("last_products") or []
    if not products:
        return None
    prices = []
    for product in products:
        try:
            prices.append(float(product.get("price")))
        except (TypeError, ValueError):
            continue
    query = str(state.get("last_query") or "").strip()
    if not prices or not query:
        return "Tell me what product you are looking for and I will find cheaper options."
    cap = round(min(prices) - 0.01, 2)
    if cap <= 0:
        return "There is no lower price available for that search."
    result = search_products(query, max_price_usd=cap, limit=5)
    _record(state, "search_products", {"query": query, "max_price_usd": cap, "limit": 5}, result)
    found = result.get("products") or []
    if not found:
        return f"I couldn't find anything cheaper than {_money(min(prices))} for that search."
    state["products"].extend(found)
    lines = [f"Here are cheaper options under {_money(cap)}:"]
    lines.extend(_product_line(product) for product in found)
    return "\n".join(lines)


def _compare_reply(message: str, state) -> str:
    last_products = state.get("last_products") or []
    selected = resolve_comparison_products(message, last_products, 3)
    asins = [str(p.get("parent_asin") or "") for p in selected if p.get("parent_asin")]
    result = compare_products(asins)
    _record(state, "compare_products", {"parent_asins": asins}, result)
    if not result.get("compared"):
        return (
            "I need two or three products to compare. Search for them first, "
            "then say 'compare these', or name the products."
        )
    state["table"] = result.get("table")
    for product in selected:
        if product not in state["products"]:
            state["products"].append(product)
    return str(result.get("summary") or "Here is the comparison.")


def _compare_refers_to_previous(message: str) -> bool:
    text = (message or "").lower()
    return any(word in text for word in ("these", "them", "both", "first two", "the two"))


def fallback_turn(
    message: str, lang: str, session: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> FallbackResult:
    """Answer one turn with rules only. Never raises and never exposes internals."""
    session = session if isinstance(session, dict) else {}
    state: Dict[str, Any] = {
        "products": [],
        "table": None,
        "actions": [],
        "tools": [],
        "trace": [],
        "last_products": session.get("last_products") or [],
        "focused_product": session.get("focused_product") or {},
        "last_query": str(session.get("last_query") or ""),
    }
    english = _english_message(message, lang)
    try:
        order_reply = _order_reply(message, state)
        if order_reply:
            reply = order_reply
            intent = "order_status"
        else:
            lower = english.lower()
            asks_warranty = "warrant" in lower or "guarantee" in lower
            warranty_reply = (
                _warranty_for_viewed(english, context, state) if asks_warranty else None
            )
            route = route_message(english)
            intent = str(route.get("intent") or "clarify")
            if warranty_reply and intent == "policy_warranty":
                reply = warranty_reply
            elif intent == "compare_products" or (
                _compare_refers_to_previous(english) and state["last_products"]
            ):
                reply = _compare_reply(english, state)
                intent = "compare_products"
            elif (cheaper := _cheaper_reply(english, state)) is not None:
                reply = cheaper
                intent = "search_product"
            elif (
                any(word in lower for word in _DETAIL_WORDS)
                and state["last_products"]
                and not any(word in lower for word in _POLICY_WORDS)
                and (detail := _detail_reply(english, state, context)) is not None
            ):
                reply = detail
                intent = "product_detail"
            elif intent in _POLICY_INTENTS:
                reply = _policy_reply(english, state)
            elif any(word in lower for word in _SHIPPING_WORDS):
                reply = _policy_reply(english, state)
                intent = "policy_shipping"
            elif any(word in lower for word in _PAYMENT_WORDS):
                reply = _policy_reply(english, state)
                intent = "policy_payment"
            elif any(word in lower for word in _SERVICE_WORDS):
                reply = _policy_reply(english, state)
                intent = "policy_promotion"
            elif intent in ("search_product", "vague_search"):
                reply = _search_reply(english, route, state)
            elif intent == "greeting":
                reply = "Hello! I can find products, compare them, or answer store policy and order questions."
            else:
                reply = (
                    "I can help with product searches, comparisons, returns, warranty, "
                    "shipping, payment and orders. Please tell me which one you need."
                )
    except Exception as exc:
        logger.exception("Fallback turn failed safely: %s", exc)
        reply = (
            "I couldn't complete that request. Please try a product search, "
            "a policy question, or an order lookup."
        )
        intent = "fallback_error"

    products = state["products"]
    table = state["table"]
    if products:
        state["last_products"] = products
        state["focused_product"] = products[0]
    if state["last_products"]:
        session["last_products"] = state["last_products"][:8]
    if state.get("focused_product"):
        session["focused_product"] = state["focused_product"]
    if state.get("last_query"):
        session["last_query"] = state["last_query"]
    suggestions = build_suggestions(
        message, products, table, False, state["tools"]
    )
    translated = _translate_reply(reply, lang)
    if lang == "ar" and not any("\u0600" <= char <= "\u06ff" for char in translated):
        translated = _ARABIC_TRANSLATION_NOTE + "\n\n" + reply
    return FallbackResult(
        reply=translated,
        products=products,
        table=table,
        actions=state["actions"],
        suggestions=suggestions,
        tools_used=state["trace"],
        intent=intent,
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": translated},
        ],
    )
