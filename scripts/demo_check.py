"""Run ten representative demo messages against ShopBot and print pass/fail.

Normal use (start the server first):
    .venv\\Scripts\\python.exe scripts/demo_check.py --port 5000

No LLM or API key (starts a local server that forces limited mode):
    .venv\\Scripts\\python.exe scripts/demo_check.py --mock --port 5099
"""

import argparse
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ARABIC_RX = re.compile(r"[\u0600-\u06ff]")
NEGATIVE_RX = re.compile(
    r"\b(not|no|cannot|can't|can not|unable|don't|do not|doesn't|only|outside|unavailable)\b",
    re.I,
)
RAW_ERROR_RX = re.compile(r"traceback|exception|api[ _-]?key|rate limit|quota", re.I)


class FailingClient:
    def generate(self, messages, tools=None, system=None):
        from llm_client import LLMRateLimitError
        raise LLMRateLimitError("demo_check mock: all models unavailable")


def start_mock_server(port):
    import app as shop
    from agent import ShopBotAgent

    shop._models_ready = True
    shop._agent = ShopBotAgent(client=FailingClient())
    shop._limited_mode = False
    shop.sessions.clear()
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", port, shop.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    return server


def post(base_url, message, session_id, context=None, timeout=180):
    payload = {"message": message, "session_id": session_id}
    if context:
        payload["context"] = context
    started = time.time()
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    body = response.json()
    return body, round(time.time() - started, 2)


def safe(body):
    text = str(body.get("text") or "")
    return (
        bool(text.strip())
        and not body.get("error")
        and not RAW_ERROR_RX.search(text)
    )


def has_products(body):
    return bool(body.get("products")) or any(
        action.get("page") == "search" for action in (body.get("actions") or [])
    )


def check_search(body):
    return safe(body) and (has_products(body) or "couldn't find" in body.get("text", "").lower())


def check_warranty(body):
    text = body.get("text", "").lower()
    return safe(body) and any(word in text for word in ("warrant", "guarantee", "not mentioned", "no specific"))


def check_compare(body):
    table = (body.get("comparison") or {}).get("table") or {}
    return safe(body) and (bool(table.get("headers")) or body.get("response_type") == "comparison")


def check_return_source(body):
    text = body.get("text", "").lower()
    return safe(body) and ("source" in text or "returns_and_refunds" in text or "docs/" in text)


def check_order_return(body):
    text = body.get("text", "").lower()
    return (
        safe(body)
        and "ord-10025" in text
        and any(word in text for word in ("processing", "shipped", "delivered", "status"))
        and "return" in text
    )


def check_arabic(body):
    return safe(body) and bool(ARABIC_RX.search(body.get("text", "")))


def check_mars(body):
    text = body.get("text", "").lower()
    claimed = re.search(r"\b(yes|we|sure).{0,30}ship.{0,20}mars\b", text, re.I)
    mentions = "mars" in text
    negative = bool(NEGATIVE_RX.search(text)) or "uae" in text
    return safe(body) and not claimed and (mentions or negative)


def check_student(body):
    text = body.get("text", "").lower()
    if "discount" not in text:
        return safe(body)
    return safe(body) and (
        bool(NEGATIVE_RX.search(text))
        or "couldn't find" in text
        or "no information" in text
    )


def check_human(body):
    text = body.get("text", "").lower()
    return safe(body) and (
        bool(body.get("escalated"))
        or any(word in text for word in ("human", "agent", "support", "representative"))
    )


def check_cod(body):
    text = body.get("text", "").lower()
    return safe(body) and any(word in text for word in ("cash on delivery", "cod", "payment"))


CASES = [
    {"name": "natural search", "message": "I need a moisturizer for dry skin under $25", "check": check_search},
    {
        "name": "warranty of viewed product",
        "setup": "find me a mascara",
        "message": "Does this product have a warranty?",
        "viewed": True,
        "check": check_warranty,
    },
    {
        "name": "compare",
        "setup": "find me two mascaras",
        "message": "compare these two",
        "check": check_compare,
    },
    {"name": "return policy with source", "message": "What is your return policy?", "check": check_return_source},
    {
        "name": "order status plus return eligibility",
        "message": "What is the status of ORD-10025 and can I return it?",
        "check": check_order_return,
    },
    {"name": "Arabic question", "message": "ما هي سياسة الإرجاع؟", "check": check_arabic},
    {"name": "ship to Mars", "message": "Do you ship to Mars?", "check": check_mars},
    {"name": "student discount", "message": "Do you offer a student discount?", "check": check_student},
    {"name": "human handoff", "message": "I want to talk to a human", "check": check_human},
    {"name": "cash on delivery", "message": "Is cash on delivery available?", "check": check_cod},
]


def main():
    parser = argparse.ArgumentParser(description="ShopBot demo safety check")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--mock", action="store_true", help="run without the LLM (limited mode)")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    server = None
    if args.mock:
        server = start_mock_server(args.port)
        print(f"MOCK MODE: limited-mode server on {base_url} (no LLM calls)\n")
    else:
        try:
            requests.get(f"{base_url}/api/health", timeout=10)
        except Exception:
            print(f"Cannot reach {base_url}. Start the server or use --mock.")
            return 2

    passed = 0
    try:
        for index, case in enumerate(CASES, start=1):
            session_id = f"demo-{uuid.uuid4().hex[:8]}"
            context = None
            try:
                if case.get("setup"):
                    setup_body, _ = post(base_url, case["setup"], session_id, timeout=args.timeout)
                    products = setup_body.get("products") or []
                    if case.get("viewed") and products:
                        context = {"page": "product", "asin": products[0].get("parent_asin")}
                body, latency = post(
                    base_url, case["message"], session_id, context, timeout=args.timeout
                )
                ok = bool(case["check"](body))
                mode = "limited" if body.get("degraded") else "agent"
            except Exception as exc:
                body, latency, ok, mode = {}, 0.0, False, "request-error"
                print(f"  request failed: {type(exc).__name__}")
            passed += int(ok)
            status = "PASS" if ok else "FAIL"
            print(f"[{index:>2}/10] {status} {case['name']} ({mode}, {latency:.1f}s)")
            print(f"  Q: {case['message']}")
            reply = str(body.get("text") or "(no reply)")
            print("  A: " + reply.replace("\n", "\n     "))
            print()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

    print(f"Result: {passed}/{len(CASES)} passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
