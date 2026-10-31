"""
llm_smoke_test.py
-----------------
Check that the LLM client works end to end against the real provider:

  1. a plain generation (no tools) -> prints the model's text;
  2. a function-calling generation -> prints the tool call the model requested.

Usage:
    .venv\\Scripts\\python.exe scripts\\llm_smoke_test.py

Requires GEMINI_API_KEY (and optionally GEMINI_MODEL) in .env. The key is never
printed.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from dotenv import load_dotenv  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Project .env wins over shell vars
load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)

from llm_client import LLMClient, LLMError  # noqa: E402

# A trivial tool so the function-calling path can be verified without the real
# agent tools. Arguments are in English.
DEMO_TOOL = {
    "name": "get_current_time",
    "description": "Return the current time for a given city. Use this whenever the user asks for the time.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name, e.g. 'Tokyo'."}
        },
        "required": ["city"],
    },
}

DEMO_SYSTEM = "You are a helpful assistant. When a tool can answer the question, call it."


def _plain_call(client: LLMClient) -> None:
    print("[1/2] Plain generation (no tools)")
    response = client.generate(
        messages=[{"role": "user", "content": "Say hello in exactly five words."}],
        system=DEMO_SYSTEM,
    )
    print(f"  model: {response.model}  latency: {response.latency_s:.2f}s")
    print(f"  text: {response.text!r}")


def _function_calling_call(client: LLMClient) -> None:
    print("[2/2] Function-calling generation")
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the current time in Tokyo? Use the tool."}
        ],
        tools=[DEMO_TOOL],
        system=DEMO_SYSTEM,
    )
    print(f"  model: {response.model}  latency: {response.latency_s:.2f}s")
    print(f"  text: {response.text!r}")
    if not response.tool_calls:
        print("  tool_calls: (none) -- the model answered in text instead of calling the tool")
        return
    for call in response.tool_calls:
        print(f"  tool_call: name={call.name!r} args={call.args}")


def main() -> int:
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            print("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
            print("Nothing to do -- stopping before the smoke test.")
            return 2
    else:
        if not os.getenv("GEMINI_API_KEY"):
            print("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
            print("Nothing to do -- stopping before the smoke test.")
            return 2

    client = LLMClient()
    print(f"Provider: {client.provider_name}   Model: {client.model}\n")

    try:
        _plain_call(client)
        _function_calling_call(client)
    except LLMError as exc:
        print(f"\nSmoke test failed: {type(exc).__name__}: {exc}")
        return 1

    print("\nSmoke test finished successfully.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
