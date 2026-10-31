"""
list_models.py
--------------
List the Gemini models your API key can use, and mark which ones support
function calling (tool calling), which System B's agent needs.

Usage:
    .venv\\Scripts\\python.exe scripts\\list_models.py
    .venv\\Scripts\\python.exe scripts\\list_models.py --probe

By default the function-calling column is a best-effort guess based on the
model's supported actions and name (Gemini text models support function calling;
embedding/imagen/veo/aqa models do not). Pass --probe to actually send a tiny
function-calling request to each candidate model and confirm it live. Probing
costs one request per model and is paced to respect the free tier.

The API key is read from .env and never printed.
"""

import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from dotenv import load_dotenv  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Project .env wins over shell vars
load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

logger = logging.getLogger("list_models")

# Name fragments of models that never do function calling.
_NON_TOOL_HINTS = ("embedding", "embed", "aqa", "imagen", "veo", "gemma", "learnlm")


def _actions(model) -> list:
    return [str(a).lower() for a in (getattr(model, "supported_actions", None) or [])]


def _is_generative(model) -> bool:
    actions = _actions(model)
    if actions:
        return any("generatecontent" in a for a in actions)
    name = (getattr(model, "name", "") or "").lower()
    return not any(h in name for h in _NON_TOOL_HINTS)


def _guess_function_calling(model) -> bool:
    name = (getattr(model, "name", "") or "").lower()
    return _is_generative(model) and not any(h in name for h in _NON_TOOL_HINTS)


def _short(name: str) -> str:
    return (name or "").split("/")[-1]


def _probe_function_calling(client, model_name: str) -> bool:
    """Send one tiny tool-calling request; True if the model calls the tool."""
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="ping",
                description="Return a pong. Always call this tool when asked to ping.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            )
        ]
    )
    config = types.GenerateContentConfig(tools=[tool])
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Ping with the value 'ok'. You must use the ping tool.",
            config=config,
        )
    except Exception as exc:
        logger.debug("probe failed for %s: %s", model_name, exc)
        return False
    return bool(getattr(response, "function_calls", None))


def main() -> int:
    parser = argparse.ArgumentParser(description="List Gemini models and tool-calling support.")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="live-test function calling on each candidate model (one request each)",
    )
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 2

    client = genai.Client(api_key=api_key)

    try:
        models = list(client.models.list())
    except Exception as exc:
        print(f"Could not list models: {type(exc).__name__}: {exc}")
        print("Check that your API key is valid and the API is enabled.")
        return 1

    rows = []
    for model in models:
        name = _short(getattr(model, "name", ""))
        rows.append(
            {
                "model": name,
                "display": getattr(model, "display_name", "") or "",
                "actions": ",".join(_actions(model)),
                "tools": _guess_function_calling(model),
                "in_limit": getattr(model, "input_token_limit", None),
                "out_limit": getattr(model, "output_token_limit", None),
            }
        )
    rows.sort(key=lambda r: r["model"])

    if args.probe:
        rpm = 10
        try:
            rpm = int(os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", "") or 10)
        except ValueError:
            pass
        interval = 60.0 / max(rpm, 1)
        print(f"Probing function calling (pacing ~{rpm}/min). This may take a while...\n")
        last = 0.0
        for row in rows:
            if not row["tools"]:
                continue
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            row["tools"] = _probe_function_calling(client, row["model"])
            last = time.monotonic()
            flag = "yes" if row["tools"] else "no"
            print(f"  probed {row['model']:<32} function_calling={flag}")

    header = f"{'model':<36} {'function_calling':<16} {'tokens(in/out)':<18} actions"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        tokens = f"{row['in_limit'] or '?'}/{row['out_limit'] or '?'}"
        print(
            f"{row['model']:<36} {'yes' if row['tools'] else 'no':<16} "
            f"{tokens:<18} {row['actions']}"
        )

    if not args.probe:
        print(
            "\nfunction_calling is a best-effort guess (supported actions + name). "
            "Re-run with --probe to confirm live."
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    raise SystemExit(main())
