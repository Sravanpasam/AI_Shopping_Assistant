#!/usr/bin/env python
"""
Show effective LLM configuration and where values come from.

Never prints GEMINI_API_KEY contents — only "set" or "not set".
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Ensure llm_client loads with our current logic
import llm_client  # noqa: E402


def _fmt_age(path: Path) -> str:
    try:
        mtime = path.stat().st_mtime
        age_s = int(time.time() - mtime)
        if age_s < 60:
            return f"{age_s}s ago"
        if age_s < 3600:
            return f"{age_s // 60}m ago"
        if age_s < 86400:
            return f"{age_s // 3600}h ago"
        return f"{age_s // 86400}d ago"
    except Exception:
        return "unknown"


def main() -> int:
    env_path = Path(llm_client._ENV_FILE)
    print(f".env path: {env_path}")
    print(f".env exists: {env_path.exists()}")
    print()

    # Check for quota/retired state files
    state_candidates = [
        ROOT_DIR / "data" / "llm_state.json",
        ROOT_DIR / "data" / "llm_pool_state.json",
        ROOT_DIR / "data" / "llm_quota.json",
        ROOT_DIR / ".llm_state.json",
    ]
    found = False
    for candidate in state_candidates:
        if candidate.exists():
            found = True
            print(f"state file: {candidate} (exists, age: {_fmt_age(candidate)})")
    if not found:
        print("state file: none found (no persisted quota/retired state)")
    print()

    # API key status only
    key_names = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    provider_name = llm_client._configured_provider()
    print(f"LLM_PROVIDER: {provider_name!r}")
    for provider, name in key_names.items():
        value = os.environ.get(name) or ""
        print(f"{name}: {'set' if value else 'not set'}")
    print()

    # Report each setting
    settings = ["GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "GROQ_MODEL",
                "GROQ_FALLBACK_MODELS", "LLM_DAILY_BUDGET", "LLM_CACHE"]
    for key in settings:
        effective = os.environ.get(key)
        if effective is None or effective == "":
            if key == "GEMINI_MODEL":
                effective = llm_client._DEFAULT_MODEL
            elif key == "GROQ_MODEL":
                effective = llm_client._DEFAULT_GROQ_MODEL
            elif key == "LLM_DAILY_BUDGET":
                effective = str(llm_client._DEFAULT_DAILY_BUDGET)
            elif key in ("GEMINI_FALLBACK_MODELS", "GROQ_FALLBACK_MODELS"):
                effective = ""
            elif key == "LLM_CACHE":
                effective = "0"
        source = "default"
        if key in os.environ and os.environ.get(key):
            source = "process environment or .env"
        print(f"{key}: {effective!r} (source: {source})")

    print()
    # Also show active model from client if exists
    try:
        active = llm_client.get_active_model_name()
        print(f"active_model_name(): {active}")
    except Exception as exc:
        print(f"active_model_name(): error {exc}")

    try:
        q = llm_client.quota_state()
        print(f"quota_state(): {q}")
    except Exception as exc:
        print(f"quota_state(): error {exc}")

    try:
        client = llm_client.get_client()
        ps = client.pool_state()
        print("pool_state():")
        print(json.dumps(ps, indent=2))
    except Exception as exc:
        print(f"pool_state(): error {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
