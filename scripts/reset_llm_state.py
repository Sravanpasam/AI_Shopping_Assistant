#!/usr/bin/env python
"""Reset LLM client/model-pool state (in-memory only in this codebase).

The ModelPool has no disk persistence; it is held in the cached LLMClient.
This script clears the shared client so the next call rebuilds from current
environment (.env with override=True)."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import llm_client  # noqa: E402


def main() -> int:
    llm_client.reset_client()
    print("Reset llm_client shared state (rebuilt from current env on next use).")
    print("ModelPool has no disk persistence; state expires at calendar day change by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
