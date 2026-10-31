"""
llm_client.py
-------------
Provider-neutral LLM client for System B (the tool-calling agent).

The agent talks to this module only through ``LLMClient.generate`` and the small
neutral data classes below, so switching provider (Google Gemini now, a local
Ollama model later) means adding one provider class and changing ``LLM_PROVIDER``.

Configuration comes from environment variables (loaded from .env at import):
    LLM_PROVIDER                  provider name (default "gemini")
    GEMINI_API_KEY                API key (never logged)
    GEMINI_MODEL                  primary model name (never hardcoded in code)
    GEMINI_FALLBACK_MODELS        comma-separated backup models, tried in order
    LLM_MAX_REQUESTS_PER_MINUTE   client-side rate limit (default 10)
    LLM_DAILY_BUDGET              max requests per model per day (default 20)
    LLM_REQUEST_TIMEOUT           per-request timeout in seconds (default 60)
    LLM_TEMPERATURE               sampling temperature (default 0.2 — low, for grounding)

Reliability:
    * a thread-safe rate limiter paces outgoing calls (shared by all callers);
    * transient failures (per-minute HTTP 429 / 5xx / timeouts) are retried up
      to ``max_retries`` times with exponential backoff;
    * a per-day quota error (RESOURCE_EXHAUSTED with a per-day metric) or a
      404 "model not found/retired" error retires that model for the day and
      switches to the next model in the list (primary, then fallbacks);
    * a per-model daily request counter stops calling a model once it reaches
      ``LLM_DAILY_BUDGET``, so a model is never pushed past its limit;
    * when every model is unavailable or out of budget, ``LLMQuotaExhaustedError``
      is raised so the app can show a clear "daily limit" message;
    * every call logs its latency and the model in use (never the key).

This module is additive (System B). It does not touch System A or app.py.
"""

import os
import time
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the .env file. Overridable so tests can point it at a nonexistent
# file and never read the developer's real .env (which holds a live key).
_ENV_FILE = os.getenv("LLM_ENV_FILE", os.path.join(ROOT_DIR, ".env"))

# Snapshot shell env for LLM settings before loading .env so we can warn if
# project .env overrides stale shell variables.
_LLM_ENV_KEYS = ("GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "LLM_DAILY_BUDGET", "LLM_CACHE")
_shell_env_snapshot = {k: os.environ.get(k) for k in _LLM_ENV_KEYS if k in os.environ}

# Load .env once with override so project .env wins over shell variables.
load_dotenv(_ENV_FILE, override=True)

# Log a one-line warning if any shell value was overridden by .env
for _key in _LLM_ENV_KEYS:
    _shell_val = _shell_env_snapshot.get(_key)
    if _shell_val is None:
        continue
    _env_val = os.environ.get(_key)
    if _env_val != _shell_val:
        logger.warning("Overriding shell %s=%r with project .env value %r", _key, _shell_val, _env_val)

# ── Defaults ────────────────────────────────────────────────────────────────
_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "gemini-2.0-flash"
_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MAX_RPM = 10
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_TEMPERATURE = 0.2
# Conservative per-model daily ceiling for the free tier.
_DEFAULT_DAILY_BUDGET = 20


# Retry policy for transient errors: 3 retries, backoff 1s, 2s, 4s.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


def _configured_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()


def get_model_name() -> str:
    """Return the configured LLM model name (never the API key).

    Exposed so the app can report which model is active in health checks
    without constructing a client or touching the provider.
    """
    if _configured_provider() == "groq":
        return os.getenv("GROQ_MODEL") or _DEFAULT_GROQ_MODEL
    return os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL


# ── Neutral request/response shapes ─────────────────────────────────────────
@dataclass
class ToolCall:
    """A tool the model wants to call."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class LLMResponse:
    """Provider-neutral result of one generation."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    model: Optional[str] = None
    latency_s: Optional[float] = None


# Neutral message format used by callers (one dict per turn):
#   {"role": "user",      "content": "the user text"}
#   {"role": "assistant", "content": "model text", "tool_calls": [{"name", "args", "id"}]}
#   {"role": "tool",      "name": "tool_name", "content": {..} | "..", "id": Optional}
# Tool arguments and tool outputs are always in English.


# ── Errors ──────────────────────────────────────────────────────────────────
class LLMError(RuntimeError):
    """Base class for every LLM failure."""

    retryable = False


class LLMConfigError(LLMError):
    """Missing/invalid configuration (e.g. no API key). Not retryable."""


class LLMResponseError(LLMError):
    """The provider rejected the request or returned something unusable."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited us. Retryable."""

    retryable = True


class LLMServerError(LLMError):
    """A transient provider-side failure (5xx). Retryable."""

    retryable = True


class LLMTimeoutError(LLMError):
    """The request timed out. Retryable."""

    retryable = True


class LLMDailyQuotaError(LLMError):
    """The model's per-day quota is exhausted. Not retryable today.

    The client retires this model for the rest of the day and switches to the
    next configured model.
    """


class LLMModelUnavailableError(LLMError):
    """The model does not exist or has been retired (HTTP 404 NOT_FOUND)."""


class LLMQuotaExhaustedError(LLMError):
    """Every configured model is retired or out of its daily budget."""

    quota_exhausted = True


# Errors that retire the current model for the day and trigger a switch.
_MODEL_ROTATION_ERRORS = (LLMDailyQuotaError, LLMModelUnavailableError)


# ── Rate limiter ────────────────────────────────────────────────────────────
class _RateLimiter:
    """Allow at most ``max_calls`` requests per ``period`` seconds.

    Shared across threads so concurrent chat requests cannot burst the provider.
    """

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max(1, int(max_calls))
        self.period = period
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and (now - self._timestamps[0]) >= self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait = self.period - (now - self._timestamps[0])
            time.sleep(max(wait, 0.001))


# ── Provider interface ──────────────────────────────────────────────────────
class _Provider:
    """Minimal provider interface. Subclasses implement ``generate``."""

    name = "base"

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class GeminiProvider(_Provider):
    """Google Gemini provider built on the official google-genai SDK."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: float, temperature: float = _DEFAULT_TEMPERATURE):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMConfigError("google-genai is not installed") from exc

        self._genai = genai
        self._types = types
        self.model = model
        self.temperature = temperature
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    # -- neutral -> Gemini conversion --------------------------------------
    def _to_contents(self, messages: List[Dict[str, Any]]) -> List[Any]:
        types = self._types
        contents: List[Any] = []
        for message in messages or []:
            role = (message.get("role") or "user").lower()
            if role == "tool":
                # Function responses travel back to the model in a user turn.
                response = message.get("content")
                if not isinstance(response, dict):
                    response = {"result": response}
                func_response = types.FunctionResponse(
                    name=message.get("name", ""), response=response
                )
                if message.get("id"):
                    func_response.id = message["id"]
                contents.append(
                    types.Content(role="user", parts=[types.Part(function_response=func_response)])
                )
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts: List[Any] = []
            text = message.get("content")
            if text:
                parts.append(types.Part(text=str(text)))
            for call in message.get("tool_calls") or []:
                call_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                func_call = types.FunctionCall(name=call_name, args=dict(call_args or {}))
                if call_id:
                    func_call.id = call_id
                parts.append(types.Part(function_call=func_call))
            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))
        return contents

    def _to_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        if not tools:
            return None
        types = self._types
        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters_json_schema=tool.get("parameters")
                    or {"type": "object", "properties": {}},
                )
            )
        return [types.Tool(function_declarations=declarations)]

    # -- generation --------------------------------------------------------
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            tools=self._to_tools(tools),
            temperature=self.temperature,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=self._to_contents(messages),
                config=config,
            )
        except Exception as exc:  # mapped to our errors below
            raise _map_gemini_error(exc) from exc

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMResponse:
        tool_calls: List[ToolCall] = []
        for call in getattr(response, "function_calls", None) or []:
            args = getattr(call, "args", None)
            tool_calls.append(
                ToolCall(
                    name=getattr(call, "name", "") or "",
                    args=dict(args) if args else {},
                    id=getattr(call, "id", None),
                )
            )
        text = None
        try:
            text = response.text
        except Exception:
            text = None
        return LLMResponse(text=text, tool_calls=tool_calls, model=self.model)


def _map_gemini_error(exc: Exception) -> LLMError:
    """Translate a google-genai / transport exception into a neutral LLMError.

    Never includes the API key. The message is kept short and safe to log.
    """
    try:
        from google.genai import errors as genai_errors
    except ImportError:  # pragma: no cover
        genai_errors = None

    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        status = (getattr(exc, "status", None) or "").upper()
        if code == 404 or status == "NOT_FOUND":
            # A retired or unknown model — retire it and fall back.
            return LLMModelUnavailableError("model not found or retired")
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            if _is_per_day_quota(exc):
                return LLMDailyQuotaError("daily quota exhausted for this model")
            return LLMRateLimitError("provider rate limit reached")
        if code in (500, 502, 503, 504) or status in (
            "INTERNAL",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        ):
            return LLMServerError(f"provider error {code or status}")
        if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
            return LLMConfigError("authentication failed (check the API key)")
        return LLMResponseError(f"request rejected ({code or status or 'unknown'})")

    if isinstance(exc, TimeoutError):
        return LLMTimeoutError("request timed out")
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
            return LLMTimeoutError("request timed out")
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        pass

    return LLMResponseError(f"{type(exc).__name__}: {exc}")


def _is_per_day_quota(exc: Exception) -> bool:
    """True when a 429 is a per-day quota (as opposed to a per-minute burst).

    The provider reports the quota metric in the error text/details, e.g.
    ``generate_requests_per_day`` or "PerDay". We only need the day/minute
    distinction: a per-day limit should switch models, a per-minute one should
    just wait and retry.
    """
    parts = [str(exc), str(getattr(exc, "message", ""))]
    try:
        parts.append(str(getattr(exc, "details", "")))
    except Exception:  # pragma: no cover - defensive
        pass
    haystack = " ".join(parts).lower()
    return any(
        marker in haystack
        for marker in ("per_day", "per day", "perday", "daily", "requests_per_day")
    )


class GroqProvider(_Provider):
    """Groq provider using its OpenAI-compatible chat-completions endpoint.

    Groq is a fast, free-tier-friendly host for open models (e.g.
    ``llama-3.3-70b-versatile``) and supports function calling. It is a drop-in
    alternative to Gemini when Google is unavailable or out of quota.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float,
        temperature: float = _DEFAULT_TEMPERATURE,
        http_client=None,
    ):
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        if http_client is not None:  # tests inject a mock transport
            self._http = http_client
            return
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - google-genai pulls it in
            raise LLMConfigError("httpx is not installed") from exc
        self._http = httpx.Client(base_url=_GROQ_BASE_URL, timeout=timeout_s)

    # -- neutral -> OpenAI/Groq conversion ---------------------------------
    def _to_messages(self, messages: List[Dict[str, Any]], system: Optional[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for message in messages or []:
            role = (message.get("role") or "user").lower()
            if role == "tool":
                content = message.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("id") or "",
                        "content": content,
                    }
                )
                continue

            if role == "assistant":
                entry: Dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
                calls = []
                for index, call in enumerate(message.get("tool_calls") or []):
                    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                    call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                    calls.append(
                        {
                            "id": call_id or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args or {}, ensure_ascii=False, default=str),
                            },
                        }
                    )
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue

            out.append({"role": role, "content": str(message.get("content") or "")})
        return out

    def _to_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

    # -- generation --------------------------------------------------------
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(messages, system),
            "temperature": self.temperature,
        }
        tool_defs = self._to_tools(tools)
        if tool_defs:
            body["tools"] = tool_defs
            body["tool_choice"] = "auto"

        try:
            response = self._http.post(
                "/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except Exception as exc:  # mapped below
            raise _map_groq_transport_error(exc) from exc

        if response.status_code >= 400:
            raise _map_groq_error(response.status_code, response.text)

        try:
            data = response.json()
        except Exception as exc:
            raise LLMResponseError("invalid JSON from provider") from exc

        return self._parse_response(data)

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else {}
        tool_calls: List[ToolCall] = []
        for call in (message or {}).get("tool_calls") or []:
            function = call.get("function") or {}
            raw_args = function.get("arguments")
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except (TypeError, ValueError):
                    args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(name=function.get("name", "") or "", args=args, id=call.get("id"))
            )
        return LLMResponse(
            text=(message or {}).get("content"),
            tool_calls=tool_calls,
            model=data.get("model") or self.model,
        )


def _map_groq_transport_error(exc: Exception) -> LLMError:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError("request timed out")
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError("request timed out")
    return LLMServerError(f"provider connection error ({type(exc).__name__})")


def _map_groq_error(status: int, body: str) -> LLMError:
    """Map an OpenAI-compatible HTTP error to a neutral LLMError (no secrets)."""
    text = (body or "")[:500]
    lowered = text.lower()
    if status == 404:
        return LLMModelUnavailableError("model not found or retired")
    if status == 429:
        if any(marker in lowered for marker in ("per day", "per_day", "daily", "tokens per day")):
            return LLMDailyQuotaError("daily quota exhausted for this model")
        return LLMRateLimitError("provider rate limit reached")
    if status in (500, 502, 503, 504):
        return LLMServerError(f"provider error {status}")
    if status in (401, 403):
        return LLMConfigError("authentication failed (check the API key)")
    return LLMResponseError(f"request rejected ({status})")


def _read_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _read_env_list(name: str) -> List[str]:
    """Parse a comma-separated env var into a list of trimmed, non-empty items."""
    raw = os.getenv(name, "") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


# ── Model pool ──────────────────────────────────────────────────────────────
class ModelPool:
    """An ordered list of models with a per-model daily request budget.

    Remembers, in memory, which models have been retired for the day (after a
    per-day quota error or a "model not found" error) and stops calling a model
    once it reaches ``daily_budget``. Thread-safe. The active model only moves
    forward within a day and resets when the calendar day changes.
    """

    def __init__(self, models: List[str], daily_budget: int, day_provider=None):
        ordered: List[str] = []
        for name in models or []:
            if name and name not in ordered:
                ordered.append(name)
        self.models = ordered or [_DEFAULT_MODEL]
        self.daily_budget = max(1, int(daily_budget))
        self._day_provider = day_provider or (
            lambda: time.strftime("%Y-%m-%d", time.localtime())
        )
        self._lock = threading.Lock()
        self._day = self._day_provider()
        self._counts: Dict[str, int] = {}
        self._down = set()
        self._active = 0
        self._last: Optional[str] = None

    def _sync_day(self) -> None:
        today = self._day_provider()
        if today != self._day:
            self._day = today
            self._counts = {}
            self._down = set()
            self._active = 0

    def current(self) -> Optional[str]:
        """The model to use now, skipping retired/budget-exhausted models."""
        with self._lock:
            self._sync_day()
            while self._active < len(self.models):
                name = self.models[self._active]
                if name in self._down or self._counts.get(name, 0) >= self.daily_budget:
                    self._active += 1
                    continue
                return name
            return None

    def has_available(self) -> bool:
        with self._lock:
            self._sync_day()
            for name in self.models:
                if name in self._down:
                    continue
                if self._counts.get(name, 0) >= self.daily_budget:
                    continue
                return True
            return False

    def last_model(self) -> str:
        with self._lock:
            return self._last or ""

    def record_call(self, model: str) -> None:
        with self._lock:
            self._sync_day()
            self._counts[model] = self._counts.get(model, 0) + 1
            self._last = model

    def retire(self, model: str) -> None:
        """Retire a model for the rest of the day (per-day quota / not found)."""
        with self._lock:
            self._sync_day()
            self._down.add(model)
            if self._active < len(self.models) and self.models[self._active] == model:
                self._active += 1

    def state(self) -> Dict[str, Any]:
        """A snapshot for health checks (model names only — never the key)."""
        with self._lock:
            self._sync_day()
            return {
                "active": self.models[self._active] if self._active < len(self.models) else None,
                "day": self._day,
                "daily_budget": self.daily_budget,
                "models": [
                    {
                        "name": name,
                        "used": self._counts.get(name, 0),
                        "retired": name in self._down,
                    }
                    for name in self.models
                ],
            }


def _set_provider_model(provider: _Provider, model: str) -> None:
    """Point a provider at ``model`` (Gemini providers read this per call)."""
    try:
        provider.model = model
    except Exception:  # pragma: no cover - defensive
        pass


# ── Client ──────────────────────────────────────────────────────────────────
class LLMClient:
    """Provider-neutral client with rate limiting, retries, model fallback and
    per-model daily budgeting."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_requests_per_minute: Optional[int] = None,
        timeout_s: Optional[float] = None,
        temperature: Optional[float] = None,
        max_retries: int = _MAX_RETRIES,
        provider_impl: Optional[_Provider] = None,
        models: Optional[List[str]] = None,
        daily_budget: Optional[int] = None,
    ):
        self.provider_name = (provider or os.getenv("LLM_PROVIDER") or _DEFAULT_PROVIDER).lower()
        if self.provider_name == "groq":
            primary = model or os.getenv("GROQ_MODEL") or _DEFAULT_GROQ_MODEL
            fallbacks = _read_env_list("GROQ_FALLBACK_MODELS")
        else:
            primary = model or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL
            fallbacks = _read_env_list("GEMINI_FALLBACK_MODELS")
        if models is not None:
            model_list = list(models)
        else:
            model_list = [primary] + fallbacks
        self.model = model_list[0] if model_list else primary
        if api_key is not None:
            self._api_key = api_key
        elif self.provider_name == "groq":
            self._api_key = os.getenv("GROQ_API_KEY")
        else:
            self._api_key = os.getenv("GEMINI_API_KEY")
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else _read_env_float("LLM_REQUEST_TIMEOUT", _DEFAULT_TIMEOUT_S)
        )
        self.temperature = (
            temperature
            if temperature is not None
            else _read_env_float("LLM_TEMPERATURE", _DEFAULT_TEMPERATURE)
        )
        self.max_retries = max(0, int(max_retries))
        rpm = (
            max_requests_per_minute
            if max_requests_per_minute is not None
            else _read_env_int("LLM_MAX_REQUESTS_PER_MINUTE", _DEFAULT_MAX_RPM)
        )
        budget = (
            daily_budget
            if daily_budget is not None
            else _read_env_int("LLM_DAILY_BUDGET", _DEFAULT_DAILY_BUDGET)
        )
        self._limiter = _RateLimiter(rpm, period=60.0)
        self._provider = provider_impl
        self._pool = ModelPool(model_list, budget)

    # -- introspection (used by health checks) -----------------------------
    def active_model(self) -> str:
        """The model currently in use (or the last one tried)."""
        return self._pool.current() or self._pool.last_model()

    def quota_exhausted(self) -> bool:
        """True when no configured model is available today."""
        try:
            return not self._pool.has_available()
        except Exception:
            return self._pool.current() is None

    def pool_state(self) -> Dict[str, Any]:
        return self._pool.state()

    def _build_provider(self) -> _Provider:
        if self._provider is not None:
            return self._provider
        if not self._api_key:
            key_name = "GROQ_API_KEY" if self.provider_name == "groq" else "GEMINI_API_KEY"
            raise LLMConfigError(
                f"{key_name} is not set (copy .env.example to .env and add your key)"
            )
        if self.provider_name == "groq":
            self._provider = GroqProvider(
                self._api_key, self.model, self.timeout_s, self.temperature
            )
        elif self.provider_name == "gemini":
            self._provider = GeminiProvider(
                self._api_key, self.model, self.timeout_s, self.temperature
            )
        else:
            raise LLMConfigError(f"unknown LLM provider '{self.provider_name}'")
        return self._provider

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Run one generation with rate limiting, retries and model fallback.

        Transient errors are retried with backoff. A per-day quota error or a
        "model not found" error retires the current model for the day and the
        same request is re-issued on the next model in the list. When no model
        is left, ``LLMQuotaExhaustedError`` is raised. The API key is never
        logged.
        """
        provider = self._build_provider()
        last_error: Optional[LLMError] = None
        attempt = 0

        while True:
            model = self._pool.current()
            if model is None:
                raise LLMQuotaExhaustedError(
                    "all configured models are retired or out of their daily budget"
                ) from last_error

            _set_provider_model(provider, model)
            self._pool.record_call(model)
            self._limiter.acquire()
            started = time.monotonic()
            try:
                response = provider.generate(messages, tools=tools, system=system)
            except LLMError as exc:
                latency = time.monotonic() - started
                last_error = exc
                logger.info(
                    "LLM call failed provider=%s model=%s attempt=%d latency=%.2fs error=%s",
                    provider.name, model, attempt + 1, latency, exc,
                )
                if isinstance(exc, _MODEL_ROTATION_ERRORS):
                    logger.warning(
                        "Retiring model %s for today (%s); switching to the next model.",
                        model, type(exc).__name__,
                    )
                    self._pool.retire(model)
                    attempt = 0  # a fresh retry budget on the next model
                    continue
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                attempt += 1
                logger.warning(
                    "LLM call retryable error (%s); retrying in %.1fs", exc, delay
                )
                time.sleep(delay)
                continue

            latency = time.monotonic() - started
            response.latency_s = latency
            response.model = response.model or model
            logger.info(
                "LLM call ok provider=%s model=%s latency=%.2fs tool_calls=%d",
                provider.name, model, latency, len(response.tool_calls),
            )
            return response


_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def get_client() -> LLMClient:
    """Return a shared ``LLMClient`` built from the environment (cached)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LLMClient()
    return _client


def reset_client() -> None:
    """Drop the cached client so a fresh one is built from the environment."""
    global _client
    with _client_lock:
        _client = None


def get_active_model_name() -> str:
    """The model in use right now, or the configured primary before a client
    has been built (never the API key)."""
    client = _client
    if client is not None:
        return client.active_model() or get_model_name()
    return get_model_name()


def quota_state() -> str:
    """``"ok"`` or ``"quota_exhausted"`` for the shared client (health checks).

    "ok" if at least one model is available and not budget-limited.
    "quota_exhausted" only when all models are retired or budget-limited.
    """
    client = _client
    if client is not None:
        try:
            ps = client.pool_state()
            models = ps.get("models") or []
            budget = ps.get("daily_budget")
            for m in models:
                name = m.get("name")
                if not name:
                    continue
                if m.get("retired"):
                    continue
                used = m.get("used", 0)
                if budget is not None and used >= budget:
                    continue
                return "ok"
            if models:
                return "quota_exhausted"
        except Exception:
            pass
        if client.quota_exhausted():
            return "quota_exhausted"
    return "ok"
