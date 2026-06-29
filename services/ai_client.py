"""Multi-provider AI client.

Single entry point ``get_ai_response(model_spec, …)`` routes to one of four
OpenAI-compatible providers based on the model name prefix:

  - ``grok-*``                  → xAI               (XAI_API_KEY)
  - ``gemini-*``                → Google            (GEMINI_API_KEY)
  - ``qwen-*``                  → Alibaba DashScope (DASHSCOPE_API_KEY)
  - ``kimi-*`` / ``moonshot-*`` → Moonshot          (KIMI_API_KEY, opt KIMI_BASE_URL)

The ``model_spec`` is a free-form string of the shape
``"<model>[, <thinking_tokens>][, <max_output_tokens>]"`` (matching
eXercise's default.env convention). Examples::

    get_ai_response("grok-4.3", "say hi")
    get_ai_response("qwen3.6-flash, 1024, 8192", "say hi")
    get_ai_response("gemini-3.5-flash, 0, 4096", "say hi")

Token usage is recorded into ``services.usage`` after every successful call
(including streamed ones — usage is read from the final chunk via
``stream_options={"include_usage": True}``). Transient SSL/5xx/429 errors
retry automatically via ``services.api_retry``.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from services.api_retry import retry_api_call
from services.usage import extract_reasoning_tokens, record_usage

load_dotenv()
logger = logging.getLogger("xbot.ai_client")


# ── Provider registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ProviderDef:
    name: str
    base_url: str
    api_key_env: str
    model_prefixes: tuple[str, ...]


_PROVIDER_REGISTRY: list[_ProviderDef] = [
    _ProviderDef(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        model_prefixes=("gemini",),
    ),
    _ProviderDef(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        model_prefixes=("grok",),
    ),
    _ProviderDef(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        model_prefixes=("qwen",),
    ),
    _ProviderDef(
        # Mainland China endpoint by default. Override via KIMI_BASE_URL=
        # https://api.moonshot.ai/v1 for the international endpoint. Keys
        # are NOT interchangeable between regions.
        name="kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        model_prefixes=("kimi", "moonshot"),
    ),
]


def _provider_def(name: str) -> _ProviderDef:
    return next(p for p in _PROVIDER_REGISTRY if p.name == name)


def provider_for_model(model: str) -> str:
    """Return the provider name for *model* based on its name prefix.

    Falls back to ``xai`` (Grok is XBot-3's default keyset) for unknown names.
    """
    m = model.lower()
    for pdef in _PROVIDER_REGISTRY:
        if any(m.startswith(pfx) for pfx in pdef.model_prefixes):
            return pdef.name
    return "xai"


# ── Model-spec parsing ────────────────────────────────────────────────────────

_LEGACY_EFFORT = {"off": 0, "low": 1024, "high": 8192}


def parse_model_spec(value: str) -> tuple[str, int | None, int | None]:
    """Parse ``"<model>[, <thinking_tokens>][, <max_output_tokens>]"``.

    Returns ``(model, thinking_tokens, max_tokens)``. ``None`` for either
    budget means "caller did not specify — use the code fallback". ``0`` for
    thinking means "explicitly off". Legacy ``off``/``low``/``high`` strings
    parse to ``0``/``1024``/``8192``. Unrecognised tokens are silently skipped.
    """
    parts = [p.strip() for p in value.split(",")]
    model = parts[0]
    nums: list[int] = []
    for p in parts[1:]:
        if not p:
            continue
        low = p.lower()
        if low in _LEGACY_EFFORT:
            nums.append(_LEGACY_EFFORT[low])
        elif p.lstrip("-").isdigit():
            nums.append(int(p))
    thinking = nums[0] if len(nums) >= 1 else None
    max_tokens = nums[1] if len(nums) >= 2 else None
    return model, thinking, max_tokens


# ── Per-provider thinking + max_tokens kwargs ─────────────────────────────────


def build_thinking_kwargs(
    provider: str, thinking_tokens: int | None
) -> tuple[bool, dict]:
    """Return ``(use_stream, extra_kwargs)`` for ``client.chat.completions.create()``.

    Mapping (integer thinking budget → provider param):

    Gemini — OpenAI-compat ``reasoning_effort`` accepts {none, low, high} only.
        0 → "none". 1..1024 → "low". >1024 → "high". None (unspecified) = provider
        default (no param); streams to show live output.
    Qwen — 0 or None disables thinking (non-streaming). Any positive value enables
        thinking (forces stream); :func:`build_completion_kwargs` also passes the
        integer through as Dashscope's ``thinking_budget``.
    Kimi — 0 or None disables (non-streaming, JSON-friendly). Positive enables via
        ``extra_body={"thinking": {"type": "enabled"}}`` AND forces streaming.
        Older ``moonshot-v1-*`` ids ignore the field.
    Grok — silently ignored; always non-streaming.
    """
    if provider == "gemini":
        if thinking_tokens is None:
            return True, {}
        if thinking_tokens == 0:
            return False, {"reasoning_effort": "none"}
        effort = "low" if thinking_tokens <= 1024 else "high"
        return True, {"reasoning_effort": effort}

    if provider == "qwen":
        if thinking_tokens is None or thinking_tokens == 0:
            return False, {"extra_body": {"enable_thinking": False}}
        return True, {"extra_body": {"enable_thinking": True}}

    if provider == "kimi":
        if thinking_tokens is None or thinking_tokens == 0:
            return False, {"extra_body": {"thinking": {"type": "disabled"}}}
        return True, {"extra_body": {"thinking": {"type": "enabled"}}}

    # xai (grok) or unknown — no thinking params
    return False, {}


def build_completion_kwargs(
    provider: str,
    thinking_tokens: int | None,
    max_tokens: int | None,
) -> tuple[bool, dict]:
    """Return ``(use_stream, kwargs)`` for ``client.chat.completions.create()``.

    Superset of :func:`build_thinking_kwargs` that also threads:

    * ``max_tokens`` — when not None, becomes the ``max_tokens=`` API param.
    * ``thinking_budget`` for Qwen — when thinking is on, the integer value is
      added inside ``extra_body`` alongside ``enable_thinking``. Other
      providers ignore it.
    """
    use_stream, kw = build_thinking_kwargs(provider, thinking_tokens)
    if max_tokens is not None:
        kw = {**kw, "max_tokens": max_tokens}
    if provider == "qwen" and thinking_tokens is not None and thinking_tokens > 0:
        eb = dict(kw.get("extra_body") or {})
        eb["thinking_budget"] = int(thinking_tokens)
        kw = {**kw, "extra_body": eb}
    return use_stream, kw


# ── Provider client cache ─────────────────────────────────────────────────────

_clients: dict[str, Any] = {}
_client_lock = threading.Lock()


def _close_clients() -> None:
    for c in list(_clients.values()):
        try:
            c.close()
        except Exception as exc:
            logger.debug("client close: %s", exc)
    _clients.clear()


atexit.register(_close_clients)


def _get_client(provider: str) -> Any:
    """Return a lazily-built OpenAI-compatible client for *provider* (cached)."""
    with _client_lock:
        if provider in _clients:
            return _clients[provider]
        from openai import OpenAI  # noqa: PLC0415

        pdef = _provider_def(provider)
        base = os.getenv("KIMI_BASE_URL") if provider == "kimi" else pdef.base_url
        if not base:
            base = pdef.base_url
        api_key = os.getenv(pdef.api_key_env)
        if not api_key:
            raise ValueError(
                f"❌ {pdef.api_key_env} not set in .env "
                f"(required for {provider} models)"
            )
        client = OpenAI(base_url=base, api_key=api_key)
        _clients[provider] = client
        return client


# ── Streaming consumer ────────────────────────────────────────────────────────


def _consume_stream(stream: Any) -> tuple[str, Any]:
    """Drain a streaming chat completion. Return (joined_text, last_usage_chunk).

    OpenAI-compat providers emit usage in the final no-choices chunk when
    ``stream_options={"include_usage": True}`` is passed; some piggyback it
    on the last content chunk. Track the latest non-None ``usage`` seen.
    """
    parts: list[str] = []
    last_usage = None
    for chunk in stream:
        u = getattr(chunk, "usage", None)
        if u is not None:
            last_usage = u
        if chunk.choices:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                parts.append(content)
    return "".join(parts).strip(), last_usage


# ── Public entry point ────────────────────────────────────────────────────────


def get_ai_response(
    model_spec: str,
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    *,
    retry_label: str | None = None,
) -> str:
    """Run a single chat completion against the model named in *model_spec*.

    *model_spec* — ``"<model>[, <thinking_tokens>][, <max_output_tokens>]"``.
    *max_tokens* (kwarg) — caller override; falls back to spec, then 1200.

    Records token usage into ``services.usage._run_usage`` and retries
    transient errors via ``services.api_retry.retry_api_call``.
    """
    model, thinking, spec_max = parse_model_spec(model_spec)
    provider = provider_for_model(model)
    use_stream, kw = build_completion_kwargs(
        provider, thinking, max_tokens or spec_max or 1200
    )
    if use_stream:
        opts = dict(kw.get("stream_options") or {})
        opts.setdefault("include_usage", True)
        kw["stream_options"] = opts

    client = _get_client(provider)
    label = retry_label or model

    def _do_call() -> tuple[str, Any]:
        logger.debug("AI request (%s, stream=%s): %.80s …", model, use_stream, user_message)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            stream=use_stream,
            **kw,
        )
        if use_stream:
            return _consume_stream(response)
        text = response.choices[0].message.content or ""
        return text, getattr(response, "usage", None)

    text, last_usage = retry_api_call(_do_call, label=label)
    if last_usage is not None:
        record_usage(
            model,
            getattr(last_usage, "prompt_tokens", 0) or 0,
            getattr(last_usage, "completion_tokens", 0) or 0,
            extract_reasoning_tokens(last_usage),
        )
    else:
        # No usage chunk (some providers omit it on streamed calls even with
        # include_usage). Log it so a missing/zero entry in the cost report is
        # explainable rather than silently under-counting spend.
        logger.debug("AI call (%s): no usage returned — cost not recorded for this call.", model)
    logger.debug("AI response (%s): %.80s …", model, text)
    return text


__all__ = [
    "build_completion_kwargs",
    "build_thinking_kwargs",
    "get_ai_response",
    "parse_model_spec",
    "provider_for_model",
]
