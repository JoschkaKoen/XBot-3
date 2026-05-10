"""Thread-safe token-usage accumulator for AI calls.

Ported from eXercise/eXercise/ai_client.py:72-152. Holds a per-cycle
running tally of (input, output, thinking) token counts per model.

Usage::

    from services.usage import record_usage, get_run_usage, reset_run_usage

    reset_run_usage()                      # at cycle start
    # … many AI calls happen, each calling record_usage(...) …
    snapshot = get_run_usage()             # at cycle end
    # snapshot: {"grok-4.3": {"input": 1234, "output": 567, "thinking": 0}, ...}
"""

from __future__ import annotations

import threading
from typing import Any

_usage_lock = threading.Lock()
_run_usage: dict[str, dict[str, int]] = {}  # model → {"input": N, "output": N, "thinking": N}


def record_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> None:
    """Accumulate token counts for *model* (thread-safe).

    *output_tokens* is the total billed output (visible + thinking), matching
    what the provider invoices. *thinking_tokens* is the thinking portion of
    *output_tokens* — informational only, not used in cost arithmetic.
    """
    with _usage_lock:
        e = _run_usage.setdefault(model, {"input": 0, "output": 0, "thinking": 0})
        e["input"] += input_tokens
        e["output"] += output_tokens
        e["thinking"] += thinking_tokens


def get_run_usage() -> dict[str, dict[str, int]]:
    """Return a snapshot of accumulated token counts since last :func:`reset_run_usage`."""
    with _usage_lock:
        return {m: dict(v) for m, v in _run_usage.items()}


def reset_run_usage() -> None:
    """Clear all accumulated token counts. Call at cycle start to isolate runs."""
    with _usage_lock:
        _run_usage.clear()


def extract_reasoning_tokens(u: Any) -> int:
    """Return reasoning/thinking token count from an OpenAI-compat usage object.

    Tries ``usage.completion_tokens_details.reasoning_tokens`` first (the
    OpenAI standard, used by Gemini OpenAI-compat and recent Qwen3 reasoning
    models), then falls back to a flat ``usage.reasoning_tokens`` shape some
    DashScope versions emit. Returns 0 if neither is present.
    """
    details = getattr(u, "completion_tokens_details", None)
    nested = (getattr(details, "reasoning_tokens", 0) if details else 0) or 0
    if nested:
        return nested
    return getattr(u, "reasoning_tokens", 0) or 0
