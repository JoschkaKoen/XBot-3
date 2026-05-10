"""Uniform retry/backoff for AI provider API calls.

Ported from eXercise/eXercise/api_retry.py. One predicate, one helper —
used by services/ai_client.get_ai_response so a transient transport error
(SSL EOF, connection drop, 429, 5xx) never silently destroys data.

Usage::

    from services.api_retry import retry_api_call

    def _do_call() -> tuple[str, Any]:
        resp = client.chat.completions.create(...)
        return resp.choices[0].message.content or "", resp.usage

    text, usage = retry_api_call(_do_call, label=f"grok-4.3 tweet")

The closure consumes any streaming iterator inside _do_call, so a
mid-stream SSL EOF triggers a retry rather than returning a partial
response.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger("xbot.api_retry")

T = TypeVar("T")


_NEVER_RETRY: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    json.JSONDecodeError,
)


def is_retryable_error(exc: BaseException) -> bool:
    """Fail-open predicate: retry anything that isn't clearly terminal.

    Catches SSL EOF (``ssl.SSLError`` wrapped in ``httpx.ReadError`` or an
    ``OpenAIError``), every transient httpx/openai/google variant, raw
    socket drops, and ``TimeoutError``.

    Excludes: cancellation signals, JSON parse errors, pydantic validation
    errors, and HTTP 4xx (auth, not-found, bad-request, unprocessable) —
    these won't be fixed by retrying.
    """
    if isinstance(exc, _NEVER_RETRY):
        return False

    # pydantic.ValidationError — optional dependency, isinstance only if importable.
    try:
        from pydantic import ValidationError  # noqa: PLC0415

        if isinstance(exc, ValidationError):
            return False
    except ImportError:
        pass

    # OpenAI APIStatusError — auth / not-found / bad-request won't be fixed by retry.
    try:
        from openai import APIStatusError  # noqa: PLC0415

        if isinstance(exc, APIStatusError):
            # DashScope wraps transient server-side flakes (image-URL parse
            # blip, content-filter race, server-overload) in HTTP 400 with
            # an `InternalError.Algo.*` body code. Same call succeeds on the
            # next attempt — let exponential backoff have a shot before the
            # broad 4xx exclusion short-circuits it.
            is_dashscope_transient_400 = (
                exc.status_code == 400 and "InternalError.Algo" in str(exc)
            )
            if not is_dashscope_transient_400 and exc.status_code in (400, 401, 403, 404, 422):
                return False
    except ImportError:
        pass

    # google.genai.errors.APIError — same exclusion for native Gemini SDK if installed.
    try:
        from google.genai.errors import APIError as _GenAIError  # noqa: PLC0415

        if isinstance(exc, _GenAIError) and getattr(exc, "code", None) in (
            400,
            401,
            403,
            404,
            422,
        ):
            return False
    except ImportError:
        pass

    return True


def retry_api_call(
    fn: Callable[[], T],
    *,
    label: str,
    max_attempts: int = 4,
    base_sleep: float = 0.1,
    backoff_factor: float = 2.0,
    max_sleep: float = 5.0,
    jitter: float = 0.25,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``fn`` with retry/backoff on transient errors.

    Sleeps grow exponentially (``base_sleep * backoff_factor**n``) capped
    at ``max_sleep``, with ±``jitter`` randomization to avoid thundering
    herds when many parallel calls hit the same endpoint blip.

    Re-raises the last exception after ``max_attempts``. Callers that want
    to degrade gracefully should wrap the call in their own try/except.

    Args:
        fn: Zero-arg callable performing the API call (and stream
            consumption, if any). A failure in either phase triggers a retry.
        label: Human-readable identifier for log lines (e.g. "grok-4.3 tweet").
        max_attempts: Total attempts (initial + retries). Default 4.
        base_sleep: Sleep before the first retry, in seconds. Default 0.1.
        backoff_factor: Multiplier for each subsequent sleep. Default 2.0.
        max_sleep: Cap on a single sleep duration. Default 5.0s.
        jitter: ±fraction randomization (0.25 = ±25%). Default 0.25.
        is_retryable: Predicate deciding which exceptions trigger a retry.
        on_retry: Optional ``(attempt_index, exc, sleep_s)`` callback for tests.
    """
    from utils.ui import info, warn  # noqa: PLC0415  (avoid circular imports)

    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:
            if attempt >= max_attempts or not is_retryable(exc):
                suffix = ", no more retries" if attempt >= max_attempts else ""
                warn(f"{label}: API error (attempt {attempt}/{max_attempts}{suffix})  —  {exc}")
                logger.warning(
                    "%s: API error (attempt %d/%d%s) — %s",
                    label, attempt, max_attempts, suffix, exc,
                )
                raise

            sleep_s = min(base_sleep * (backoff_factor ** (attempt - 1)), max_sleep)
            if jitter:
                sleep_s *= random.uniform(1.0 - jitter, 1.0 + jitter)
            # Interim retries are routine (transient SSL hiccups, brief 5xx);
            # log as info so a recovered call does not surface as a warning.
            info(f"{label}: API error (attempt {attempt}/{max_attempts})  —  {exc}")
            info(f"{label}: retrying in {sleep_s:.2f}s …")
            logger.info(
                "%s: API error (attempt %d/%d) — %s ; retrying in %.2fs",
                label, attempt, max_attempts, exc, sleep_s,
            )
            if on_retry is not None:
                on_retry(attempt, exc, sleep_s)
            time.sleep(sleep_s)
