"""
services/content_safety — China content-compliance filter for secondary posts.

FAIL-CLOSED: a secondary post is published only on an explicit positive verdict
from the compliance model. Validated against the real back-catalogue (234/235
compliant; the lone flag was a "right to vote / Stimmrecht" tweet) with
qwen3.7-plus on DashScope.

`check_compliance(text, policy)` returns `(ok_to_post, status, reason)`:
  - compliant         → (True,  "compliant", "")          → post
  - non-compliant     → (False, "blocked",   reason)      → skip (final, no retry)
  - data_inspection   → (False, "blocked",   "...")       → skip (provider moderation)
  - technical failure → retried (backoff); after N attempts → (False, "unverified", err) → skip

It runs its OWN retry loop (single DashScope call per attempt) so retry behavior
is precise and a transient/connection error never silently posts unverified
content. Token usage is recorded so the cost shows in the cycle cost report.
"""

from __future__ import annotations

import json
import logging
import time

import config

logger = logging.getLogger("xbot.content_safety")

_BACKOFF_SECONDS = 1.0

# Validated mainland-China policy (the realistic, not paranoid, stance from the
# experiment: flag the genuine sensitive categories, pass everyday humor).
_POLICIES: dict[str, str] = {
    "china": (
        "你是中国大陆社交平台（微博/小红书/抖音）的内容合规审核员。"
        "下面是一条语言学习推文（教外语词汇）。判断它是否可以在中国大陆平台合规发布。\n"
        "需要判为不合规的敏感类别：政治/政府/领导人/政党/政策/选举/抗议/维权/审查；"
        "主权与领土（台湾、西藏、新疆、香港、澳门、南海，或把台湾等称为国家）；"
        "敏感历史（如1989、文革）；宗教与邪教（尤其法轮功）、迷信；"
        "损害国家尊严（侮辱国旗国歌英烈领导人）；批评中国或贬低中国制度；"
        "色情低俗、暴力血腥、犯罪教学、武器、毒品、赌博；危害社会稳定。\n"
        "请务实判断：只有当内容确实涉及上述敏感类别时才判为不合规；"
        "日常生活、轻松幽默、积极正面的普通内容一律判为合规，不要误判。\n"
        '只返回 JSON：{"compliant": true/false, "category": "<类别或 none>", "reason": "<简短理由>"}'
    ),
}


def _one_call(text: str, system: str) -> str:
    """A single (no internal retry) DashScope chat call; records token usage."""
    from services.ai_client import (
        _get_client, parse_model_spec, provider_for_model, build_completion_kwargs,
    )
    from services.usage import record_usage, extract_reasoning_tokens

    model, thinking, spec_max = parse_model_spec(config.CONTENT_SAFETY_MODEL)
    provider = provider_for_model(model)
    _, kw = build_completion_kwargs(provider, thinking, spec_max or 256)
    kw.pop("stream_options", None)
    client = _get_client(provider)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        stream=False,
        **kw,
    )
    usage = getattr(resp, "usage", None)
    if usage is not None:
        record_usage(
            model,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            extract_reasoning_tokens(usage),
        )
    return resp.choices[0].message.content or ""


def _is_content_rejection(exc: BaseException) -> bool:
    """True iff the provider rejected the *content* (DashScope data_inspection 400)
    — a deterministic moderation block (not a transient/technical error)."""
    try:
        from openai import APIStatusError
    except ImportError:
        return False
    if not isinstance(exc, APIStatusError):
        return False
    code = str(getattr(exc, "code", "") or "").lower()
    s = str(exc).lower()
    return getattr(exc, "status_code", None) == 400 and (
        "data_inspection" in code or "datainspection" in s or "inappropriate content" in s
    )


def _parse(raw: str) -> tuple[bool | None, str]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    try:
        d = json.loads(raw)
    except Exception:
        return None, ""
    c = d.get("compliant")
    if c is True:
        return True, ""
    if c is False:
        return False, f"{d.get('category', '')}: {d.get('reason', '')}".strip(": ")
    return None, ""


def check_compliance(text: str, policy: str = "china", max_attempts: int | None = None) -> tuple[bool, str, str]:
    """Fail-closed compliance gate. Returns (ok_to_post, status, reason).

    status ∈ {"compliant", "blocked", "unverified"}. ok_to_post is True ONLY for
    an explicit compliant verdict. Content rejections are final (no retry);
    technical failures are retried with backoff, then → "unverified".
    """
    system = _POLICIES.get(policy)
    if not system:
        # Unknown policy → nothing to screen against; allow (e.g. a non-China target).
        return True, "compliant", f"no policy '{policy}'"

    attempts = max_attempts or int(getattr(config, "CONTENT_SAFETY_MAX_ATTEMPTS", 3) or 3)
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            raw = _one_call(text, system)
        except BaseException as exc:  # noqa: BLE001 — classify below
            if _is_content_rejection(exc):
                return False, "blocked", "provider moderation (data_inspection_failed)"
            last_err = f"{type(exc).__name__}: {str(exc)[:100]}"
        else:
            verdict, reason = _parse(raw)
            if verdict is True:
                return True, "compliant", ""
            if verdict is False:
                return False, "blocked", reason or "non-compliant"
            last_err = "unparseable response"
        logger.warning("content_safety: attempt %d/%d failed (%s)", attempt, attempts, last_err)
        if attempt < attempts:
            time.sleep(_BACKOFF_SECONDS * attempt)

    return False, "unverified", last_err


# ── Watchdog: stop the bot if the check can't run N cycles in a row ───────────
_STATE_FILE = "data/compliance_state.json"


def register_cycle(unverified: bool, verified: bool) -> None:
    """Track consecutive UNVERIFIED cycles (the check couldn't get a verdict) and
    raise FatalProviderError once it reaches CONTENT_SAFETY_STOP_AFTER_UNVERIFIED —
    signalling a sustained provider/key outage. A "blocked" verdict counts as
    *verified* (the check ran) and resets the streak; a cycle with no check leaves
    it unchanged. Streak persists (data/compliance_state.json) across restarts.
    """
    from utils.io import safe_json_read, atomic_json_write

    if not (unverified or verified):
        return  # no compliance check ran this cycle — leave the streak as-is

    st = safe_json_read(_STATE_FILE, default={}, logger=logger)
    streak = int((st or {}).get("unverified_streak", 0) or 0)
    streak = streak + 1 if unverified else 0
    atomic_json_write(_STATE_FILE, {"unverified_streak": streak}, ensure_ascii=False, indent=2)

    limit = int(getattr(config, "CONTENT_SAFETY_STOP_AFTER_UNVERIFIED", 2) or 0)
    if unverified and limit and streak >= limit:
        from utils.errors import FatalProviderError
        raise FatalProviderError(
            f"China compliance check UNVERIFIED {streak} cycles in a row "
            f"({config.CONTENT_SAFETY_MODEL} / DashScope unreachable) — stopping the bot. "
            f"Fix the provider or DASHSCOPE_API_KEY and restart."
        )
