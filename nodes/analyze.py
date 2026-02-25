"""
Node: analyze_and_improve

Reads the last N posts from history, asks the LLM to identify patterns
(which CEFR levels, themes, sentence styles perform best), and outputs
an updated strategy dict that is fed into generate_content.

Strategy is persisted to data/strategy.json so it survives restarts.
Every version is also appended to data/strategy_history.json.
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import HISTORY_FILE, ANALYZE_LAST_N, STRATEGY_FILE, STRATEGY_HISTORY_FILE, STRATEGY_MODEL, AI_PROVIDER
from services.ai_client import get_ai_response
from nodes.score import _load_history


def _get_strategy_ai() -> callable:
    """Return the AI function to use for strategy analysis."""
    if AI_PROVIDER == "grok" and STRATEGY_MODEL == "reasoning":
        from services.grok_ai import get_grok_reasoning_response
        return get_grok_reasoning_response
    return get_ai_response
from utils.retry import retry_call
from utils.ui import stage_banner, ok, info as ui_info, warn as ui_warn

logger = logging.getLogger("german_bot.analyze")


_SYSTEM_PROMPT = (
    "You are a data-driven social media strategist for a German language learning X (Twitter) account. "
    "You analyse past post performance and output a JSON strategy to improve future posts."
)


_DEFAULT_STRATEGY = {
    "preferred_cefr": "A1, A2, B1, B2, C1, C2",
    "preferred_themes": "food, travel, daily life, emotions, work, weather, relationships",
    "focus": "",
    "avoid_words": [],
}


# ── strategy persistence ───────────────────────────────────────────────────────

def load_strategy() -> dict:
    """Load strategy from data/strategy.json, or return defaults if missing."""
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                return {**_DEFAULT_STRATEGY, **data}
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not read strategy file: %s — using defaults.", exc)
    return dict(_DEFAULT_STRATEGY)


def _save_strategy(strategy: dict) -> None:
    os.makedirs(os.path.dirname(STRATEGY_FILE) or ".", exist_ok=True)
    with open(STRATEGY_FILE, "w", encoding="utf-8") as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)


def _append_strategy_history(strategy: dict) -> None:
    history: list = []
    if os.path.exists(STRATEGY_HISTORY_FILE):
        try:
            with open(STRATEGY_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **strategy}
    history.append(entry)
    history = history[-50:]   # keep last 50 versions
    with open(STRATEGY_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── diff logging ───────────────────────────────────────────────────────────────

def _log_strategy_diff(old: dict, new: dict) -> bool:
    """
    Print a human-readable diff to the terminal.
    Returns True if anything changed, False if identical.
    """
    scalar_keys = ["preferred_cefr", "preferred_themes", "focus"]
    changed = False
    lines = []

    for key in scalar_keys:
        old_val = str(old.get(key, "")).strip()
        new_val = str(new.get(key, "")).strip()
        if old_val != new_val:
            changed = True
            lines.append(f"   {key:<20} \"{old_val}\"  →  \"{new_val}\"")

    # avoid_words: show add/remove counts rather than the full list
    old_words = set(old.get("avoid_words") or [])
    new_words = set(new.get("avoid_words") or [])
    added   = new_words - old_words
    removed = old_words - new_words
    if added or removed:
        changed = True
        parts = []
        if added:
            parts.append(f"added {len(added)}")
        if removed:
            parts.append(f"removed {len(removed)}")
        lines.append(f"   avoid_words            {', '.join(parts)}")

    print("\n  📋  Strategy Update:", flush=True)
    if changed:
        for line in lines:
            print(line, flush=True)
    else:
        print("   No changes — keeping current approach.", flush=True)
    print(flush=True)

    return changed


# ── analysis prompt ────────────────────────────────────────────────────────────

def _build_analysis_prompt(history_slice: list) -> str:
    posts_summary = json.dumps(
        [
            {
                "word": r.get("german_word"),
                "cefr": r.get("cefr_level"),
                "sentence_de": r.get("example_sentence_de"),
                "score": r.get("engagement_score", 0),
                "likes": r.get("metrics", {}).get("like_count", 0),
                "reposts": r.get("metrics", {}).get("retweet_count", 0),
                "replies": r.get("metrics", {}).get("reply_count", 0),
            }
            for r in history_slice
        ],
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"Here are the last {len(history_slice)} posts with their engagement scores:\n\n"
        f"{posts_summary}\n\n"
        "Based on this data, output a JSON object with these keys:\n"
        '  "preferred_cefr":    (string) comma-separated CEFR levels that performed best, '
        'chosen from A1, A2, B1, B2, C1, C2, e.g. "A2, B1, C1"\n'
        '  "preferred_themes":  (string) comma-separated themes to focus on next, e.g. "food, travel, emotions"\n'
        '  "focus":             (string) a short additional instruction for the next post, e.g. "use funnier sentences"\n'
        '  "avoid_words":       (array)  list of German words recently used that should not be repeated\n\n'
        "Return ONLY the raw JSON object. No markdown, no explanation."
    )


def _parse_strategy(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse strategy JSON: %s — using defaults.", exc)
        return {}


# ── node ──────────────────────────────────────────────────────────────────────

def analyze_and_improve(state: dict) -> dict:
    stage_banner(2)
    logger.info("Node: analyze_and_improve")

    # Load the previous strategy to diff against
    old_strategy = load_strategy()

    history = _load_history()
    if len(history) < 2:
        ui_info(f"Not enough history yet ({len(history)} record(s)) — keeping current strategy.")
        logger.info("Not enough history (%d records) — using current strategy.", len(history))
        return {**state, "strategy": old_strategy}

    history_slice = history[-ANALYZE_LAST_N:]
    prompt = _build_analysis_prompt(history_slice)

    strategy_ai = _get_strategy_ai()
    model_label = "grok-reasoning" if (AI_PROVIDER == "grok" and STRATEGY_MODEL == "reasoning") else "default"
    logger.info("Running strategy analysis with model: %s", model_label)

    raw_strategy: str = retry_call(
        strategy_ai,
        prompt,
        _SYSTEM_PROMPT,
        max_tokens=1200,
        temperature=0.4,
        label="analyze_strategy",
    )
    logger.debug("Raw strategy response: %s", raw_strategy)

    new_strategy = _parse_strategy(raw_strategy)

    # Merge with defaults so all keys are always present
    merged = {**_DEFAULT_STRATEGY, **new_strategy}

    # Always accumulate avoid_words from the full history (last 30)
    recent_words = [r.get("german_word", "") for r in history[-30:] if r.get("german_word")]
    merged["avoid_words"] = list(dict.fromkeys(recent_words))   # deduplicate, preserve order

    # Log diff vs previous strategy
    _log_strategy_diff(old_strategy, merged)

    # Persist new strategy
    _save_strategy(merged)
    _append_strategy_history(merged)

    ok(f"Strategy saved — CEFR: {merged['preferred_cefr']} | Focus: {merged['focus'] or 'none'}")
    ui_info(f"Themes: {merged['preferred_themes']}")
    ui_info(f"Avoiding {len(merged['avoid_words'])} recent word(s)")
    logger.info("Updated strategy: %s", {k: v for k, v in merged.items() if k != "avoid_words"})
    logger.info("Avoid words (%d): %s", len(merged["avoid_words"]), merged["avoid_words"][-10:])

    return {**state, "strategy": merged}
