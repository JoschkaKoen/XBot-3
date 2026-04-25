"""
Node: analyze_and_improve

Reads the last N posts from history (ANALYZE_LAST_N), asks the LLM to identify
patterns (CEFR, themes, sentence styles that perform best), and outputs an
updated strategy dict fed into generate_content.

================================================================================
 FILES & SECTIONS YOU MAY EDIT
================================================================================
  data/strategy.json       — current strategy (also written by this node)
  data/strategy_history.json — append-only log of every strategy update
  _system_prompt()         — system prompt for the strategy LLM
  _DEFAULT_STRATEGY        — default keys when no history exists
  _DEFAULT_SCAFFOLD        — example tweet format shown to the LLM
================================================================================

================================================================================
 RELATED MODULES
================================================================================
  - nodes.score:        Provides load_history() and tweet performance metrics
  - nodes.generate_content: Consumes strategy for word/tweet generation
  - services.ai_client: AI response handling
  - config:             ANALYZE_LAST_N, STRATEGY_MODEL settings
================================================================================

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   metrics_refreshed (bool)
  Writes to state:    strategy (dict)
  Side effects:       writes data/strategy.json, appends data/strategy_history.json
================================================================================
"""

import json
import logging
import os
from datetime import datetime, timezone

import config
from config import HISTORY_FILE, STRATEGY_FILE, STRATEGY_HISTORY_FILE
from services.ai_client import get_ai_response
from nodes.score import load_history
from utils.io import atomic_json_write, safe_json_read
from utils.retry import retry_call
from utils.ui import stage_banner, ok, info as ui_info, warn as ui_warn


def _get_strategy_ai() -> callable:
    """Return the AI function to use for strategy analysis."""
    if config.AI_PROVIDER == "grok" and config.STRATEGY_MODEL == "reasoning":
        from services.grok_ai import get_grok_reasoning_response
        return get_grok_reasoning_response
    return get_ai_response


logger = logging.getLogger("xbot.analyze")


def _system_prompt() -> str:
    return (
        f"You are a data-driven social media strategist for a {config.SOURCE_LANGUAGE} language learning "
        "X (Twitter) account. "
        "You analyse past post performance and output a JSON strategy to improve future posts."
    )


_DEFAULT_SCAFFOLD = (
    "#Learn[SOURCE_LANGUAGE] [LEVEL]\n\n"
    "[SOURCE_FLAG]  [ARTICLE] [SOURCE_WORD]\n"
    "[TARGET_FLAG]  [TARGET_TRANSLATION]  [EMOJI][EMOJI]\n\n"
    "[SOURCE_FLAG]  [SHORT_FUNNY_SOURCE_SENTENCE]\n"
    "[TARGET_FLAG]  [TARGET_TRANSLATION_OF_SENTENCE]  [EMOJI][EMOJI]"
)

_ALL_CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
_CEFR_MIN_TWEETS = 10   # minimum tweets per level before CEFR bias is allowed

_DEFAULT_STRATEGY = {
    "preferred_cefr": "A1, A2, B1, B2, C1, C2",
    "next_topic": "",
    "style": "",
    "avoid_words": [],
    "scaffold": _DEFAULT_SCAFFOLD,
}


# ── strategy persistence ───────────────────────────────────────────────────────

def load_strategy() -> dict:
    """Load strategy from data/strategy.json, or return defaults if missing."""
    data = safe_json_read(STRATEGY_FILE, default=None, logger=logger)
    if data:
        return {**_DEFAULT_STRATEGY, **data}
    return dict(_DEFAULT_STRATEGY)


def _save_strategy(strategy: dict) -> None:
    atomic_json_write(STRATEGY_FILE, strategy, ensure_ascii=False, indent=2)


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
    atomic_json_write(STRATEGY_HISTORY_FILE, history, ensure_ascii=False, indent=2)


# ── diff logging ───────────────────────────────────────────────────────────────

_R     = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_GRAY  = "\033[90m"
_CYAN  = "\033[96m"


def _print_scaffold_diff(old_scaffold: str, new_scaffold: str) -> None:
    """
    Print a line-by-line diff of two scaffolds.
    Lines only in old  → red   with  - prefix
    Lines only in new  → green with  + prefix
    Lines in both      → gray  with    prefix
    """
    old_lines = old_scaffold.splitlines()
    new_lines = new_scaffold.splitlines()

    # Build sets for quick membership checks
    old_set = set(old_lines)
    new_set = set(new_lines)

    print(f"\n  {_CYAN}{_BOLD}📐  Scaffold updated:{_R}", flush=True)

    # Show removed lines first (from old, not in new)
    for line in old_lines:
        if line not in new_set:
            display = line if line.strip() else "(blank)"
            print(f"  {_RED}- {display}{_R}", flush=True)

    print(flush=True)

    # Show new scaffold with added lines highlighted
    for line in new_lines:
        if line in old_set:
            display = line if line.strip() else ""
            print(f"  {_GRAY}  {display}{_R}", flush=True)
        else:
            display = line if line.strip() else "(blank)"
            print(f"  {_GREEN}+ {display}{_R}", flush=True)

    print(flush=True)


def _log_strategy_diff(old: dict, new: dict) -> bool:
    """
    Print a human-readable diff to the terminal.
    Returns True if anything changed, False if identical.
    """
    _Y    = "\033[93m"

    scalar_keys = ["preferred_cefr", "next_topic", "style"]
    changed = False
    lines = []

    for key in scalar_keys:
        old_val = str(old.get(key, "")).strip()
        new_val = str(new.get(key, "")).strip()
        if old_val != new_val:
            changed = True
            lines.append(
                f"   {_GRAY}{key:<20}{_R} {_RED}\"{old_val}\"{_R}  →  {_GREEN}\"{new_val}\"{_R}"
            )

    # avoid_words: show add/remove counts rather than the full list
    old_words = set(old.get("avoid_words") or [])
    new_words = set(new.get("avoid_words") or [])
    added_w   = new_words - old_words
    removed_w = old_words - new_words
    if added_w or removed_w:
        changed = True
        parts = []
        if added_w:
            parts.append(f"{_GREEN}+{len(added_w)}{_R}")
        if removed_w:
            parts.append(f"{_RED}-{len(removed_w)}{_R}")
        lines.append(f"   {_GRAY}{'avoid_words':<20}{_R} {', '.join(parts)}")

    # Check scaffold before printing the summary so "No changes" is never printed
    # when only the scaffold changed.
    old_scaffold = old.get("scaffold", "")
    new_scaffold = new.get("scaffold", "")
    scaffold_changed = old_scaffold != new_scaffold and bool(new_scaffold)
    if scaffold_changed:
        changed = True

    print(f"\n  {_BOLD}📋  Strategy Update:{_R}", flush=True)
    if changed:
        for line in lines:
            print(line, flush=True)
        if not lines and scaffold_changed:
            pass   # scaffold diff will be printed below
    else:
        print(f"   {_GRAY}No changes — keeping current approach.{_R}", flush=True)
    print(flush=True)

    if scaffold_changed:
        _print_scaffold_diff(old_scaffold, new_scaffold)
    else:
        print(f"  {_CYAN}📐  Scaffold:{_R} {_GRAY}unchanged{_R}\n", flush=True)

    return changed


# ── analysis prompt ────────────────────────────────────────────────────────────

def _build_analysis_prompt(history_slice: list, current_scaffold: str, funny_mode: bool = False, cefr_frozen: bool = False) -> str:
    from nodes.score import tweet_age_hours, normalized_score

    posts_summary = json.dumps(
        [
            {
                "word": r.get("source_word"),
                "cefr": r.get("cefr_level"),
                "sentence_source": r.get("example_sentence_source"),
                "age_hours": tweet_age_hours(r),
                "score_raw": r.get("engagement_score", 0),
                "score_per_hour": normalized_score(r),
                "likes": r.get("metrics", {}).get("like_count", 0),
                "reposts": r.get("metrics", {}).get("retweet_count", 0),
                "replies": r.get("metrics", {}).get("reply_count", 0),
            }
            for r in history_slice
        ],
        ensure_ascii=False,
        indent=2,
    )

    # Summarise which theme-words have appeared recently so the AI can avoid them
    recent_words = [r.get("source_word", "") for r in history_slice if r.get("source_word")]
    recent_words_str = ", ".join(recent_words) if recent_words else "none"

    return (
        f"Here are the last {len(history_slice)} posts with their engagement data:\n\n"
        f"{posts_summary}\n\n"
        "IMPORTANT: use 'score_per_hour' (engagement divided by tweet age in hours) as the primary "
        "performance metric — NOT 'score_raw'. A tweet that is 10 hours old with score_per_hour=0.5 "
        "outperforms a tweet that is 100 hours old with score_per_hour=0.1, even if the older tweet "
        "has a higher raw score.\n\n"
        f"Recently used words (avoid clustering around the same topics): {recent_words_str}\n\n"
        "Based on this data, output a JSON object with these keys:\n"
        + (
            '  "preferred_cefr":  (string) IGNORE — this field is controlled by the system and will be overridden. '
            'Set it to "A1, A2, B1, B2, C1, C2" — there is not yet enough data per level to bias it.\n'
            if cefr_frozen else
            '  "preferred_cefr":  (string) comma-separated CEFR levels that performed best, '
            'chosen from A1, A2, B1, B2, C1, C2, e.g. "A2, B1, C1"\n'
        )
        +         f'  "next_topic":      (string) ONE fresh topic or angle for the next tweet that has NOT been covered '
        f'in recent posts and that you anticipate will resonate with {config.TARGET_LANGUAGE}-speaking {config.SOURCE_LANGUAGE} learners. '
        'Pick something new and specific — do NOT repeat any theme already seen in the recent post list. '
        f'Examples: "{config.SOURCE_LANGUAGE} workplace culture", "untranslatable {config.SOURCE_LANGUAGE} concepts", "{config.SOURCE_LANGUAGE} food idioms", '
        f'"emotions expressed in {config.SOURCE_LANGUAGE} that {config.TARGET_LANGUAGE} lacks a word for". '
        'Leave empty string "" if you cannot identify a genuinely fresh angle.\n'
        '  "style":           (string) a short instruction about sentence style, tone, grammatical patterns, '
        'or humour style for the next tweet — STYLE ONLY, NO topics or themes. '
        + (
            'DO NOT mention any specific CEFR levels (A1/A2/B1/B2/C1/C2) here. '
            if cefr_frozen else
            ('NOTE: tone/humour direction is controlled externally — do NOT override it here. '
             if funny_mode else '')
        )
        + 'Examples of good style instructions: "use a twist ending", "use self-aware irony", '
        '"write in second person (du)", "use short punchy sentences". '
        'Examples of BAD style instructions (DO NOT do this): "focus on food topics", "use daily life themes".\n'
        f'  "avoid_words":     (array)  list of {config.SOURCE_LANGUAGE} words recently used that should not be repeated\n\n'
        "Return ONLY the raw JSON object. No markdown, no explanation."
    )


def _cefr_counts(history: list) -> dict:
    """Return a dict of {level: tweet_count} for all CEFR levels."""
    counts = {level: 0 for level in _ALL_CEFR_LEVELS}
    for r in history:
        level = (r.get("cefr_level") or "").strip().upper()
        if level in counts:
            counts[level] += 1
    return counts


def _cefr_frozen(history: list) -> tuple[bool, dict]:
    """
    Return (frozen, counts).
    frozen=True when any CEFR level has fewer than _CEFR_MIN_TWEETS tweets,
    meaning we don't yet have enough data to bias towards a level.
    """
    counts = _cefr_counts(history)
    frozen = any(c < _CEFR_MIN_TWEETS for c in counts.values())
    return frozen, counts


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

    # Skip full strategy analysis if metrics were not refreshed this cycle,
    # but always refresh avoid_words from post history regardless.
    if not state.get("metrics_refreshed", True):
        old_strategy = load_strategy()
        if not getattr(config, "STRATEGY_METRICS_UPDATES_ENABLED", True):
            ui_info(
                "Strategy analysis skipped — STRATEGY_UPDATE_INTERVAL_HOURS=false "
                "(metrics + strategy updates disabled)."
            )
        else:
            ui_info("Metrics not refreshed — skipping strategy analysis, reusing current strategy.")
        logger.info("Strategy analysis skipped (metrics_refreshed=False).")

        # Always keep avoid_words current from post history.
        history = load_history()
        recent_words = [r.get("source_word", "") for r in history[-30:] if r.get("source_word")]
        updated_strategy = {**old_strategy, "avoid_words": list(dict.fromkeys(recent_words))}
        if updated_strategy["avoid_words"] != old_strategy.get("avoid_words", []):
            _save_strategy(updated_strategy)
            logger.info(
                "avoid_words refreshed from history (%d words).",
                len(updated_strategy["avoid_words"]),
            )
        return {**state, "strategy": updated_strategy}

    # Load the previous strategy to diff against
    old_strategy = load_strategy()

    history = load_history()
    if len(history) < 2:
        ui_info(f"Not enough history yet ({len(history)} record(s)) — keeping current strategy.")
        logger.info("Not enough history (%d records) — using current strategy.", len(history))
        return {**state, "strategy": old_strategy}

    history_slice = history[-config.ANALYZE_LAST_N:]
    current_scaffold = old_strategy.get("scaffold", _DEFAULT_SCAFFOLD)
    frozen, cefr_counts_pre = _cefr_frozen(history)
    prompt = _build_analysis_prompt(history_slice, current_scaffold, funny_mode="funny" in config.TWEET_STYLE_CYCLE, cefr_frozen=frozen)

    strategy_ai = _get_strategy_ai()
    model_label = "grok-reasoning" if (config.AI_PROVIDER == "grok" and config.STRATEGY_MODEL == "reasoning") else "default"
    logger.info("Running strategy analysis with model: %s", model_label)

    raw_strategy: str = retry_call(
        strategy_ai,
        prompt,
        _system_prompt(),
        max_tokens=1200,
        temperature=0.4,
        label="analyze_strategy",
    )
    logger.debug("Raw strategy response: %s", raw_strategy)

    new_strategy = _parse_strategy(raw_strategy)

    # Merge with defaults so all keys are always present.
    # Scaffold is locked — always keep the current scaffold unchanged.
    merged = {**_DEFAULT_STRATEGY, **new_strategy}
    merged["scaffold"] = current_scaffold

    # Freeze CEFR bias until every level has at least _CEFR_MIN_TWEETS tweets.
    if frozen:
        merged["preferred_cefr"] = ", ".join(_ALL_CEFR_LEVELS)
        # Safety-net: strip any stray CEFR level tokens the LLM may have written into 'focus'
        import re as _re
        merged["style"] = _re.sub(
            r"\b(A1|A2|B1|B2|C1|C2)\b[\s,/]*",
            "",
            merged.get("style", ""),
            flags=_re.IGNORECASE,
        ).strip(" ,;—-")
        under = {lvl: n for lvl, n in cefr_counts_pre.items() if n < _CEFR_MIN_TWEETS}
        ui_info(
            f"CEFR bias frozen — insufficient data for: "
            + ", ".join(f"{lvl}({n})" for lvl, n in sorted(under.items()))
            + f" (need {_CEFR_MIN_TWEETS} each)"
        )
        logger.info("CEFR bias frozen. counts=%s", cefr_counts_pre)

    # Always accumulate avoid_words from the full history (last 30)
    recent_words = [r.get("source_word", "") for r in history[-30:] if r.get("source_word")]
    merged["avoid_words"] = list(dict.fromkeys(recent_words))   # deduplicate, preserve order

    # Log diff vs previous strategy
    _log_strategy_diff(old_strategy, merged)

    # Persist new strategy
    _save_strategy(merged)
    _append_strategy_history(merged)

    ok(f"Strategy saved — CEFR: {merged['preferred_cefr']} | Style: {merged['style'] or 'none'}")
    ui_info(f"Next topic: {merged['next_topic'] or 'none'}")
    ui_info(f"Avoiding {len(merged['avoid_words'])} recent word(s)")
    logger.info("Updated strategy:")
    logger.info("  topic : %s", merged.get("next_topic", "—"))
    logger.info("  style : %s", merged.get("style", "—"))
    logger.info("  CEFR  : %s", merged.get("preferred_cefr", "—"))
    logger.info("Avoid words (%d): %s", len(merged["avoid_words"]), merged["avoid_words"][-10:])

    return {**state, "strategy": merged}
