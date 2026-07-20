"""
Tweet scaffold pool and rotation.

================================================================================
 WHERE TO EDIT TEMPLATES
================================================================================
  data/scaffolds.json  — one object per scaffold with "name" and "template".
  Each "template" is an array of lines (joined with newlines at runtime).

================================================================================
 PLACEHOLDERS (substituted in generate_content._expand_scaffold)
================================================================================
  [LEVEL]                        — CEFR level, e.g. A1, B2
  [ARTICLE]                      — grammatical article (omitted for non-nouns)
  [SOURCE_WORD]                  — the source-language word
  [TARGET_TRANSLATION]            — translation of the word
  [SHORT_FUNNY_SOURCE_SENTENCE]   — example sentence (source language)
  [TARGET_TRANSLATION_OF_SENTENCE]— translation of the example sentence
  [EMOJI1] / [EMOJI2]            — emojis chosen by the LLM for the tweet
  [SOURCE_FLAG] / [TARGET_FLAG]   — from config (e.g. 🇩🇪, 🇺🇸)
  [SOURCE_LANGUAGE] / [TARGET_LANGUAGE] — from config (e.g. German, English)

Rotation is round-robin and persisted to data/scaffold_state.json so the
sequence survives restarts.
"""

import json
import logging
import re

logger = logging.getLogger("xbot.scaffolds")

_SCAFFOLDS_FILE = "data/scaffolds.json"
_STATE_FILE     = "data/scaffold_state.json"


# ── pool ──────────────────────────────────────────────────────────────────────

def _load_pool() -> list[tuple[str, str]]:
    """Load scaffolds from data/scaffolds.json and return as (name, template) tuples.

    Templates are stored as arrays of lines for readability and joined here.
    """
    with open(_SCAFFOLDS_FILE, encoding="utf-8") as f:
        entries = json.load(f)
    return [
        (e["name"], "\n".join(e["template"]) if isinstance(e["template"], list) else e["template"])
        for e in entries
    ]


# ── rotation ──────────────────────────────────────────────────────────────────

def _load_index() -> int:
    """Read the last-used scaffold index from disk (0-based)."""
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            return int(json.load(f).get("last_index", -1))
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return -1


def _save_index(idx: int) -> None:
    """Persist the current scaffold index to disk (atomic write-then-rename)."""
    from utils.io import atomic_json_write
    atomic_json_write(_STATE_FILE, {"last_index": idx})


def next_scaffold() -> tuple[str, str]:
    """
    Return the next (name, template) in round-robin order and advance the
    persisted index so the next call picks the following scaffold.
    """
    pool = _load_pool()
    last = _load_index()
    idx  = (last + 1) % len(pool)
    _save_index(idx)
    name, template = pool[idx]
    logger.info("Scaffold rotation: %d/%d — %s", idx + 1, len(pool), name)
    return name, template


# ── stateless helpers (shared with the secondary fan-out) ──────────────────────

def scaffold_at(index: int) -> tuple[str, str]:
    """Return the (name, template) at *index* modulo the pool — no state I/O.

    Lets a caller drive rotation from its own counter (a secondary target
    rotates on len(its own history)) without touching the primary path's
    persisted data/scaffold_state.json.
    """
    pool = _load_pool()
    name, template = pool[index % len(pool)]
    logger.info("Scaffold pick: %d/%d — %s", (index % len(pool)) + 1, len(pool), name)
    return name, template


def fill_scaffold(template: str, values: dict) -> str:
    """Substitute ``[PLACEHOLDER]`` tokens in *template* from *values*.

    *values* keys are bare placeholder names without brackets, e.g.
    ``{"SOURCE_FLAG": "🇬🇧", "SOURCE_WORD": "notebook", ...}``.

    - An empty/absent ``ARTICLE`` drops the token *and* one trailing space, so
      ``"[SOURCE_FLAG]  [ARTICLE] [SOURCE_WORD]"`` collapses cleanly instead of
      leaving a stray gap (English has no article; German nouns do). The
      deliberate two-space gaps after flags are preserved.
    - Bracket-closed tokens can't prefix-collide (``[TARGET_TRANSLATION]`` is not
      a substring of ``[TARGET_TRANSLATION_OF_SENTENCE]``), so fill order is free.
    - Any placeholder left unfilled is stripped and logged, so a future scaffold
      edit adding an unsupplied token can't ship literal ``[FOO]`` text.
    - Collapses 3+ consecutive newlines to 2.
    """
    article = (values.get("ARTICLE") or "").strip()
    out = template
    if article:
        out = out.replace("[ARTICLE]", article)
    else:
        out = out.replace("[ARTICLE] ", "").replace("[ARTICLE]", "")

    for key, val in values.items():
        if key == "ARTICLE":
            continue
        out = out.replace(f"[{key}]", str(val))

    leftovers = sorted(set(re.findall(r"\[[A-Z0-9_]+\]", out)))
    if leftovers:
        logger.warning("fill_scaffold: unfilled placeholders stripped: %s", ", ".join(leftovers))
        for tok in leftovers:
            out = out.replace(tok, "")

    # Trim end-of-line whitespace (e.g. a dropped [EMOJI] leaving a trailing gap);
    # the deliberate two-space gaps are mid-line, so they survive. Then collapse
    # blank runs the substitutions may have opened up.
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", out).strip()
