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
import os

logger = logging.getLogger("lang_bot.scaffolds")

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
    """Persist the current scaffold index to disk."""
    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_index": idx}, f)


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
