"""
Tweet scaffold pool and rotation.

Scaffold templates are defined in data/scaffolds.json.
Each entry has a "name" and a "template" string with these placeholders:
  [LEVEL]                              — CEFR level, e.g. A1, B2
  [ARTICLE]                            — grammatical article (omitted for non-nouns)
  [SOURCE_WORD]                        — the bare source-language word
  [TARGET_TRANSLATION]                 — translation of the word in the target language
  [SHORT_FUNNY_SOURCE_SENTENCE]        — example sentence in the source language
  [TARGET_TRANSLATION_OF_SENTENCE]     — target-language translation of the sentence
  [EMOJI1][EMOJI1]                     — first emoji pair (two identical emojis)
  [EMOJI2][EMOJI2]                     — second emoji pair (two identical emojis, may differ from EMOJI1)
  [SOURCE_FLAG] / [TARGET_FLAG]        — flag emojis substituted at runtime from config
  [SOURCE_LANGUAGE] / [TARGET_LANGUAGE]— language names substituted at runtime from config

Rotation is round-robin and persisted to data/scaffold_state.json so the
sequence survives restarts and is predictable.
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
