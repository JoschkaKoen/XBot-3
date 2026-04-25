"""
Curated random theme pool for vocabulary tweets (German → English learners).

Themes are short English angle strings injected as an ephemeral next_topic in
generate_content when USE_TRENDS cycle includes "pool".

Recent picks are tracked in data/theme_recent.json (gitignored) to reduce repeats.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from utils.io import atomic_json_write

logger = logging.getLogger("xbot.theme_random")

_THEMES_FILE = Path(__file__).resolve().parent.parent / "data" / "themes_german_for_english_learners.json"
_RECENT_FILE = Path(__file__).resolve().parent.parent / "data" / "theme_recent.json"
_RECENT_MAX = 28  # avoid reusing same theme for this many picks


def _load_themes() -> list[str]:
    try:
        with open(_THEMES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load theme bank %s: %s", _THEMES_FILE, exc)
    return []


def _load_recent() -> list[str]:
    try:
        with open(_RECENT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save_recent(items: list[str]) -> None:
    os.makedirs(_RECENT_FILE.parent, exist_ok=True)
    atomic_json_write(str(_RECENT_FILE), items[-_RECENT_MAX:], ensure_ascii=False, indent=2)


def pick_theme() -> str:
    """
    Return a random theme string not among the last _RECENT_MAX picks.
    Falls back to any theme if the bank is small or exhausted.
    """
    bank = _load_themes()
    if not bank:
        logger.warning("Theme bank empty — returning empty string.")
        return ""

    recent = _load_recent()
    recent_set = set(recent)
    candidates = [t for t in bank if t not in recent_set]
    if not candidates:
        candidates = bank

    choice = random.choice(candidates)
    recent.append(choice)
    _save_recent(recent)
    logger.info("Pool theme picked: %s", choice[:80] + ("…" if len(choice) > 80 else ""))
    return choice
