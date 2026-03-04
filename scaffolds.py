"""
Tweet scaffold pool and rotation.

Each scaffold is a (name, template) tuple.  Placeholders the AI must fill:
  [LEVEL]                      — CEFR level, e.g. A1, B2
  [ARTICLE]                    — der / die / das  (omitted for non-nouns)
  [GERMAN_WORD]                — the bare German word
  [ENGLISH_TRANSLATION]        — English translation of the word
  [SHORT_FUNNY_GERMAN_SENTENCE]— example sentence in German
  [ENGLISH_TRANSLATION_OF_SENTENCE] — English translation of the sentence
  [EMOJI1][EMOJI1]             — first emoji pair (two identical emojis)
  [EMOJI2][EMOJI2]             — second emoji pair (two identical emojis, may differ from EMOJI1)

Rotation is round-robin and persisted to data/scaffold_state.json so the
sequence survives restarts and is predictable.
"""

import json
import logging
import os

logger = logging.getLogger("german_bot.scaffolds")

_STATE_FILE = "data/scaffold_state.json"

# ── pool ──────────────────────────────────────────────────────────────────────

SCAFFOLD_POOL: list[tuple[str, str]] = [
    (
        "Classic",
        "#DeutschLernen [LEVEL]\n\n"
        "🇩🇪  [ARTICLE] [GERMAN_WORD]\n"
        "🇺🇸  [ENGLISH_TRANSLATION]  [EMOJI1][EMOJI1]\n\n"
        "🇩🇪  [SHORT_FUNNY_GERMAN_SENTENCE]\n"
        "🇺🇸  [ENGLISH_TRANSLATION_OF_SENTENCE]  [EMOJI2][EMOJI2]",
    ),
    (
        "Flip Card",
        "🇩🇪 → 🇺🇸\n\n"
        "[ARTICLE] [GERMAN_WORD] = [ENGLISH_TRANSLATION]  [EMOJI1][EMOJI1]\n\n"
        "📝 [SHORT_FUNNY_GERMAN_SENTENCE]\n"
        "✏️ [ENGLISH_TRANSLATION_OF_SENTENCE]  [EMOJI2][EMOJI2]\n\n"
        "#DeutschLernen [LEVEL]",
    ),
    (
        "Did You Know",
        "Did you know? 🤔\n\n"
        "🇩🇪  [ARTICLE] [GERMAN_WORD]  [EMOJI1][EMOJI1]\n"
        "🇺🇸  [ENGLISH_TRANSLATION]\n\n"
        "🗣️  [SHORT_FUNNY_GERMAN_SENTENCE]\n"
        "💬  [ENGLISH_TRANSLATION_OF_SENTENCE]  [EMOJI2][EMOJI2]\n\n"
        "#DeutschLernen [LEVEL]",
    ),
    (
        "Word of the Day",
        "🇩🇪 Word of the Day  [EMOJI1][EMOJI1]\n\n"
        "[ARTICLE] [GERMAN_WORD]\n"
        "→ [ENGLISH_TRANSLATION]\n\n"
        "🇩🇪  [SHORT_FUNNY_GERMAN_SENTENCE]\n"
        "🇺🇸  [ENGLISH_TRANSLATION_OF_SENTENCE]  [EMOJI2][EMOJI2]\n\n"
        "#DeutschLernen [LEVEL]",
    ),
    (
        "Reply Challenge",
        "🧠 New German word!  #DeutschLernen [LEVEL]\n\n"
        "🇩🇪  [ARTICLE] [GERMAN_WORD]  [EMOJI1][EMOJI1]\n"
        "🇺🇸  [ENGLISH_TRANSLATION]\n\n"
        "🇩🇪  [SHORT_FUNNY_GERMAN_SENTENCE]\n"
        "🇺🇸  [ENGLISH_TRANSLATION_OF_SENTENCE]  [EMOJI2][EMOJI2]\n\n"
        "↩️ Reply with your own sentence!",
    ),
    (
        "Pro Tip",
        "💡 #DeutschLernen [LEVEL]\n\n"
        "🇩🇪  [ARTICLE] [GERMAN_WORD]\n"
        "🇺🇸  [ENGLISH_TRANSLATION]  [EMOJI1][EMOJI1]\n\n"
        "\u201e[SHORT_FUNNY_GERMAN_SENTENCE]\u201c\n"
        "= [ENGLISH_TRANSLATION_OF_SENTENCE]  [EMOJI2][EMOJI2]\n\n"
        "🔖 Save for later!",
    ),
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
    last = _load_index()
    idx = (last + 1) % len(SCAFFOLD_POOL)
    _save_index(idx)
    name, template = SCAFFOLD_POOL[idx]
    logger.info("Scaffold rotation: %d/%d — %s", idx + 1, len(SCAFFOLD_POOL), name)
    return name, template
