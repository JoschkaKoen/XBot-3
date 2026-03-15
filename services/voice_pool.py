"""
Dynamic ElevenLabs voice pool.

On each bot run the pool is loaded from data/voice_pool.json.
If the pool has fewer than TARGET_POOL_SIZE voices for the configured language,
new voices are discovered via the ElevenLabs Shared Voices API, added to the
account, and appended to the pool — up to GROW_BATCH_SIZE new voices per run.

Pool JSON schema (one file, all languages):
  [
    {
      "name":            str,
      "voice_id":        str,
      "public_owner_id": str,
      "description":     str,
      "language":        str,   e.g. "de"
      "gender":          str,   e.g. "female" / "male"
      "added_at":        ISO-timestamp
    },
    ...
  ]
"""

import json
import logging
import os
from datetime import datetime, timezone

import requests

from config import ELEVENLABS_API_KEY
from utils.io import atomic_json_write

logger = logging.getLogger("german_bot.voice_pool")

_POOL_FILE      = "data/voice_pool.json"
_EL_BASE        = "https://api.elevenlabs.io/v1"
TARGET_POOL_SIZE = 1000  # voices per language
_GROW_BATCH      = 50   # max new voices added per bot run


# ── Persistence ───────────────────────────────────────────────────────────────

def load_pool(language: str = "de") -> list:
    """Return pool entries for *language* from disk. Empty list if file missing."""
    try:
        with open(_POOL_FILE, encoding="utf-8") as f:
            pool = json.load(f)
        return [v for v in pool if v.get("language") == language]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_full_pool() -> list:
    """Return the entire pool (all languages)."""
    try:
        with open(_POOL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_pool(pool: list) -> None:
    atomic_json_write(_POOL_FILE, pool, ensure_ascii=False, indent=2)


# ── ElevenLabs API helpers ────────────────────────────────────────────────────

def _headers() -> dict:
    return {"xi-api-key": ELEVENLABS_API_KEY}


def _search_shared_voices(language: str, page_size: int = 100) -> list:
    """
    Search the ElevenLabs shared voice library.
    Sorted by yearly usage so the most popular / battle-tested voices come first.
    Returns raw voice dicts from the API; empty list on any error.
    """
    try:
        resp = requests.get(
            f"{_EL_BASE}/shared-voices",
            headers=_headers(),
            params={
                "language":  language,
                "page_size": page_size,
                "sort":      "usage_character_count_1y",
            },
            timeout=20,
        )
        resp.raise_for_status()
        voices = resp.json().get("voices", [])
        logger.info("Shared voices search (%s): %d results.", language, len(voices))
        return voices
    except Exception as exc:
        logger.warning("Shared voices search failed: %s", exc)
        return []


def _add_voice_to_account(public_user_id: str, voice_id: str, name: str) -> bool:
    """
    Add a shared voice to the account so it can be used in TTS calls.
    Returns True on success (including "already in account" 422).
    """
    try:
        resp = requests.post(
            f"{_EL_BASE}/voices/add/{public_user_id}/{voice_id}",
            headers={**_headers(), "Content-Type": "application/json"},
            json={"new_name": name},
            timeout=20,
        )
        if resp.status_code == 200:
            logger.info("Added voice to account: %s (%s)", name, voice_id)
            return True
        if resp.status_code == 422:
            # Voice already present in the account — still usable
            logger.info("Voice already in account: %s (%s)", name, voice_id)
            return True
        logger.warning(
            "Add voice failed (HTTP %d): %s", resp.status_code, resp.text[:200]
        )
        return False
    except Exception as exc:
        logger.warning("Add voice request failed: %s", exc)
        return False


# ── Pool growth ────────────────────────────────────────────────────────────────

def grow_pool(language: str = "de", target_size: int = TARGET_POOL_SIZE) -> list:
    """
    Ensure the pool has at least *target_size* voices for *language*.

    - If already at target: returns the existing pool immediately (no API calls).
    - Otherwise: searches the shared library, skips voices already in the pool,
      adds up to _GROW_BATCH new voices to the account, persists the pool, and
      returns the updated language slice.
    """
    full_pool = _load_full_pool()
    lang_pool = [v for v in full_pool if v.get("language") == language]

    if len(lang_pool) >= target_size:
        return lang_pool

    slots_needed = target_size - len(lang_pool)
    to_add = min(slots_needed, _GROW_BATCH)
    logger.info(
        "Voice pool (%s): %d/%d — searching for up to %d new voices …",
        language, len(lang_pool), target_size, to_add,
    )

    existing_ids = {v["voice_id"] for v in full_pool}
    candidates   = _search_shared_voices(language, page_size=50)
    added        = 0

    for c in candidates:
        if added >= to_add:
            break

        voice_id = c.get("voice_id", "")
        if not voice_id or voice_id in existing_ids:
            continue

        name           = c.get("name", voice_id)
        public_user_id = c.get("public_owner_id") or c.get("public_user_id", "")
        if not public_user_id:
            logger.warning("No public_owner_id for voice '%s' — skipping.", name)
            continue

        # Build a human-readable description from all available fields so the
        # AI voice picker has rich signal (accent, age, use-case, tone, etc.)
        labels = c.get("labels") or {}
        parts = []
        for field in ("gender", "age", "accent"):
            val = c.get(field) or labels.get(field)
            if val:
                parts.append(str(val).strip())
        use_case = c.get("use_case") or labels.get("use case") or labels.get("use_case")
        if use_case:
            parts.append(str(use_case).strip())
        # labels may also carry a free-form "description" key
        for desc_src in (labels.get("description"), c.get("description")):
            if desc_src:
                parts.append(str(desc_src).strip()[:100])
                break
        description = ", ".join(parts) if parts else "no description"

        if not _add_voice_to_account(public_user_id, voice_id, name):
            continue

        entry = {
            "name":            name,
            "voice_id":        voice_id,
            "public_owner_id": public_user_id,
            "description":     description,
            "language":        language,
            "gender":          c.get("gender", ""),
            "added_at":        datetime.now(timezone.utc).isoformat(),
        }
        full_pool.append(entry)
        existing_ids.add(voice_id)
        added += 1
        logger.info("Pool +1: %s (%s) — %s", name, voice_id, description)

    if added:
        _save_pool(full_pool)
        logger.info(
            "Voice pool saved: %d total voices (%d new this run).",
            len(full_pool), added,
        )
    else:
        logger.info("Voice pool: no new voices added this run.")

    return [v for v in full_pool if v.get("language") == language]
