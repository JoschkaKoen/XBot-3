"""
Auto-derive language-pair configuration from SOURCE_LANGUAGE / TARGET_LANGUAGE.

The derived parameters (language codes, flag emojis, flag colors, trends country
slug) are resolved once via the Grok non-reasoning model and cached in
data/language_config.json.  On subsequent runs the cache is reused as long as the
language pair hasn't changed.  If the user switches languages, the cache is
invalidated and a fresh AI call is made.
"""

import json
import logging
import os

logger = logging.getLogger("xbot.language_config")

_CACHE_FILE = "data/language_config.json"
_GROK_MODEL = "grok-4-1-fast-non-reasoning"
_GROK_BASE  = "https://api.x.ai/v1"


def _load_cache() -> dict:
    from utils.io import safe_json_read
    return safe_json_read(_CACHE_FILE, default={}, logger=logger)


def _save_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_FILE) or ".", exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _call_ai(source_language: str, target_language: str) -> dict:
    """Use Grok non-reasoning to derive all language-pair config fields."""
    from openai import OpenAI
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        raise ValueError("XAI_API_KEY not set — cannot auto-derive language config.")

    client = OpenAI(base_url=_GROK_BASE, api_key=api_key)

    prompt = (
        f"I am building a language learning X (Twitter) bot.\n"
        f"Source language (being taught): {source_language}\n"
        f"Target language (learner's native language): {target_language}\n\n"
        "Return ONLY a valid JSON object with exactly these fields:\n"
        "{\n"
        f'  "source_language_code": "<ISO 639-1 two-letter code for {source_language}, e.g. de>",\n'
        f'  "target_language_code": "<ISO 639-1 two-letter code for {target_language}, e.g. en>",\n'
        f'  "source_flag": "<single flag emoji for the primary country where {source_language} is the official language>",\n'
        f'  "target_flag": "<single flag emoji for the primary country where {target_language} is the official language>",\n'
        f'  "trends_country": "<lowercase country slug used on getdaytrends.com for {source_language}, e.g. germany, france, spain, japan>",\n'
        f'  "source_flag_colors": "<three representative colors of the {source_language} country flag, as six-digit hex RGB values, comma-separated, e.g. 000000,DD0000,FFCE00>",\n'
        f'  "target_flag_colors": "<three representative colors of the {target_language} country flag, as six-digit hex RGB values, comma-separated, e.g. B22234,FFFFFF,3C3B6E>"\n'
        "}\n\n"
        "Rules:\n"
        "- Use only the JSON object, no explanation or markdown.\n"
        "- Flag colors: pick the three most visually distinctive colors of the national flag.\n"
        "- trends_country: use the exact slug that appears in getdaytrends.com URLs, e.g. getdaytrends.com/germany/ → germany.\n"
    )

    response = client.chat.completions.create(
        model=_GROK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a language configuration assistant. Reply only with a valid JSON object — no markdown, no explanation.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)


def resolve(source_language: str, target_language: str) -> dict:
    """
    Return the derived language-pair config dict.

    Uses the cache when the language pair matches the cached one.
    Re-calls the AI (and updates the cache) when the language pair has changed.
    """
    cache = _load_cache()

    if (
        cache.get("source_language", "").strip().lower() == source_language.strip().lower()
        and cache.get("target_language", "").strip().lower() == target_language.strip().lower()
    ):
        logger.info(
            "Language config loaded from cache: %s → %s.",
            source_language, target_language,
        )
        return cache

    logger.info(
        "Language pair changed or no cache found — deriving config for %s → %s via AI …",
        source_language, target_language,
    )
    print(
        f"\n  🌐  Auto-configuring language pair: {source_language} → {target_language} …",
        flush=True,
    )

    derived = _call_ai(source_language, target_language)
    derived["source_language"] = source_language
    derived["target_language"] = target_language
    _save_cache(derived)

    print(
        f"  ✅  Language config saved  "
        f"({derived.get('source_flag','')} {derived.get('source_language_code','')} / "
        f"{derived.get('target_flag','')} {derived.get('target_language_code','')})",
        flush=True,
    )
    logger.info("Language config derived and cached: %s", derived)
    return derived
