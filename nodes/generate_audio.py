"""
Node: generate_audio

Generates TTS audio via the xAI (Grok) Voice API for the source-language
example sentence. Uses generate_source_audio_with_timings() (ktv mode, returns
word timing data) or generate_source_audio() (simple mode, audio only).

================================================================================
 TUNABLE CONSTANTS
================================================================================
  config.TTS_SPEED  — TTS playback speed (xAI accepts 0.7–1.5; 0.70 = ~30 %
                      slower than normal, readable for language learners).
                      Honored directly by the xAI REST endpoint.

================================================================================
 VOICES
================================================================================
  xAI exposes a built-in voice library (GET /v1/tts/voices): named, native
  per-language voices (with gender + age) plus 5 expressive multilingual voices
  (eve, ara, rex, sal, leo). services/xai_tts.list_voices() fetches that library
  and filters it to the configured source language. The AI picks the best voice
  for each tweet based on the tweet's mood and topic; when the image prompt
  depicts a clearly male or female subject, the list is restricted to that
  gender first so TTS matches the image.
================================================================================

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   example_sentence_source, full_tweet, midjourney_prompt
  Writes to state:    clean_audio_path, word_timings, image_subject_gender
  Side effects:       writes MP3 to Voices/
================================================================================
"""

import os
import re
import logging
from datetime import datetime

import config
from config import VOICES_DIR
from utils.ui import stage_banner, ok, warn as ui_warn, info as ui_info
from services.ai_client import get_ai_response
from services.xai_tts import tts_to_file, list_voices

logger = logging.getLogger("xbot.generate_audio")

os.makedirs(VOICES_DIR, exist_ok=True)


def _parse_subject_gender(raw: str) -> str:
    """Normalize model output to male | female | neutral."""
    if not raw or not raw.strip():
        return "neutral"
    m = re.search(r"\b(female|male|neutral)\b", raw.lower())
    if m:
        return m.group(1)
    return "neutral"


def _infer_subject_gender_from_prompt(image_prompt: str) -> str:
    """
    Infer whether the focal subject in the image prompt is male, female, or neutral
    (no clear person / objects only), so TTS can match the voice gender.
    """
    if not image_prompt or not str(image_prompt).strip():
        return "neutral"
    user_msg = (
        "Classify the primary depicted subject for voice casting (spoken audio should match "
        "the apparent gender of the main character or person in the image).\n\n"
        "Rules:\n"
        "- One clear focal human or gendered character (man, woman, boy, girl, businessman, "
        "grandmother, prince, etc.) → male or female.\n"
        "- Several people: use the single clearest focal subject (usually foreground / center).\n"
        "- No people, only objects, landscapes, food, or animals with no clear gender → neutral.\n"
        "- If uncertain → neutral.\n\n"
        f"Image generation prompt:\n{str(image_prompt)[:6000]}\n\n"
        "Reply with exactly one word: male, female, or neutral."
    )
    system = "Reply with exactly one word: male, female, or neutral. No punctuation or explanation."
    try:
        raw = get_ai_response(
            config.SUBJECT_GENDER_MODEL,
            user_msg,
            system,
            max_tokens=15,
            temperature=0.0,
            retry_label="subject_gender",
        ).strip()
        g = _parse_subject_gender(raw)
        logger.info("Image subject gender for TTS: %s (model raw: %r)", g, raw[:120])
        return g
    except Exception as exc:
        logger.warning("Subject gender inference failed (%s) — using neutral.", exc)
        ui_warn(f"Could not infer image subject gender ({exc}) — voice gender not filtered.")
        return "neutral"


def _filter_pool_for_subject_gender(pool: list, subject_gender: str) -> list:
    """Restrict the voice list to male or female when the image subject is clearly gendered."""
    g = (subject_gender or "neutral").lower().strip()
    if g not in ("male", "female"):
        return pool
    filtered = [v for v in pool if (v.get("gender") or "").lower().strip() == g]
    if not filtered:
        logger.warning(
            "No %s voices available (%d total) — using full voice list for selection.",
            g,
            len(pool),
        )
        ui_warn(f"No {g} voices available — using full voice list.")
        return pool
    logger.info(
        "Voice list filtered to %s voices: %d of %d.",
        g,
        len(filtered),
        len(pool),
    )
    return filtered


def _pick_random_voice(pool: list) -> tuple:
    """Return a random (name, voice_id) from the voice list."""
    import random
    v = random.choice(pool)
    return v["name"], v["voice_id"]


def _pick_voice_by_ai(full_tweet: str, pool: list, subject_gender: str = "neutral") -> tuple:
    """
    Use AI to pick the most suitable voice for the given tweet.
    Returns (name, voice_id). Falls back to a random entry on any error.
    """
    import re as _re
    voice_list = "\n".join(
        f"{i + 1}. {v['name']} -- {v['description']}"
        for i, v in enumerate(pool)
    )
    sg = (subject_gender or "neutral").lower().strip()
    gender_line = ""
    if sg == "male":
        gender_line = (
            "The image for this post depicts a male subject — only male voices are listed; "
            "pick the best match for tweet mood among them.\n\n"
        )
    elif sg == "female":
        gender_line = (
            "The image for this post depicts a female subject — only female voices are listed; "
            "pick the best match for tweet mood among them.\n\n"
        )
    prompt = (
        f"You are selecting the best {config.SOURCE_LANGUAGE} text-to-speech voice for a tweet in a "
        "language-learning context.\n\n"
        + gender_line
        + f"Tweet:\n{full_tweet}\n\n"
        f"Available voices:\n{voice_list}\n\n"
        "Each voice entry shows: name -- gender, age, accent, use-case, and tone description.\n"
        "Choose the voice whose gender, age, accent, use-case, and overall character best fit "
        "the mood and content of the tweet (e.g. a playful tweet suits a young, warm voice; "
        "a news-style tweet suits a clear professional narrator; a calm reflective tweet suits "
        "a middle-aged, soothing voice).\n"
        "Reply with ONLY the number of the chosen voice (e.g. '7'). Nothing else."
    )
    system = (
        "You are a voice casting expert. "
        "Reply with only the number of the best-matching voice -- no explanation."
    )
    try:
        raw = get_ai_response(
            config.VOICE_PICKER_MODEL,
            prompt,
            system,
            max_tokens=10,
            temperature=0.0,
            retry_label="voice_pick",
        ).strip()
        m = _re.search(r'\b(\d+)\b', raw)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(pool):
                v = pool[idx]
                logger.info("AI picked voice %d: %s", idx + 1, v["name"])
                ui_info(f"AI selected voice: {v['name']} -- {v['description']}")
                return v["name"], v["voice_id"]
        logger.warning("Voice picker returned unparseable response %r -- using random.", raw)
        ui_warn(f"Voice picker returned unexpected response ({raw!r}) -- falling back to random voice.")
    except Exception as exc:
        logger.warning("Voice picker failed (%s) -- using random voice.", exc)
        ui_warn(f"Voice picker failed ({exc}) -- falling back to random voice.")
    name, voice_id = _pick_random_voice(pool)
    ui_info(f"Random voice selected: {name}")
    return name, voice_id


def generate_source_audio(
    text: str,
    output_file: str = None,
    voice_id: str = None,
    speed: float = None,
) -> str:
    """Generate TTS audio. Returns path to saved MP3."""
    if speed is None:
        speed = config.TTS_SPEED

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(VOICES_DIR, f"{config.SOURCE_LANGUAGE_CODE}_{ts}.mp3")

    logger.info("xAI TTS | voice_id=%s speed=%.2f | %.60s", voice_id, speed, text)
    path, _ = tts_to_file(
        text,
        voice_id,
        language=config.SOURCE_LANGUAGE_CODE or "auto",
        out_path=output_file,
        with_timestamps=False,
        speed=speed,
    )
    logger.info("Audio saved -> %s", path)
    return path


def generate_source_audio_with_timings(
    text: str,
    output_file: str = None,
    voice_id: str = None,
    speed: float = None,
) -> tuple:
    """
    Generate TTS audio with word-level timings.
    Returns (audio_path, word_timings).
    word_timings = [{'word': str, 'start': float, 'end': float}, ...]
    """
    if speed is None:
        speed = config.TTS_SPEED

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(VOICES_DIR, f"{config.SOURCE_LANGUAGE_CODE}_ktv_{ts}.mp3")

    logger.info("xAI TTS (timings) | voice_id=%s speed=%.2f | %.60s", voice_id, speed, text)
    path, alignment = tts_to_file(
        text,
        voice_id,
        language=config.SOURCE_LANGUAGE_CODE or "auto",
        out_path=output_file,
        with_timestamps=True,
        speed=speed,
    )
    logger.info("Audio (with timings) saved -> %s", path)

    word_timings = _character_alignment_to_word_timings(text, alignment)
    return path, word_timings


def _character_alignment_to_word_timings(original_text: str, alignment) -> list:
    """Convert per-character alignment -> clean word timings."""
    if not alignment:
        return _fallback_timings(original_text)

    # xAI normalizes to a dict {"characters": [...], "character_start_times_seconds": [...]};
    # also accept an attribute-style object defensively.
    if hasattr(alignment, "characters"):
        chars  = list(alignment.characters or [])
        starts = list(alignment.character_start_times_seconds or [])
    else:
        if hasattr(alignment, "__dict__"):
            alignment = alignment.__dict__
        chars  = alignment.get("characters", [])
        starts = alignment.get("character_start_times_seconds", [])

    if not chars or not starts:
        return _fallback_timings(original_text)

    words = original_text.split()
    word_timings = []
    char_idx = 0

    for word in words:
        word_start = None
        word_end = None
        for i in range(char_idx, len(chars)):
            remaining = original_text[char_idx:].lstrip()
            if remaining.startswith(word):
                word_start = starts[i] if i < len(starts) else 0.0
                for j in range(i + len(word), len(chars)):
                    if chars[j].strip() == "" or chars[j] in ".,!?":
                        word_end = starts[j] if j < len(starts) else starts[-1] + 0.4
                        break
                else:
                    word_end = starts[-1] + 0.4 if starts else 3.0
                break
            char_idx += 1

        if word_start is not None:
            word_timings.append({"word": word, "start": word_start, "end": word_end})
        char_idx += len(word) + 1

    return word_timings if word_timings else _fallback_timings(original_text)


def _fallback_timings(text: str) -> list:
    words = text.split()
    dur = 3.0 / max(len(words), 1)
    return [{"word": w, "start": i * dur, "end": (i + 1) * dur} for i, w in enumerate(words)]


# ---- node --------------------------------------------------------------------

def generate_audio(state: dict) -> dict:
    stage_banner(5)
    logger.info("Node: generate_audio")

    text: str       = state["example_sentence_source"]
    full_tweet: str = state.get("full_tweet", "")
    style: str      = config.VIDEO_STYLE

    pool = list_voices(config.SOURCE_LANGUAGE_CODE)
    ui_info(f"Voice library: {len(pool)} voices for {config.SOURCE_LANGUAGE}.")

    image_prompt = state.get("midjourney_prompt", "") or ""
    # Reuse a precomputed gender if the caller supplied one (the secondary-target
    # fan-out passes the base cycle's value to skip a redundant inference call).
    subject_gender = state.get("image_subject_gender") or _infer_subject_gender_from_prompt(image_prompt)
    ui_info(f"Image subject gender (voice match): {subject_gender}")
    pool_for_voice = _filter_pool_for_subject_gender(pool, subject_gender)

    if full_tweet:
        ui_info("Selecting voice with AI ...")
        voice_name, voice_id = _pick_voice_by_ai(full_tweet, pool_for_voice, subject_gender)
    else:
        voice_name, voice_id = _pick_random_voice(pool_for_voice)
        ui_info(f"Voice: {voice_name}")
    logger.info("Selected voice: %s (%s)", voice_name, voice_id)

    try:
        if style == "ktv":
            audio_path, word_timings = generate_source_audio_with_timings(text, voice_id=voice_id)
            ok(f"Audio + {len(word_timings)} word timings -> {os.path.basename(audio_path)}")
            return {
                **state,
                "clean_audio_path": audio_path,
                "word_timings": word_timings,
                "image_subject_gender": subject_gender,
            }
        else:
            audio_path = generate_source_audio(text, voice_id=voice_id)
            ok(f"Audio -> {os.path.basename(audio_path)}")
            return {
                **state,
                "clean_audio_path": audio_path,
                "word_timings": [],
                "image_subject_gender": subject_gender,
            }

    except Exception as exc:
        logger.warning("xAI TTS failed (%s) — no fallback audio used; re-raising.", exc)
        ui_warn(f"xAI TTS unavailable ({exc}) — skipping this cycle (no audio generated).")
        raise
