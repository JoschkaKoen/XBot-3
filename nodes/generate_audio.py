"""
Node: generate_audio

Generates TTS audio via ElevenLabs.
Uses generate_german_audio_with_timings() for ktv mode,
or generate_german_audio() for simple mode.

Voice pool
----------
Voices are stored in data/voice_pool.json and grown automatically each run
via services/voice_pool.py (ElevenLabs Shared Voices API).
The AI picks the most suitable voice from the pool for each tweet.
"""

import os
import base64
import logging
from datetime import datetime

import config
from config import VOICES_DIR, ELEVENLABS_API_KEY
from utils.retry import with_retry, retry_call
from utils.ui import stage_banner, ok, warn as ui_warn, info as ui_info

from elevenlabs.client import ElevenLabs
from elevenlabs import save
from elevenlabs.types import VoiceSettings

logger = logging.getLogger("german_bot.generate_audio")

_DEFAULT_SPEED  = 0.70
_VOICE_LANGUAGE = "de"   # ElevenLabs language code for the TTS voice pool

os.makedirs(VOICES_DIR, exist_ok=True)


def _get_client() -> ElevenLabs:
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not found in .env!")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


def _voice_settings(speed: float) -> VoiceSettings:
    return VoiceSettings(stability=0.75, similarity_boost=0.85, speed=speed)


def _get_voice_picker_ai():
    """Return the AI function to use for voice selection."""
    from nodes.generate_content import _model_to_ai_fn
    return _model_to_ai_fn(config.VOICE_PICKER_MODEL)


def _pick_random_voice(pool: list) -> tuple:
    """Return a random (name, voice_id) from the pool."""
    import random
    v = random.choice(pool)
    return v["name"], v["voice_id"]


def _pick_voice_by_ai(full_tweet: str, pool: list) -> tuple:
    """
    Use AI to pick the most suitable voice for the given tweet.
    Returns (name, voice_id). Falls back to a random pool entry on any error.
    """
    import re as _re
    voice_list = "\n".join(
        f"{i + 1}. {v['name']} -- {v['description']}"
        for i, v in enumerate(pool)
    )
    prompt = (
        "You are selecting the best German text-to-speech voice for a tweet in a "
        "language-learning context.\n\n"
        f"Tweet:\n{full_tweet}\n\n"
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
        raw = retry_call(
            _get_voice_picker_ai(),
            prompt,
            system,
            max_tokens=10,
            temperature=0.0,
            label="voice_pick",
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


@with_retry(max_attempts=4, base_delay=3.0, label="elevenlabs_simple")
def generate_german_audio(
    text: str,
    output_file: str = None,
    voice_id: str = None,
    speed: float = _DEFAULT_SPEED,
) -> str:
    """Generate TTS audio. Returns path to saved MP3."""
    client = _get_client()
    if not (0.7 <= speed <= 1.2):
        logger.warning("Speed %.2f out of range, clamping to 0.70.", speed)
        speed = 0.70

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(VOICES_DIR, f"german_{ts}.mp3")

    logger.info("ElevenLabs TTS | voice_id=%s speed=%.2f | %.60s", voice_id, speed, text)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=_voice_settings(speed),
    )
    save(audio, output_file)
    logger.info("Audio saved -> %s", output_file)
    return output_file


@with_retry(max_attempts=4, base_delay=3.0, label="elevenlabs_timings")
def generate_german_audio_with_timings(
    text: str,
    output_file: str = None,
    voice_id: str = None,
    speed: float = 0.70,
) -> tuple:
    """
    Generate TTS audio with word-level timings.
    Returns (audio_path, word_timings).
    word_timings = [{'word': str, 'start': float, 'end': float}, ...]
    """
    client = _get_client()

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(VOICES_DIR, f"german_ktv_{ts}.mp3")

    logger.info("ElevenLabs TTS (timings) | voice_id=%s speed=%.2f | %.60s", voice_id, speed, text)
    result = client.text_to_speech.convert_with_timestamps(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=_voice_settings(speed),
    )
    # SDK >= 2.x returns audio_base_64 (str); older versions returned .audio (bytes)
    if hasattr(result, "audio_base_64") and result.audio_base_64:
        audio_bytes = base64.b64decode(result.audio_base_64)
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
    else:
        save(result.audio, output_file)
    logger.info("Audio (with timings) saved -> %s", output_file)

    word_timings = _character_alignment_to_word_timings(text, result.alignment)
    return output_file, word_timings


def _character_alignment_to_word_timings(original_text: str, alignment) -> list:
    """Convert ElevenLabs character alignment -> clean word timings."""
    if not alignment:
        return _fallback_timings(original_text)

    # SDK >= 2.x: alignment is a CharacterAlignmentResponseModel object with direct attributes
    # Older SDK: alignment is a plain dict -- handle both
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

    text: str       = state["example_sentence_de"]
    full_tweet: str = state.get("full_tweet", "")
    style: str      = config.VIDEO_STYLE

    # Grow the voice pool passively (no-op once it reaches TARGET_POOL_SIZE)
    from services.voice_pool import grow_pool, TARGET_POOL_SIZE
    pool = grow_pool(language=_VOICE_LANGUAGE, target_size=TARGET_POOL_SIZE)
    ui_info(f"Voice pool: {len(pool)} voices available.")

    if not pool:
        raise RuntimeError(
            "Voice pool is empty and could not be populated. "
            "Check ELEVENLABS_API_KEY and network connectivity."
        )

    if full_tweet:
        ui_info("Selecting voice with AI ...")
        voice_name, voice_id = _pick_voice_by_ai(full_tweet, pool)
    else:
        voice_name, voice_id = _pick_random_voice(pool)
        ui_info(f"Voice: {voice_name}")
    logger.info("Selected voice: %s (%s)", voice_name, voice_id)

    try:
        if style == "ktv":
            audio_path, word_timings = generate_german_audio_with_timings(text, voice_id=voice_id)
            ok(f"Audio + {len(word_timings)} word timings -> {os.path.basename(audio_path)}")
            return {**state, "clean_audio_path": audio_path, "word_timings": word_timings}
        else:
            audio_path = generate_german_audio(text, voice_id=voice_id)
            ok(f"Audio -> {os.path.basename(audio_path)}")
            return {**state, "clean_audio_path": audio_path, "word_timings": []}

    except Exception as exc:
        logger.warning("ElevenLabs failed (%s). Attempting fallback to last known audio.", exc)
        existing = sorted(
            [f for f in os.listdir(VOICES_DIR) if f.endswith(".mp3")],
            reverse=True,
        )
        if existing:
            fallback_path = os.path.join(VOICES_DIR, existing[0])
            ui_warn(f"ElevenLabs unavailable -- using fallback audio: {os.path.basename(fallback_path)}")
            logger.warning("Using fallback audio: %s", fallback_path)
            return {**state, "clean_audio_path": fallback_path, "word_timings": []}
        raise
