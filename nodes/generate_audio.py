"""
Node: generate_audio

Generates German TTS audio via ElevenLabs.
Uses generate_german_audio_with_timings() for ktv mode,
or generate_german_audio() for simple mode.
"""

import os
import base64
import logging
from datetime import datetime

from config import VIDEO_STYLE, VOICES_DIR, ELEVENLABS_API_KEY
from utils.retry import with_retry
from utils.ui import stage_banner, ok, warn as ui_warn

from elevenlabs.client import ElevenLabs
from elevenlabs import save
from elevenlabs.types import VoiceSettings

logger = logging.getLogger("german_bot.generate_audio")

_VOICES = {
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "Sarah":   "EXAVITQu4vr4xnSDxMaL",
    "Serena":  "pMsXgVXv3BLzUgSXRplE",
    "Freya":   "jsCqWAovK2LkecY7zXl4",
    "Adam":    "pNInz6obpgDQGcFmaJgB",
}

_DEFAULT_VOICE = "Matilda"
_DEFAULT_SPEED = 0.70

os.makedirs(VOICES_DIR, exist_ok=True)


def _get_client() -> ElevenLabs:
    if not ELEVENLABS_API_KEY:
        raise ValueError("❌ ELEVENLABS_API_KEY not found in .env!")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


def _voice_settings(speed: float) -> VoiceSettings:
    return VoiceSettings(stability=0.75, similarity_boost=0.85, speed=speed)


@with_retry(max_attempts=4, base_delay=3.0, label="elevenlabs_simple")
def generate_german_audio(
    text: str,
    output_file: str = None,
    voice: str = _DEFAULT_VOICE,
    speed: float = _DEFAULT_SPEED,
) -> str:
    """Generate German TTS audio. Returns path to saved MP3."""
    client = _get_client()
    if voice not in _VOICES:
        logger.warning("Unknown voice '%s', falling back to Matilda.", voice)
        voice = _DEFAULT_VOICE
    if not (0.7 <= speed <= 1.2):
        logger.warning("Speed %.2f out of range, clamping to 0.70.", speed)
        speed = 0.70

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(VOICES_DIR, f"german_{ts}.mp3")

    logger.info("ElevenLabs TTS | voice=%s speed=%.2f | %.60s", voice, speed, text)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=_VOICES[voice],
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=_voice_settings(speed),
    )
    save(audio, output_file)
    logger.info("Audio saved → %s", output_file)
    return output_file


@with_retry(max_attempts=4, base_delay=3.0, label="elevenlabs_timings")
def generate_german_audio_with_timings(
    text: str,
    output_file: str = None,
    voice: str = _DEFAULT_VOICE,
    speed: float = 0.70,
) -> tuple:
    """
    Generate German TTS audio with word-level timings.
    Returns (audio_path, word_timings).
    word_timings = [{'word': str, 'start': float, 'end': float}, ...]
    """
    client = _get_client()
    if voice not in _VOICES:
        voice = _DEFAULT_VOICE

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(VOICES_DIR, f"german_ktv_{ts}.mp3")

    logger.info("ElevenLabs TTS (timings) | voice=%s speed=%.2f | %.60s", voice, speed, text)
    result = client.text_to_speech.convert_with_timestamps(
        text=text,
        voice_id=_VOICES[voice],
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
    logger.info("Audio (with timings) saved → %s", output_file)

    word_timings = _character_alignment_to_word_timings(text, result.alignment)
    return output_file, word_timings


def _character_alignment_to_word_timings(original_text: str, alignment) -> list:
    """Convert ElevenLabs character alignment → clean word timings."""
    if not alignment:
        return _fallback_timings(original_text)

    # SDK >= 2.x: alignment is a CharacterAlignmentResponseModel object with direct attributes
    # Older SDK: alignment is a plain dict — handle both
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


# ── node ──────────────────────────────────────────────────────────────────────

def generate_audio(state: dict) -> dict:
    stage_banner(5)
    logger.info("Node: generate_audio")

    text: str = state["example_sentence_de"]
    style: str = VIDEO_STYLE

    try:
        if style == "ktv":
            audio_path, word_timings = generate_german_audio_with_timings(text)
            ok(f"Audio + {len(word_timings)} word timings → {os.path.basename(audio_path)}")
            return {**state, "clean_audio_path": audio_path, "word_timings": word_timings}
        else:
            audio_path = generate_german_audio(text)
            ok(f"Audio → {os.path.basename(audio_path)}")
            return {**state, "clean_audio_path": audio_path, "word_timings": []}

    except Exception as exc:
        logger.warning("ElevenLabs failed (%s). Attempting fallback to last known audio.", exc)
        existing = sorted(
            [f for f in os.listdir(VOICES_DIR) if f.endswith(".mp3")],
            reverse=True,
        )
        if existing:
            fallback_path = os.path.join(VOICES_DIR, existing[0])
            ui_warn(f"ElevenLabs unavailable — using fallback audio: {os.path.basename(fallback_path)}")
            logger.warning("Using fallback audio: %s", fallback_path)
            return {**state, "clean_audio_path": fallback_path, "word_timings": []}
        raise
