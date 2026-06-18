"""
xAI (Grok) Text-to-Speech client — POST https://api.x.ai/v1/tts.

Replaces the former ElevenLabs integration. Reuses the project-wide
``XAI_API_KEY`` (the same key that powers tweets, strategy, images and video
via :mod:`services.ai_client`), so no separate TTS credential is needed.

Two concerns:
  * :func:`list_voices`  — fetch + filter xAI's built-in voice library for the
                           configured source language (native voices first, plus
                           the 5 expressive multilingual voices for variety and
                           guaranteed gender coverage).
  * :func:`tts_to_file`  — synthesize an MP3 (optionally with per-character
                           timings for KTV word-highlighting).

xAI TTS response shapes (verified against the live API):
  * ``with_timestamps=False`` → raw MP3 bytes (Content-Type ``audio/mpeg``).
  * ``with_timestamps=True``  → JSON ``{audio: <base64 mp3>, content_type,
                                 audio_timestamps: {graph_chars: [str],
                                 graph_times: [[start, end], …]}, duration}``.
"""

import base64
import logging

import requests

import config
from services.api_retry import retry_api_call, is_retryable_error
from utils.errors import FatalProviderError

logger = logging.getLogger("xbot.xai_tts")

_BASE        = "https://api.x.ai/v1"
_VOICES_URL  = f"{_BASE}/tts/voices"
_TTS_URL     = f"{_BASE}/tts"
_TIMEOUT     = 60

# Speed bounds accepted by the xAI TTS API.
_SPEED_MIN, _SPEED_MAX = 0.7, 1.5

# The 5 expressive multilingual voices. Always offered (alongside any native
# voices) so the AI voice-picker has both genders to choose from, and used as a
# standalone fallback if the voice library can't be fetched.
_MULTILINGUAL = [
    {"name": "Ara", "voice_id": "ara", "gender": "female", "description": "multilingual, expressive (warm)"},
    {"name": "Eve", "voice_id": "eve", "gender": "female", "description": "multilingual, expressive (energetic)"},
    {"name": "Leo", "voice_id": "leo", "gender": "male",   "description": "multilingual, expressive (authoritative)"},
    {"name": "Rex", "voice_id": "rex", "gender": "male",   "description": "multilingual, expressive (confident)"},
    {"name": "Sal", "voice_id": "sal", "gender": "male",   "description": "multilingual, expressive (balanced)"},
]

_voices_cache: list | None = None   # raw API voice list, fetched once per process


# ── helpers ───────────────────────────────────────────────────────────────────

def _headers() -> dict:
    if not config.XAI_API_KEY:
        raise FatalProviderError("XAI_API_KEY not set — required for xAI TTS.")
    return {"Authorization": f"Bearer {config.XAI_API_KEY}", "Content-Type": "application/json"}


def _is_retryable(exc: BaseException) -> bool:
    """Terminal on auth/4xx; otherwise defer to the shared transient predicate."""
    if isinstance(exc, FatalProviderError):
        return False
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and 400 <= resp.status_code < 500:
            return False
    return is_retryable_error(exc)


def _describe(v: dict) -> str:
    """Build a short voice description from the library metadata."""
    lang = v.get("language")
    parts = [v.get("gender"), v.get("age"),
             "multilingual, expressive" if lang == "multilingual" else lang]
    return ", ".join(p for p in parts if p)


# ── voice library ─────────────────────────────────────────────────────────────

def _fetch_voice_library() -> list:
    """GET /v1/tts/voices once per process; [] (→ fallback) on failure."""
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache

    def _do() -> list:
        r = requests.get(_VOICES_URL, headers=_headers(), timeout=_TIMEOUT)
        if r.status_code in (401, 403):
            raise FatalProviderError(f"xAI auth failed (HTTP {r.status_code}) fetching voices.")
        r.raise_for_status()
        return r.json().get("voices", [])

    try:
        _voices_cache = retry_api_call(_do, label="xai_tts voices", is_retryable=_is_retryable)
    except Exception as exc:
        logger.warning("Voice library fetch failed (%s) — using multilingual fallback.", exc)
        _voices_cache = []
    return _voices_cache


def list_voices(language: str) -> list:
    """
    Return voice dicts ``{name, voice_id, gender, description}`` for the source
    *language*: native-language voices first, then the 5 expressive multilingual
    voices. Falls back to the multilingual set if the library can't be fetched.
    """
    lib = _fetch_voice_library()
    if not lib:
        return [dict(v) for v in _MULTILINGUAL]

    code = (language or "").lower().strip()
    native, multi = [], []
    for v in lib:
        vid = v.get("voice_id")
        if not vid:
            continue
        vlang = (v.get("language") or "").lower()
        entry = {
            "name":        v.get("name") or vid,
            "voice_id":    vid,
            "gender":      (v.get("gender") or "").lower(),
            "description": _describe(v),
        }
        if vlang == "multilingual":
            multi.append(entry)
        elif code and vlang.split("-")[0] == code:   # "de" matches "de"; "sv" matches "sv-SE"
            native.append(entry)

    pool = native + multi
    if pool:
        logger.info("xAI voices for '%s': %d native + %d multilingual.",
                    code, len(native), len(multi))
        return pool
    logger.warning("No voices matched language '%s' — using multilingual fallback.", code)
    return [dict(v) for v in _MULTILINGUAL]


# ── synthesis ─────────────────────────────────────────────────────────────────

def tts_to_file(
    text: str,
    voice_id: str,
    *,
    language: str,
    out_path: str,
    with_timestamps: bool = False,
    speed: float = 1.0,
) -> tuple[str, dict | None]:
    """
    Synthesize *text* with *voice_id* to an MP3 at *out_path*.

    Returns ``(out_path, alignment)`` where *alignment* is ``None`` unless
    *with_timestamps* is set, in which case it is a dict
    ``{"characters": [...], "character_start_times_seconds": [...]}`` —
    the shape consumed by
    ``nodes.generate_audio._character_alignment_to_word_timings``.
    """
    speed = max(_SPEED_MIN, min(_SPEED_MAX, float(speed)))
    payload = {
        "text":            text,
        "voice_id":        voice_id,
        "language":        language or "auto",
        "speed":           speed,
        "with_timestamps": bool(with_timestamps),
        "output_format": {
            "codec":       "mp3",
            "sample_rate": config.TTS_SAMPLE_RATE,
            "bit_rate":    config.TTS_BIT_RATE,
        },
    }

    def _do() -> requests.Response:
        r = requests.post(_TTS_URL, headers=_headers(), json=payload, timeout=_TIMEOUT)
        if r.status_code in (401, 403):
            raise FatalProviderError(f"xAI TTS auth failed (HTTP {r.status_code}).")
        if r.status_code == 400:
            raise FatalProviderError(f"xAI TTS bad request (HTTP 400): {r.text[:200]}")
        r.raise_for_status()
        return r

    resp = retry_api_call(_do, label=f"xai_tts {voice_id}", is_retryable=_is_retryable)

    if not with_timestamps:
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path, None

    data = resp.json()
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(data.get("audio") or ""))

    ts     = data.get("audio_timestamps") or {}
    chars  = ts.get("graph_chars") or []
    times  = ts.get("graph_times") or []
    starts = [(t[0] if isinstance(t, (list, tuple)) and t else 0.0) for t in times]
    alignment = {"characters": chars, "character_start_times_seconds": starts}
    return out_path, alignment
