"""
Service: grok_video

Generates an 8-second animated video from a still image using the
xAI Grok Imagine API (grok-imagine-video model), then returns the
path to the downloaded MP4.

Daily gate: only one Grok video generation is made per calendar day.
The gate is persisted to data/video_state.json alongside the scaffold
state so all daily counters live in one place.
"""

import base64
import json
import logging
import os
import time
from pathlib import Path

import requests

from utils.ui import info, ok

logger = logging.getLogger("xbot.grok_video")

_XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
_BASE_URL = "https://api.x.ai/v1"
_VIDEO_MODEL = "grok-imagine-video"
_VIDEO_STATE_FILE = "data/video_state.json"
_VIDEOS_DIR = "Videos"  # mirrored from config to avoid a circular import


# ── cycle-frequency gate ───────────────────────────────────────────────────────

def _load_state() -> dict:
    from utils.io import safe_json_read
    return safe_json_read(_VIDEO_STATE_FILE, default={}, logger=logger)


def _save_state(data: dict) -> None:
    os.makedirs(os.path.dirname(_VIDEO_STATE_FILE), exist_ok=True)
    with open(_VIDEO_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def should_generate_video() -> bool:
    """Return True if this cycle should generate a Grok video.

    The cycle index stored in state is 0-based; video is generated when it
    equals 0 (i.e., on the first cycle after the counter rolls over).
    """
    from config import GROK_VIDEO_FREQUENCY
    if GROK_VIDEO_FREQUENCY <= 1:
        return True
    data = _load_state()
    cycle_index = data.get("grok_video_cycle_index", 0)
    return cycle_index == 0


def advance_cycle() -> None:
    """Advance the cycle counter by 1, rolling over at GROK_VIDEO_FREQUENCY.

    Must be called once per cycle (regardless of whether a video was generated).
    """
    from config import GROK_VIDEO_FREQUENCY
    data = _load_state()
    current = data.get("grok_video_cycle_index", 0)
    data["grok_video_cycle_index"] = (current + 1) % max(GROK_VIDEO_FREQUENCY, 1)
    _save_state(data)
    logger.info(
        "Grok video cycle advanced: index now %d (frequency=%d)",
        data["grok_video_cycle_index"], GROK_VIDEO_FREQUENCY,
    )


# ── motion prompt generation ──────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert cinematographer and AI video prompt engineer. "
    "You write motion descriptions for image-to-video (I2V) diffusion models. "
    "You understand that I2V models animate a single still frame — they cannot "
    "add new objects, change the scene, or teleport the camera. "
    "Your prompts describe ONLY physically plausible motion that grows "
    "naturally from the existing image composition.\n\n"
    "Output structure (3 layers, each one short clause):\n"
    "  1. CAMERA — one slow, continuous move (dolly, truck, crane, push-in, or static)\n"
    "  2. SUBJECT — the main figure's subtle living motion (breathing, blinking, shifting weight, gesture)\n"
    "  3. ENVIRONMENT — ambient background motion (wind, water, light, particles, cloth)\n\n"
    "Rules:\n"
    "- Every element must be slow, gentle, and continuous — no sudden cuts, whip pans, or fast zooms\n"
    "- Describe motion the model can infer from the still frame (e.g. hair blowing IF wind is visible)\n"
    "- Never request new objects, scene changes, text overlays, or impossible physics\n"
    "- Never mention 'camera' by brand or technical specs — just the movement type\n"
    "- Use present tense, active voice\n"
    "- Output ONLY the motion description — no preamble, labels, numbering, or quotes"
)


def build_motion_prompt(
    example_en: str,
    midjourney_prompt: str,
    *,
    engine: str = "grok",
    image_style: str = "photographic",
) -> str:
    """
    Use the LLM to generate a cinematic motion prompt for I2V generation.

    The prompt is tailored to the video *engine* (grok vs wan2.1) and the
    *image_style* (photographic vs disney) so the motion feels natural for
    the source material and stays within each engine's strengths.
    """
    from services.ai_client import get_ai_response

    # Engine-specific constraints
    if engine == "wan2.1":
        engine_note = (
            "Target engine: Wan2.1 (local I2V diffusion, 480p, ~5 s clip).\n"
            "Wan2.1 excels at: slow dolly/push-in, natural hair & cloth physics, "
            "water ripples, flickering light, gentle parallax.\n"
            "Wan2.1 struggles with: fast motion, large camera orbits, full-body walking, "
            "complex hand gestures, multiple independently moving subjects.\n"
            "Keep motion minimal and grounded — less is more."
        )
    else:
        engine_note = (
            "Target engine: Grok Imagine (API, 720p, 8 s clip).\n"
            "Grok handles broader motion well: slow tracking shots, gentle subject "
            "animation, atmospheric effects. Still avoid fast or erratic movement."
        )

    # Style-specific tone guidance
    if image_style == "disney":
        style_note = (
            "The image is a 3D CGI / Pixar-Disney animated scene. "
            "Motion should feel like a held frame from an animated feature: "
            "slightly exaggerated but smooth character motion (a slow blink, "
            "a gentle head tilt, a soft smile forming), with lush environmental "
            "animation (leaves drifting, light rays shifting, dust motes floating). "
            "Keep the storybook warmth — nothing jarring."
        )
    else:
        style_note = (
            "The image is photorealistic / editorial photography. "
            "Motion should feel like a locked-off cinema camera that barely moves: "
            "a very slow push-in or static shot with shallow depth-of-field rack. "
            "Subject motion is restrained and lifelike (breathing, a micro-expression, "
            "weight shift). Environmental motion is naturalistic (wind in foliage, "
            "steam rising, light flicker through clouds)."
        )

    user_msg = (
        f"IMAGE DESCRIPTION:\n{midjourney_prompt}\n\n"
        f"SCENE CONTEXT (the sentence this image illustrates):\n{example_en}\n\n"
        f"ENGINE:\n{engine_note}\n\n"
        f"STYLE:\n{style_note}\n\n"
        "Write the motion prompt now. Keep it to 2–3 short sentences covering "
        "camera, subject, and environment motion — nothing else."
    )

    prompt = get_ai_response(
        user_msg, _SYSTEM_PROMPT, max_tokens=180, temperature=0.6
    ).strip()

    # Strip stray labels/quotes the LLM may have added despite instructions
    for prefix in ("Camera:", "Subject:", "Environment:", "Motion:", "Prompt:"):
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].strip()
    prompt = prompt.strip('"\'')

    logger.info("Motion prompt (%s/%s): %s", engine, image_style, prompt)
    return prompt


# ── xAI video API helpers ─────────────────────────────────────────────────────

def _headers() -> dict:
    if not _XAI_API_KEY:
        raise ValueError("XAI_API_KEY not set — cannot call Grok Imagine video API.")
    return {
        "Authorization": f"Bearer {_XAI_API_KEY}",
        "Content-Type": "application/json",
    }


def _image_to_data_url(image_path: str) -> str:
    """Read a local image file and return a base64 data URL."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"


def _submit_generation(image_path: str, motion_prompt: str, duration: int = 8) -> str:
    """Submit an image-to-video request; returns the request_id."""
    payload = {
        "model": _VIDEO_MODEL,
        "prompt": motion_prompt,
        "image": {"url": _image_to_data_url(image_path)},   # API expects an object
        "duration": duration,
        "aspect_ratio": "16:9",
        "resolution": "720p",
    }
    resp = requests.post(
        f"{_BASE_URL}/videos/generations",
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    request_id = data.get("request_id") or data.get("id")
    if not request_id:
        raise RuntimeError(f"xAI video API returned no request_id: {data}")
    logger.info("Grok video submitted, request_id=%s", request_id)
    info(f"Grok Imagine job queued (request_id={request_id[:8]}…)")
    return request_id


def _poll_generation(request_id: str, timeout_sec: int = 600, interval: int = 5) -> str:
    """Poll the status endpoint until done; returns the video URL."""
    url = f"{_BASE_URL}/videos/{request_id}"
    start = time.time()
    dots = 0
    while time.time() - start < timeout_sec:
        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
            if resp.status_code >= 500:
                # Transient server error — wait and retry
                time.sleep(interval)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("Poll request error (%s) — retrying", exc)
            time.sleep(interval)
            continue
        data = resp.json()
        status = data.get("status", "")

        if status == "done":
            print()
            # Response shape: {"status":"done","video":{"url":"https://..."}}
            video_url = (
                (data.get("video") or {}).get("url")
                or data.get("video_url")
                or data.get("url")
            )
            if not video_url:
                raise RuntimeError(f"Grok video done but no URL in response: {data}")
            logger.info("Grok video ready: %s", video_url)
            return video_url

        if status in ("expired", "failed"):
            print()
            raise RuntimeError(f"Grok video generation {status}: {data}")

        dots += 1
        print(
            f"\r  ⏳  Generating video with Grok Imagine{'.' * (dots % 4):<4}",
            end="", flush=True,
        )
        time.sleep(interval)

    raise TimeoutError(f"Grok video generation timed out after {timeout_sec}s")


def _download_video(url: str) -> str:
    """Download video from URL to the Videos/ folder; returns local path."""
    from datetime import datetime
    os.makedirs(_VIDEOS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_VIDEOS_DIR, f"grok_{ts}.mp4")
    info("Downloading Grok video …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info("Grok video downloaded → %s (%.1f MB)", path, size_mb)
    ok(f"Grok video saved ({size_mb:.1f} MB) → {os.path.basename(path)}")
    return path


# ── public interface ──────────────────────────────────────────────────────────

def generate_video(image_path: str, motion_prompt: str, duration: int = 8) -> str:
    """
    Animate a still image using the Grok Imagine API.

    Args:
        image_path:    Path to the source PNG/JPEG (the ImageReward-selected image).
        motion_prompt: Cinematic motion description (generated by build_motion_prompt).
        duration:      Clip length in seconds (default 8, max 10).

    Returns:
        Local path to the downloaded silent MP4.
    """
    request_id = _submit_generation(image_path, motion_prompt, duration)
    video_url = _poll_generation(request_id)
    return _download_video(video_url)
