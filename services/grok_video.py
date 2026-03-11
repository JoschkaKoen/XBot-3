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

logger = logging.getLogger("german_bot.grok_video")

_XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
_BASE_URL = "https://api.x.ai/v1"
_VIDEO_MODEL = "grok-imagine-video"
_VIDEO_STATE_FILE = "data/video_state.json"
_VIDEOS_DIR = "Videos"  # mirrored from config to avoid a circular import


# ── cycle-frequency gate ───────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        with open(_VIDEO_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {}


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

def build_motion_prompt(example_en: str, midjourney_prompt: str) -> str:
    """
    Use the Grok text API to generate a concise cinematic motion prompt
    suitable for image-to-video generation.
    """
    from services.ai_client import get_ai_response

    user_msg = (
        f"Image description: {midjourney_prompt}\n\n"
        f"Scene context: {example_en}\n\n"
        "Write a SHORT, vivid motion prompt (max 2 sentences) for animating this still "
        "image into an 8-second cinematic video clip. Focus on realistic, subtle motion "
        "that matches the scene: camera movement, subject animation, environmental motion "
        "(wind, light, steam, etc.). "
        "Output ONLY the motion description — no preamble, no quotes."
    )
    system = (
        "You are a cinematographer writing motion descriptions for AI video generation. "
        "Be specific and concise. Describe camera movement and subject motion naturally."
    )
    prompt = get_ai_response(user_msg, system, max_tokens=120, temperature=0.7).strip()
    logger.info("Motion prompt: %s", prompt)
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
    print(f"  ✉   Grok Imagine job queued (request_id={request_id[:8]}…)", flush=True)
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
    print(f"  ⬇   Downloading Grok video …", flush=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info("Grok video downloaded → %s (%.1f MB)", path, size_mb)
    print(f"  ✅  Grok video saved ({size_mb:.1f} MB) → {os.path.basename(path)}", flush=True)
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
