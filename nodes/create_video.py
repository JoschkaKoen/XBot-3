"""
Node: create_video

================================================================================
 VIDEO PIPELINE
================================================================================

This node creates the final MP4 video posted to X.  It combines:
    1. Audio: voice recording (generate_audio), optionally mixed with background music
       (ENABLE_BACKGROUND_MUSIC in settings.env — on/off)
    2. Visual: generated image (generate_image) — possibly animated

VIDEO ENGINE OPTIONS (set ENABLE_VIDEO in settings.env):
    "off"    → static image + optional Ken Burns zoom/pan
    "grok"   → Grok Imagine API (8s, 720p) animation
    "WAN2.1" → local Wan2.1 model via Wan2GP (5s, 480p)

KTV OVERLAY:
    The "ktv" VIDEO_STYLE adds a karaoke-style word-by-word highlight bar
    at the bottom of the video, synced to the audio timing.

KEN BURNS:
    ENABLE_KEN_BURNS adds a slow zoom+pan effect to static images (works
    independently of ENABLE_VIDEO).

================================================================================
 RELATED MODULES
================================================================================
  - nodes.generate_audio: Provides clean_audio_path and word_timings
  - nodes.generate_image: Provides image_path and midjourney_prompt
  - services.grok_video:  Grok Imagine I2V video generation
  - services.wan_video:   Wan2.1 local I2V video generation
  - config:               KTV_FONT, VIDEO_STYLE, ENABLE_VIDEO, ENABLE_BACKGROUND_MUSIC, etc.
================================================================================

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   image_path, midjourney_prompt, clean_audio_path,
                      word_timings, example_sentence_source,
                      example_sentence_target, cycle
  Writes to state:    mixed_audio_path, video_path
  Side effects:       writes MP3 to Voices with Background Music/,
                      writes MP4 to Videos/
================================================================================
"""

import io
import math
import os
import logging
from datetime import datetime
from pydub import AudioSegment
import numpy as np
import requests
from PIL import Image as _PILImage, ImageFont, ImageDraw
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    VideoFileClip,
)

import config
from config import (
    BACKGROUND_MUSIC_PATH,
    ENABLE_BACKGROUND_MUSIC,
    VOICES_MUSIC_DIR,
    VIDEOS_DIR,
    KTV_FONT,
)
from services.ktv_renderer import build_ktv_overlay_clips as _build_ktv_overlay_clips
from utils.retry import with_retry
from utils.ui import stage_banner, ok, warn as ui_warn

_FONT = KTV_FONT

logger = logging.getLogger("xbot.create_video")

os.makedirs(VOICES_MUSIC_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


# ── flag badge helpers (composited onto video frames) ────────────────────────

_FLAGCDN_URL    = "https://flagcdn.com/w{w}/{code}.png"
_FLAGCDN_WIDTHS = [20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 320]
_flag_cache: dict = {}   # (code, fetch_w) → PIL Image, avoids re-downloading each cycle


def _flag_emoji_to_country_code(emoji: str) -> str:
    """Extract the ISO 3166-1 alpha-2 country code from a flag emoji (e.g. 🇩🇪 → 'de')."""
    chars = [
        chr(ord(c) - 0x1F1E6 + ord("A"))
        for c in emoji
        if 0x1F1E6 <= ord(c) <= 0x1F1FF
    ]
    return "".join(chars).lower()


@with_retry(max_attempts=3, base_delay=0.1, backoff=1.0, label="fetch_flag")
def _fetch_flag(code: str, desired_width: int) -> "_PILImage":
    """Download a flag PNG from flagcdn.com at the nearest supported width (cached)."""
    fetch_w = next((w for w in _FLAGCDN_WIDTHS if w >= desired_width), 320)
    key = (code.lower(), fetch_w)
    if key not in _flag_cache:
        url = _FLAGCDN_URL.format(w=fetch_w, code=code.lower())
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        _flag_cache[key] = _PILImage.open(io.BytesIO(resp.content)).convert("RGBA")
    return _flag_cache[key].copy()


def _fit_flag(flag: "_PILImage", w: int, h: int) -> "_PILImage":
    """Scale-to-fill: zoom until both dimensions are covered, then centre-crop."""
    ratio = max(w / flag.width, h / flag.height)
    new_w = int(flag.width  * ratio)
    new_h = int(flag.height * ratio)
    flag = flag.resize((new_w, new_h), _PILImage.LANCZOS)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    return flag.crop((x0, y0, x0 + w, y0 + h))


def _create_flag_badge(badge_w: int, badge_h: int) -> "_PILImage":
    """
    Gradient-blended badge: target flag (left) → source flag (right).
    Cosine ease-in/out gradient for a seamless blend.
    """
    from PIL import ImageDraw
    src_code = _flag_emoji_to_country_code(config.SOURCE_FLAG)
    tgt_code = _flag_emoji_to_country_code(config.TARGET_FLAG)

    src_img = _fit_flag(_fetch_flag(src_code, badge_w * 2), badge_w, badge_h).convert("RGB")
    tgt_img = _fit_flag(_fetch_flag(tgt_code, badge_w * 2), badge_w, badge_h).convert("RGB")

    gradient = bytes(
        [int(128 + 127 * math.cos(math.pi * x / max(badge_w - 1, 1)))
         for x in range(badge_w)] * badge_h
    )
    mask  = _PILImage.frombytes("L", (badge_w, badge_h), gradient)
    badge = _PILImage.composite(tgt_img, src_img, mask)

    radius = max(int(badge_h * 0.20), 4)
    # White border drawn before rounding so it is clipped cleanly
    ImageDraw.Draw(badge).rounded_rectangle(
        [0, 0, badge_w - 1, badge_h - 1], radius=radius,
        outline=(255, 255, 255), width=2,
    )

    # Rounded corners via alpha mask
    badge = badge.convert("RGBA")
    corner_mask = _PILImage.new("L", (badge_w, badge_h), 0)
    ImageDraw.Draw(corner_mask).rounded_rectangle(
        [0, 0, badge_w - 1, badge_h - 1], radius=radius, fill=255
    )
    badge.putalpha(corner_mask)

    # 90 % opacity
    r, g, b, a = badge.split()
    a = a.point(lambda v: int(v * 0.90))
    return _PILImage.merge("RGBA", (r, g, b, a))


def _make_badge_clip(frame_w: int, frame_h: int, duration: float, fps: int) -> ImageClip:
    """
    Build a flag-badge ImageClip positioned in the top-right corner.

    Badge metrics mirror the original _overlay_flags() sizing so the badge
    looks identical to how it did when burned into the still image.
    """
    badge_w = max(int(frame_w * 0.10), 100)
    badge_h = int(badge_w * 0.60)
    padding = max(int(frame_w * 0.015), 12)

    badge_img = _create_flag_badge(badge_w, badge_h)
    x = frame_w - badge_w - padding
    y = padding

    return (
        ImageClip(np.array(badge_img), is_mask=False)
        .with_duration(duration)
        .with_position((x, y))
        .with_fps(fps)
    )


# ── combine_audio ─────────────────────────────────────────────────────────────

def combine_audio(
    sentence_file_path: str,
    background_file_path: str,
    gain_reduction: float = -7,
    fade_out_duration: int = 1350,
) -> str:
    """
    Overlay voice MP3 on top of background music.
    Returns the path to the saved combined MP3.
    """
    logger.info("Combining audio: %s + %s", sentence_file_path, background_file_path)

    voice = AudioSegment.from_mp3(sentence_file_path)
    silence = AudioSegment.silent(duration=fade_out_duration)
    voice_padded = voice + silence

    music = AudioSegment.from_mp3(background_file_path)
    music = music.apply_gain(gain_reduction)

    if len(music) > len(voice_padded):
        music = music[: len(voice_padded)]

    music = music.fade_out(fade_out_duration)
    combined = voice_padded.overlay(music)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(VOICES_MUSIC_DIR, f"{ts}.mp3")
    combined.export(filepath, format="mp3")
    logger.info("Combined audio saved → %s", filepath)
    return filepath


# ── KTV overlay builder (shared by both image and video paths) ────────────────
# KTV overlay rendering lives in services/ktv_renderer.py — imported above as
# _build_ktv_overlay_clips. Ken Burns + flag overlay still local to this node.


# ── Ken Burns effect (PIL AFFINE, sub-pixel smooth) ──────────────────────────

# Zoom and pan constants — kept here so they're easy to tune.
# These values create a subtle, professional-looking motion that enhances
# static images without being distracting.
_KB_ZOOM_START = 1.0    # 1.0 = full frame visible at start
_KB_ZOOM_END   = 1.08   # 1.08 = 8 % zoom-in at end
_KB_PAN_X      = 0.3    # fraction of available slack to drift rightward
_KB_PAN_Y      = 0.2    # fraction of available slack to drift downward


def _make_ken_burns_clip(image_path: str, duration: float, fps: int = 24) -> VideoClip:
    """
    Return a moviepy VideoClip with Ken Burns slow zoom+pan applied to the image.

    Uses PIL img.transform(AFFINE, BICUBIC) — true float-space sampling, no
    integer rounding, no frame stutter. This produces buttery-smooth motion
    that looks professional on social media.

    The pan is expressed as a fraction of the available slack (iw * (1 - scale)).
    At ZOOM_START=1.0 slack is 0, so tx=ty=0 regardless of PAN values — the
    view is guaranteed to start exactly at the full image with no offset.
    
    The motion follows a linear interpolation from start to end, creating
    a smooth, predictable zoom that stays centered-ish while drifting slightly.

    Args:
        image_path: Path to the source image file.
        duration: Length of the output video in seconds.
        fps: Frames per second for the output video (default: 24).
    
    Returns:
        A VideoClip with the Ken Burns effect applied, ready for audio overlay.
    """
    img = _PILImage.open(image_path).convert("RGB")
    iw, ih = img.size

    def make_frame(t: float) -> np.ndarray:
        # Calculate progress through the video (0.0 at start, 1.0 at end)
        progress = max(0.0, min(1.0, t / max(duration, 1.0 / fps)))
        
        # Interpolate zoom level: starts at 1.0 (full image), ends at 1.08 (8% zoomed in)
        zoom  = _KB_ZOOM_START + (_KB_ZOOM_END - _KB_ZOOM_START) * progress
        scale = 1.0 / zoom

        # Calculate how much "slack" (unused space) the zoom creates
        # At zoom=1.0, slack=0 (full image fills frame)
        # At zoom=1.08, slack allows 8% of the image to be cropped
        slack_x = iw * (1.0 - scale)
        slack_y = ih * (1.0 - scale)

        # Calculate pan offset: drifts rightward and downward as zoom increases
        # The (progress - 0.5) term makes pan start centered and drift outward
        tx = slack_x * (0.5 + _KB_PAN_X * (progress - 0.5))
        ty = slack_y * (0.5 + _KB_PAN_Y * (progress - 0.5))
        
        # Clamp to valid range (don't pan beyond the image bounds)
        tx = max(0.0, min(tx, slack_x))
        ty = max(0.0, min(ty, slack_y))

        # Apply affine transformation: scale + translate
        # The matrix (scale, 0, tx, 0, scale, ty) defines:
        #   x' = scale * x + tx
        #   y' = scale * y + ty
        frame = img.transform(
            (iw, ih),
            _PILImage.AFFINE,
            (scale, 0, tx, 0, scale, ty),
            resample=_PILImage.BICUBIC,
        )
        return np.array(frame)

    return VideoClip(make_frame, duration=duration).with_fps(fps)


# ── simple video (static image) ───────────────────────────────────────────────

def create_simple_video(image_path: str, audio_path: str) -> str:
    """Create a static image + audio MP4. Returns video path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(VIDEOS_DIR, f"{ts}.mp4")
    logger.info("Creating simple video → %s", video_path)

    fps = config.VIDEO_FPS
    audio = AudioFileClip(audio_path)
    if config.ENABLE_KEN_BURNS:
        base = _make_ken_burns_clip(image_path, audio.duration)
    else:
        base = ImageClip(image_path).with_duration(audio.duration).with_fps(fps)

    if config.FLAG_OVERLAY:
        try:
            badge = _make_badge_clip(base.w, base.h, audio.duration, fps)
            final = CompositeVideoClip([base, badge]).with_audio(audio)
        except Exception as exc:
            logger.warning("Flag badge skipped in simple video — could not fetch flag images: %s", exc)
            final = base.with_audio(audio)
    else:
        final = base.with_audio(audio)

    final.write_videofile(
        video_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        bitrate="8000k",
        preset="medium",
        threads=4,
        logger=None,
    )
    logger.info("Simple video ready: %s", video_path)
    return video_path


# ── KTV video (static image base) ────────────────────────────────────────────

def create_ktv_video(
    image_path: str,
    audio_path: str,
    german_text: str,
    word_timings: list = None,
) -> str:
    """
    Create a KTV-style MP4 with word-by-word highlighting synced to audio.
    Base layer is a still image.  Returns video path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(VIDEOS_DIR, f"ktv_{ts}.mp4")
    logger.info("Creating KTV video (image base) → %s", output_path)

    fps      = config.VIDEO_FPS
    audio    = AudioFileClip(audio_path)
    duration = audio.duration
    if config.ENABLE_KEN_BURNS:
        base = _make_ken_burns_clip(image_path, duration)
    else:
        base = ImageClip(image_path).with_duration(duration).with_fps(fps)

    if config.FLAG_OVERLAY:
        try:
            badge_layers = [_make_badge_clip(base.w, base.h, duration, fps)]
        except Exception as exc:
            logger.warning("Flag badge skipped in KTV video — could not fetch flag images: %s", exc)
            badge_layers = []
    else:
        badge_layers = []
    overlays = _build_ktv_overlay_clips(base, duration, german_text, word_timings or [])
    final = CompositeVideoClip([base] + badge_layers + overlays).with_audio(audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        bitrate="8000k",
        preset="medium",
        threads=4,
        logger=None,
    )
    logger.info("KTV video (image) ready: %s", output_path)
    return output_path


# ── KTV video (Grok animated video base) ─────────────────────────────────────

def create_ktv_video_from_motion(
    base_video_path: str,
    audio_path: str,
    german_text: str,
    word_timings: list = None,
) -> str:
    """
    Apply KTV text overlay on top of a Grok-animated video clip.

    The Grok video is trimmed / padded to match the audio duration so
    neither track is cut short.  Returns the path to the final MP4.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(VIDEOS_DIR, f"ktv_grok_{ts}.mp4")
    logger.info("Creating KTV video (Grok base) → %s", output_path)

    audio         = AudioFileClip(audio_path)
    audio_dur     = audio.duration
    raw_video     = VideoFileClip(base_video_path)

    # Preserve the source video's native FPS for compositing so MoviePy does
    # not drop/duplicate frames. Only the final encode is written at VIDEO_FPS.
    # (Forcing with_fps to a lower value before write caused frame-drops.)
    src_fps   = raw_video.fps
    encode_fps = config.VIDEO_FPS
    duration  = max(audio_dur, raw_video.duration)
    base      = raw_video.with_duration(duration)
    logger.info(
        "create_ktv_video_from_motion: src_fps=%.2f encode_fps=%d duration=%.2fs",
        src_fps, encode_fps, duration,
    )

    if config.FLAG_OVERLAY:
        try:
            badge_layers = [_make_badge_clip(base.w, base.h, duration, src_fps)]
        except Exception as exc:
            logger.warning("Flag badge skipped in KTV motion video — could not fetch flag images: %s", exc)
            badge_layers = []
    else:
        badge_layers = []
    overlays = _build_ktv_overlay_clips(base, duration, german_text, word_timings or [])
    final = CompositeVideoClip([base] + badge_layers + overlays).with_audio(audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=encode_fps,
        bitrate="8000k",
        preset="medium",
        threads=4,
        logger=None,
    )
    logger.info("KTV video (Grok) ready: %s", output_path)
    return output_path


# ── node ──────────────────────────────────────────────────────────────────────

def create_video(state: dict) -> dict:
    stage_banner(6)
    logger.info("Node: create_video")

    # If image generation was skipped (e.g. ComfyUI unavailable), there is
    # nothing to animate.  Return early so the rest of the cycle (publish,
    # score_and_store) can still run without a video.
    image_path: str | None = state.get("image_path")
    if not image_path:
        ui_warn("No image available — skipping video generation for this cycle.")
        logger.warning("create_video: image_path missing — video skipped.")
        return {**state, "video_path": None}

    clean_audio: str  = state["clean_audio_path"]
    german_text: str  = state["example_sentence_source"]
    word_timings: list = state.get("word_timings", [])
    style: str        = config.VIDEO_STYLE

    # ── Step 1: Mix voice with background music (optional) ────────────────────
    if not ENABLE_BACKGROUND_MUSIC:
        mixed_audio = clean_audio
        logger.info("Background music disabled — using voice-only audio for video.")
    elif not os.path.exists(BACKGROUND_MUSIC_PATH):
        ui_warn("Background music enabled but file not found — using voice-only audio.")
        logger.warning(
            "Background music not found at '%s'. Using voice-only audio.",
            BACKGROUND_MUSIC_PATH,
        )
        mixed_audio = clean_audio
    else:
        mixed_audio = combine_audio(
            clean_audio,
            BACKGROUND_MUSIC_PATH,
            gain_reduction=-7,
            fade_out_duration=1350,
        )
        ok("Audio mixed with background music")

    # ── Step 2: Render video ──────────────────────────────────────────────────
    animated_video_path: str | None = None

    if config.ENABLE_VIDEO in ("grok", "wan2.1"):
        _svc = None
        if config.ENABLE_VIDEO == "grok":
            from services import grok_video as _svc
            engine_label = "Grok Imagine"
        else:
            from services import wan_video as _svc
            engine_label = "Wan2.1 (local)"

        freq = config.VIDEO_FREQUENCY
        if not _svc.should_generate_video():
            freq_str = f"every {freq} tweets" if freq > 1 else "every tweet"
            logger.info("%s frequency gate: skipping this cycle (%s).", engine_label, freq_str)
            ui_warn(f"{engine_label}: skipping this cycle ({freq_str}) — using static KTV video.")
            _svc.advance_cycle()
        else:
            try:
                from utils.ui import info as ui_info
                ui_info(f"🎬  {engine_label} video enabled — animating image …")

                example_en: str = state.get("example_sentence_target", "")
                mj_prompt: str  = state.get("midjourney_prompt", "")
                _img_style: str = config.resolve_image_style(state.get("cycle", 0))

                ui_info("  Step 1/3  Generating cinematic motion prompt …")
                motion_prompt = _svc.build_motion_prompt(
                    example_en, mj_prompt, image_style=_img_style
                )
                ui_info(f"  Motion prompt: {motion_prompt[:80]}{'…' if len(motion_prompt) > 80 else ''}")

                ui_info(f"  Step 2/3  Generating animated video with {engine_label} …")
                animated_video_path = _svc.generate_video(image_path, motion_prompt)
                ok(f"  Step 2/3  Video ready → {os.path.basename(animated_video_path)}")

                # ── Real-ESRGAN upscale (WAN2.1 only, before KTV compositing) ──
                if config.ENABLE_VIDEO == "wan2.1" and config.ENABLE_REALESRGAN:
                    from services.realesrgan_upscale import upscale_video
                    ui_info(
                        f"  ⬆   Real-ESRGAN: upscaling 480p → "
                        f"{int(480 * config.REALESRGAN_OUTSCALE):.0f}p …"
                    )
                    try:
                        animated_video_path = upscale_video(animated_video_path)
                        ok(f"  ⬆   Upscaled → {os.path.basename(animated_video_path)}")
                    except Exception as _upscale_exc:
                        logger.warning(
                            "Real-ESRGAN upscale failed (%s) — using original 480p video.",
                            _upscale_exc,
                        )
                        ui_warn(
                            f"Real-ESRGAN upscale failed ({_upscale_exc}) "
                            "— continuing with original 480p video."
                        )

                ui_info("  Step 3/3  Applying KTV overlay on animated video …")
            except Exception as exc:
                logger.warning("%s video generation failed (%s) — falling back to static.", engine_label, exc)
                ui_warn(f"{engine_label} video failed ({exc}) — falling back to static KTV video.")
                animated_video_path = None
            finally:
                _svc.advance_cycle()

    if animated_video_path and style == "ktv":
        video_path = create_ktv_video_from_motion(
            base_video_path=animated_video_path,
            audio_path=mixed_audio,
            german_text=german_text,
            word_timings=word_timings,
        )
        ok(f"  Step 3/3  KTV overlay applied → {os.path.basename(video_path)}")
    elif animated_video_path:
        # Simple style with animated base: overlay audio only, no text
        video_path = create_ktv_video_from_motion(
            base_video_path=animated_video_path,
            audio_path=mixed_audio,
            german_text="",
            word_timings=[],
        )
    elif style == "ktv":
        video_path = create_ktv_video(
            image_path=image_path,
            audio_path=mixed_audio,
            german_text=german_text,
            word_timings=word_timings,
        )
    else:
        video_path = create_simple_video(image_path=image_path, audio_path=mixed_audio)

    ok(f"Video ready → {os.path.basename(video_path)}")
    return {**state, "mixed_audio_path": mixed_audio, "video_path": video_path}
