"""
Node: create_video

1. Combines voice MP3 with background music (pydub).
2. Creates the final MP4:
   - If ENABLE_VIDEO=grok: calls Grok Imagine to animate the image (8 s, 720p)
   - If ENABLE_VIDEO=wan:  calls local Wan2.1 via Wan2GP (~5 s, 480p)
   - Both animated paths apply the same KTV text overlay on the result.
   - Otherwise (ENABLE_VIDEO=off):
       • Creates a static KTV or simple video from the still image
         (existing behaviour, unchanged)
"""

import os
import logging
from datetime import datetime
from pydub import AudioSegment
import numpy as np
from PIL import Image as _PILImage
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
from config import BACKGROUND_MUSIC_PATH, VOICES_MUSIC_DIR, VIDEOS_DIR, KTV_FONT
from utils.ui import stage_banner, ok, warn as ui_warn

_FONT = KTV_FONT

logger = logging.getLogger("german_bot.create_video")

os.makedirs(VOICES_MUSIC_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


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

def _build_ktv_overlay_clips(
    base_clip,
    duration: float,
    german_text: str,
    word_timings: list,
) -> list:
    """
    Return the list of overlay clips (bar + text + highlights) to composite
    on top of *base_clip*.  Does NOT include base_clip itself.

    Args:
        base_clip:    The underlying ImageClip or VideoFileClip (already sized).
        duration:     Final video duration in seconds.
        german_text:  Full German sentence shown as the base caption.
        word_timings: Optional list of {word, start, end} dicts for highlighting.
    """
    _BAR_H  = 180
    _TEXT_W = int(base_clip.w * 0.90)
    _TEXT_H = 148
    _TEXT_X = int(base_clip.w * 0.05)
    _TEXT_Y = base_clip.h - _TEXT_H - 20

    bar = (
        ColorClip(size=(base_clip.w, _BAR_H), color=(0, 0, 0))
        .with_opacity(0.68)
        .with_position(("center", "bottom"))
        .with_duration(duration)
    )

    full_text = (
        TextClip(
            font=_FONT,
            text=german_text,
            font_size=58,
            color="#9FD8E8",
            stroke_color="black",
            stroke_width=3,
            size=(_TEXT_W, _TEXT_H),
            method="caption",
            text_align="left",
            horizontal_align="left",
            vertical_align="top",
        )
        .with_position((_TEXT_X, _TEXT_Y))
        .with_duration(duration)
    )

    overlays = [bar, full_text]

    if word_timings:
        logger.info("Adding %d KTV word highlights", len(word_timings))
        for i, wt in enumerate(word_timings):
            prefix = " ".join(w["word"] for w in word_timings[: i + 1])
            start_t = wt["start"]
            end_t = (
                word_timings[i + 1]["start"] if i + 1 < len(word_timings) else duration
            )
            highlight = (
                TextClip(
                    font=_FONT,
                    text=prefix,
                    font_size=58,
                    color="white",
                    stroke_color="black",
                    stroke_width=3,
                    size=(_TEXT_W, _TEXT_H),
                    method="caption",
                    text_align="left",
                    horizontal_align="left",
                    vertical_align="top",
                )
                .with_position((_TEXT_X, _TEXT_Y))
                .with_start(start_t)
                .with_end(end_t)
            )
            overlays.append(highlight)

    return overlays


# ── Ken Burns effect (PIL AFFINE, sub-pixel smooth) ──────────────────────────

# Zoom and pan constants — kept here so they're easy to tune.
_KB_ZOOM_START = 1.0    # 1.0 = full frame visible at start
_KB_ZOOM_END   = 1.08   # 1.08 = 8 % zoom-in at end
_KB_PAN_X      = 0.3    # fraction of available slack to drift rightward
_KB_PAN_Y      = 0.2    # fraction of available slack to drift downward


def _make_ken_burns_clip(image_path: str, duration: float, fps: int = 24) -> VideoClip:
    """
    Return a moviepy VideoClip with Ken Burns slow zoom+pan applied to the image.

    Uses PIL img.transform(AFFINE, BICUBIC) — true float-space sampling, no
    integer rounding, no frame stutter.

    Pan is expressed as a fraction of the available slack (iw * (1 - scale)).
    At ZOOM_START=1.0 slack is 0, so tx=ty=0 regardless of PAN values — the
    view is guaranteed to start exactly at the full image with no offset.
    """
    img = _PILImage.open(image_path).convert("RGB")
    iw, ih = img.size

    def make_frame(t: float) -> np.ndarray:
        progress = max(0.0, min(1.0, t / max(duration, 1.0 / fps)))
        zoom  = _KB_ZOOM_START + (_KB_ZOOM_END - _KB_ZOOM_START) * progress
        scale = 1.0 / zoom

        slack_x = iw * (1.0 - scale)
        slack_y = ih * (1.0 - scale)

        tx = slack_x * (0.5 + _KB_PAN_X * (progress - 0.5))
        ty = slack_y * (0.5 + _KB_PAN_Y * (progress - 0.5))
        tx = max(0.0, min(tx, slack_x))
        ty = max(0.0, min(ty, slack_y))

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

    audio = AudioFileClip(audio_path)
    if config.ENABLE_KEN_BURNS:
        base = _make_ken_burns_clip(image_path, audio.duration)
    else:
        base = ImageClip(image_path).with_duration(audio.duration).with_fps(24)
    video = base.with_audio(audio)
    video.write_videofile(
        video_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
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

    audio    = AudioFileClip(audio_path)
    duration = audio.duration
    if config.ENABLE_KEN_BURNS:
        base = _make_ken_burns_clip(image_path, duration)
    else:
        base = ImageClip(image_path).with_duration(duration).with_fps(24)

    overlays = _build_ktv_overlay_clips(base, duration, german_text, word_timings or [])
    final = CompositeVideoClip([base] + overlays).with_audio(audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
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

    # Use the longer of the two so neither track gets cut off.
    # with_duration() trims if audio_dur < video, or holds last frame if longer.
    duration = max(audio_dur, raw_video.duration)
    base     = raw_video.with_duration(duration).with_fps(24)

    overlays = _build_ktv_overlay_clips(base, duration, german_text, word_timings or [])
    final = CompositeVideoClip([base] + overlays).with_audio(audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
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

    clean_audio: str  = state["clean_audio_path"]
    image_path: str   = state["image_path"]
    german_text: str  = state["example_sentence_source"]
    word_timings: list = state.get("word_timings", [])
    style: str        = config.VIDEO_STYLE

    # ── Step 1: Mix voice with background music ───────────────────────────────
    if not os.path.exists(BACKGROUND_MUSIC_PATH):
        ui_warn("Background music not found — using voice-only audio.")
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

    if config.ENABLE_VIDEO in ("grok", "wan"):
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

                ui_info("  Step 1/3  Generating cinematic motion prompt …")
                motion_prompt = _svc.build_motion_prompt(example_en, mj_prompt)
                ui_info(f"  Motion prompt: {motion_prompt[:80]}{'…' if len(motion_prompt) > 80 else ''}")

                ui_info(f"  Step 2/3  Generating animated video with {engine_label} …")
                animated_video_path = _svc.generate_video(image_path, motion_prompt)
                ok(f"  Step 2/3  Video ready → {os.path.basename(animated_video_path)}")

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
