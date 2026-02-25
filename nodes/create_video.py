"""
Node: create_video

1. Combines voice MP3 with background music (pydub).
2. Creates the final MP4 — either a simple static video or a ktv-style video
   with word highlighting, depending on VIDEO_STYLE config.
"""

import os
import logging
from datetime import datetime
from pydub import AudioSegment
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, ColorClip

from config import VIDEO_STYLE, BACKGROUND_MUSIC_PATH, VOICES_MUSIC_DIR, VIDEOS_DIR, KTV_FONT
from utils.ui import stage_banner, ok, warn as ui_warn

_FONT = KTV_FONT

logger = logging.getLogger("german_bot.create_video")

os.makedirs(VOICES_MUSIC_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


# ── combine_audio (copied from combine_audio.py) ──────────────────────────────

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


# ── simple video (copied from audio_video1.py) ────────────────────────────────

def create_simple_video(image_path: str, audio_path: str) -> str:
    """Create a static image + audio MP4. Returns video path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(VIDEOS_DIR, f"{ts}.mp4")
    logger.info("Creating simple video → %s", video_path)

    audio = AudioFileClip(audio_path)
    video = (
        ImageClip(image_path)
        .with_duration(audio.duration)
        .with_fps(24)
        .with_audio(audio)
    )
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


# ── ktv video (word-highlighting style) ──────────────────────────────────────

def create_ktv_video(
    image_path: str,
    audio_path: str,
    german_text: str,
    word_timings: list = None,
) -> str:
    """
    Create a KTV-style MP4 with word-by-word highlighting synced to audio.
    Returns video path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(VIDEOS_DIR, f"ktv_{ts}.mp4")
    logger.info("Creating KTV video → %s", output_path)

    audio = AudioFileClip(audio_path)
    duration = audio.duration

    video = ImageClip(image_path).with_duration(duration).with_fps(24)

    # Semi-transparent bar at the bottom — 180 px tall
    _BAR_H  = 180
    # Text area: 90 % of frame width, fixed height (2 lines at font_size 58 ≈ 148 px).
    # Left-edge x so both clips are anchored identically → yellow prefix always
    # starts at the same pixel as the first character of the white sentence.
    _TEXT_W = int(video.w * 0.90)
    _TEXT_H = 148
    _TEXT_X = int(video.w * 0.05)          # 5 % left margin
    _TEXT_Y = video.h - _TEXT_H - 20       # 20 px bottom padding inside the bar

    bar = (
        ColorClip(size=(video.w, _BAR_H), color=(0, 0, 0))
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

    clips = [video, bar, full_text]

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
            clips.append(highlight)

    final = CompositeVideoClip(clips).with_audio(audio)
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
    logger.info("KTV video ready: %s", output_path)
    return output_path


# ── node ──────────────────────────────────────────────────────────────────────

def create_video(state: dict) -> dict:
    stage_banner(6)
    logger.info("Node: create_video")

    clean_audio: str = state["clean_audio_path"]
    image_path: str = state["image_path"]
    german_text: str = state["example_sentence_de"]
    word_timings: list = state.get("word_timings", [])
    style: str = VIDEO_STYLE

    # Step 1: Mix voice with background music
    if not os.path.exists(BACKGROUND_MUSIC_PATH):
        ui_warn("Background music not found — using voice-only audio.")
        logger.warning("Background music not found at '%s'. Using voice-only audio.", BACKGROUND_MUSIC_PATH)
        mixed_audio = clean_audio
    else:
        mixed_audio = combine_audio(
            clean_audio,
            BACKGROUND_MUSIC_PATH,
            gain_reduction=-7,
            fade_out_duration=1350,
        )
        ok("Audio mixed with background music")

    # Step 2: Render video
    if style == "ktv":
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
