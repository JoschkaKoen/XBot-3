"""
KTV subtitle renderer — moviepy overlays for the karaoke-style word-by-word
highlight bar at the bottom of the video.

Extracted from nodes/create_video.py so the bar layout, font scaling, and
adaptive 2→3-line overflow logic live in one focused module. The video node
calls build_ktv_overlay_clips() and composites the returned clips on top of
the base video/image.

Reference resolution is 720p (Grok Imagine standard). All dimensions scale
proportionally to the actual base_clip height so the overlay looks consistent
on 480p (Wan2.1 local) and tall static images.
"""

import logging
import textwrap
from typing import Optional

from PIL import Image as _PILImage, ImageFont, ImageDraw
from moviepy import ColorClip, TextClip

import config

logger = logging.getLogger("xbot.ktv_renderer")

# Reference dimensions (at 720p output); scaled to actual frame height at runtime.
_KTV_REF_FRAME_HEIGHT: float = 720.0
_KTV_BAR_H_REF: int = 270             # bar height
_KTV_TEXT_H_REF: int = 222            # text box height
_KTV_FONT_DEFAULT: float = 87.0       # default font size in px
_KTV_STROKE_REF: int = 5
_KTV_BOTTOM_INSET_REF: int = 30

# Adaptive layout when the sentence overflows two lines.
_FONT_SHRINK = 0.75       # first attempt: shrink font to 75 % of configured size
_MAX_LINES   = 3          # never grow the bar past three lines
_LINE_SPACING = 1.35      # line-height multiplier used when sizing the expanded bar


def _scale_factors(base_clip) -> tuple[float, int, int, int, int, int, int, int]:
    """Return overlay metrics scaled from the 720p reference to base_clip's height.

    Returns (scale_factor, bar_h, text_h, font_size, stroke_w, text_w, text_x, text_y).
    """
    user_font = float(config.KTV_FONT_SIZE)
    relative = user_font / _KTV_FONT_DEFAULT
    s = float(base_clip.h) / _KTV_REF_FRAME_HEIGHT
    bar_h = max(1, int(round(_KTV_BAR_H_REF * s * relative)))
    text_h = max(1, int(round(_KTV_TEXT_H_REF * s * relative)))
    font_size = max(12, int(round(user_font * s)))
    stroke_w = max(1, int(round(_KTV_STROKE_REF * s * relative)))
    bottom_inset = max(4, int(round(_KTV_BOTTOM_INSET_REF * s * relative)))
    text_w = int(base_clip.w * 0.90)
    text_x = int(base_clip.w * 0.05)
    text_y = base_clip.h - text_h - bottom_inset
    return s, bar_h, text_h, font_size, stroke_w, text_w, text_x, text_y


def _count_wrapped_lines(text: str, font_path: str, font_size: int, max_width: int) -> int:
    """Count wrapped lines for *text* at *font_size* within *max_width* pixels."""
    try:
        pil_font = ImageFont.truetype(font_path, font_size)
        img = _PILImage.new("RGB", (max_width * 2, font_size * 10))
        draw = ImageDraw.Draw(img)
        words = text.split()
        lines = 0
        current_line = ""
        for word in words:
            test = (current_line + " " + word).strip()
            bbox = draw.textbbox((0, 0), test, font=pil_font)
            if bbox[2] <= max_width:
                current_line = test
            else:
                if current_line:
                    lines += 1
                current_line = word
        if current_line:
            lines += 1
        return max(lines, 1)
    except (OSError, ValueError) as exc:
        logger.warning("PIL textbbox measurement failed (%s) — using char-width fallback.", exc)
        chars_per_line = max(1, int(max_width / (font_size * 0.55)))
        return len(textwrap.wrap(text, width=chars_per_line)) or 1


def build_ktv_overlay_clips(
    base_clip,
    duration: float,
    sentence_text: str,
    word_timings: Optional[list],
) -> list:
    """
    Build the overlay clip list (bar + caption + per-word highlights) for the
    KTV subtitle effect. Caller composites these on top of *base_clip*.

    If the sentence overflows 2 lines at the configured font size, the font is
    shrunk to 75 %. If the text still needs 3 lines, the bar height is expanded
    so the third line is visible rather than cut off.
    """
    font_path = config.KTV_FONT
    s, bar_h, text_h, font_size, stroke_w, text_w, text_x, text_y = _scale_factors(base_clip)

    n_lines = _count_wrapped_lines(sentence_text, font_path, font_size, text_w)

    if n_lines > 2:
        reduced_size = max(12, int(round(font_size * _FONT_SHRINK)))
        n_lines_reduced = _count_wrapped_lines(sentence_text, font_path, reduced_size, text_w)
        if n_lines_reduced <= 2:
            font_size = reduced_size
            stroke_w  = max(1, int(round(stroke_w * _FONT_SHRINK)))
            logger.info(
                "KTV: text needs %d lines at default size — shrinking font to %dpx (fits in 2 lines).",
                n_lines, font_size,
            )
        else:
            font_size = reduced_size
            stroke_w  = max(1, int(round(stroke_w * _FONT_SHRINK)))
            n_lines   = min(n_lines_reduced, _MAX_LINES)
            line_h    = int(round(font_size * _LINE_SPACING))
            extra     = line_h
            bar_h    += extra
            text_h   += extra
            text_y    = base_clip.h - bar_h - max(4, int(round(_KTV_BOTTOM_INSET_REF * s)))
            logger.info(
                "KTV: text needs %d lines even at %dpx — expanding bar to 3 lines.",
                n_lines_reduced, font_size,
            )

    bar = (
        ColorClip(size=(base_clip.w, bar_h), color=(0, 0, 0))
        .with_opacity(0.68)
        .with_position(("center", "bottom"))
        .with_duration(duration)
    )

    full_text = (
        TextClip(
            font=font_path,
            text=sentence_text,
            font_size=font_size,
            color="#9FD8E8",
            stroke_color="black",
            stroke_width=stroke_w,
            size=(text_w, text_h),
            method="caption",
            text_align="left",
            horizontal_align="left",
            vertical_align="top",
        )
        .with_position((text_x, text_y))
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
                    font=font_path,
                    text=prefix,
                    font_size=font_size,
                    color="white",
                    stroke_color="black",
                    stroke_width=stroke_w,
                    size=(text_w, text_h),
                    method="caption",
                    text_align="left",
                    horizontal_align="left",
                    vertical_align="top",
                )
                .with_position((text_x, text_y))
                .with_start(start_t)
                .with_end(end_t)
            )
            overlays.append(highlight)

    return overlays
