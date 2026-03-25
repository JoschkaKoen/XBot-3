"""
Node: interpolate_video

Interpolates the generated video to VIDEO_UPLOAD_FPS using Practical-RIFE
before it is uploaded to X.

Only active when ENABLE_VIDEO=wan2.1 and VIDEO_INTERPOLATION=true.
Grok videos are already at their native FPS and are passed through unchanged.

On any failure (RIFE not set up, VRAM OOM, etc.) the node logs a warning
and passes the original video path through unchanged — the tweet still goes out.
"""

import logging
import os

import config
from services.rife_video import RIFENotConfiguredError, interpolate
from utils.ui import stage_banner, ok, warn as ui_warn

logger = logging.getLogger("german_bot.interpolate_video")


def interpolate_video(state: dict) -> dict:
    stage_banner(7)   # sits between create_video (6) and publish (8 after this insert)
    logger.info("Node: interpolate_video")

    video_path: str | None = state.get("video_path")

    if not video_path:
        logger.info("interpolate_video: no video_path — skipping.")
        return state

    if config.ENABLE_VIDEO != "wan2.1":
        logger.info("interpolate_video: ENABLE_VIDEO=%s (not wan2.1) — skipping.", config.ENABLE_VIDEO)
        return state

    if not config.VIDEO_INTERPOLATION:
        logger.info("interpolate_video: VIDEO_INTERPOLATION=false — skipping.")
        return state

    print(
        f"  ⏳  Interpolating video {config.VIDEO_FPS} fps → {config.VIDEO_UPLOAD_FPS} fps …",
        flush=True,
    )

    try:
        interpolated_path = interpolate(video_path)
        ok(
            f"  Interpolated video ready → {os.path.basename(interpolated_path)} "
            f"({config.VIDEO_FPS} fps → {config.VIDEO_UPLOAD_FPS} fps)"
        )
        logger.info("interpolate_video: done → %s", interpolated_path)
        return {**state, "video_path": interpolated_path}

    except RIFENotConfiguredError as exc:
        ui_warn(
            f"RIFE not configured — uploading original {config.VIDEO_FPS} fps video. "
            f"({exc})"
        )
        logger.warning("interpolate_video: RIFE not configured: %s", exc)
        return state

    except Exception as exc:
        ui_warn(
            f"RIFE interpolation failed ({exc}) — uploading original "
            f"{config.VIDEO_FPS} fps video."
        )
        logger.warning("interpolate_video: interpolation failed: %s", exc, exc_info=True)
        return state
