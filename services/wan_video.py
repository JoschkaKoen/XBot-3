"""
Service: wan_video

Generates a ~5-second animated video from a still image using the local
Wan2.1 model via Wan2GP (run_i2v.py).  The motion prompt is produced by the
same AI call used for the Grok video path, so the two engines are drop-in
swappable from create_video.py's perspective.

Cycle-frequency gate (should_generate_video / advance_cycle) is shared with
grok_video so a single VIDEO_FREQUENCY setting governs both engines.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("german_bot.wan_video")


# ── motion prompt (re-exported from grok_video so callers need only one import) ─

def build_motion_prompt(example_en: str, midjourney_prompt: str) -> str:
    """Delegate to grok_video.build_motion_prompt — identical logic for both engines."""
    from services.grok_video import build_motion_prompt as _build
    return _build(example_en, midjourney_prompt)


# ── cycle-frequency gate (shared state with grok_video) ──────────────────────

def should_generate_video() -> bool:
    """Return True if this cycle should generate a Wan video."""
    from services.grok_video import should_generate_video as _should
    return _should()


def advance_cycle() -> None:
    """Advance the shared video cycle counter."""
    from services.grok_video import advance_cycle as _advance
    _advance()


# ── Wan2GP runner ─────────────────────────────────────────────────────────────

def _wan_dir() -> Path:
    import config
    d = Path(config.WAN_VIDEO_DIR)
    if not d.exists():
        raise FileNotFoundError(
            f"Wan2GP directory not found: {d}\n"
            "Set WAN_VIDEO_DIR in settings.env to the correct path."
        )
    return d


def _find_venv_python(wan_dir: Path) -> str:
    venv_py = wan_dir / "venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def generate_video(image_path: str, motion_prompt: str) -> str:
    """
    Animate *image_path* with Wan2.1 using *motion_prompt*.

    Calls run_i2v.py inside the Wan2GP directory as a subprocess so it runs
    in the correct venv and working directory.  Streams all output to the
    console in real time.

    Returns the local path to the generated MP4.
    """
    wan_dir   = _wan_dir()
    python    = _find_venv_python(wan_dir)
    script    = wan_dir / "run_i2v.py"

    if not script.exists():
        raise FileNotFoundError(
            f"run_i2v.py not found at {script}\n"
            "Make sure the Wan2GP directory contains run_i2v.py."
        )

    cmd = [
        python, str(script),
        str(Path(image_path).resolve()),
        motion_prompt,
        "--no-nvfp4",   # use int8 (no RTX 50xx required)
    ]

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }

    logger.info("Starting Wan2.1 I2V generation …")
    logger.info("  Image : %s", image_path)
    logger.info("  Prompt: %s", motion_prompt[:100])

    result = subprocess.run(
        cmd,
        cwd=str(wan_dir),
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Wan2GP exited with code {result.returncode}")

    # run_i2v.py saves to XBot 3/Videos/ if it exists, otherwise Wan2GP/outputs/.
    # Check the XBot Videos dir first, then fall back to Wan2GP outputs/.
    import config as _cfg
    xbot_videos = Path(_cfg.VIDEOS_DIR).resolve()
    outputs_dir = xbot_videos if xbot_videos.exists() else wan_dir / "outputs"
    mp4_files = sorted(outputs_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4_files:
        raise RuntimeError(f"Wan2GP finished but no MP4 found in {outputs_dir}")

    video_path = str(mp4_files[0])
    size_mb = os.path.getsize(video_path) / 1024 / 1024
    logger.info("Wan video ready → %s (%.1f MB)", os.path.basename(video_path), size_mb)
    return video_path
