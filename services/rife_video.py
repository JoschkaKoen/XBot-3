"""
Service: rife_video

Interpolates a video to a higher frame rate using Practical-RIFE before upload.

Setup is complete:
  - Repo cloned to RIFE_DIR (/home/y/Programming/Practical-RIFE)
  - v4.25 model weights in train_log/
  - skvideo, opencv-python, moviepy installed in Wan2GP venv
  - RIFE_PYTHON points to Wan2GP venv Python (PyTorch 2.10+cu130, RTX 5070 support)

Process (3 ffmpeg/RIFE steps):
    1. ffmpeg: pad video to nearest multiple of 32 (RIFE requirement)
    2. RIFE:   interpolate frames to target FPS (--exp 1 for exact 2x, e.g. 16→32)
    3. ffmpeg: crop back to original size + re-attach audio, encode yuv420p CRF 18
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from subprocess import CalledProcessError

import config

logger = logging.getLogger("german_bot.rife_video")


class RIFENotConfiguredError(RuntimeError):
    """Raised when RIFE_DIR is not set up correctly."""


def _check_setup(rife_dir: Path) -> Path:
    """
    Validate the Practical-RIFE installation and return the Python interpreter path.

    The interpreter is taken from config.RIFE_PYTHON (default: RIFE venv).
    For RTX 5070 / Blackwell GPUs, set RIFE_PYTHON to a Python with PyTorch 2.10+cu130,
    e.g. the Wan2GP venv: /home/y/Programming/Wan2GP/venv/bin/python

    Raises RIFENotConfiguredError with a helpful message if anything is missing.
    """
    python_path = Path(config.RIFE_PYTHON)
    train_log   = rife_dir / "train_log"
    script      = rife_dir / "inference_video.py"

    if not rife_dir.exists():
        raise RIFENotConfiguredError(
            f"RIFE_DIR not found: {rife_dir}\n"
            "  git clone https://github.com/hzwer/Practical-RIFE.git"
        )
    if not python_path.exists():
        raise RIFENotConfiguredError(
            f"RIFE_PYTHON not found: {python_path}\n"
            "  Set RIFE_PYTHON in settings.env to a Python with PyTorch+CUDA installed.\n"
            "  Wan2GP venv works: RIFE_PYTHON=/home/y/Programming/Wan2GP/venv/bin/python"
        )
    if not train_log.exists():
        raise RIFENotConfiguredError(
            f"RIFE train_log/ not found at {train_log}\n"
            "  Download v4.25 model weights into train_log/ (see Practical-RIFE README)."
        )
    if not script.exists():
        raise RIFENotConfiguredError(
            f"inference_video.py not found at {script}\n"
            "  Ensure Practical-RIFE is fully cloned."
        )
    return python_path


def _get_resolution(video_path: str) -> tuple[int, int]:
    """Return (width, height) of video using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"])
    raise RuntimeError(f"Could not detect video resolution: {video_path}")


def interpolate(input_path: str) -> str:
    """
    Interpolate *input_path* to config.VIDEO_UPLOAD_FPS using Practical-RIFE.

    Returns the path to the new interpolated video.
    Raises RIFENotConfiguredError if RIFE is not set up.
    Raises RuntimeError / subprocess.CalledProcessError on processing failure.
    """
    rife_dir   = Path(config.RIFE_DIR)
    target_fps = config.VIDEO_UPLOAD_FPS
    src_fps    = config.VIDEO_FPS

    venv_python = _check_setup(rife_dir)

    input_abs = Path(input_path).resolve()
    if not input_abs.exists():
        raise FileNotFoundError(f"Input video not found: {input_abs}")

    w, h = _get_resolution(str(input_abs))
    pad_w = ((w + 31) // 32) * 32
    pad_h = ((h + 31) // 32) * 32

    output_path = str(input_abs.parent / (input_abs.stem + f"_{target_fps}fps.mp4"))

    logger.info(
        "RIFE interpolation: %s  %dfps → %dfps  (%dx%d → padded %dx%d)",
        input_abs.name, src_fps, target_fps, w, h, pad_w, pad_h,
    )

    # Use a temp dir so partial files are cleaned up even on failure
    with tempfile.TemporaryDirectory(prefix="rife_") as tmp:
        padded   = os.path.join(tmp, "padded.mp4")
        interped = os.path.join(tmp, "interped.mp4")

        def _run(label: str, cmd: list, cwd=None):
            """Run a subprocess, raising with stderr included on failure."""
            try:
                subprocess.run(cmd, check=True, capture_output=True, cwd=cwd)
            except CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"RIFE {label} failed (exit {exc.returncode}):\n{stderr}"
                ) from exc

        # ── Step 1: pad to multiple of 32 ────────────────────────────────────
        print(f"  ⏳  RIFE step 1/3 — padding to {pad_w}×{pad_h} …", flush=True)
        _run("step 1 (ffmpeg pad)", [
            "ffmpeg", "-y", "-i", str(input_abs),
            "-vf", f"pad={pad_w}:{pad_h}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-crf", "0",
            "-an",      # drop audio — RIFE ignores it
            padded,
        ])

        # ── Step 2: RIFE interpolation ────────────────────────────────────────
        # --exp 1 = 2x multiplier (16→32fps exactly).
        # If target_fps is not exactly 2x, fall back to --fps flag.
        if target_fps == src_fps * 2:
            rife_fps_args = ["--exp", "1"]
        else:
            rife_fps_args = ["--fps", str(target_fps)]

        print(f"  ⏳  RIFE step 2/3 — interpolating to {target_fps} fps …", flush=True)
        # --model expects the directory containing flownet.pkl + RIFE_HDv3.py (default: train_log)
        _run("step 2 (RIFE inference)", [
            str(venv_python),
            "inference_video.py",
            "--video",  padded,
            "--output", interped,
            "--model",  "train_log",
            "--scale",  "1.0",
        ] + rife_fps_args, cwd=str(rife_dir))

        # ── Step 3: crop back + re-attach audio + final encode ────────────────
        print(f"  ⏳  RIFE step 3/3 — encoding final video …", flush=True)
        _run("step 3 (ffmpeg encode)", [
            "ffmpeg", "-y",
            "-i", interped,
            "-i", str(input_abs),
            "-vf", f"crop={w}:{h}",
            "-map", "0:v:0",
            "-map", "1:a:0?",       # re-attach audio from original (optional)
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ])

    logger.info("RIFE done → %s", output_path)
    return output_path
