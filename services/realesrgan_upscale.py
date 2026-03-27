"""
Real-ESRGAN video upscaler service.

Upscales WAN2.1 480p videos to 720p (or any scale) before the KTV/badge
overlay compositing step.

================================================================================
 ONE-TIME SETUP  (non-trivial — read carefully)
================================================================================
  git clone https://github.com/xinntao/Real-ESRGAN.git ~/Programming/Real-ESRGAN
  cd ~/Programming/Real-ESRGAN

  # ⚠  basicsr has a known compatibility issue with modern torchvision:
  #    torchvision.transforms.functional_tensor was removed in recent versions.
  #    The safest fix is to patch basicsr after install:
  #      pip install basicsr facexlib gfpgan -r requirements.txt
  #      python setup.py develop
  #      # Then open the file that errors and replace the broken import, e.g.:
  #      #   from torchvision.transforms.functional import rgb_to_grayscale
  #      # instead of the removed functional_tensor import.
  #
  #  Alternatively create a dedicated venv with pinned versions; put the
  #  venv at Real-ESRGAN/venv/ and this service will use it automatically.

  wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/

Then set in settings.env:
  ENABLE_REALESRGAN=true
  REALESRGAN_DIR=/home/y/Programming/Real-ESRGAN

================================================================================
 VRAM NOTE
================================================================================
  --tile 256 keeps peak VRAM low (well under 4 GB).  fp16 is the default
  precision for inference_realesrgan_video.py — there is no --fp16 flag to
  pass; use --fp32 only if you need to force full precision.
  WAN2.1 has already exited and released its VRAM before this runs, so there
  is no contention.

 OUTSCALE NOTE
================================================================================
  --outscale 1.5 tells Real-ESRGAN to run the neural net at 4× then resize
  the result down to 1.5× with LANCZOS4.  The AI detail enhancement comes
  from the 4× pass; the final resize to 720p is a cheap bicubic/LANCZOS step.
  The output is still noticeably sharper and cleaner than the raw 480p input,
  but "exactly 720p via neural upscale" is an overstatement.
================================================================================
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import config

logger = logging.getLogger("german_bot.realesrgan_upscale")


def _find_python(realesrgan_dir: Path) -> str:
    """Prefer Real-ESRGAN's own venv, fall back to XBot's venv Python."""
    venv_py = realesrgan_dir / "venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def upscale_video(input_path: str) -> str:
    """
    Upscale *input_path* with Real-ESRGAN and return the path to the result.

    The upscaled file is saved alongside the original in the same directory
    with an ``_up`` suffix before the extension, e.g.
    ``2026-video.mp4`` → ``2026-video_up.mp4``.

    If Real-ESRGAN is not available or fails, a RuntimeError is raised so the
    caller can decide whether to fall back to the original.
    """
    realesrgan_dir = Path(config.REALESRGAN_DIR).resolve()
    if not realesrgan_dir.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN directory not found: {realesrgan_dir}\n"
            "Set REALESRGAN_DIR in settings.env to the correct path.\n"
            "Setup: git clone https://github.com/xinntao/Real-ESRGAN.git"
        )

    script = realesrgan_dir / "inference_realesrgan_video.py"
    if not script.exists():
        raise FileNotFoundError(
            f"inference_realesrgan_video.py not found in {realesrgan_dir}.\n"
            "Make sure REALESRGAN_DIR points to the Real-ESRGAN repository root."
        )

    weights_dir = realesrgan_dir / "weights"
    model_file  = weights_dir / f"{config.REALESRGAN_MODEL}.pth"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_file}\n"
            f"Download with:\n"
            f"  wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
            f"{config.REALESRGAN_MODEL}.pth -P {weights_dir}/"
        )

    input_p    = Path(input_path).resolve()
    output_dir = input_p.parent

    # Real-ESRGAN names the output {stem}_out.{ext} by default.
    # We snapshot the directory first so we can detect it robustly even if
    # the naming convention differs between repo versions.
    before = {p.resolve() for p in output_dir.glob("*.mp4")}

    python = _find_python(realesrgan_dir)
    cmd = [
        python,
        str(script),
        "-i",         str(input_p),
        "-o",         str(output_dir),   # folder, not file — Real-ESRGAN saves {stem}_out.mp4 inside
        "-n",         config.REALESRGAN_MODEL,
        "--outscale", str(config.REALESRGAN_OUTSCALE),
        "--tile",     str(config.REALESRGAN_TILE),
        # fp16 is the default — do NOT pass --fp16 (the flag doesn't exist).
        # Pass --fp32 here if you ever need to force full precision.
    ]

    logger.info(
        "Real-ESRGAN: %s → %sx  model=%s  tile=%d",
        input_p.name, config.REALESRGAN_OUTSCALE,
        config.REALESRGAN_MODEL, config.REALESRGAN_TILE,
    )

    result = subprocess.run(cmd, cwd=str(realesrgan_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"Real-ESRGAN exited with code {result.returncode}. "
            f"Check the output above for details."
        )

    # Find newly created file.
    after     = {p.resolve() for p in output_dir.glob("*.mp4")}
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_files:
        raise RuntimeError(
            f"Real-ESRGAN finished successfully but no new MP4 found in {output_dir}."
        )

    upscaled = new_files[-1]
    size_mb  = upscaled.stat().st_size / 1024 / 1024
    logger.info(
        "Real-ESRGAN done: %s (%.1f MB, was %.1f MB)",
        upscaled.name, size_mb, input_p.stat().st_size / 1024 / 1024,
    )
    return str(upscaled)
