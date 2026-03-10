#!/usr/bin/env python3
"""
run_i2v.py — Wan2.1 14B Image-to-Video wrapper (RTX 50xx / 8 GB VRAM)

Usage:
    python run_i2v.py <image_path> "<prompt>" [options]

Examples:
    python run_i2v.py photo.jpg "a person walking through a forest"
    python run_i2v.py photo.png "timelapse clouds" --steps 20 --seed 42
    python run_i2v.py photo.png "waves on the beach" --no-nvfp4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
CONFIG_DIR = SCRIPT_DIR / "_i2v_config"


def write_wgp_config(quant: str) -> None:
    """Write wgp_config.json tuned for 8 GB VRAM into the config directory."""
    CONFIG_DIR.mkdir(exist_ok=True)
    cfg = {
        # NVFP4 uses the lightx2v kernel (RTX 50xx only).
        # Fall back to int8 with --no-nvfp4 flag.
        "transformer_quantization": quant,
        "text_encoder_quantization": "int8",
        # Profile 4 = LowRAM_LowVRAM: loads model parts on demand — safest for 8 GB VRAM.
        "profile": 4,
        "video_profile": 4,
        "attention_mode": "sdpa",    # safe default; swap for "sage2" if Triton is installed
        "save_path": str(OUTPUTS_DIR),
        "image_save_path": str(OUTPUTS_DIR),
        "audio_save_path": str(OUTPUTS_DIR),
        "boost": 1,
        "enable_int8_kernels": 0,
        "lm_decoder_engine": "",
        "compile": "",
        "mmaudio_mode": 0,
        "transformer_types": [],
    }
    config_file = CONFIG_DIR / "wgp_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def build_task_settings(image_path: Path, prompt: str, steps: int, seed: int) -> dict:
    """Return the WanGP task params dict for one I2V generation."""
    params: dict = {
        "model_type": "image2video_720p",
        "resolution": "1280x720",
        # 129 frames ≈ 8.06 s at 16 fps
        "video_length": 129,
        "num_inference_steps": steps,
        # image_start must be an absolute path or relative to the WanGP folder
        "image_start": str(image_path),
        "prompt": prompt,
    }
    if seed >= 0:
        params["seed"] = seed
    return params


def find_python() -> str:
    """Return path to the venv Python if present, otherwise the current interpreter."""
    venv_python = SCRIPT_DIR / "venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an 8-second 1280×720 video from an image using Wan2.1 14B I2V"
    )
    parser.add_argument("image", help="Path to the input image (JPG/PNG)")
    parser.add_argument("prompt", help="Text prompt describing the desired motion")
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Denoising steps (default: 30; use 20 for faster/lower quality)"
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Random seed; -1 = random (default)"
    )
    parser.add_argument(
        "--no-nvfp4", action="store_true",
        help="Disable NVFP4 quantization and fall back to int8 "
             "(use if lightx2v kernel is not installed)"
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    quant = "int8" if args.no_nvfp4 else "nvfp4"
    write_wgp_config(quant)

    task = build_task_settings(image_path, args.prompt, args.steps, args.seed)

    # Write a temp settings JSON that WanGP will consume via --process
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_file = SCRIPT_DIR / f"_i2v_task_{timestamp}.json"
    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)

        python_exe = find_python()
        wgp_script = SCRIPT_DIR / "wgp.py"
        if not wgp_script.exists():
            print(
                f"Error: wgp.py not found at {wgp_script}\n"
                "Make sure run_i2v.py is inside the Wan2GP directory.",
                file=sys.stderr,
            )
            sys.exit(1)

        cmd = [
            python_exe,
            str(wgp_script),
            "--process", str(settings_file),
            "--output-dir", str(OUTPUTS_DIR),
            "--config", str(CONFIG_DIR),
            "--verbose", "1",
        ]

        print("\n" + "=" * 60)
        print("  Wan2GP  Image-to-Video  (Wan2.1 14B)")
        print("=" * 60)
        print(f"  Image   : {image_path}")
        print(f"  Prompt  : {args.prompt[:72]}{'…' if len(args.prompt) > 72 else ''}")
        print(f"  Frames  : 129  (~8 s @ 16 fps)")
        print(f"  Size    : 1280×720")
        print(f"  Steps   : {args.steps}")
        print(f"  Quant   : {quant}")
        print(f"  Output  : {OUTPUTS_DIR}")
        print("=" * 60 + "\n")

        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        sys.exit(result.returncode)

    finally:
        if settings_file.exists():
            settings_file.unlink()


if __name__ == "__main__":
    main()
