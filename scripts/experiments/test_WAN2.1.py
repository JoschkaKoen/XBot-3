#!/usr/bin/env python3
"""
test_WAN2.1.py — Wan2.1 Image-to-Video  (480p, 8 GB VRAM)

Run from the XBot 3 project root.  Reads WAN_VIDEO_DIR from settings.env
(or the environment) to locate the Wan2GP installation, then drives wgp.py
directly with the standard 480p 14B int8 parameters.

Usage:
    <WAN_VENV>/python test_WAN2.1.py <image_path> "<prompt>" [options]

Examples:
    /home/y/Programming/Wan2GP/venv/bin/python test_WAN2.1.py Images/photo.png "waves on the beach"
    /home/y/Programming/Wan2GP/venv/bin/python test_WAN2.1.py Images/photo.png "subtle motion" --steps 20 --seed 42
    /home/y/Programming/Wan2GP/venv/bin/python test_WAN2.1.py Images/photo.png "subtle motion" --steps 20 --frames 81

Requirements:
    WAN_VIDEO_DIR must point to a Wan2GP installation that contains:
      - wgp.py
      - venv/  (with torch, av, open_clip installed)
      - the wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors model

    Set WAN_VIDEO_DIR in settings.env (auto-loaded) or export it as an
    environment variable.  No edits to this script are needed.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Load settings.env so WAN_VIDEO_DIR is available without manual export ─────
# Uses only stdlib — python-dotenv is not installed in Wan2GP's venv.
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_settings_env = _PROJECT_DIR / "settings.env"
if _settings_env.exists():
    with open(_settings_env, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Wan2GP location ───────────────────────────────────────────────────────────
_wan_dir_raw = os.getenv(
    "WAN_VIDEO_DIR",
    str(Path.home() / "Programming" / "Wan2GP"),
)
WAN_DIR = Path(_wan_dir_raw).resolve()

if not WAN_DIR.exists():
    print(
        f"Error: Wan2GP directory not found: {WAN_DIR}\n"
        "Set WAN_VIDEO_DIR in settings.env or as an environment variable.",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR      = _PROJECT_DIR / "Videos"                          # XBot Videos folder
CONFIG_DIR       = WAN_DIR      / "_i2v_config"                     # wgp config written here
RUN_HISTORY_FILE = _PROJECT_DIR / "data" / "wan_video_history.jsonl"  # XBot history

MODEL_FILENAME = "wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors"
RESOLUTION     = "832x480"
PRELOAD_MB     = 1500
PERC_RESERVED  = 0.85


# ── Video reward scorer ───────────────────────────────────────────────────────

def score_video(video_path: Path, prompt: str, n_frames: int = 16) -> dict:
    """
    Evaluate a generated video using open_clip.

    Returns a dict with:
      clip_score           – mean frame–prompt cosine similarity (0–1).
      temporal_consistency – mean consecutive-frame similarity (0–1).
      motion_score         – 1 - temporal_consistency (0–1).

    Requires torch, av, open_clip — available in Wan2GP's venv.
    Run this script with Wan2GP's venv python:
        WAN_VIDEO_DIR/venv/bin/python run_i2v.py ...
    """
    import torch
    import av
    import open_clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or 0
    step = max(1, total // n_frames)
    frames: list[Image.Image] = []
    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            frames.append(frame.to_image())
        if len(frames) >= n_frames:
            break
    container.close()

    if not frames:
        return {"clip_score": 0.0, "temporal_consistency": 0.0, "motion_score": 0.0}

    frame_tensors = torch.stack([preprocess(f) for f in frames]).to(device)
    with torch.no_grad():
        frame_feats = model.encode_image(frame_tensors)
        frame_feats = frame_feats / frame_feats.norm(dim=-1, keepdim=True)

    text_tokens = tokenizer([prompt]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    clip_score = float((frame_feats @ text_feat.T).mean().cpu())

    if len(frame_feats) > 1:
        sims = (frame_feats[:-1] * frame_feats[1:]).sum(dim=-1)
        temporal = float(sims.mean().cpu())
    else:
        temporal = 1.0

    return {
        "clip_score":            round(clip_score, 4),
        "temporal_consistency":  round(temporal, 4),
        "motion_score":          round(1.0 - temporal, 4),
    }


# ── wgp helpers ───────────────────────────────────────────────────────────────

def find_python() -> str:
    """Return the Wan2GP venv python, falling back to the current interpreter."""
    venv = WAN_DIR / "venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def write_wgp_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    cfg = {
        "transformer_quantization": "int8",
        "text_encoder_quantization": "int8",
        # Profile 2 = HighRAM_LowVRAM: 64 GB RAM + 8 GB VRAM.
        "profile": 2,
        "video_profile": 2,
        "attention_mode": "sage",
        "save_path":       str(OUTPUTS_DIR),
        "image_save_path": str(OUTPUTS_DIR),
        "audio_save_path": str(OUTPUTS_DIR),
        # boost=1: fuses attention + FF into one GPU pass.
        "boost": 1,
        "enable_int8_kernels": 0,
        "lm_decoder_engine": "",
        "compile":         "",
        "mmaudio_mode":    0,
        "transformer_types": [],
    }
    with open(CONFIG_DIR / "wgp_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def build_task(image_path: Path, prompt: str, steps: int, seed: int, frames: int) -> dict:
    params: dict = {
        "model_type":            "i2v",
        "model_filename":        MODEL_FILENAME,
        "resolution":            RESOLUTION,
        "video_length":          frames,
        "num_inference_steps":   steps,
        "image_start":           str(image_path),
        "prompt":                prompt,
        "skip_steps_cache_type": "",
    }
    if seed >= 0:
        params["seed"] = seed
    return params


def append_run_history(record: dict) -> None:
    """Append one JSON line per run to RUN_HISTORY_FILE."""
    RUN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── banner ────────────────────────────────────────────────────────────────────

def _banner_line(label: str, value: object, width: int = 22) -> None:
    print(f"  {label:<{width}}  {value}")


_PROFILE_NAMES = {
    1: "HighRAM_HighVRAM",
    2: "HighRAM_LowVRAM",
    3: "LowRAM_HighVRAM",
    4: "LowRAM_LowVRAM",
}


def _profile_label(profile_val: object) -> str:
    try:
        pi = int(float(profile_val))
    except (TypeError, ValueError):
        return str(profile_val)
    name = _PROFILE_NAMES.get(pi, "")
    return f"{profile_val}  ({name})" if name else str(profile_val)


def print_run_banner(
    *,
    started_at: str,
    image_path: Path,
    prompt: str,
    args: argparse.Namespace,
    task: dict,
    wgp_config: dict,
    preload_mb: int,
    perc_reserved: float,
    fps: int,
) -> None:
    prof_str = _profile_label(wgp_config.get("profile"))

    tea = (task.get("skip_steps_cache_type") or "").strip()
    if tea:
        tea_mult  = task.get("skip_steps_multiplier", "")
        tea_start = task.get("skip_steps_start_step_perc", "")
        tea_str = f"{tea}  (×{tea_mult}, start {tea_start}%)" if tea_mult != "" else tea
    else:
        tea_str = "off"

    seed_disp = args.seed if args.seed >= 0 else "(random)"

    print()
    print("═" * 64)
    print("  Wan2.1 I2V  ·  Wan2GP  ·  480p 14B int8")
    print("═" * 64)
    print()
    print("  RUN")
    _banner_line("started",          started_at)
    _banner_line("history (append)", RUN_HISTORY_FILE)
    print()
    print("  PATHS")
    _banner_line("project root",     _PROJECT_DIR)
    _banner_line("Wan2GP dir",       WAN_DIR)
    print()
    print("  INPUT")
    _banner_line("image",  image_path)
    _banner_line("prompt", prompt)
    print()
    print("  VIDEO")
    _banner_line("resolution",           RESOLUTION)
    _banner_line("frames",               args.frames)
    _banner_line(f"duration @{fps} fps", f"~{args.frames / fps:.2f} s")
    _banner_line("output folder",        OUTPUTS_DIR)
    print()
    print("  MODEL")
    _banner_line("model_type",   task.get("model_type"))
    _banner_line("weights file", MODEL_FILENAME)
    print()
    print("  SAMPLING")
    _banner_line("denoising steps", args.steps)
    _banner_line("seed",            seed_disp)
    _banner_line("TeaCache",        tea_str)
    print()
    print("  MEMORY / SPEED")
    _vp = wgp_config.get("video_profile")
    _p  = wgp_config.get("profile")
    if _vp == _p:
        _banner_line("MMGP profile", prof_str)
    else:
        _banner_line("MMGP profile", _profile_label(_p))
        _banner_line("video profile", _profile_label(_vp))
    _banner_line("attention",          wgp_config.get("attention_mode"))
    _banner_line("transformer quant",  wgp_config.get("transformer_quantization"))
    _banner_line("text encoder quant", wgp_config.get("text_encoder_quantization"))
    _banner_line("boost",              wgp_config.get("boost"))
    _banner_line("preload (MB)",       preload_mb)
    _banner_line("reserved RAM cap",   perc_reserved)
    _banner_line("int8 Triton kernels",
                 "disabled" if not wgp_config.get("enable_int8_kernels") else "on")
    print()
    print("  WGP CONFIG")
    _banner_line("config dir", CONFIG_DIR)
    _banner_line("verbose",    "1")
    print("═" * 64)
    print()
    sys.stdout.flush()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wan2.1 480p 14B Image-to-Video — run from XBot 3 project root"
    )
    parser.add_argument("image",   help="Input image path (JPG/PNG)")
    parser.add_argument("prompt",  help="Motion prompt")
    parser.add_argument("--steps",  type=int, default=20,
                        help="Denoising steps (default: 20)")
    parser.add_argument("--frames", type=int, default=81,
                        help="Frames (default: 81 = ~5 s @ 16 fps)")
    parser.add_argument("--seed",   type=int, default=-1,
                        help="Seed (-1 = random)")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    wgp_script = WAN_DIR / "wgp.py"
    if not wgp_script.exists():
        print(
            f"Error: wgp.py not found at {wgp_script}\n"
            "Check that WAN_VIDEO_DIR in settings.env is correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    wgp_config = write_wgp_config()

    task = build_task(image_path, args.prompt, args.steps, args.seed, args.frames)

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_file = WAN_DIR / f"_i2v_task_{timestamp}.json"

    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)

        cmd = [
            find_python(), str(wgp_script),
            "--process",               str(settings_file),
            "--output-dir",            str(OUTPUTS_DIR),
            "--config",                str(CONFIG_DIR),
            "--perc-reserved-mem-max", str(PERC_RESERVED),
            "--preload",               str(PRELOAD_MB),
            "--verbose",               "1",
        ]

        fps = 16
        print_run_banner(
            started_at=started_at,
            image_path=image_path,
            prompt=args.prompt,
            args=args,
            task=task,
            wgp_config=wgp_config,
            preload_mb=PRELOAD_MB,
            perc_reserved=PERC_RESERVED,
            fps=fps,
        )

        before = set(OUTPUTS_DIR.glob("*.mp4"))

        result = subprocess.run(
            cmd,
            cwd=str(WAN_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        finished_at  = datetime.now().isoformat(timespec="seconds")
        video_out:    str | None  = None
        video_reward: dict | None = None
        reward_error: str | None  = None

        if result.returncode == 0:
            after      = set(OUTPUTS_DIR.glob("*.mp4"))
            new_videos = sorted(after - before, key=lambda p: p.stat().st_mtime)
            if new_videos:
                video     = new_videos[-1]
                video_out = str(video.resolve())

                print("\n" + "─" * 60)
                print("  Video Reward Evaluation  (open_clip ViT-B-32)")
                print("─" * 60)
                try:
                    scores       = score_video(video, args.prompt)
                    video_reward = scores
                    print(f"  CLIP score            : {scores['clip_score']:.4f}  "
                          f"(prompt alignment — higher is better)")
                    print(f"  Temporal consistency  : {scores['temporal_consistency']:.4f}  "
                          f"(frame smoothness  — higher is better)")
                    print(f"  Motion score          : {scores['motion_score']:.4f}  "
                          f"(amount of motion  — higher means more movement)")
                except Exception as exc:
                    reward_error = str(exc)
                    print(f"  (Scoring failed: {exc})", file=sys.stderr)
                print("─" * 60)

                if not os.environ.get("WAN_I2V_NO_OPEN"):
                    print(f"\n  Opening: {video.name}")
                    subprocess.Popen(["xdg-open", str(video)])
            else:
                print("\n  (No new .mp4 found.)", file=sys.stderr)

        record = {
            "engine":       "WAN2.1",
            "started_at":   started_at,
            "finished_at":  finished_at,
            "wgp_exit_code": result.returncode,
            "reward_model": "open_clip ViT-B-32 laion2b_s34b_b79k",
            "input_image":  str(image_path),
            "prompt":       args.prompt,
            "cli": {
                "steps":  args.steps,
                "frames": args.frames,
                "seed":   args.seed,
            },
            "task":       task,
            "wgp_config": wgp_config,
            "wgp_cli_flags": {
                "perc_reserved_mem_max": str(PERC_RESERVED),
                "preload_mb":            PRELOAD_MB,
                "output_dir":            str(OUTPUTS_DIR),
                "config_dir":            str(CONFIG_DIR),
                "verbose":               1,
            },
            "wgp_command":          cmd,
            "output_video_path":    video_out,
            "output_video_filename": Path(video_out).name if video_out else None,
            "video_reward":         video_reward,
            "video_reward_error":   reward_error,
            "run_history_file":     str(RUN_HISTORY_FILE.resolve()),
        }
        append_run_history(record)
        print(f"\n  Run log appended → {RUN_HISTORY_FILE}")

        sys.exit(result.returncode)

    finally:
        if settings_file.exists():
            settings_file.unlink()


if __name__ == "__main__":
    main()
