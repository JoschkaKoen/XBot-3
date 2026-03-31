"""
Service: wan_video

Generates a ~5-second animated video from a still image using the local
Wan2.1 model via Wan2GP.

================================================================================
 SETUP REQUIREMENTS
================================================================================
  1. Clone and set up Wan2GP:  https://github.com/deepbeepmeep/Wan2GP
  2. Set WAN_VIDEO_DIR in settings.env to the Wan2GP directory, e.g.:
       WAN_VIDEO_DIR=/home/user/Programming/Wan2GP
  3. Set ENABLE_VIDEO=WAN2.1 in settings.env.
  4. A dedicated venv inside Wan2GP/venv/ is used automatically if present.

This service drives wgp.py directly (no intermediate run_i2v.py needed):
  1. Writes a wgp config JSON to WAN_VIDEO_DIR/_i2v_config/
  2. Writes a task JSON and calls wgp.py as a subprocess (Wan2GP venv)
  3. Detects the newly produced MP4 via before/after set comparison
  4. Scores the video with open_clip via a second Wan2GP venv subprocess
  5. Appends generation params + reward scores to data/wan_video_history.jsonl

Motion prompt and cycle-frequency gate are shared with grok_video, so both
engines are drop-in swappable and VIDEO_FREQUENCY governs both.
================================================================================
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger("german_bot.wan_video")

# ── Video reward scorer ───────────────────────────────────────────────────────
# Executed inside Wan2GP's venv (which has torch, av, open_clip installed).
# Passed verbatim to `python -c`. Reads video_path and prompt from sys.argv.
# Prints a single JSON object to stdout.
_SCORE_CODE = """\
import sys, json, torch, av, open_clip
from PIL import Image as _PIL

def _score(video_path, prompt, n_frames=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    step = max(1, total // n_frames)
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            frames.append(frame.to_image())
        if len(frames) >= n_frames:
            break
    container.close()
    if not frames:
        print(json.dumps({"clip_score": 0.0, "temporal_consistency": 0.0, "motion_score": 0.0}))
        return
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
    print(json.dumps({
        "clip_score":           round(clip_score, 4),
        "temporal_consistency": round(temporal, 4),
        "motion_score":         round(1.0 - temporal, 4),
    }))

_score(sys.argv[1], sys.argv[2])
"""


# ── motion prompt (re-exported so callers need only one import) ───────────────

def build_motion_prompt(
    example_en: str,
    midjourney_prompt: str,
    *,
    engine: str = "wan2.1",
    image_style: str = "photographic",
) -> str:
    """Delegate to grok_video.build_motion_prompt with Wan2.1-specific defaults."""
    from services.grok_video import build_motion_prompt as _build
    return _build(example_en, midjourney_prompt, engine=engine, image_style=image_style)


# ── cycle-frequency gate (shared state with grok_video) ──────────────────────

def should_generate_video() -> bool:
    """Return True if this cycle should generate a Wan video."""
    from services.grok_video import should_generate_video as _should
    return _should()


def advance_cycle() -> None:
    """Advance the shared video cycle counter."""
    from services.grok_video import advance_cycle as _advance
    _advance()


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── main generator ────────────────────────────────────────────────────────────

def _prepare_wan_image(src: Path, target_w: int, target_h: int) -> Path:
    """
    Return a path to an image that is exactly *target_w* × *target_h*.

    If *src* is already the right size it is returned unchanged.
    Otherwise a centre-crop + resize copy is saved next to the original with
    a ``_wan{W}x{H}`` suffix and that path is returned.

    Centre-crop preserves composition: scale so the short side fills the
    frame, then crop the excess from the long side symmetrically.
    """
    from PIL import Image as _PIL

    with _PIL.open(src) as im:
        iw, ih = im.size

    if iw == target_w and ih == target_h:
        return src

    # Scale so both dimensions are >= target, then centre-crop.
    scale  = max(target_w / iw, target_h / ih)
    new_w  = round(iw * scale)
    new_h  = round(ih * scale)
    left   = (new_w - target_w) // 2
    top    = (new_h - target_h) // 2

    with _PIL.open(src) as im:
        resized  = im.convert("RGB").resize((new_w, new_h), _PIL.LANCZOS)
        cropped  = resized.crop((left, top, left + target_w, top + target_h))

    out = src.parent / f"{src.stem}_wan{target_w}x{target_h}{src.suffix}"
    cropped.save(out, format="PNG")
    logger.info(
        "_prepare_wan_image: %s (%dx%d) → %s (%dx%d)",
        src.name, iw, ih, out.name, target_w, target_h,
    )
    return out


def generate_video(image_path: str, motion_prompt: str) -> str:
    """
    Animate *image_path* with Wan2.1 using *motion_prompt*.

    Drives wgp.py directly: writes wgp config + task JSON, runs wgp.py as a
    subprocess (Wan2GP venv), detects the new MP4, scores it with open_clip
    (second Wan2GP venv subprocess), and appends a record to
    data/wan_video_history.jsonl.

    Returns the local path to the generated MP4.
    """
    import config as _cfg

    wan_dir    = _wan_dir()
    python     = _find_venv_python(wan_dir)
    wgp_script = wan_dir / "wgp.py"

    if not wgp_script.exists():
        raise FileNotFoundError(
            f"wgp.py not found at {wgp_script}\n"
            "Make sure WAN_VIDEO_DIR points to a valid Wan2GP installation."
        )

    steps  = getattr(_cfg, "WAN_VIDEO_STEPS",  20)
    frames = getattr(_cfg, "WAN_VIDEO_FRAMES", 81)

    # Output dir: XBot's Videos folder (wgp.py respects --output-dir).
    outputs_dir = Path(_cfg.VIDEOS_DIR).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Write wgp config ──────────────────────────────────────────────
    config_dir = wan_dir / "_i2v_config"
    config_dir.mkdir(exist_ok=True)
    wgp_cfg = {
        "transformer_quantization": "int8",
        "text_encoder_quantization": "int8",
        "profile":        2,
        "video_profile":  2,
        "attention_mode": "sage",
        "save_path":       str(outputs_dir),
        "image_save_path": str(outputs_dir),
        "audio_save_path": str(outputs_dir),
        "boost": 1,
        "enable_int8_kernels": 0,
        "lm_decoder_engine": "",
        "compile":         "",
        "mmaudio_mode":    0,
        "transformer_types": [],
    }
    (config_dir / "wgp_config.json").write_text(
        json.dumps(wgp_cfg, indent=2), encoding="utf-8"
    )

    # ── Step 2: Prepare input image at WAN2.1's exact resolution ─────────────
    # Parse target resolution from the task config ("832x480" → 832, 480).
    _wan_resolution = "832x480"
    _wan_w, _wan_h  = (int(x) for x in _wan_resolution.split("x"))

    image_abs = _prepare_wan_image(Path(image_path).resolve(), _wan_w, _wan_h)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now().isoformat(timespec="seconds")

    task = {
        "model_type":            "i2v",
        "model_filename":        "wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors",
        "resolution":            _wan_resolution,
        "video_length":          frames,
        "num_inference_steps":   steps,
        "image_start":           str(image_abs),
        "prompt":                motion_prompt,
        "skip_steps_cache_type": "",
        "force_fps":             str(config.VIDEO_FPS),
    }
    settings_file = wan_dir / f"_i2v_task_{timestamp}.json"
    settings_file.write_text(json.dumps(task, indent=2), encoding="utf-8")

    cmd = [
        python, str(wgp_script),
        "--process",               str(settings_file),
        "--output-dir",            str(outputs_dir),
        "--config",                str(config_dir),
        "--perc-reserved-mem-max", "0.85",
        "--preload",               "1500",
        "--verbose",               "1",
    ]

    logger.info("Starting Wan2.1 I2V generation (%d steps, %d frames) …", steps, frames)
    logger.info("  Image : %s", image_path)
    logger.info("  Prompt: %s", motion_prompt[:100])

    before = set(outputs_dir.glob("*.mp4"))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(wan_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        settings_file.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"wgp.py exited with code {result.returncode}")

    # ── Step 3: Find new MP4 ──────────────────────────────────────────────────
    after      = set(outputs_dir.glob("*.mp4"))
    new_videos = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_videos:
        raise RuntimeError(f"wgp.py finished but no new MP4 found in {outputs_dir}")

    video_file     = new_videos[-1]
    video_path_str = str(video_file.resolve())
    size_mb        = video_file.stat().st_size / 1024 / 1024
    logger.info("Wan video ready → %s (%.1f MB)", video_file.name, size_mb)

    # ── Step 4: Score video (Wan2GP venv) ─────────────────────────────────────
    logger.info("Scoring video with open_clip …")
    video_reward: dict | None = None
    reward_error: str | None  = None
    try:
        score_proc = subprocess.run(
            [python, "-c", _SCORE_CODE, video_path_str, motion_prompt],
            cwd=str(wan_dir),
            capture_output=True,
            text=True,
        )
        if score_proc.returncode == 0 and score_proc.stdout.strip():
            video_reward = json.loads(score_proc.stdout.strip())
            logger.info(
                "  CLIP score: %.4f  temporal: %.4f  motion: %.4f",
                video_reward["clip_score"],
                video_reward["temporal_consistency"],
                video_reward["motion_score"],
            )
        else:
            reward_error = score_proc.stderr.strip() or f"exit code {score_proc.returncode}"
            logger.warning("Video scoring failed: %s", reward_error)
    except Exception as exc:
        reward_error = str(exc)
        logger.warning("Video scoring exception: %s", exc)

    # ── Step 5: Append JSONL history ──────────────────────────────────────────
    record = {
        "engine":             "WAN2.1",
        "started_at":         started_at,
        "finished_at":        datetime.now().isoformat(timespec="seconds"),
        "input_image":        str(image_abs),
        "prompt":             motion_prompt,
        "cli":                {"steps": steps, "frames": frames},
        "output_video_path":  video_path_str,
        "video_reward":       video_reward,
        "video_reward_error": reward_error,
    }
    history_file = Path(getattr(_cfg, "WAN_VIDEO_HISTORY_FILE", "data/wan_video_history.jsonl"))
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("History appended → %s", history_file)

    return video_path_str
