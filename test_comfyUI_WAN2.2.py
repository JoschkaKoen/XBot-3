#!/usr/bin/env python3
"""
test_comfyUI_WAN2.2.py — Wan2.2 TI2V-5B (720p) via ComfyUI HTTP API
================================================================================

Animates a still image using the Wan 2.2 5B Text-Image-to-Video model running
inside ComfyUI.  ComfyUI handles all model loading and VRAM management; this
script is a thin HTTP client that uploads the image, patches the workflow JSON,
submits the job, polls for completion, copies the MP4, scores it, and logs.

PREREQUISITES (one-time, see setup_comfyui.sh):
  1. ComfyUI installed at ~/ComfyUI with Wan 2.2 5B models in place.
  2. Workflow JSON exported from ComfyUI and saved as:
       XBot 3/workflows/comfyui_wan2.2.json
     (Workflow → Browse Templates → Video → Wan2.2 5B video generation
      → menu → Export (API format))
  3. COMFYUI_DIR and COMFYUI_URL set in settings.env (already done).

USAGE (no changes needed — run from XBot 3 project root):
  # 1. Start ComfyUI in a separate terminal (leave it running):
  #    cd ~/ComfyUI && source venv/bin/activate && python main.py --normalvram --fp16-vae

  # 2. Run this script with any Python (torch not required in this venv):
  python test_comfyUI_WAN2.2.py Images/photo.png "subtle camera drift, leaves rustle"
  python test_comfyUI_WAN2.2.py Images/photo.png "gentle waves" --steps 50 --frames 49
  python test_comfyUI_WAN2.2.py Images/photo.png "cinematic pan" --seed 42

Settings are read automatically from settings.env — no manual exports needed.
================================================================================
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Load settings.env (stdlib only — no python-dotenv needed) ─────────────────
_PROJECT_DIR = Path(__file__).parent.resolve()
_settings_env = _PROJECT_DIR / "settings.env"
if _settings_env.exists():
    with open(_settings_env, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Constants ─────────────────────────────────────────────────────────────────
COMFYUI_URL        = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_DIR        = Path(os.getenv("COMFYUI_DIR", str(Path.home() / "ComfyUI")))
_default_output    = str(Path.home() / "ComfyUI" / "output")
COMFYUI_OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", _default_output))
WAN_VIDEO_DIR = Path(os.getenv("WAN_VIDEO_DIR", str(Path.home() / "Programming" / "Wan2GP")))
WORKFLOW_FILE = _PROJECT_DIR / "workflows" / "comfyui_wan2.2.json"
OUTPUTS_DIR   = _PROJECT_DIR / "Videos"
HISTORY_FILE  = _PROJECT_DIR / "data" / "wan_video_history.jsonl"
ENGINE_TAG    = "ComfyUI-WAN2.2-5B"
BANNER        = "Wan2.2 TI2V-5B  ·  ComfyUI  ·  720p"
FPS           = 16

# ── Inline video scorer (runs in Wan2GP venv which has torch/av/open_clip) ───
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


# ── ComfyUI helpers ───────────────────────────────────────────────────────────

def check_server() -> None:
    """Abort with a clear message if ComfyUI is not reachable."""
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
    except Exception:
        print(
            f"\n  ERROR: ComfyUI server not reachable at {COMFYUI_URL}\n\n"
            "  Start it first (leave running in a separate terminal):\n"
            f"    cd {COMFYUI_DIR} && source venv/bin/activate\n"
            "    python main.py --normalvram --fp16-vae\n",
            file=sys.stderr,
        )
        sys.exit(1)


def upload_image(image_path: Path) -> str:
    """Upload *image_path* to ComfyUI via multipart POST. Returns the server filename."""
    url = f"{COMFYUI_URL}/upload/image"
    boundary = "XBot3ComfyBoundary"
    with open(image_path, "rb") as fh:
        image_data = fh.read()

    suffix = image_path.suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/png")

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["name"]


def patch_workflow(
    workflow: dict,
    image_name: str,
    prompt: str,
    steps: int,
    frames: int,
    seed: int,
) -> dict:
    """
    Patch a ComfyUI API-format workflow dict with runtime values.

    Scans every node by class_type and updates the relevant inputs.
    Works with the standard Wan 2.2 ComfyUI template and similar layouts.
    """
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})

        # Input image
        if ct == "LoadImage":
            inp["image"] = image_name

        # Text prompt — positive encoding nodes only (skip negative prompt nodes)
        meta_title = node.get("_meta", {}).get("title", "").lower()
        if ct in ("CLIPTextEncode", "CLIPTextEncodeWan2", "WanTextEncode",
                  "CLIPTextEncodeWan", "TextEncode"):
            if "negative" not in meta_title and "text" in inp:
                inp["text"] = prompt

        # Sampler: steps and seed
        if ct in ("KSampler", "KSamplerAdvanced", "WanVideoSampler",
                  "SamplerCustom", "SamplerCustomAdvanced"):
            if "steps" in inp:
                inp["steps"] = steps
            if seed >= 0:
                for seed_key in ("seed", "noise_seed"):
                    if seed_key in inp:
                        inp[seed_key] = seed

        # Frame / video length nodes (explicit class types)
        if ct in ("WanVideoLatent", "WanImageToVideo", "WanImageToVideoLatent",
                  "Wan22ImageToVideoLatent", "EmptyHunyuanLatentVideo", "EmptyLatentVideo"):
            for length_key in ("video_length", "length", "num_frames"):
                if length_key in inp and isinstance(inp[length_key], int):
                    inp[length_key] = frames

        # Also patch bare "video_length" / "length" on any non-image node
        if ct not in ("LoadImage",):
            for length_key in ("video_length", "length"):
                if length_key in inp and isinstance(inp[length_key], int):
                    inp[length_key] = frames

        # Output fps — CreateVideo, VHS_VideoCombine, and similar nodes
        if ct in ("CreateVideo", "VHS_VideoCombine", "SaveVideo",
                  "SaveAnimatedWEBP", "SaveAnimatedPNG"):
            for fps_key in ("frame_rate", "fps"):
                if fps_key in inp and isinstance(inp[fps_key], (int, float)):
                    inp[fps_key] = FPS

    return wf


def submit_prompt(workflow: dict) -> str:
    """POST workflow to /prompt, return prompt_id."""
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def poll_until_done(prompt_id: str, poll_interval: int = 5) -> dict:
    """
    Poll /history/{prompt_id} until the job completes or errors.
    Returns the history entry dict.
    """
    dots = 0
    while True:
        time.sleep(poll_interval)
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
            history = json.loads(resp.read())

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print()  # newline after dots
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI reported an error for prompt {prompt_id}: {status}"
                )

        dots += 1
        print(f"\r  ... generating {'.' * (dots % 4):<4}", end="", flush=True)


def find_output_video(history_entry: dict) -> Path | None:
    """
    Locate the generated video file from a completed history entry.
    ComfyUI stores output files under COMFYUI_DIR/output/.
    """
    for node_output in history_entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []):
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                if not filename:
                    continue
                if not filename.lower().endswith((".mp4", ".webm", ".gif")):
                    continue
                candidate = COMFYUI_OUTPUT_DIR / subfolder / filename
                if candidate.exists():
                    return candidate
    return None


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_video(video_path: str, prompt: str) -> tuple[dict | None, str | None]:
    """
    Run the inline scorer in Wan2GP's venv.
    Returns (reward_dict, error_str) — exactly one will be None.
    """
    wan_python = WAN_VIDEO_DIR / "venv" / "bin" / "python"
    if not wan_python.exists():
        return None, f"Wan2GP venv not found at {wan_python}"

    try:
        proc = subprocess.run(
            [str(wan_python), "-c", _SCORE_CODE, video_path, prompt],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip()), None
        error = proc.stderr.strip() or f"exit code {proc.returncode}"
        return None, error
    except Exception as exc:
        return None, str(exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wan2.2 TI2V-5B via ComfyUI — run from XBot 3 project root",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_comfyUI_WAN2.2.py Images/photo.png \"subtle motion\"\n"
            "  python test_comfyUI_WAN2.2.py Images/photo.png \"leaves rustle\" --steps 40\n"
            "  python test_comfyUI_WAN2.2.py Images/photo.png \"gentle pan\" --seed 42\n"
        ),
    )
    parser.add_argument("image",  help="Input image path")
    parser.add_argument("prompt", help="Motion / animation prompt")
    parser.add_argument("--steps",  type=int, default=30,
                        help="Denoising steps (default: 30)")
    parser.add_argument("--frames", type=int, default=80,
                        help="Number of video frames (default: 80 = 5s @ 16 fps)")
    parser.add_argument("--seed",   type=int, default=-1,
                        help="RNG seed (-1 = random, default: -1)")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if not WORKFLOW_FILE.exists():
        print(
            f"\n  ERROR: workflow file not found:\n    {WORKFLOW_FILE}\n\n"
            "  Export it from ComfyUI:\n"
            "    Workflow → Browse Templates → Video → Wan2.2 5B video generation\n"
            "    → menu → Export (API format)\n"
            f"    → save as {WORKFLOW_FILE}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    print(f"\n{'═' * 62}")
    print(f"  {BANNER}")
    print(f"{'═' * 62}\n")
    print(f"  Image : {image_path}")
    print(f"  Prompt: {args.prompt[:80]}")
    print(f"  Steps : {args.steps}   Frames: {args.frames}   FPS: {FPS}   "
          f"Seed: {args.seed if args.seed >= 0 else '(random)'}")
    print(f"  Server: {COMFYUI_URL}")
    print()

    # 1. Verify server
    check_server()
    print("  ComfyUI server: OK")

    # 2. Upload image
    print(f"  Uploading image ({image_path.name}) ...")
    image_name = upload_image(image_path)
    print(f"  Uploaded as: {image_name}")

    # 3. Load and patch workflow
    with open(WORKFLOW_FILE, encoding="utf-8") as fh:
        workflow = json.load(fh)
    workflow = patch_workflow(
        workflow, image_name, args.prompt, args.steps, args.frames, args.seed
    )

    # 4. Submit
    print("  Submitting workflow to ComfyUI ...")
    prompt_id = submit_prompt(workflow)
    print(f"  prompt_id: {prompt_id}")
    print("  ", end="", flush=True)

    # 5. Poll
    history_entry = poll_until_done(prompt_id)
    finished_at = datetime.now().isoformat(timespec="seconds")

    # 6. Retrieve output
    comfy_video = find_output_video(history_entry)
    if comfy_video is None:
        raise RuntimeError(
            "Generation completed but no MP4 found in ComfyUI output.\n"
            f"Check ComfyUI output folder: {COMFYUI_OUTPUT_DIR}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = OUTPUTS_DIR / f"comfyui_wan22_{ts}{comfy_video.suffix}"
    shutil.copy2(comfy_video, dest)
    video_path = str(dest.resolve())
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"\n  Video ready  →  {dest.name}  ({size_mb:.1f} MB)")

    # 7. Score
    print("  Scoring video ...")
    video_reward, reward_error = score_video(video_path, args.prompt)
    if video_reward:
        print(
            f"  CLIP: {video_reward['clip_score']:.4f}  "
            f"temporal: {video_reward['temporal_consistency']:.4f}  "
            f"motion: {video_reward['motion_score']:.4f}"
        )
    else:
        print(f"  (Scoring skipped: {reward_error})")

    # 8. History
    record = {
        "engine":             ENGINE_TAG,
        "started_at":         started_at,
        "finished_at":        finished_at,
        "input_image":        str(image_path),
        "prompt":             args.prompt,
        "cli": {
            "steps":  args.steps,
            "frames": args.frames,
            "fps":    FPS,
            "seed":   args.seed,
        },
        "comfyui_url":        COMFYUI_URL,
        "workflow_file":      str(WORKFLOW_FILE),
        "comfyui_prompt_id":  prompt_id,
        "output_video_path":  video_path,
        "video_reward":       video_reward,
        "video_reward_error": reward_error,
    }
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  History     →  {HISTORY_FILE}")
    print()


if __name__ == "__main__":
    main()
